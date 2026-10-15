"""
异步路由器（AsyncRouter）

基于 asyncio.Queue 的异步消息路由，解决同步 Router 的递归调用栈溢出问题。

核心区别：
- 同步 Router：route_message 直接调 agent.receive()，receive 内部再 send
  又调 route_message（现已改为迭代泵投递，栈深恒定）。
- AsyncRouter：route 只把消息投递到队列就返回（await 让出控制权），
  后台消费协程逐条取出处理。receive 内部再 send 时，新消息回到队列排队，
  不在调用栈上嵌套——并发与长链路场景更自然。

用法：
    async def main():
        router = AsyncRouter()
        router.register_agent(agent_a)
        router.register_agent(agent_b)
        await router.start()                # 启动后台消费协程
        await router.route(message)         # 投递消息（异步）
        await router.drain()                # 等待所有消息处理完
        await router.stop()                 # 停止

兼容性：
- Agent 不需改造即可使用：AsyncRouter 优先调 agent.receive_async()，
  若不存在则回退到同步 receive()（在执行器线程中运行，并复制上下文，
  使消息跳数与 Tracer 追踪在执行器线程中依然可见）。
- 与 Router 接口大体平行；routing_rules / add_routing_rule 尚未提供异步版本。
- 消息携带 hops/max_hops（见 core/message.py），超过上限的转发被丢弃。

关闭语义（stop）：
- 先等待同步桥在途消息入队，再向队列投入关闭哨兵；消费协程处理完
  剩余消息后遇到哨兵退出。因此 stop() 不会在消息未处理完时死等，
  也不会丢弃队列中已入队的消息。
"""

import asyncio
import contextvars
import logging
import threading
from collections import deque
from typing import Any

from .agent import Agent
from .message import Message, reset_message_hops, set_message_hops

logger = logging.getLogger(__name__)

# 消息历史上限（防止长驻进程内存无限增长；超出后丢弃最旧记录）
MAX_MESSAGE_HISTORY = 1000

# 关闭哨兵：消费者取到它即退出（区别于任何合法消息对象）
_SHUTDOWN = object()


class AsyncRouter:
    """
    基于 asyncio.Queue 的异步路由器

    消息投递与处理解耦：route() 入队即返回，后台协程串行处理。
    这样 receive→send→route 不在调用栈上嵌套，而是队列排队。
    """

    def __init__(self, max_concurrent: int = 1):
        """
        Args:
            max_concurrent: 消费协程并发数。默认 1（串行，保证消息顺序与
                            完全无递归）。>1 可并发处理不同消息，但同一
                            消息链仍按队列出队顺序调度。
        """
        self.agents: dict[str, Agent] = {}
        self.message_history: deque[Message] = deque(maxlen=MAX_MESSAGE_HISTORY)
        self.stats: dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "routing_errors": 0,
            "hop_limit_drops": 0,
        }
        self._max_concurrent = max_concurrent
        # 队列与消费任务在 start() 中创建（需要事件循环）
        self._queue: asyncio.Queue | None = None
        self._consumer_tasks: list[asyncio.Task] = []
        self._pending_tasks: set[asyncio.Task] = set()  # 同步桥的入队 task（防 GC）
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None  # start() 时保存
        # drain 的等待条件：队列空且无在处理消息时通知
        self._idle_event: asyncio.Event | None = None
        self._in_flight = 0
        # stats 的读-改-写可能来自调用线程/事件循环/executor 线程，加锁保证原子
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Agent 管理（同步，同 Router）
    # ------------------------------------------------------------------
    def register_agent(self, agent: Agent) -> None:
        """注册 Agent，自动绑定本路由器"""
        self.agents[agent.agent_id] = agent
        agent.router = self  # 复用 Agent.router 字段；Agent.send 会检测异步

    def unregister_agent(self, agent_id: str) -> None:
        if agent_id in self.agents:
            self.agents[agent_id].router = None
            del self.agents[agent_id]

    def get_agent(self, agent_id: str) -> Agent | None:
        return self.agents.get(agent_id)

    def list_agents(self) -> list[str]:
        return list(self.agents.keys())

    def _bump(self, key: str, delta: int = 1) -> None:
        """线程安全地累加统计计数"""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + delta

    def get_router_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = self.stats.copy()
        return {
            "agent_count": len(self.agents),
            "message_history_count": len(self.message_history),
            "stats": stats,
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动后台消费协程（必须在事件循环中调用）"""
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # 初始无消息，视为空闲
        self._running = True
        self._consumer_tasks = [
            asyncio.create_task(self._consumer_loop()) for _ in range(self._max_concurrent)
        ]
        logger.debug("AsyncRouter 已启动，消费协程数=%d", self._max_concurrent)

    async def stop(self) -> None:
        """
        停止路由器（优雅关闭）

        消息链的处理会产生新的在途消息（Agent 处理完一条又转发下一条），
        因此关闭按轮进行：投递哨兵 → 等消费者排空本轮队列 → 检查是否又
        有同步桥消息入队 → 有则重启消费者再来一轮，直到完全清空。
        不会死等，也不丢已入队消息。
        """
        if self._queue is None:
            return  # 未启动或已停止
        self._running = False
        # 先等同步桥已调度的消息入队
        await self._wait_pending_puts()
        while True:
            if not self._consumer_tasks:
                # 消费者已退出但仍有残留消息：重启消费者清空本轮
                self._consumer_tasks = [
                    asyncio.create_task(self._consumer_loop()) for _ in range(self._max_concurrent)
                ]
            for _ in self._consumer_tasks:
                self._queue.put_nowait(_SHUTDOWN)
            results = await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    logger.error("AsyncRouter 消费协程异常退出: %r", r)
            self._consumer_tasks.clear()
            # 让 call_soon_threadsafe 回调有机会执行，再检查是否有新的在途消息
            await asyncio.sleep(0)
            if self._pending_tasks:
                await self._wait_pending_puts()
            if self._queue.qsize() == 0 and not self._pending_tasks:
                break
            logger.debug("AsyncRouter 停止中：仍有 %d 条残留消息，继续排空", self._queue.qsize())
        self._queue = None
        self._idle_event = None
        logger.debug("AsyncRouter 已停止")

    async def _wait_pending_puts(self) -> None:
        """等待同步桥已调度的入队 task 全部完成（新产生的也会被等到）"""
        while self._pending_tasks:
            tasks = list(self._pending_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self) -> None:
        """
        等待队列中所有消息处理完毕（用于同步等待异步结果）

        覆盖同步桥在途消息：等待已调度未入队的 task 入队后，再次确认
        队列清空，避免消息链中途"看起来空闲"导致的提前返回。
        """
        if self._queue is not None:
            await self._queue.join()
            while self._pending_tasks:
                await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)
                await self._queue.join()
        # 等待空闲信号（处理完最后一帧）
        if self._idle_event is not None:
            await self._idle_event.wait()

    # ------------------------------------------------------------------
    # 路由（异步，核心）
    # ------------------------------------------------------------------
    def route_message(self, message: Message) -> bool:
        """
        同步路由入口（兼容现有 Agent.send 的调用）

        AsyncRouter 是异步的，但现有 Agent.send 是同步的、会调用
        router.route_message。这里提供线程安全的同步桥接：
        用 call_soon_threadsafe 在事件循环线程中调度入队（线程安全 + 防 GC）。

        Returns:
            False 表示接收者不存在或路由器未运行；True 表示已调度入队
            （实际处理异步完成，结果反映在 stats 中）
        """
        self._ensure_started()
        if message.receiver not in self.agents:
            logger.warning("[AsyncRouter] 无法路由到 %s：Agent 不存在", message.receiver)
            self._bump("routing_errors")
            return False
        self.message_history.append(message)
        self._bump("messages_sent")
        # 线程安全地调度入队（无论从事件循环线程还是 executor 线程调用都安全）
        self._loop.call_soon_threadsafe(self._schedule_put, message)
        return True

    def _ensure_started(self) -> None:
        if self._queue is None or self._loop is None:
            raise RuntimeError("AsyncRouter 未启动或已停止，请先 await router.start()")

    def _schedule_put(self, message: Message) -> None:
        """在事件循环线程中调度异步入队（保留 task 引用防 GC）"""
        task = asyncio.ensure_future(self._put(message))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _put(self, message: Message) -> None:
        """异步入队辅助"""
        await self._queue.put(message)

    async def route(self, message: Message) -> bool:
        """
        异步路由一条消息：投递到队列即返回，不阻塞等待处理完成。

        消息的实际处理由后台消费协程异步完成。调用方若需等待结果，
        应在 route 后调用 drain()。

        Returns:
            False 表示接收者不存在；True 表示已成功入队（不代表已处理）
        """
        self._ensure_started()
        if message.receiver not in self.agents:
            logger.warning("[AsyncRouter] 无法路由到 %s：Agent 不存在", message.receiver)
            self._bump("routing_errors")
            return False
        self.message_history.append(message)
        self._bump("messages_sent")
        await self._queue.put(message)
        return True

    async def broadcast(
        self,
        message: Message,
        exclude_sender: bool = True,
        target_agents: list[str] | set[str] | None = None,
    ) -> dict[str, bool]:
        """
        异步广播：把消息分发给目标 Agent（每个作为独立消息入队）

        Returns:
            广播结果字典（与同步 Router 一致）：键为 agent_id，
            值为是否成功入队（排除发送者的目标为 False）
        """
        self._ensure_started()
        if target_agents is None:
            target_agents = list(self.agents.keys())
        results: dict[str, bool] = {}
        for agent_id in target_agents:
            if exclude_sender and agent_id == message.sender:
                results[agent_id] = False
                continue
            if agent_id not in self.agents:
                self._bump("routing_errors")
                results[agent_id] = False
                continue
            results[agent_id] = await self.route(
                Message(
                    sender=message.sender,
                    receiver=agent_id,
                    content=message.content,
                    msg_type=message.type,
                    metadata=message.metadata.copy(),
                    max_hops=message.max_hops,
                )
            )
        return results

    # ------------------------------------------------------------------
    # 消费循环（后台协程）
    # ------------------------------------------------------------------
    async def _consumer_loop(self) -> None:
        """后台消费协程：从队列取消息，调用 Agent 异步接收；取到哨兵则退出"""
        assert self._queue is not None
        while True:
            message = await self._queue.get()
            if message is _SHUTDOWN:
                self._queue.task_done()
                return
            self._in_flight += 1
            if self._idle_event is not None:
                self._idle_event.clear()
            try:
                await self._deliver(message)
            except Exception as e:
                # _deliver 内部已有异常边界；这里兜底防消费者整体退出
                logger.exception("[AsyncRouter] 消息处理异常: %s", e)
            finally:
                self._in_flight -= 1
                self._queue.task_done()
                if self._in_flight == 0 and self._queue.empty() and self._idle_event is not None:
                    self._idle_event.set()

    async def _deliver(self, message: Message) -> None:
        """把单条消息投递给目标 Agent（跳数检查 + 异常边界保护）"""
        if message.hops >= message.max_hops:
            logger.warning(
                "[AsyncRouter] 消息 %s 达到最大转发跳数 %d（sender=%s receiver=%s），已丢弃",
                message.id[:8],
                message.max_hops,
                message.sender,
                message.receiver,
            )
            self._bump("hop_limit_drops")
            return

        receiver_id = message.receiver
        agent = self.agents.get(receiver_id)
        if agent is None:
            logger.warning("[AsyncRouter] 无法路由到 %s：Agent 不存在", receiver_id)
            self._bump("routing_errors")
            return

        # 让本消息处理期间 Agent.send 创建的新消息继承 hops+1 与链路预算
        token = set_message_hops(message.hops + 1, message.max_hops)
        try:
            # 优先用异步接收方法；回退到同步 receive（在执行器中跑，
            # 复制当前上下文使跳数/追踪在执行器线程可见）
            receive_async = getattr(agent, "receive_async", None)
            if receive_async is not None:
                await receive_async(message)
            else:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                await loop.run_in_executor(None, lambda: ctx.run(agent.receive, message))
            self._bump("messages_received")
        except Exception as e:
            logger.exception("[AsyncRouter] Agent '%s' 处理消息异常: %s", receiver_id, e)
            self._bump("routing_errors")
        finally:
            reset_message_hops(token)
