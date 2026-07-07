"""
异步路由器（AsyncRouter）

基于 asyncio.Queue 的异步消息路由，解决同步 Router 的递归调用栈溢出问题。

核心区别：
- 同步 Router：route_message 直接调 agent.receive()，receive 内部再 send
  又调 route_message，形成无深度限制的栈递归（receive→send→route→receive）。
- AsyncRouter：route 只把消息投递到队列就返回（await 让出控制权），
  后台消费协程逐条取出处理。receive 内部再 send 时，新消息回到队列排队，
  不再嵌套在调用栈上——彻底打破递归。

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
  若不存在则回退到同步 receive()（在执行器线程中运行）。
- 与现有 Router 接口平行，可作为可选替代。
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging

from .agent import Agent
from .message import Message

logger = logging.getLogger(__name__)


class AsyncRouter:
    """
    基于 asyncio.Queue 的异步路由器

    消息投递与处理解耦：route() 入队即返回，后台协程串行处理。
    这样 receive→send→route 不再是调用栈递归，而是队列排队。
    """

    def __init__(self, max_concurrent: int = 1):
        """
        Args:
            max_concurrent: 消费协程并发数。默认 1（串行，保证消息顺序与
                            完全无递归）。>1 可并发处理不同消息，但同一
                            消息链仍按队列出队顺序调度。
        """
        self.agents: Dict[str, Agent] = {}
        self.message_history: List[Message] = []
        self.stats: Dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "routing_errors": 0,
        }
        self._max_concurrent = max_concurrent
        # 队列与消费任务在 start() 中创建（需要事件循环）
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_tasks: List[asyncio.Task] = []
        self._running = False
        # drain 的等待条件：队列空且无在处理消息时通知
        self._idle_event: Optional[asyncio.Event] = None
        self._in_flight = 0

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

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def get_router_stats(self) -> Dict[str, Any]:
        return {
            "agent_count": len(self.agents),
            "message_history_count": len(self.message_history),
            "stats": self.stats.copy(),
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动后台消费协程（必须在事件循环中调用）"""
        if self._running:
            return
        self._queue = asyncio.Queue()
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # 初始无消息，视为空闲
        self._running = True
        self._consumer_tasks = [
            asyncio.create_task(self._consumer_loop())
            for _ in range(self._max_concurrent)
        ]
        logger.debug("AsyncRouter 已启动，消费协程数=%d", self._max_concurrent)

    async def stop(self) -> None:
        """停止消费协程，等待队列处理完"""
        self._running = False
        if self._queue is not None:
            await self._queue.join()  # 等待所有已入队消息处理完
        for task in self._consumer_tasks:
            task.cancel()
        for task in self._consumer_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._consumer_tasks.clear()
        logger.debug("AsyncRouter 已停止")

    async def drain(self) -> None:
        """等待队列中所有消息处理完毕（用于同步等待异步结果）"""
        if self._queue is not None:
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
        router.route_message。这里提供同步桥接：把消息投递到队列。
        若在事件循环线程中，用 call_soon 异步入队；否则直接 put_nowait。

        这样同步 Agent（receive 内部用 self.send）在 AsyncRouter 下也能工作，
        且不会同步递归——消息入队后立即返回，实际处理在消费协程。
        """
        if self._queue is None:
            raise RuntimeError("AsyncRouter 未启动，请先 await router.start()")
        self.message_history.append(message)
        self.stats["messages_sent"] += 1
        try:
            loop = asyncio.get_running_loop()
            # 在事件循环线程内：调度异步入队
            loop.create_task(self._put(message))
        except RuntimeError:
            # 无运行中的事件循环（不应发生于 AsyncRouter 场景），直接入队
            self._queue.put_nowait(message)
        return True

    async def _put(self, message: Message) -> None:
        """异步入队辅助"""
        await self._queue.put(message)

    async def route(self, message: Message) -> bool:
        """
        异步路由一条消息：投递到队列即返回，不阻塞等待处理完成。

        消息的实际处理由后台消费协程异步完成。调用方若需等待结果，
        应在 route 后调用 drain()。

        Returns:
            True 表示已成功入队（不代表已处理）
        """
        if self._queue is None:
            raise RuntimeError("AsyncRouter 未启动，请先 await router.start()")
        self.message_history.append(message)
        self.stats["messages_sent"] += 1
        await self._queue.put(message)
        return True

    async def broadcast(
        self,
        message: Message,
        exclude_sender: bool = True,
        target_agents: Optional[List[str]] = None,
    ) -> None:
        """异步广播：把消息分发给目标 Agent（每个作为独立消息入队）"""
        if target_agents is None:
            target_agents = list(self.agents.keys())
        for agent_id in target_agents:
            if exclude_sender and agent_id == message.sender:
                continue
            if agent_id not in self.agents:
                self.stats["routing_errors"] += 1
                continue
            await self.route(Message(
                sender=message.sender,
                receiver=agent_id,
                content=message.content,
                msg_type=message.type,
                metadata=message.metadata.copy(),
            ))

    # ------------------------------------------------------------------
    # 消费循环（后台协程）
    # ------------------------------------------------------------------
    async def _consumer_loop(self) -> None:
        """后台消费协程：从队列取消息，调用 Agent 异步接收"""
        assert self._queue is not None
        while self._running:
            try:
                message = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                self._in_flight += 1
                if self._idle_event is not None:
                    self._idle_event.clear()
                await self._deliver(message)
            finally:
                self._in_flight -= 1
                self._queue.task_done()
                if self._in_flight == 0 and self._queue.empty():
                    if self._idle_event is not None:
                        self._idle_event.set()

    async def _deliver(self, message: Message) -> None:
        """把单条消息投递给目标 Agent（异步，异常边界保护）"""
        receiver_id = message.receiver
        agent = self.agents.get(receiver_id)
        if agent is None:
            logger.warning("[AsyncRouter] 无法路由到 %s：Agent 不存在", receiver_id)
            self.stats["routing_errors"] += 1
            return

        try:
            # 优先用异步接收方法；回退到同步 receive（在执行器中跑）
            receive_async = getattr(agent, "receive_async", None)
            if receive_async is not None:
                await receive_async(message)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, agent.receive, message)
            self.stats["messages_received"] += 1
        except Exception as e:
            logger.exception("[AsyncRouter] Agent '%s' 处理消息异常: %s", receiver_id, e)
            self.stats["routing_errors"] += 1
