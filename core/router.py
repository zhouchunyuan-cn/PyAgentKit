"""
同步消息路由器（Router）

集中式消息总线，管理 Agent 间的通信：点对点路由、广播、规则路由。
带异常边界（单个 Agent 崩溃不中断整个消息流）。
异步场景请用 core/async_router.py（AsyncRouter）。

防栈溢出与防环路：
- 投递采用迭代泵（pending 队列）：Agent.receive 内部再 send 时，新消息进入
  待投递队列而非递归调用 route_message，调用栈深度恒定，长转发链不再
  RecursionError。
- 消息携带 hops/max_hops（见 core/message.py），超过上限的转发被丢弃，
  防止 A↔B 互发回复的无限循环。
"""

import json
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

from .agent import Agent
from .message import Message, reset_message_hops, set_message_hops

logger = logging.getLogger(__name__)

# 消息历史上限（防止长驻进程内存无限增长；超出后丢弃最旧记录）
MAX_MESSAGE_HISTORY = 1000


class Router:
    """
    PyAgentKit中的消息路由器

    功能:
    - 点对点路由
    - 广播路由
    - 条件路由（rule-based 或 LLM-driven）
    - 转发跳数上限（防环路）
    """

    def __init__(self):
        """
        初始化路由器
        """
        self.agents: dict[str, Agent] = {}
        self.routing_rules: list[Callable[[Message], str | None]] = []
        self.message_history: deque[Message] = deque(maxlen=MAX_MESSAGE_HISTORY)
        self.stats: dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "routing_errors": 0,
            "hop_limit_drops": 0,
        }
        # 迭代泵状态：receive 期间嵌套 send 产生的消息进入 _pending，
        # 由外层 route_message 的循环统一投递（不递归）
        self._pending: deque[tuple[str, Message]] = deque()
        self._pumping = False

    def register_agent(self, agent: Agent) -> None:
        """
        注册Agent到路由器

        Args:
            agent: 要注册的Agent实例
        """
        self.agents[agent.agent_id] = agent
        # 自动设置Agent的路由器
        agent.set_router(self)

    def unregister_agent(self, agent_id: str) -> None:
        """
        从路由器注销Agent

        Args:
            agent_id: 要注销的Agent ID
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.set_router(None)  # 清除路由器引用
            del self.agents[agent_id]

    def route_message(self, message: Message) -> bool:
        """
        路由消息到指定的接收者

        消息实际投递通过迭代泵完成：若在 Agent.receive 执行期间有新消息
        被路由（Agent 转发回复），它们排队等待当前调用栈返回后统一投递，
        因此任意长度的转发链都不会递归加深调用栈。

        Args:
            message: 要路由的消息

        Returns:
            是否成功路由（嵌套调用时表示已成功入队）
        """
        receiver_id = message.receiver

        # 检查是否有特定的路由规则
        for rule in self.routing_rules:
            target_agent_id = rule(message)
            if target_agent_id:
                receiver_id = target_agent_id
                message.receiver = receiver_id
                break

        # 添加到消息历史
        self.message_history.append(message)
        self.stats["messages_sent"] += 1

        # 入队；若已在泵中（嵌套调用），由外层循环继续投递
        self._pending.append((receiver_id, message))
        if self._pumping:
            return True

        # 迭代泵：逐条投递，投递期间产生的新消息继续入队
        self._pumping = True
        delivered_any = False
        try:
            while self._pending:
                rid, msg = self._pending.popleft()
                if self._deliver_one(rid, msg):
                    delivered_any = True
            return delivered_any
        finally:
            self._pumping = False

    def _deliver_one(self, receiver_id: str, message: Message) -> bool:
        """
        投递单条消息给目标 Agent（跳数检查 + 异常边界）

        Returns:
            是否投递成功
        """
        # 转发跳数上限：防止 Agent 间互相回复形成无限循环
        if message.hops >= message.max_hops:
            logger.warning(
                "消息 %s 达到最大转发跳数 %d（sender=%s receiver=%s），已丢弃",
                message.id[:8],
                message.max_hops,
                message.sender,
                receiver_id,
            )
            self.stats["hop_limit_drops"] += 1
            return False

        agent = self.agents.get(receiver_id)
        if agent is None:
            logger.warning("无法路由消息到 %s：Agent 不存在", receiver_id)
            self.stats["routing_errors"] += 1
            return False

        # 让本消息处理期间 Agent.send 创建的新消息继承 hops+1 与链路预算
        token = set_message_hops(message.hops + 1, message.max_hops)
        try:
            agent.receive(message)
        except Exception as e:
            # 异常边界：单个 Agent 处理失败不应中断整个消息流
            logger.exception("Agent '%s' 处理消息时抛出异常: %s", receiver_id, e)
            self.stats["routing_errors"] += 1
            return False
        finally:
            reset_message_hops(token)
        self.stats["messages_received"] += 1
        return True

    def broadcast(
        self, message: Message, exclude_sender: bool = True, target_agents: set[str] | None = None
    ) -> dict[str, bool]:
        """
        广播消息给所有Agent

        Args:
            message: 要广播的消息
            exclude_sender: 是否排除发送者
            target_agents: 指定要广播的目标Agents，如果为None则广播给所有agents

        Returns:
            广播结果字典，键为agent_id，值为是否成功发送
        """
        # 添加到消息历史
        self.message_history.append(message)

        results = {}

        # 确定目标agents
        if target_agents is None:
            target_agents = set(self.agents.keys())

        for agent_id in target_agents:
            # 检查是否排除发送者（排除 ≠ 投递失败，但保持旧契约：False）
            if exclude_sender and agent_id == message.sender:
                results[agent_id] = False
                continue

            self.stats["messages_sent"] += 1
            if agent_id in self.agents:
                # 创建新的消息实例以避免引用问题
                broadcast_msg = Message(
                    sender=message.sender,
                    receiver=agent_id,
                    content=message.content,
                    msg_type=message.type,
                    metadata=message.metadata.copy(),
                    max_hops=message.max_hops,
                )
                results[agent_id] = self._deliver_one(agent_id, broadcast_msg)
            else:
                results[agent_id] = False
                self.stats["routing_errors"] += 1

        return results

    def add_routing_rule(self, rule_func: Callable[[Message], str | None]) -> None:
        """
        添加条件路由规则

        Args:
            rule_func: 路由规则函数，接受消息作为参数并返回目标Agent ID或None
        """
        self.routing_rules.append(rule_func)

    def get_message_history(self, limit: int | None = None) -> list[Message]:
        """
        获取消息历史

        Args:
            limit: 限制返回的消息数量

        Returns:
            消息历史列表
        """
        history = list(self.message_history)
        if limit:
            return history[-limit:]
        return history

    def get_agent(self, agent_id: str) -> Agent | None:
        """
        获取指定ID的Agent

        Args:
            agent_id: Agent ID

        Returns:
            Agent实例或None
        """
        return self.agents.get(agent_id)

    def list_agents(self) -> list[str]:
        """
        列出所有已注册的Agents

        Returns:
            Agent ID列表
        """
        return list(self.agents.keys())

    def get_router_stats(self) -> dict[str, Any]:
        """
        获取路由器统计信息

        Returns:
            路由器统计信息字典
        """
        return {
            "agent_count": len(self.agents),
            "routing_rule_count": len(self.routing_rules),
            "message_history_count": len(self.message_history),
            "stats": self.stats.copy(),
        }

    def export_message_log(self, filepath: str) -> bool:
        """
        导出消息日志到文件

        Args:
            filepath: 导出文件路径

        Returns:
            是否导出成功
        """
        try:
            log_data = []
            for msg in self.message_history:
                log_data.append(
                    {
                        "id": msg.id,
                        "timestamp": msg.timestamp.isoformat(),
                        "sender": msg.sender,
                        "receiver": msg.receiver,
                        "type": msg.type,
                        "content": str(msg.content),
                        "metadata": msg.metadata,
                        "hops": msg.hops,
                    }
                )

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error("导出消息日志失败: %s", e)
            return False
