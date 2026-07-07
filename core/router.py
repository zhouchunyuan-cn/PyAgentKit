"""
同步消息路由器（Router）

集中式消息总线，管理 Agent 间的通信：点对点路由、广播、规则路由。
带异常边界（单个 Agent 崩溃不中断整个消息流）。
异步场景请用 core/async_router.py（AsyncRouter）。
"""

from typing import Dict, List, Callable, Optional, Union, Set, Any
from .agent import Agent
from .message import Message
import json
import logging

logger = logging.getLogger(__name__)


class Router:
    """
    PyAgentKit中的消息路由器
    
    功能:
    - 点对点路由
    - 广播路由
    - 条件路由（rule-based 或 LLM-driven）
    """

    def __init__(self):
        """
        初始化路由器
        """
        self.agents: Dict[str, Agent] = {}
        self.routing_rules: List[Callable[[Message], Optional[str]]] = []
        self.message_history: List[Message] = []
        self.stats: Dict[str, int] = {
            "messages_sent": 0,
            "messages_received": 0,
            "routing_errors": 0
        }

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
        
        Args:
            message: 要路由的消息
            
        Returns:
            是否成功路由
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
        
        # 路由消息
        if receiver_id in self.agents:
            try:
                self.agents[receiver_id].receive(message)
                self.stats["messages_received"] += 1
                return True
            except Exception as e:
                # 异常边界：单个 Agent 处理失败不应中断整个消息流
                logger.exception("Agent '%s' 处理消息时抛出异常: %s", receiver_id, e)
                self.stats["routing_errors"] += 1
                return False
        else:
            logger.warning("无法路由消息到 %s：Agent 不存在", receiver_id)
            self.stats["routing_errors"] += 1
            return False

    def broadcast(self, message: Message, exclude_sender: bool = True, 
                  target_agents: Optional[Set[str]] = None) -> Dict[str, bool]:
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
            # 检查是否排除发送者
            if exclude_sender and agent_id == message.sender:
                results[agent_id] = False
                continue
                
            # 检查agent是否存在
            if agent_id in self.agents:
                # 创建新的消息实例以避免引用问题
                broadcast_msg = Message(
                    sender=message.sender,
                    receiver=agent_id,
                    content=message.content,
                    msg_type=message.type,
                    metadata=message.metadata.copy()
                )
                try:
                    self.agents[agent_id].receive(broadcast_msg)
                    results[agent_id] = True
                    self.stats["messages_sent"] += 1
                    self.stats["messages_received"] += 1
                except Exception as e:
                    logger.exception("广播时 Agent '%s' 处理异常: %s", agent_id, e)
                    results[agent_id] = False
                    self.stats["routing_errors"] += 1
            else:
                results[agent_id] = False
                self.stats["routing_errors"] += 1
                
        return results

    def add_routing_rule(self, rule_func: Callable[[Message], Optional[str]]) -> None:
        """
        添加条件路由规则
        
        Args:
            rule_func: 路由规则函数，接受消息作为参数并返回目标Agent ID或None
        """
        self.routing_rules.append(rule_func)

    def get_message_history(self, limit: Optional[int] = None) -> List[Message]:
        """
        获取消息历史
        
        Args:
            limit: 限制返回的消息数量
            
        Returns:
            消息历史列表
        """
        if limit:
            return self.message_history[-limit:]
        return self.message_history.copy()

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        获取指定ID的Agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent实例或None
        """
        return self.agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """
        列出所有已注册的Agents
        
        Returns:
            Agent ID列表
        """
        return list(self.agents.keys())

    def get_router_stats(self) -> Dict[str, Any]:
        """
        获取路由器统计信息
        
        Returns:
            路由器统计信息字典
        """
        return {
            "agent_count": len(self.agents),
            "routing_rule_count": len(self.routing_rules),
            "message_history_count": len(self.message_history),
            "stats": self.stats.copy()
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
                log_data.append({
                    "id": msg.id,
                    "timestamp": msg.timestamp.isoformat(),
                    "sender": msg.sender,
                    "receiver": msg.receiver,
                    "type": msg.type,
                    "content": str(msg.content),
                    "metadata": msg.metadata
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error("导出消息日志失败: %s", e)
            return False