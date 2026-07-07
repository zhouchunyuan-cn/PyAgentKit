"""
动态协作系统

基于 Agent 的【能力声明】(capabilities) 进行任务路由与协作编排，
而非硬编码 Agent 名称。每个 Agent 在初始化时声明自己具备的能力
（如 ["search","write"]），协作策略据此匹配。

核心组件：
- CollaborationStrategy: 协作策略抽象基类
- CapabilityCollaborationStrategy: 基于 required_capabilities 的通用策略
- DynamicCollaborationManager: 任务分析 + 策略调度 + 消息路由
"""
from typing import Dict, List, Any, Optional, Callable
from .agent import Agent
from .message import Message
from .router import Router
import json
import logging

logger = logging.getLogger(__name__)


class CollaborationStrategy:
    """
    协作策略基类
    定义 Agent 间协作的规则和逻辑
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def should_collaborate(self, task: Dict[str, Any], agents: List[Agent]) -> List[Agent]:
        """
        判断哪些 Agent 应该参与协作

        Args:
            task: 任务信息
            agents: 可用的 Agent 列表

        Returns:
            应该参与协作的 Agent 列表
        """
        raise NotImplementedError

    def assign_roles(self, task: Dict[str, Any], collaborating_agents: List[Agent]) -> Dict[str, str]:
        """
        为协作 Agent 分配角色

        Args:
            task: 任务信息
            collaborating_agents: 参与协作的 Agent 列表

        Returns:
            Agent ID 到角色的映射
        """
        raise NotImplementedError

    def get_entry_agent(self, collaborating_agents: List[Agent]) -> Optional[Agent]:
        """
        确定协作流程的入口 Agent（第一个接收任务消息的 Agent）

        默认返回参与协作的第一个 Agent；子类可重写以指定特定入口。

        Args:
            collaborating_agents: 参与协作的 Agent 列表

        Returns:
            入口 Agent，无可用时返回 None
        """
        return collaborating_agents[0] if collaborating_agents else None


class CapabilityCollaborationStrategy(CollaborationStrategy):
    """
    基于能力声明的通用协作策略

    通过 required_capabilities 指定本策略需要哪些能力，
    系统自动筛选声明了相应能力的 Agent 参与，不再依赖 Agent 名称。
    """

    def __init__(
        self,
        name: str,
        description: str,
        required_capabilities: List[str],
        entry_capability: Optional[str] = None,
    ):
        """
        Args:
            name: 策略名称
            description: 策略描述
            required_capabilities: 本策略需要的能力列表（如 ["search","write"]），
                                   具备其中任意一项的 Agent 都会参与
            entry_capability: 协作入口 Agent 应具备的能力（如 "search"），
                              None 则取参与列表中的第一个
        """
        super().__init__(name, description)
        self.required_capabilities = required_capabilities
        self.entry_capability = entry_capability

    def should_collaborate(self, task: Dict[str, Any], agents: List[Agent]) -> List[Agent]:
        """筛选声明了所需能力的 Agent（保留原始顺序）"""
        matched = [a for a in agents if a.has_any_capability(self.required_capabilities)]
        logger.debug(
            "策略 '%s' 匹配到 %d/%d 个 Agent: %s",
            self.name, len(matched), len(agents),
            [a.agent_id for a in matched],
        )
        return matched

    def assign_roles(self, task: Dict[str, Any], collaborating_agents: List[Agent]) -> Dict[str, str]:
        """按 Agent 声明的能力分配角色（取第一个匹配能力作为角色名）"""
        roles = {}
        for agent in collaborating_agents:
            for cap in self.required_capabilities:
                if agent.has_capability(cap):
                    roles[agent.agent_id] = cap
                    break
        return roles

    def get_entry_agent(self, collaborating_agents: List[Agent]) -> Optional[Agent]:
        """优先选具备入口能力的 Agent 作为流程起点"""
        if self.entry_capability:
            for a in collaborating_agents:
                if a.has_capability(self.entry_capability):
                    return a
        return super().get_entry_agent(collaborating_agents)


# --------------------------------------------------------------------
# 向后兼容：保留旧的策略类名，但内部改为基于能力匹配
# --------------------------------------------------------------------

class ResearchCollaborationStrategy(CapabilityCollaborationStrategy):
    """研究任务协作策略（需要 search + write 能力，search 为入口）"""

    def __init__(self):
        super().__init__(
            name="research",
            description="研究任务协作策略（基于能力匹配）",
            required_capabilities=["search", "write"],
            entry_capability="search",
        )


class AnalysisCollaborationStrategy(CapabilityCollaborationStrategy):
    """分析任务协作策略（需要 analysis 能力）"""

    def __init__(self):
        super().__init__(
            name="analysis",
            description="分析任务协作策略（基于能力匹配）",
            required_capabilities=["analysis", "calculate"],
            entry_capability="analysis",
        )


class DynamicCollaborationManager:
    """
    动态协作管理器

    根据任务需求自动选择合适的协作策略，并按 Agent 能力匹配组织协作。
    任务类型分析当前用关键词匹配；可后续替换为 LLM 分类。
    """

    def __init__(self, router: Router):
        self.router = router
        self.strategies: Dict[str, CollaborationStrategy] = {}
        self.task_history: List[Dict[str, Any]] = []

        # 注册默认的协作策略
        self.register_strategy(ResearchCollaborationStrategy())
        self.register_strategy(AnalysisCollaborationStrategy())

    def register_strategy(self, strategy: CollaborationStrategy) -> None:
        """注册协作策略"""
        self.strategies[strategy.name] = strategy

    def get_strategy(self, strategy_name: str) -> Optional[CollaborationStrategy]:
        """获取协作策略"""
        return self.strategies.get(strategy_name)

    def analyze_task(self, task_description: str) -> str:
        """
        分析任务类型（关键词匹配；后续可替换为 LLM 分类）

        Args:
            task_description: 任务描述

        Returns:
            任务类型名称（对应已注册的策略名）
        """
        task_lower = task_description.lower()
        if any(keyword in task_lower for keyword in ["研究", "搜索", "查找", "了解", "调研"]):
            return "research"
        elif any(keyword in task_lower for keyword in ["分析", "计算", "统计", "数据", "测算"]):
            return "analysis"
        else:
            # 默认使用研究策略
            return "research"

    def initiate_collaboration(self, task_description: str, sender_id: str = "user") -> bool:
        """
        启动协作流程

        Args:
            task_description: 任务描述
            sender_id: 发送者 ID

        Returns:
            是否成功启动协作
        """
        # 1. 分析任务类型
        task_type = self.analyze_task(task_description)

        # 2. 获取对应的协作策略
        strategy = self.get_strategy(task_type)
        if not strategy:
            logger.warning("未找到适用于任务类型 '%s' 的协作策略", task_type)
            return False

        # 3. 基于能力匹配确定参与协作的 Agent
        agents = list(self.router.agents.values())
        collaborating_agents = strategy.should_collaborate(
            {"description": task_description, "type": task_type},
            agents,
        )

        if not collaborating_agents:
            logger.warning("没有找到具备所需能力的 Agent（任务类型 '%s'，需要能力 %s）",
                           task_type,
                           getattr(strategy, 'required_capabilities', []))
            return False

        # 4. 分配角色
        roles = strategy.assign_roles(
            {"description": task_description, "type": task_type},
            collaborating_agents,
        )

        # 5. 记录任务
        task_record = {
            "description": task_description,
            "type": task_type,
            "collaborating_agents": [a.agent_id for a in collaborating_agents],
            "roles": roles,
        }
        self.task_history.append(task_record)

        # 6. 确定入口 Agent 并发送启动消息
        entry_agent = strategy.get_entry_agent(collaborating_agents)
        if entry_agent is None:
            logger.warning("无法确定协作入口 Agent")
            return False

        # 根据任务类型选择消息类型
        msg_type = "research_request" if task_type == "research" else "analyze_request"
        message = Message(
            sender=sender_id,
            receiver=entry_agent.agent_id,
            content=task_description,
            msg_type=msg_type,
        )
        self.router.route_message(message)
        logger.info("协作已启动：任务类型=%s，入口=%s，参与=%s",
                    task_type, entry_agent.agent_id,
                    [a.agent_id for a in collaborating_agents])
        return True

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """获取协作统计信息"""
        return {
            "strategy_count": len(self.strategies),
            "task_history_count": len(self.task_history),
            "recent_tasks": self.task_history[-5:] if len(self.task_history) > 5 else self.task_history
        }
