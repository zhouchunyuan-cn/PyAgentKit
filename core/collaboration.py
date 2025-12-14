from typing import Dict, List, Any, Optional, Callable
from .agent import Agent
from .message import Message
from .router import Router
import json


class CollaborationStrategy:
    """
    协作策略基类
    定义Agent间协作的规则和逻辑
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def should_collaborate(self, task: Dict[str, Any], agents: List[Agent]) -> List[Agent]:
        """
        判断哪些Agent应该参与协作
        
        Args:
            task: 任务信息
            agents: 可用的Agent列表
            
        Returns:
            应该参与协作的Agent列表
        """
        raise NotImplementedError

    def assign_roles(self, task: Dict[str, Any], collaborating_agents: List[Agent]) -> Dict[str, str]:
        """
        为协作Agent分配角色
        
        Args:
            task: 任务信息
            collaborating_agents: 参与协作的Agent列表
            
        Returns:
            Agent ID到角色的映射
        """
        raise NotImplementedError


class ResearchCollaborationStrategy(CollaborationStrategy):
    """
    研究任务协作策略
    """

    def __init__(self):
        super().__init__("research", "研究任务协作策略")

    def should_collaborate(self, task: Dict[str, Any], agents: List[Agent]) -> List[Agent]:
        """
        对于研究任务，需要研究员和作家Agent参与
        """
        collaborating_agents = []
        for agent in agents:
            if agent.name in ["Researcher", "Writer"]:
                collaborating_agents.append(agent)
        return collaborating_agents

    def assign_roles(self, task: Dict[str, Any], collaborating_agents: List[Agent]) -> Dict[str, str]:
        """
        分配角色：研究员负责研究，作家负责撰写报告
        """
        roles = {}
        for agent in collaborating_agents:
            if agent.name == "Researcher":
                roles[agent.agent_id] = "researcher"
            elif agent.name == "Writer":
                roles[agent.agent_id] = "writer"
        return roles


class AnalysisCollaborationStrategy(CollaborationStrategy):
    """
    分析任务协作策略
    """

    def __init__(self):
        super().__init__("analysis", "分析任务协作策略")

    def should_collaborate(self, task: Dict[str, Any], agents: List[Agent]) -> List[Agent]:
        """
        对于分析任务，需要分析员Agent参与
        """
        collaborating_agents = []
        for agent in agents:
            if agent.name == "Analyzer":
                collaborating_agents.append(agent)
        return collaborating_agents

    def assign_roles(self, task: Dict[str, Any], collaborating_agents: List[Agent]) -> Dict[str, str]:
        """
        分配角色：分析员负责分析
        """
        roles = {}
        for agent in collaborating_agents:
            if agent.name == "Analyzer":
                roles[agent.agent_id] = "analyzer"
        return roles


class DynamicCollaborationManager:
    """
    动态协作管理器
    根据任务需求自动选择合适的Agent进行协作
    """

    def __init__(self, router: Router):
        self.router = router
        self.strategies: Dict[str, CollaborationStrategy] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        # 注册默认的协作策略
        self.register_strategy(ResearchCollaborationStrategy())
        self.register_strategy(AnalysisCollaborationStrategy())

    def register_strategy(self, strategy: CollaborationStrategy) -> None:
        """
        注册协作策略
        
        Args:
            strategy: 协作策略实例
        """
        self.strategies[strategy.name] = strategy

    def get_strategy(self, strategy_name: str) -> Optional[CollaborationStrategy]:
        """
        获取协作策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            协作策略实例或None
        """
        return self.strategies.get(strategy_name)

    def analyze_task(self, task_description: str) -> str:
        """
        分析任务类型
        
        Args:
            task_description: 任务描述
            
        Returns:
            任务类型
        """
        # 简单的关键词匹配来确定任务类型
        task_lower = task_description.lower()
        if any(keyword in task_lower for keyword in ["研究", "搜索", "查找", "了解"]):
            return "research"
        elif any(keyword in task_lower for keyword in ["分析", "计算", "统计", "数据"]):
            return "analysis"
        else:
            # 默认使用研究策略
            return "research"

    def initiate_collaboration(self, task_description: str, sender_id: str = "user") -> bool:
        """
        启动协作流程
        
        Args:
            task_description: 任务描述
            sender_id: 发送者ID
            
        Returns:
            是否成功启动协作
        """
        # 分析任务类型
        task_type = self.analyze_task(task_description)
        
        # 获取对应的协作策略
        strategy = self.get_strategy(task_type)
        if not strategy:
            print(f"未找到适用于任务类型 '{task_type}' 的协作策略")
            return False
            
        # 获取所有可用的Agent
        agents = list(self.router.agents.values())
        
        # 确定参与协作的Agent
        collaborating_agents = strategy.should_collaborate(
            {"description": task_description, "type": task_type}, 
            agents
        )
        
        if not collaborating_agents:
            print(f"没有找到适合参与 '{task_type}' 类型任务的Agent")
            return False
            
        # 分配角色
        roles = strategy.assign_roles(
            {"description": task_description, "type": task_type}, 
            collaborating_agents
        )
        
        # 记录任务
        task_record = {
            "description": task_description,
            "type": task_type,
            "collaborating_agents": [agent.agent_id for agent in collaborating_agents],
            "roles": roles
        }
        self.task_history.append(task_record)
        
        # 启动协作流程
        if task_type == "research":
            # 发送研究请求给第一个研究员
            researcher = None
            for agent in collaborating_agents:
                if agent.name == "Researcher":
                    researcher = agent
                    break
                    
            if researcher:
                message = Message(
                    sender=sender_id,
                    receiver=researcher.agent_id,
                    content=task_description,
                    msg_type="research_request"
                )
                self.router.route_message(message)
                return True
                
        elif task_type == "analysis":
            # 发送分析请求给分析员
            analyzer = None
            for agent in collaborating_agents:
                if agent.name == "Analyzer":
                    analyzer = agent
                    break
                    
            if analyzer:
                message = Message(
                    sender=sender_id,
                    receiver=analyzer.agent_id,
                    content=task_description,
                    msg_type="analyze_request"
                )
                self.router.route_message(message)
                return True
        
        return False

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """
        获取协作统计信息
        
        Returns:
            协作统计信息
        """
        return {
            "strategy_count": len(self.strategies),
            "task_history_count": len(self.task_history),
            "recent_tasks": self.task_history[-5:] if len(self.task_history) > 5 else self.task_history
        }