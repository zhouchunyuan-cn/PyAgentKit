"""
任务调度器（Orchestrator）

控制 Agent 的执行顺序与并发：顺序执行、广播、独立线程运行。
基于 Router 提供更高级的执行流程控制。
"""

from typing import List, Dict, Any, Set, Optional
from .agent import Agent
from .router import Router
from .message import Message
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor


class Orchestrator:
    """
    PyAgentKit中的调度器
    
    功能:
    - 控制 Agent 执行顺序
    - 支持并发 / 异步
    """

    def __init__(self, router: Router):
        """
        初始化调度器
        
        Args:
            router: 消息路由器实例
        """
        self.router = router
        self.executor = ThreadPoolExecutor(max_workers=4)

    def add_agent(self, agent: Agent) -> None:
        """
        添加Agent到调度器
        
        Args:
            agent: Agent实例
        """
        self.router.register_agent(agent)

    def remove_agent(self, agent_id: str) -> None:
        """
        从调度器移除Agent
        
        Args:
            agent_id: Agent ID
        """
        self.router.unregister_agent(agent_id)

    def sequential_execution(self, agent_ids: List[str], initial_message: Message) -> None:
        """
        顺序执行Agents
        
        Args:
            agent_ids: Agent ID列表，按执行顺序排列
            initial_message: 初始消息
        """
        current_message = initial_message
        
        for agent_id in agent_ids:
            if agent_id in self.router.agents:
                # 更新消息的接收者
                current_message.receiver = agent_id
                # 发送消息给当前Agent
                self.router.route_message(current_message)
            else:
                print(f"Warning: Agent {agent_id} not found")

    def broadcast_message(self, message: Message, exclude_sender: bool = True, 
                         target_agents: Optional[Set[str]] = None) -> Dict[str, bool]:
        """
        广播消息
        
        Args:
            message: 要广播的消息
            exclude_sender: 是否排除发送者
            target_agents: 指定要广播的目标Agents，如果为None则广播给所有agents
            
        Returns:
            广播结果字典
        """
        return self.router.broadcast(message, exclude_sender, target_agents)

    async def async_send_message(self, message: Message) -> bool:
        """
        异步发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            是否成功发送
        """
        # 在线程池中执行消息路由，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.router.route_message, message)
        return result

    def run_agent_in_thread(self, agent_id: str, message: Message) -> threading.Thread:
        """
        在独立线程中运行Agent
        
        Args:
            agent_id: Agent ID
            message: 发送给Agent的消息
            
        Returns:
            线程对象
        """
        def target():
            if agent_id in self.router.agents:
                message.receiver = agent_id
                self.router.route_message(message)
            else:
                print(f"Warning: Agent {agent_id} not found")
                
        thread = threading.Thread(target=target)
        thread.start()
        return thread

    def get_router(self) -> Router:
        """
        获取路由器实例
        
        Returns:
            Router实例
        """
        return self.router

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            调度器统计信息字典
        """
        router_stats = self.router.get_router_stats()
        return {
            "router_stats": router_stats,
            "thread_pool_max_workers": self.executor._max_workers
        }