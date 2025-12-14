from typing import Dict, List, Any, Optional
from .router import Router
from .agent import Agent
import json
import time


class AgentMonitor:
    """
    Agent监控系统
    用于监控和调试多Agent交互
    """

    def __init__(self, router: Router):
        """
        初始化监控系统
        
        Args:
            router: 路由器实例
        """
        self.router = router
        self.monitoring_data: Dict[str, List[Dict[str, Any]]] = {}
        self.is_monitoring = False

    def start_monitoring(self) -> None:
        """
        开始监控
        """
        self.is_monitoring = True
        print("Agent监控系统已启动")

    def stop_monitoring(self) -> None:
        """
        停止监控
        """
        self.is_monitoring = False
        print("Agent监控系统已停止")

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定Agent的状态信息
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent状态信息
        """
        agent = self.router.get_agent(agent_id)
        if not agent:
            return None
            
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "system_prompt": agent.system_prompt,
            "tool_count": len(agent.tools),
            "memory_stats": agent.memory.get_memory_stats() if agent.memory else {},
            "tools": agent.list_tools()
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """
        获取系统概览
        
        Returns:
            系统概览信息
        """
        router_stats = self.router.get_router_stats()
        agent_statuses = {}
        
        for agent_id in self.router.list_agents():
            agent_statuses[agent_id] = self.get_agent_status(agent_id)
            
        return {
            "router_stats": router_stats,
            "agent_statuses": agent_statuses,
            "monitoring_active": self.is_monitoring
        }

    def export_monitoring_report(self, filepath: str) -> bool:
        """
        导出监控报告
        
        Args:
            filepath: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            report_data = self.get_system_overview()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"导出监控报告失败: {e}")
            return False

    def print_real_time_status(self) -> None:
        """
        打印实时状态信息
        """
        if not self.is_monitoring:
            print("监控未启动，请先调用start_monitoring()")
            return
            
        overview = self.get_system_overview()
        
        print("\n=== Agent系统实时状态 ===")
        print(f"注册Agent数量: {overview['router_stats']['agent_count']}")
        print(f"消息历史数量: {overview['router_stats']['message_history_count']}")
        print(f"路由规则数量: {overview['router_stats']['routing_rule_count']}")
        print(f"发送消息数: {overview['router_stats']['stats']['messages_sent']}")
        print(f"接收消息数: {overview['router_stats']['stats']['messages_received']}")
        print(f"路由错误数: {overview['router_stats']['stats']['routing_errors']}")
        
        print("\n--- Agent详情 ---")
        for agent_id, status in overview['agent_statuses'].items():
            print(f"\nAgent ID: {agent_id}")
            print(f"  名称: {status['name']}")
            print(f"  工具数量: {status['tool_count']}")
            print(f"  短期记忆项数: {status['memory_stats'].get('short_term_count', 0)}")
            print(f"  长期记忆项数: {status['memory_stats'].get('long_term_count', 0)}")