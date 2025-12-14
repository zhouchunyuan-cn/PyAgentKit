#!/usr/bin/env python3
"""
PyAgentKit 主程序
综合演示所有功能
"""

from core import Router, Message, Orchestrator
from core.monitor import AgentMonitor
from core.collaboration import DynamicCollaborationManager
from core.mcp_tools import MCPTool
from agents import UserAgent, ResearchAgent, WriterAgent, AnalyzerAgent


def main():
    """
    主函数 - 综合演示所有功能
    """
    print("PyAgentKit 综合演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    orchestrator = Orchestrator(router)
    monitor = AgentMonitor(router)
    collaboration_manager = DynamicCollaborationManager(router)
    
    # 创建Agents
    user_agent = UserAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    analyzer_agent = AnalyzerAgent()
    
    # 通过调度器注册Agents
    orchestrator.add_agent(user_agent)
    orchestrator.add_agent(research_agent)
    orchestrator.add_agent(writer_agent)
    orchestrator.add_agent(analyzer_agent)
    
    # 创建并注册MCP工具
    file_system_tool = MCPTool(
        name="file_system",
        description="文件系统操作工具",
        mcp_endpoint="http://localhost:8000/mcp"
    )
    
    # 为AnalyzerAgent注册MCP工具
    analyzer_agent.tool_registry.register_mcp_tool(file_system_tool)
    
    # 启动监控
    monitor.start_monitoring()
    
    print("1. 演示研究流程:")
    # 模拟用户提问
    user_question = "人工智能"
    print(f"[System] 用户提问: {user_question}")
    
    # 创建初始消息
    initial_message = Message(
        sender="user",
        receiver="researcher",
        content=user_question,
        msg_type="research_request"
    )
    
    # 路由消息启动研究流程
    print("\n开始处理用户研究请求...")
    router.route_message(initial_message)
    
    print("\n" + "="*50)
    print("2. 演示计算分析流程:")
    # 演示计算功能
    calculation_request = "2 * 3 + 5"
    print(f"[System] 计算请求: {calculation_request}")
    
    calc_message = Message(
        sender="user",
        receiver="analyzer",
        content=calculation_request,
        msg_type="calculate"
    )
    
    # 路由消息启动计算流程
    print("\n开始处理计算请求...")
    router.route_message(calc_message)
    
    print("\n" + "="*50)
    print("3. 演示动态协作:")
    # 演示动态协作功能
    collaboration_task = "研究机器学习的应用"
    print(f"[System] 协作任务: {collaboration_task}")
    
    # 启动协作
    success = collaboration_manager.initiate_collaboration(collaboration_task)
    if success:
        print("动态协作已启动")
    else:
        print("动态协作启动失败")
    
    print("\n" + "="*50)
    print("4. 系统状态:")
    # 显示系统状态
    monitor.print_real_time_status()
    
    # 显示协作统计信息
    stats = collaboration_manager.get_collaboration_stats()
    print(f"\n协作策略数量: {stats['strategy_count']}")
    print(f"任务历史数量: {stats['task_history_count']}")
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 导出监控报告
    monitor.export_monitoring_report("monitoring_report.json")
    router.export_message_log("message_log.json")
    
    print("\n监控报告已导出到 monitoring_report.json")
    print("消息日志已导出到 message_log.json")
    
    print("\n综合演示完成")


if __name__ == "__main__":
    main()