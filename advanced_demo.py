#!/usr/bin/env python3
"""
PyAgentKit 高级多智能体对话系统演示
展示更复杂的多智能体协作场景
"""

from core import Router, Message, Orchestrator
from core.mcp_tools import MCPTool
from agents import UserAgent, ResearchAgent, WriterAgent, AnalyzerAgent


def main():
    """
    主函数 - 演示高级多智能体对话流程
    """
    print("PyAgentKit 高级多智能体对话系统演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    orchestrator = Orchestrator(router)
    
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
    print("3. 演示MCP工具集成:")
    # 演示MCP工具使用
    mcp_command = {
        "tool_name": "file_system",
        "command": "read_file",
        "params": {"path": "example.txt"}
    }
    
    print("[System] MCP工具请求: 读取example.txt文件")
    
    mcp_message = Message(
        sender="user",
        receiver="analyzer",
        content=mcp_command,
        msg_type="mcp_command"
    )
    
    # 路由消息启动MCP工具流程
    print("\n开始处理MCP工具请求...")
    router.route_message(mcp_message)
    
    print("\n" + "="*50)
    print("4. 消息历史记录:")
    # 显示消息历史
    history = router.get_message_history()
    for i, msg in enumerate(history[-8:], 1):  # 显示最近8条消息
        print(f"  {i}. {msg.sender} -> {msg.receiver} [{msg.type}]: {str(msg.content)[:50]}...")
    
    print("\n演示完成")


if __name__ == "__main__":
    main()