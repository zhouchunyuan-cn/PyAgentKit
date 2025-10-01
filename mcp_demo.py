#!/usr/bin/env python3
"""
PyAgentKit MCP工具集成演示
展示如何接入和使用其他MCP工具
"""

from core import Router, Message
from agents import UserAgent, ResearchAgent, WriterAgent, AnalyzerAgent
from core.mcp_tools import MCPTool


def create_sample_mcp_tool():
    """
    创建一个示例MCP工具
    """
    mcp_tool = MCPTool(
        name="sample_file_system",
        description="示例文件系统工具，支持文件读写操作",
        mcp_endpoint="http://localhost:8000/mcp"
    )
    return mcp_tool


def main():
    """
    主函数 - 演示MCP工具集成
    """
    print("PyAgentKit MCP工具集成演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    
    # 创建Agents
    user_agent = UserAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    analyzer_agent = AnalyzerAgent()
    
    # 注册Agents
    router.register_agent(user_agent)
    router.register_agent(research_agent)
    router.register_agent(writer_agent)
    router.register_agent(analyzer_agent)
    
    # 为AnalyzerAgent添加MCP工具支持
    sample_mcp_tool = create_sample_mcp_tool()
    analyzer_agent.tool_registry.register_mcp_tool(sample_mcp_tool)
    
    print("1. 演示MCP工具注册:")
    mcp_tools = analyzer_agent.tool_registry.mcp_registry.list_tools()
    for name, desc in mcp_tools.items():
        print(f"  - {name}: {desc}")
    
    print("\n2. 演示MCP工具执行:")
    # 演示文件读取MCP工具
    result = analyzer_agent.tool_registry.execute_mcp_tool(
        "sample_file_system",
        command="read_file",
        params={"path": "example.txt"}
    )
    print(f"  文件读取结果: {result}")
    
    # 演示文件写入MCP工具
    result = analyzer_agent.tool_registry.execute_mcp_tool(
        "sample_file_system",
        command="write_file",
        params={"path": "output.txt", "content": "Hello from MCP tool!"}
    )
    print(f"  文件写入结果: {result}")
    
    # 演示命令执行MCP工具
    result = analyzer_agent.tool_registry.execute_mcp_tool(
        "sample_file_system",
        command="execute_command",
        params={"cmd": "ls -la"}
    )
    print(f"  命令执行结果: {result}")
    
    print("\n3. 演示传统工作流程:")
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
    
    print("\n演示完成")


if __name__ == "__main__":
    main()