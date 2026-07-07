#!/usr/bin/env python3
"""
PyAgentKit 主程序
综合演示所有功能（GLM 驱动的 ReAct Agent 版本）

运行前请确保：
1. pip install -r requirements.txt
2. 已设置环境变量 ZHIPUAI_API_KEY（可复制 .env.example 为 .env 并填写）
"""

import logging
import os

# 自动加载 .env 文件中的环境变量（如 ZHIPUAI_API_KEY）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 未安装 python-dotenv 时静默跳过，依赖系统环境变量
    pass

from core import Router, Message, Orchestrator
from core.llm import GLMClient
from core.logging_config import setup_logging
from core.monitor import AgentMonitor
from core.collaboration import DynamicCollaborationManager
from core.mcp_tools import MCPTool
from agents import UserAgent, ResearchAgent, WriterAgent, AnalyzerAgent


def create_llm_client() -> GLMClient:
    """
    创建共享的 GLM 客户端

    API Key 从环境变量 ZHIPUAI_API_KEY 读取，缺失时给出清晰提示。
    模型默认 glm-4.5，可通过环境变量 GLM_MODEL 切换（如 glm-4.5-air）。
    """
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("=" * 60)
        print("⚠️  未检测到 ZHIPUAI_API_KEY 环境变量")
        print("请执行以下步骤：")
        print("  1. 复制 .env.example 为 .env")
        print("  2. 在 .env 中填入你的 API Key（获取地址：https://open.bigmodel.cn/）")
        print("  3. 确保运行前已加载 .env（或手动 export ZHIPUAI_API_KEY=...）")
        print("=" * 60)
        raise SystemExit(1)

    model = os.environ.get("GLM_MODEL", "glm-4-flash")
    print(f"[System] 使用模型: {model}")
    return GLMClient(model=model)


def main():
    """
    主函数 - 综合演示所有功能
    """
    setup_logging()
    print("PyAgentKit 综合演示（GLM 驱动）")
    print("=" * 60)

    # 创建共享的 LLM 客户端（所有 Agent 复用同一实例，节省资源）
    llm_client = create_llm_client()

    # 创建核心组件
    router = Router()
    orchestrator = Orchestrator(router)
    monitor = AgentMonitor(router)
    collaboration_manager = DynamicCollaborationManager(router)

    # 创建 Agents（注入共享的 LLM 客户端）
    user_agent = UserAgent()
    research_agent = ResearchAgent(llm_client=llm_client)
    writer_agent = WriterAgent(llm_client=llm_client)
    analyzer_agent = AnalyzerAgent(llm_client=llm_client)

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

    # ----------------------------------------------------------
    # 1. 演示研究流程（LLM 驱动：Researcher → Writer → User）
    # ----------------------------------------------------------
    print("\n1. 演示研究流程:")
    user_question = "人工智能的发展现状"
    print(f"[System] 用户提问: {user_question}")

    initial_message = Message(
        sender="user",
        receiver="researcher",
        content=user_question,
        msg_type="research_request"
    )
    print("\n开始处理用户研究请求...")
    router.route_message(initial_message)

    # ----------------------------------------------------------
    # 2. 演示计算分析流程（LLM 自主选择 calculator 工具）
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. 演示计算分析流程:")
    calculation_request = "计算 (123 + 456) * 2 的结果"
    print(f"[System] 分析请求: {calculation_request}")

    calc_message = Message(
        sender="user",
        receiver="analyzer",
        content=calculation_request,
        msg_type="calculate"
    )
    print("\n开始处理分析请求...")
    router.route_message(calc_message)

    # ----------------------------------------------------------
    # 3. 演示动态协作
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. 演示动态协作:")
    collaboration_task = "研究机器学习的主要应用领域"
    print(f"[System] 协作任务: {collaboration_task}")

    success = collaboration_manager.initiate_collaboration(collaboration_task)
    if success:
        print("动态协作已启动")
    else:
        print("动态协作启动失败")

    # ----------------------------------------------------------
    # 4. 系统状态
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. 系统状态:")
    monitor.print_real_time_status()

    stats = collaboration_manager.get_collaboration_stats()
    print(f"\n协作策略数量: {stats['strategy_count']}")
    print(f"任务历史数量: {stats['task_history_count']}")

    # 停止监控并导出报告
    monitor.stop_monitoring()
    monitor.export_monitoring_report("monitoring_report.json")
    router.export_message_log("message_log.json")

    print("\n监控报告已导出到 monitoring_report.json")
    print("消息日志已导出到 message_log.json")
    print("\n综合演示完成")


if __name__ == "__main__":
    main()
