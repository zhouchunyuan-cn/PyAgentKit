#!/usr/bin/env python3
"""
PyAgentKit 真 MCP 客户端演示

按官方协议（Model Context Protocol）接入 MCP 服务器：
initialize 握手 → tools/list 发现工具 → call_tool 调用。
支持两种官方传输：Streamable HTTP 与 stdio 子进程。

前置条件：
1. API Key：复制 .env.example 为 .env 并填入 ZHIPUAI_API_KEY
2. 可选依赖：pip install "mcp>=1.20"（或 pip install -e ".[mcp]"）
3. 一个运行中的 MCP 服务器（下方 URL / 命令按你的环境修改）

用法：
    python mcp_demo.py               # 连接 Streamable HTTP 服务器
    python mcp_demo.py --stdio       # 连接 stdio 子进程服务器
"""

import asyncio
import sys

from core import MCPServerConfig


async def demo_discover_and_call(config: MCPServerConfig) -> None:
    """异步 API：发现远端工具并调用（也可用同步版 provider.discover()）"""
    from core.mcp_client import MCPToolProvider

    provider = MCPToolProvider(config)
    try:
        tools = await provider.adiscover()
        print(f"已连接 {config.name}，发现 {len(tools)} 个工具：")
        for t in tools:
            print(f"  - {t.name}: {t.description[:60]}")

        if not tools:
            print("服务器上没有工具，结束演示")
            return

        # 调用第一个工具做冒烟测试（参数按它的 schema 给；无参工具直接调）
        first = tools[0]
        params = {k: f"demo-{k}" for k in (first.parameters.get("properties") or {})}
        result = await provider.acall_tool(first.name, params)
        print(f"\n调用 {first.name} 的结果：{result}")
    finally:
        provider.close()


def demo_registry_integration() -> None:
    """Agent 集成方式：一步完成 连接 → 发现 → 注册为本地 Tool"""
    from core.agent import Agent
    from core.llm import GLMClient

    llm = GLMClient(model="glm-4-flash")

    class MCPAgent(Agent):
        """挂载 MCP 服务器工具的通用 Agent"""

        def __init__(self, config: MCPServerConfig):
            super().__init__(
                agent_id="mcp_assistant",
                name="MCP 助手",
                system_prompt="你可以调用 MCP 服务器上的工具完成任务。",
                llm_client=llm,
                capabilities=["mcp"],
            )
            # 关键一行：连接服务器，把它的工具自动注册进来（schema 随迁）
            names = self.tool_registry.register_mcp_server(config)
            print(f"[{self.name}] 已挂载 MCP 工具：{names}")

        def receive(self, message):
            pass  # 演示场景直接用 think

    agent = MCPAgent(MCPServerConfig(url="http://localhost:8000/mcp"))
    print(agent.think("用 MCP 工具帮我处理一个示例任务"))
    agent.tool_registry.close_mcp_servers()  # 退出前断开（释放 stdio 子进程）


def main() -> None:
    print("PyAgentKit 真 MCP 客户端演示")
    print("=" * 50)

    try:
        import mcp  # noqa: F401
    except ImportError:
        print("未安装官方 mcp SDK。请运行：pip install 'mcp>=1.20'")
        print("（或安装项目可选依赖：pip install -e '.[mcp]'）")
        sys.exit(1)

    use_stdio = "--stdio" in sys.argv
    if use_stdio:
        # stdio 传输：框架拉起子进程并与之按协议通信（命令按你的服务器修改）
        config = MCPServerConfig(command="python", args=["my_mcp_server.py"], name="stdio-server")
    else:
        # Streamable HTTP 传输
        config = MCPServerConfig(url="http://localhost:8000/mcp", name="http-server")

    try:
        asyncio.run(demo_discover_and_call(config))
        demo_registry_integration()
    except ConnectionError as e:
        print(f"\n连接 MCP 服务器失败：{e}")
        print("请确认服务器已启动，并修改 mcp_demo.py 里的 url / command 配置。")


if __name__ == "__main__":
    main()
