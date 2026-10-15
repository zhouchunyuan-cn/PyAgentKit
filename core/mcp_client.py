"""
真 MCP 客户端（Model Context Protocol）

按官方协议连接 MCP 服务器：initialize 握手 → tools/list 发现 → call_tool 调用，
支持两种官方传输：
- Streamable HTTP：MCPServerConfig(url="https://host/mcp")
- stdio 子进程：MCPServerConfig(command="python", args=["server.py"])

依赖官方 mcp SDK（可选依赖，未安装时给出清晰指引）：
    pip install "mcp>=1.20"     # 或 pip install -e ".[mcp]"

线程模型：provider 拥有一条专属事件循环线程，会话在其上建立并复用
（stdio 服务器不因每次调用重启）。同步 API（discover / call_tool_sync）
阻塞等待结果；异步 API（adiscover / acall_tool）在任何事件循环中可用
（内部跨循环提交）。同一会话的调用在循环上天然串行化。

用法：
    from core import MCPServerConfig
    from core.mcp_client import MCPToolProvider

    provider = MCPToolProvider(MCPServerConfig(url="http://localhost:8000/mcp"))
    tools = provider.discover()          # [RemoteMCPTool...]（已带 schema）
    result = tools[0].run(**{"query": "..."})
    provider.close()

    # 或一步接入 Agent 的工具注册表
    agent.tool_registry.register_mcp_server(MCPServerConfig(command="python", args=["math_server.py"]))
"""

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .tools import Tool

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """MCP 客户端错误（依赖缺失、连接失败等）"""


@dataclass
class MCPServerConfig:
    """
    MCP 服务器连接配置（url 与 command 二选一）

    Attributes:
        url: Streamable HTTP 端点（如 "http://localhost:8000/mcp"）
        command: stdio 方式的服务器启动命令
        args: command 的参数列表
        env: stdio 子进程环境变量
        name: 服务器显示名（用于日志与线程命名）
    """

    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    name: str = "mcp"

    def __post_init__(self):
        if bool(self.url) == bool(self.command):
            raise ValueError("MCPServerConfig 需要且只需要 url 或 command 之一")
        if self.command and not self.url:
            self.name = self.name if self.name != "mcp" else self.command


class RemoteMCPTool(Tool):
    """远端 MCP 工具的本地适配器（schema 来自服务器 tools/list，调用走 call_tool）"""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        provider: "MCPToolProvider",
    ):
        super().__init__(name, description, input_schema or {"type": "object"})
        self.provider = provider

    def run(self, **kwargs) -> Any:
        return self.provider.call_tool_sync(self.name, kwargs)


class MCPToolProvider:
    """
    连接单个 MCP 服务器：发现工具并代理调用

    会话懒建立（首次 discover/call 时连接）并复用；close() 断开并停止
    专属事件循环线程。
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._session: Any | None = None
        self._stack: AsyncExitStack | None = None

    # ------------------------------------------------------------------
    # 专属事件循环线程
    # ------------------------------------------------------------------
    def _ensure_loop(self) -> None:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return
            self._loop_ready.clear()
            threading.Thread(
                target=self._run_loop, name=f"mcp-{self.config.name}", daemon=True
            ).start()
            if not self._loop_ready.wait(timeout=10):
                raise MCPClientError("MCP 事件循环线程启动超时")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _submit(self, coro) -> Any:
        """把协程提交到专属循环并同步等待结果（在非该循环的线程中调用）"""
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _submit_async(self, coro) -> Any:
        """把协程提交到专属循环，返回可 await 的 future（供异步调用方使用）"""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    async def _open(self) -> tuple[Any, AsyncExitStack]:
        """
        建立连接并完成 initialize 握手（测试可通过替换本方法注入假会话）

        Returns:
            (ClientSession, AsyncExitStack)
        """
        try:
            from mcp import ClientSession
        except ImportError as e:
            raise MCPClientError(
                "未安装官方 mcp SDK。请运行：pip install 'mcp>=1.20'（或 pip install -e '.[mcp]'）"
            ) from e

        stack = AsyncExitStack()
        try:
            if self.config.url:
                from mcp.client.streamable_http import streamablehttp_client

                transport = await stack.enter_async_context(streamablehttp_client(self.config.url))
                read_stream, write_stream = transport[0], transport[1]
            else:
                from mcp.client.stdio import StdioServerParameters, stdio_client

                params = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args or []),
                    env=self.config.env,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            return session, stack
        except BaseException:
            await stack.aclose()
            raise

    async def _ensure_connected(self) -> Any:
        if self._session is not None:
            return self._session
        session, stack = await self._open()
        self._session, self._stack = session, stack
        logger.info("[MCP:%s] 已连接", self.config.name)
        return session

    def close(self) -> None:
        """断开会话并停止专属事件循环线程（可再次使用，会重新连接）"""
        with self._lock:
            stack, loop = self._stack, self._loop
            self._session = None
            self._stack = None
            self._loop = None
        if stack is not None and loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(stack.aclose(), loop).result(timeout=10)
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        logger.info("[MCP:%s] 已断开", self.config.name)

    # ------------------------------------------------------------------
    # 工具发现与调用（同步 + 异步双入口）
    # ------------------------------------------------------------------
    def discover(self) -> list[RemoteMCPTool]:
        """连接服务器并发现工具（同步阻塞版）"""
        return self._submit(self._discover_once())

    async def adiscover(self) -> list[RemoteMCPTool]:
        """连接服务器并发现工具（异步版，可在任何事件循环中 await）"""
        return await self._submit_async(self._discover_once())

    async def _discover_once(self) -> list[RemoteMCPTool]:
        session = await self._ensure_connected()
        result = await session.list_tools()
        tools: list[RemoteMCPTool] = []
        for t in getattr(result, "tools", None) or []:
            schema = getattr(t, "inputSchema", None) or {"type": "object"}
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            tools.append(
                RemoteMCPTool(
                    name=t.name,
                    description=getattr(t, "description", "") or "",
                    input_schema=dict(schema),
                    provider=self,
                )
            )
        logger.info("[MCP:%s] 发现 %d 个工具", self.config.name, len(tools))
        return tools

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用远端工具（同步阻塞版）"""
        return self._submit(self._call_once(name, arguments))

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用远端工具（异步版）"""
        return await self._submit_async(self._call_once(name, arguments))

    async def _call_once(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = await self._ensure_connected()
        result = await session.call_tool(name, dict(arguments or {}))
        texts = [
            item.text
            for item in (getattr(result, "content", None) or [])
            if getattr(item, "text", None)
        ]
        payload: dict[str, Any] = {}
        if texts:
            payload["content"] = "\n".join(texts)
        if getattr(result, "isError", False):
            payload["error"] = payload.pop("content", None) or "MCP 工具执行失败"
        return payload
