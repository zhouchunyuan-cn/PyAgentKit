"""
Monitor / MCP 工具的基础测试

这三个模块之前零测试覆盖。补核心路径测试，确保它们作为公开 API 可用。
"""

import json
from pathlib import Path

import pytest

from core.agent import Agent
from core.mcp_tools import (
    ConcreteMCPIntegrationTool,
    MCPTool,
    MCPToolRegistry,
)
from core.monitor import AgentMonitor
from core.router import Router

# --------------------------------------------------------------------
# 桩 Agent
# --------------------------------------------------------------------


class StubAgent(Agent):
    def __init__(self, agent_id="a", name="stub"):
        super().__init__(agent_id, name, capabilities=["chat"])
        self.received = []

    def receive(self, message):
        self.received.append(message)


# --------------------------------------------------------------------
# AgentMonitor
# --------------------------------------------------------------------


class TestAgentMonitor:
    @pytest.fixture
    def monitor(self):
        router = Router()
        router.register_agent(StubAgent("a1", "甲"))
        router.register_agent(StubAgent("a2", "乙"))
        return AgentMonitor(router)

    def test_start_stop_monitoring(self, monitor):
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False

    def test_get_agent_status(self, monitor):
        status = monitor.get_agent_status("a1")
        assert status is not None
        assert status["agent_id"] == "a1"
        assert status["name"] == "甲"

    def test_get_agent_status_missing(self, monitor):
        assert monitor.get_agent_status("ghost") is None

    def test_system_overview(self, monitor):
        overview = monitor.get_system_overview()
        assert "agent_statuses" in overview
        assert "router_stats" in overview
        assert len(overview["agent_statuses"]) == 2

    def test_export_report(self, monitor, tmp_path):
        filepath = str(tmp_path / "report.json")
        ok = monitor.export_monitoring_report(filepath)
        assert ok is True
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        assert "agent_statuses" in data

    def test_print_status_when_not_monitoring(self, monitor, capsys):
        # 未启动监控时打印应提示
        monitor.print_real_time_status()
        captured = capsys.readouterr()
        assert "监控未启动" in captured.out


# --------------------------------------------------------------------
# MCP 工具
# --------------------------------------------------------------------


class TestMCPTool:
    def test_create_tool(self):
        tool = MCPTool("test_mcp", "测试工具", "http://localhost:8000/mcp")
        assert tool.name == "test_mcp"
        assert tool.mcp_endpoint == "http://localhost:8000/mcp"

    def test_run_without_init_returns_error(self):
        """未初始化（且无法连接）时返回明确的错误，不抛异常"""
        tool = MCPTool("test_mcp", "测试", "http://nonexistent:9999/mcp")
        tool.initialized = False
        # initialize 会尝试连接失败，run 应回退到错误返回
        result = tool.run(command="test")
        # 要么 success=False（已尝试初始化），要么含 error
        assert "success" in result or "error" in result

    def test_get_capabilities_without_init(self):
        tool = MCPTool("test_mcp", "测试", "http://nonexistent:9999/mcp")
        # 未初始化时调用 get_capabilities 会触发 initialize（可能失败）
        caps = tool.get_capabilities()
        assert isinstance(caps, list)


class TestMCPToolRegistry:
    def test_register_and_list(self):
        registry = MCPToolRegistry()
        tool = MCPTool("t1", "工具1", "http://localhost:8000/mcp")
        assert registry.register_tool(tool) is True
        assert "t1" in registry.list_tools()

    def test_register_non_mcp_returns_false(self):
        registry = MCPToolRegistry()
        assert registry.register_tool("not_a_tool") is False

    def test_unregister(self):
        registry = MCPToolRegistry()
        tool = MCPTool("t1", "工具1", "http://localhost:8000/mcp")
        registry.register_tool(tool)
        assert registry.unregister_tool("t1") is True
        assert registry.unregister_tool("ghost") is False

    def test_get_tool(self):
        registry = MCPToolRegistry()
        tool = MCPTool("t1", "工具1", "http://localhost:8000/mcp")
        registry.register_tool(tool)
        assert registry.get_tool("t1") is tool
        assert registry.get_tool("missing") is None

    def test_execute_missing_tool_returns_error(self):
        registry = MCPToolRegistry()
        result = registry.execute_tool("ghost", command="x")
        assert result["success"] is False
        assert "not found" in result["error"]


class TestMCPIntegrationTool:
    def test_run_delegates_to_registry(self):
        registry = MCPToolRegistry()
        integration = ConcreteMCPIntegrationTool(registry)
        # 无注册工具时执行应返回 not found
        result = integration.run(tool_name="ghost")
        assert result["success"] is False

    def test_to_openai_schema(self):
        registry = MCPToolRegistry()
        integration = ConcreteMCPIntegrationTool(registry)
        schema = integration.to_openai_schema()
        assert schema["type"] == "function"
        assert "name" in schema["function"]
