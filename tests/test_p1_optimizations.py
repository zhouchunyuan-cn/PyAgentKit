"""
P1 优化项的单元测试

覆盖：
- 异步化：LLMClient.achat/achat_stream 线程桥接、Agent.think_async
  （contextvars 传播：异步路径 Tracer 不丢 span）
- 并行执行：ReAct 循环内多个 tool_calls 并发执行（Barrier 验证）；
  HierarchicalProcess parallel_subtasks 按成员分组 fan-out；
  YAML team.parallel 配置
- 上下文工程：长期记忆注入的条数/字符预算；ConversationSession
  滚动摘要压缩（summarize_on_trim，失败回退硬截断）
- 真 MCP 客户端：MCPServerConfig 校验、工具发现/调用（假会话注入）、
  ToolRegistry.register_mcp_server 一步接入、异步入口、依赖缺失提示
"""

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

from core.agent import Agent
from core.config import ConfigLoader
from core.llm import ChatResult, LLMClient, StreamChunk, TokenUsage, ToolCall
from core.mcp_client import (
    MCPClientError,
    MCPServerConfig,
    MCPToolProvider,
    RemoteMCPTool,
)
from core.memory_manager import MemoryManager
from core.session import ConversationSession
from core.team import HierarchicalProcess, Team
from core.tools import Tool, ToolRegistry
from core.trace import Tracer


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _agent_with(llm, tools=None):
    class A(Agent):
        def __init__(self):
            super().__init__("a1", "测试", system_prompt="你是助手", llm_client=llm)
            for t in tools or []:
                self.tool_registry.register(t)

        def receive(self, message):
            pass

    return A()


class EchoLLM(LLMClient):
    """按序返回预置结果的 mock"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.results.pop(0)


# ====================================================================
# P1-1：异步化
# ====================================================================


class TestAsyncLLM:
    def test_achat_thread_bridge(self):
        llm = EchoLLM([ChatResult(content="异步结果", usage=TokenUsage(1, 2, 3))])
        result = run_async(llm.achat([{"role": "user", "content": "q"}]))
        assert result.content == "异步结果"
        assert result.usage.total_tokens == 3

    def test_achat_stream_bridge_preserves_order(self):
        class StreamMock(EchoLLM):
            def chat_stream(self, messages, tools=None, tool_choice="auto"):
                for text in ["A", "B", "C"]:
                    yield StreamChunk(delta_content=text)

        async def collect():
            return [c.delta_content async for c in llm.achat_stream([])]

        llm = StreamMock([])
        assert run_async(collect()) == ["A", "B", "C"]

    def test_think_async_runs_engine(self):
        llm = EchoLLM([ChatResult(content="异步推理结果")])
        agent = _agent_with(llm)
        result = run_async(agent.think_async("你好"))
        assert result == "异步推理结果"

    def test_think_async_propagates_contextvars(self):
        """to_thread 复制上下文：异步路径中 Tracer 不丢 span"""
        llm = EchoLLM([ChatResult(content="ok")])
        agent = _agent_with(llm)
        Tracer.enable()
        try:
            trace = Tracer.start_trace("async-trace")
            run_async(agent.think_async("q"))
            ended = Tracer.end_trace()
            assert ended is trace
            # think 的 span 在异步路径中被记录（修复前 threading.local 会丢失）
            assert any(s.name.startswith("think:") for s in trace.spans)
        finally:
            Tracer.disable()

    def test_think_async_with_stream_callback(self):
        llm = EchoLLM([ChatResult(content="流式内容")])
        agent = _agent_with(llm)
        collected = []
        run_async(agent.think_async("q", stream_callback=collected.append))
        assert "".join(collected) == "流式内容"


# ====================================================================
# P1-2a：循环内并行工具执行
# ====================================================================


class BarrierTool(Tool):
    """在共享 Barrier 上会合：若两个工具被串行执行会超时返回错误标记"""

    def __init__(self, name, barrier, marker):
        super().__init__(name, f"{name} 工具", {"type": "object", "properties": {}, "required": []})
        self.barrier = barrier
        self.marker = marker

    def run(self):
        # Barrier.wait 成功返回本方序号（int），超时返回 False
        reached = self.barrier.wait(timeout=8)
        return {"marker": self.marker, "met": reached is not False}


class TestParallelToolCalls:
    def test_two_tool_calls_run_concurrently(self):
        barrier = threading.Barrier(2)
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="tool_a", arguments={}),
                        ToolCall(id="c2", name="tool_b", arguments={}),
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="两个工具都完成了"),
            ]
        )
        agent = _agent_with(
            llm, tools=[BarrierTool("tool_a", barrier, "A"), BarrierTool("tool_b", barrier, "B")]
        )
        result = agent.think("并发执行")
        assert result == "两个工具都完成了"
        # 模型在第二步看到了两个工具结果，且都在 Barrier 上会合（并发证明）
        tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        for m in tool_messages:
            payload = json.loads(m["content"])
            assert payload["met"] is True

    def test_results_keep_submission_order(self):
        class MarkerTool(Tool):
            def __init__(self, name, marker):
                super().__init__(name, name, {"type": "object", "properties": {}})
                self.marker = marker

            def run(self):
                return {"marker": self.marker}

        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="second", arguments={}),
                        ToolCall(id="c2", name="first", arguments={}),
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="done"),
            ]
        )
        agent = _agent_with(llm, tools=[MarkerTool("first", "1st"), MarkerTool("second", "2nd")])
        agent.think("排序")
        tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
        # 结果顺序与 tool_calls 声明顺序一致（second, first），与工具注册顺序无关
        assert json.loads(tool_messages[0]["content"])["marker"] == "2nd"
        assert json.loads(tool_messages[1]["content"])["marker"] == "1st"
        assert tool_messages[0]["tool_call_id"] == "c1"

    def test_single_tool_call_no_thread_overhead(self):
        """单个工具调用走直接执行路径（结果正确即可）"""
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="only", arguments={})],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="done"),
            ]
        )
        agent = _agent_with(llm, tools=[MarkerTool_Single("only", "solo")])
        assert agent.think("q") == "done"


class MarkerTool_Single(Tool):
    def __init__(self, name, marker):
        super().__init__(name, name, {"type": "object", "properties": {}})
        self.marker = marker

    def run(self):
        return {"marker": self.marker}


# ====================================================================
# P1-2b：HierarchicalProcess 并行子任务
# ====================================================================


def _member(agent_id, capability, llm):
    class M(Agent):
        def __init__(self):
            super().__init__(agent_id, agent_id, llm_client=llm, capabilities=[capability])

        def receive(self, message):
            pass

    return M()


class BarrierLLM(LLMClient):
    """chat 时在共享 Barrier 上会合：若两个成员被串行执行会超时"""

    def __init__(self, barrier, marker):
        self.barrier = barrier
        self.marker = marker

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        met = self.barrier.wait(timeout=8)
        return ChatResult(content=f"{self.marker}-{'met' if met else 'TIMEOUT'}")


class PlanLeaderLLM(LLMClient):
    """规划返回固定 JSON；汇总返回固定文本"""

    def __init__(self, plan):
        self.plan = plan
        self.chat_calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.chat_calls += 1
        last = messages[-1]["content"]
        if "项目经理" in last:
            return ChatResult(content=self.plan)
        return ChatResult(content="汇总结果")


class TestParallelSubtasks:
    def test_subtasks_fan_out_across_members(self):
        barrier = threading.Barrier(2)
        plan = json.dumps(
            {
                "subtasks": [
                    {"subtask": "检索", "capability": "search"},
                    {"subtask": "写作", "capability": "write"},
                ]
            }
        )

        class L(Agent):
            def __init__(self):
                super().__init__(
                    "leader", "经理", llm_client=PlanLeaderLLM(plan), capabilities=["plan"]
                )

            def receive(self, m):
                pass

        researcher = _member("researcher", "search", BarrierLLM(barrier, "研究"))
        writer = _member("writer", "write", BarrierLLM(barrier, "写作"))
        team = Team(
            name="并行组",
            members=[researcher, writer],
            process=HierarchicalProcess(parallel_subtasks=True),
            leader=L(),
        )
        result = team.run("任务")
        assert result == "汇总结果"
        summary = team.task_history[-1]["context_summary"]
        assert set(summary["output_agents"]) == {"researcher", "writer"}
        # 两个成员的产出都不含 TIMEOUT → 它们确实并发执行（Barrier 会合成功）
        outputs_text = json.dumps(team.task_history[-1], ensure_ascii=False)
        assert "TIMEOUT" not in outputs_text

    def test_same_member_subtasks_stay_serial_within_group(self):
        """并行模式下同一成员的多个子任务组内串行（会话状态不并发写）"""
        plan = json.dumps(
            {
                "subtasks": [
                    {"subtask": "任务一", "capability": "work"},
                    {"subtask": "任务二", "capability": "work"},
                ]
            }
        )
        results = iter(["产出1", "产出2", "汇总"])
        lock = threading.Lock()
        active = {"n": 0}

        class SerialCheckLLM(LLMClient):
            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                with lock:
                    active["n"] += 1
                    assert active["n"] == 1, "同一成员被并发执行"
                    current = next(results)
                    active["n"] -= 1
                return ChatResult(content=current)

        class L(Agent):
            def __init__(self):
                super().__init__(
                    "leader", "经理", llm_client=PlanLeaderLLM(plan), capabilities=["plan"]
                )

            def receive(self, m):
                pass

        worker = _member("worker", "work", SerialCheckLLM())
        team = Team(
            name="同成员组",
            members=[worker],
            process=HierarchicalProcess(parallel_subtasks=True),
            leader=L(),
        )
        assert team.run("任务") == "汇总结果"
        assert team.task_history[-1]["context_summary"]["output_count"] == 2

    def test_config_parses_parallel_option(self):
        loader = ConfigLoader()
        config = loader.parse(
            {
                "llm": {"model": "glm-4-flash"},
                "agents": [
                    {"id": "leader", "capabilities": ["plan"]},
                    {"id": "w1", "capabilities": ["work"]},
                ],
                "team": {
                    "name": "并行团队",
                    "process": "hierarchical",
                    "leader": "leader",
                    "parallel": True,
                },
            }
        )
        assert config.team.parallel is True
        # 默认不并行
        config2 = loader.parse(
            {
                "llm": {},
                "agents": [{"id": "a"}],
                "team": {"process": "hierarchical", "leader": "a"},
            }
        )
        assert config2.team.parallel is False


# ====================================================================
# P1-3：上下文工程
# ====================================================================


class TestLongTermContextBudget:
    def test_entry_count_cap(self):
        mm = MemoryManager(max_long_term_in_context=5)
        for i in range(30):
            mm.remember(f"key{i}", f"值{i}", memory_type="long")
        messages = mm.build_context(user_input="q", system_prompt="s")
        brief = [m for m in messages if "长期记忆" in m["content"]][0]["content"]
        assert "key29" in brief  # 最新的保留
        assert "key25" in brief  # 最近 5 条边界
        assert "key24" not in brief  # 第 6 条起丢弃
        assert "key0" not in brief

    def test_char_budget(self):
        mm = MemoryManager(max_long_term_in_context=20, long_term_char_budget=1000)
        for i in range(6):
            mm.remember(f"k{i}", "长" * 400, memory_type="long")
        messages = mm.build_context(user_input="q", system_prompt="s")
        brief = [m for m in messages if "长期记忆" in m["content"]][0]["content"]
        assert len(brief) < 1000 + 200  # 预算约束生效（留出模板余量）
        assert "k5" in brief and "k4" in brief  # 最近的保留
        assert "k0" not in brief and "k1" not in brief  # 更旧的被裁

    def test_non_serializable_value_degrades(self):
        mm = MemoryManager()
        mm.memory.long_term_memory["obj"] = object()  # 绕过校验直接注入
        messages = mm.build_context(user_input="q", system_prompt="s")
        assert any("长期记忆" in m["content"] for m in messages)  # 不抛异常


class SummaryAwareLLM(LLMClient):
    """think 调用返回固定回复；摘要压缩调用返回固定摘要（按 prompt 内容区分）"""

    def __init__(self):
        self.calls = []
        self.summarize_fail = False

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append([dict(m) for m in messages])
        last = messages[-1]["content"]
        if "简短摘要" in last:
            if self.summarize_fail:
                raise RuntimeError("摘要服务不可用")
            return ChatResult(content="用户偏好深色主题，讨论了预算问题")
        return ChatResult(content="正常回复")


class TestSessionSummarizeOnTrim:
    def _make_session(self, llm, **kw):
        agent = _agent_with(llm)
        return ConversationSession(agent, max_history=4, summarize_on_trim=True, **kw)

    def test_trim_produces_rolling_summary(self):
        llm = SummaryAwareLLM()
        session = self._make_session(llm)
        for i in range(4):
            session.chat(f"第{i}轮问题")  # 超过 max_history=4 条消息即触发裁剪+摘要
        assert len(session.history) <= 4
        # 后续 think 的上下文里注入了滚动摘要
        # （calls[-1] 是裁剪时的摘要压缩调用，think 调用取多消息的那次）
        think_calls = [c for c in llm.calls if len(c) > 1]
        last_think = think_calls[-1]
        system_msgs = [
            m
            for m in last_think
            if m.get("role") == "system" and "此前对话的摘要" in m.get("content", "")
        ]
        assert system_msgs
        assert "深色主题" in system_msgs[0]["content"]

    def test_summarize_failure_falls_back_to_hard_trim(self):
        llm = SummaryAwareLLM()
        llm.summarize_fail = True
        session = self._make_session(llm)
        for i in range(6):
            reply = session.chat(f"第{i}轮")
            assert reply == "正常回复"  # 摘要失败不影响主流程
        assert len(session.history) <= 4
        assert session._summary is None

    def test_clear_history_resets_summary(self):
        llm = SummaryAwareLLM()
        session = self._make_session(llm)
        for i in range(6):
            session.chat(f"第{i}轮")
        session.clear_history()
        assert session._summary is None
        ctx = session._build_context_messages()
        assert not any("摘要" in m.get("content", "") for m in ctx)

    def test_off_by_default(self):
        llm = SummaryAwareLLM()
        agent = _agent_with(llm)
        session = ConversationSession(agent, max_history=2)
        for i in range(4):
            session.chat(f"第{i}轮")
        assert session._summary is None  # 默认不摘要
        assert len(session.history) == 2


# ====================================================================
# P1-4：真 MCP 客户端
# ====================================================================


class FakeMCPSession:
    """注入用的假 ClientSession（避免真实网络/子进程）"""

    def __init__(self, fail_calls=False):
        self.calls = []
        self.fail_calls = fail_calls

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="add",
                    description="加法",
                    inputSchema={
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                ),
                SimpleNamespace(name="echo", description="回声", inputSchema={"type": "object"}),
            ]
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail_calls:
            return SimpleNamespace(content=[SimpleNamespace(text="boom")], isError=True)
        if name == "add":
            return SimpleNamespace(
                content=[SimpleNamespace(text=str(arguments["a"] + arguments["b"]))], isError=False
            )
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)


def _patch_open(monkeypatch, session):
    async def fake_open(self):
        return session, AsyncExitStack()

    monkeypatch.setattr(MCPToolProvider, "_open", fake_open)
    return session


class TestMCPClient:
    def test_config_requires_exactly_one_endpoint(self):
        with pytest.raises(ValueError):
            MCPServerConfig()  # 都不填
        with pytest.raises(ValueError):
            MCPServerConfig(url="http://x/mcp", command="python")  # 都填

    def test_discover_converts_tools_with_schema(self, monkeypatch):
        _patch_open(monkeypatch, FakeMCPSession())
        provider = MCPToolProvider(MCPServerConfig(url="http://fake/mcp"))
        try:
            tools = provider.discover()
            assert [t.name for t in tools] == ["add", "echo"]
            assert tools[0].parameters["required"] == ["a", "b"]
            assert isinstance(tools[0], RemoteMCPTool)
            # 会话复用：discover 只建立一次连接
            provider.discover()
        finally:
            provider.close()

    def test_call_tool_sync(self, monkeypatch):
        session = _patch_open(monkeypatch, FakeMCPSession())
        provider = MCPToolProvider(MCPServerConfig(url="http://fake/mcp"))
        try:
            result = provider.call_tool_sync("add", {"a": 2, "b": 3})
            assert result == {"content": "5"}
            assert session.calls == [("add", {"a": 2, "b": 3})]
        finally:
            provider.close()

    def test_call_tool_error_surfaced(self, monkeypatch):
        _patch_open(monkeypatch, FakeMCPSession(fail_calls=True))
        provider = MCPToolProvider(MCPServerConfig(url="http://fake/mcp"))
        try:
            result = provider.call_tool_sync("add", {"a": 1, "b": 1})
            assert "error" in result
        finally:
            provider.close()

    def test_async_api(self, monkeypatch):
        _patch_open(monkeypatch, FakeMCPSession())

        async def run():
            provider = MCPToolProvider(MCPServerConfig(url="http://fake/mcp"))
            try:
                tools = await provider.adiscover()
                result = await provider.acall_tool("add", {"a": 20, "b": 22})
                return tools, result
            finally:
                provider.close()

        tools, result = run_async(run())
        assert len(tools) == 2
        assert result == {"content": "42"}

    def test_registry_one_step_integration(self, monkeypatch):
        """ToolRegistry.register_mcp_server：发现→注册→参数校验→执行 全链路"""
        session = _patch_open(monkeypatch, FakeMCPSession())
        registry = ToolRegistry()
        try:
            names = registry.register_mcp_server(MCPServerConfig(url="http://fake/mcp"))
            assert names == ["add", "echo"]
            # 参数校验同样作用于远端工具：字符串数字被矫正后执行
            result = registry.execute("add", a="2", b=3)
            assert result == {"content": "5.0"}
            assert session.calls[-1] == ("add", {"a": 2.0, "b": 3.0})
        finally:
            registry.close_mcp_servers()

    def test_missing_sdk_gives_clear_hint(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "mcp", None)  # 模拟未安装
        provider = MCPToolProvider(MCPServerConfig(url="http://fake/mcp"))
        with pytest.raises(MCPClientError, match="pip install"):
            provider.discover()
