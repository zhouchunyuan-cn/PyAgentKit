"""
Trace 系统单元测试

验证：
- Tracer enable/disable 与零开销
- TraceSpan start/end/duration
- Trace span 记录、嵌套、token 汇总
- Agent.think 与 Team.run 的 trace 插桩
- format_text / to_dict 序列化
"""

import pytest

from core.agent import Agent
from core.llm import ChatResult, LLMClient, TokenUsage
from core.team import SequentialProcess, Team
from core.trace import Tracer, TraceSpan


@pytest.fixture(autouse=True)
def reset_tracer():
    """每个测试前后都重置 tracer 状态"""
    Tracer.disable()
    yield
    Tracer.disable()


class FakeLLM(LLMClient):
    def chat(self, messages, tools=None, tool_choice="auto"):
        return ChatResult(content="reply", usage=TokenUsage(5, 7, 12))


def make_agent(aid="a", name="助手"):
    class A(Agent):
        def __init__(self):
            super().__init__(aid, name, llm_client=FakeLLM())

        def receive(self, m):
            pass

    return A()


# --------------------------------------------------------------------
# Tracer 开关
# --------------------------------------------------------------------


class TestTracerToggle:
    def test_disabled_by_default(self):
        assert not Tracer.is_enabled()

    def test_enable_disable(self):
        Tracer.enable()
        assert Tracer.is_enabled()
        Tracer.disable()
        assert not Tracer.is_enabled()

    def test_zero_overhead_when_disabled(self):
        """关闭时 start_span/end_span 返回 None，不记录"""
        assert Tracer.start_span("x") is None
        Tracer.end_span(None)
        Tracer.record("y")


# --------------------------------------------------------------------
# TraceSpan
# --------------------------------------------------------------------


class TestTraceSpan:
    def test_duration_when_finished(self):
        s = TraceSpan(name="x", start=100.0)
        s.end = 100.5
        assert s.duration == 0.5

    def test_duration_none_when_unfinished(self):
        s = TraceSpan(name="x", start=100.0)
        assert s.duration is None

    def test_set_attribute(self):
        s = TraceSpan(name="x")
        s.set("tokens", 42)
        assert s.attributes["tokens"] == 42

    def test_auto_id(self):
        s1 = TraceSpan(name="a")
        s2 = TraceSpan(name="b")
        assert s1.id != s2.id


# --------------------------------------------------------------------
# Trace 记录与汇总
# --------------------------------------------------------------------


class TestTraceRecording:
    def test_start_end_span_recorded(self):
        Tracer.enable()
        Tracer.start_trace("t")
        span = Tracer.start_span("step1")
        assert span is not None
        Tracer.end_span(span)
        trace = Tracer.end_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "step1"
        assert trace.spans[0].duration is not None

    def test_span_nesting(self):
        Tracer.enable()
        Tracer.start_trace("t")
        outer = Tracer.start_span("outer")
        inner = Tracer.start_span("inner")
        assert inner.parent_id == outer.id
        Tracer.end_span(inner)
        Tracer.end_span(outer)
        trace = Tracer.end_trace()
        assert len(trace.spans) == 2

    def test_total_tokens(self):
        Tracer.enable()
        Tracer.start_trace("t")
        s = Tracer.start_span("a")
        Tracer.end_span(s, total_tokens=10)
        s2 = Tracer.start_span("b")
        Tracer.end_span(s2, total_tokens=25)
        trace = Tracer.end_trace()
        assert trace.total_tokens() == 35

    def test_record_immediate_span(self):
        Tracer.enable()
        Tracer.start_trace("t")
        Tracer.record("instant", duration=1.5, key="v")
        trace = Tracer.end_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].duration == 1.5

    def test_no_trace_when_disabled(self):
        """未启用时即便调用也不报错、不记录"""
        Tracer.start_span("x")  # 不报错
        assert Tracer.current_trace() is None


# --------------------------------------------------------------------
# Agent.think 插桩
# --------------------------------------------------------------------


class TestAgentThinkTrace:
    def test_think_recorded_when_enabled(self):
        Tracer.enable()
        Tracer.start_trace("task")
        agent = make_agent()
        agent.think("hello")
        trace = Tracer.end_trace()
        # 应有一个 think span
        think_spans = [s for s in trace.spans if s.name.startswith("think:")]
        assert len(think_spans) == 1
        assert think_spans[0].attributes.get("agent") == "a"
        assert think_spans[0].attributes.get("total_tokens") == 12

    def test_think_not_recorded_when_disabled(self):
        # 不启用 tracer，think 仍正常工作，无 trace
        agent = make_agent()
        result = agent.think("hi")
        assert result == "reply"
        assert Tracer.current_trace() is None


# --------------------------------------------------------------------
# Team.run 插桩
# --------------------------------------------------------------------


class TestTeamTrace:
    def test_team_run_creates_trace(self):
        Tracer.enable()
        # 不手动 start_trace，team.run 应自动开一个
        members = [make_agent("a", "甲"), make_agent("b", "乙")]
        team = Team(name="组", members=members, process=SequentialProcess())
        team.run("任务")
        # team.run 内部已 start/end trace，结束后 current 应为 None
        # 通过 task_history 间接验证执行成功
        assert len(team.task_history) == 1

    def test_team_trace_has_member_spans(self):
        Tracer.enable()
        members = [make_agent("a", "甲"), make_agent("b", "乙")]
        team = Team(name="组", members=members, process=SequentialProcess())

        # 手动开 trace，覆盖整个 team.run
        Tracer.start_trace("outer")
        team.run("任务")
        trace = Tracer.end_trace()
        # 应有 2 个成员 span
        member_spans = [s for s in trace.spans if s.name.startswith("sequential:")]
        assert len(member_spans) == 2


# --------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------


class TestSerialization:
    def test_format_text(self):
        Tracer.enable()
        Tracer.start_trace("demo")
        s = Tracer.start_span("step")
        Tracer.end_span(s, total_tokens=10)
        trace = Tracer.end_trace()
        text = trace.format_text()
        assert "step" in text
        assert "总 token: 10" in text

    def test_to_dict(self):
        Tracer.enable()
        Tracer.start_trace("d")
        s = Tracer.start_span("x")
        Tracer.end_span(s)
        trace = Tracer.end_trace()
        d = trace.to_dict()
        assert d["name"] == "d"
        assert len(d["spans"]) == 1
        import json

        json.dumps(d)  # 可 JSON 序列化
