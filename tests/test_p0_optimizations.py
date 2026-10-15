"""
P0 优化项的单元测试

覆盖 2026-10 架构升级的四个主题：
- 真流式：think() 主路径走 chat_stream，文本/思考增量实时回调；
  流式工具调用碎片累积拼装；流失败安全回退
- 混合推理模型：ChatResult.reasoning_content 透传 + 多轮回传；
  GLMClient 的 thinking 开关与 response_format
- 结构化规划：HierarchicalProcess 优先 response_format JSON 模式，
  不支持的客户端自动降级，解析兼容对象/数组两种格式
- 工具参数校验：按工具 schema 做必填检查与类型矫正，错误回传模型自纠
"""

import json
from types import SimpleNamespace

import pytest

from core.agent import Agent
from core.llm import (
    ChatResult,
    GLMClient,
    LLMClient,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from core.react_engine import AgentStreamEvent
from core.team import HierarchicalProcess, SharedContext, Team
from core.tools import CalculatorTool, ToolRegistry, WebSearchTool


def _agent_with(llm, tools=None, name="测试"):
    class A(Agent):
        def __init__(self):
            super().__init__("a1", name, system_prompt="你是助手", llm_client=llm)
            for t in tools or []:
                self.tool_registry.register(t)

        def receive(self, message):
            pass

    return A()


# ====================================================================
# GLMClient：thinking 开关 / response_format / reasoning 提取
# ====================================================================


def _fake_chat_response(content="答复", reasoning="", tool_calls=None):
    """构造 zhipuai 非流式响应的同构对象（不联网）"""
    msg = SimpleNamespace(
        content=content, reasoning_content=reasoning or None, tool_calls=tool_calls
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


class _FakeCompletions:
    """记录 create 收到的 kwargs 并返回预置响应（流式/非流式两用）"""

    def __init__(self, response=None, stream_chunks=None):
        self.kwargs_seen = []
        self._response = response
        self._stream_chunks = stream_chunks

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._stream_chunks or [])
        return self._response


def _make_glm(fake, **client_kwargs):
    llm = GLMClient(api_key="test-key", **client_kwargs)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return llm


class TestGLMClientModernFeatures:
    def test_thinking_enabled_kwarg(self):
        fake = _FakeCompletions(response=_fake_chat_response())
        llm = _make_glm(fake, thinking=True)
        llm.chat([{"role": "user", "content": "q"}])
        assert fake.kwargs_seen[0]["thinking"] == {"type": "enabled"}

    def test_thinking_disabled_kwarg(self):
        fake = _FakeCompletions(response=_fake_chat_response())
        llm = _make_glm(fake, thinking=False)
        llm.chat([])
        assert fake.kwargs_seen[0]["thinking"] == {"type": "disabled"}

    def test_thinking_default_not_sent(self):
        fake = _FakeCompletions(response=_fake_chat_response())
        llm = _make_glm(fake)
        llm.chat([])
        assert "thinking" not in fake.kwargs_seen[0]

    def test_response_format_passthrough(self):
        fake = _FakeCompletions(response=_fake_chat_response(content='{"a":1}'))
        llm = _make_glm(fake)
        llm.chat([], response_format={"type": "json_object"})
        assert fake.kwargs_seen[0]["response_format"] == {"type": "json_object"}

    def test_reasoning_content_extracted(self):
        fake = _FakeCompletions(response=_fake_chat_response(reasoning="内心思考"))
        llm = _make_glm(fake)
        result = llm.chat([])
        assert result.reasoning_content == "内心思考"
        assert result.content == "答复"

    def test_thinking_sent_in_stream(self):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="你"), finish_reason=None)],
            usage=None,
        )
        fake = _FakeCompletions(stream_chunks=[chunk])
        llm = _make_glm(fake, thinking=True)
        list(llm.chat_stream([]))
        assert fake.kwargs_seen[0]["thinking"] == {"type": "enabled"}

    def test_stream_yields_reasoning_and_tool_fragments(self):
        """流式 chunk 正确转出思考增量与工具调用碎片"""
        tc_frag = SimpleNamespace(
            index=0,
            id="c1",
            function=SimpleNamespace(name="calculator", arguments='{"express'),
        )
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, reasoning_content="思考A", tool_calls=None
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, reasoning_content=None, tool_calls=[tc_frag]
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
        ]
        fake = _FakeCompletions(stream_chunks=chunks)
        llm = _make_glm(fake)
        out = list(llm.chat_stream([]))
        assert out[0].delta_reasoning == "思考A"
        assert out[1].delta_tool_calls[0]["name"] == "calculator"
        assert out[1].delta_tool_calls[0]["arguments"] == '{"express'


# ====================================================================
# 真流式主路径（engine）
# ====================================================================


class ToolStreamingLLM(LLMClient):
    """两步流式 mock：第 1 步思考+叙述+工具调用（参数分片），第 2 步最终答复"""

    def __init__(self):
        self.stream_calls = 0
        self.chat_calls = 0
        self.messages_per_call: list[list] = []

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.chat_calls += 1
        return ChatResult(content="非流式结果", usage=TokenUsage(1, 1, 2))

    def chat_stream(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.stream_calls += 1
        self.messages_per_call.append([dict(m) for m in messages])
        if self.stream_calls == 1:
            yield StreamChunk(delta_reasoning="让我想想")
            yield StreamChunk(delta_content="我先计算。")
            yield StreamChunk(
                delta_tool_calls=[
                    {"index": 0, "id": "c1", "name": "calculator", "arguments": '{"expression":'}
                ]
            )
            yield StreamChunk(delta_tool_calls=[{"index": 0, "arguments": ' "2**10"}'}])
            yield StreamChunk(finish_reason="tool_calls", usage=TokenUsage(10, 5, 15))
        else:
            yield StreamChunk(delta_content="结果是 ")
            yield StreamChunk(delta_content="1024", finish_reason="stop", usage=TokenUsage(5, 3, 8))


class TestEngineTrueStreaming:
    def test_streaming_main_path_with_tool_calls(self):
        """流式贯穿主循环：文本实时到达、工具碎片拼装、结果正确"""
        llm = ToolStreamingLLM()
        agent = _agent_with(llm, tools=[CalculatorTool()])

        text_chunks, events = [], []
        result = agent.think(
            "计算 2**10",
            stream_callback=text_chunks.append,
            events_callback=events.append,
        )

        # 全程流式（无非流式 chat 调用），共 2 次流式调用
        assert llm.chat_calls == 0
        assert llm.stream_calls == 2
        # 文本增量实时送达：叙述 + 最终答复
        assert "".join(text_chunks) == "我先计算。结果是 1024"
        assert result == "结果是 1024"
        # token 按流式 usage 累计（15 + 8）
        assert agent.get_token_usage().total_tokens == 23

        # 结构化事件：思考 → 叙述文本 → 工具调用 → 工具结果 → 最终文本×2
        seq = [e.type for e in events]
        assert seq == ["reasoning", "text", "tool_call", "tool_result", "text", "text"]
        tool_event = events[2]
        assert tool_event.content["name"] == "calculator"
        assert tool_event.content["arguments"] == {"expression": "2**10"}
        tool_result_event = events[3]
        assert tool_result_event.content["result"]["result"] == 1024

    def test_reasoning_roundtrip_in_history(self):
        """思考内容随 assistant 消息回传（Preserved Thinking），工具结果带 tool_call_id"""
        llm = ToolStreamingLLM()
        agent = _agent_with(llm, tools=[CalculatorTool()])
        agent.think("计算 2**10", stream_callback=lambda s: None)

        second_call_messages = llm.messages_per_call[1]
        assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
        assert assistant_msgs, "第二步应能看到第一步的 assistant 消息"
        first_assistant = assistant_msgs[-1]
        assert first_assistant.get("reasoning_content") == "让我想想"
        assert first_assistant["tool_calls"][0]["function"]["name"] == "calculator"

        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "c1"
        assert json.loads(tool_msgs[0]["content"])["result"] == 1024

    def test_stream_failure_before_output_falls_back_to_chat(self):
        """流在产出任何内容前失败：安全回退一次非流式调用"""

        class BrokenStreamLLM(LLMClient):
            def __init__(self):
                self.chat_calls = 0

            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                self.chat_calls += 1
                return ChatResult(content="回退结果", usage=TokenUsage(1, 1, 2))

            def chat_stream(self, messages, tools=None, tool_choice="auto", **kwargs):
                raise ConnectionError("流建立失败")
                yield  # pragma: no cover

        llm = BrokenStreamLLM()
        agent = _agent_with(llm)
        result = agent.think("q", stream_callback=lambda s: None)
        assert result == "回退结果"
        assert llm.chat_calls == 1

    def test_events_callback_exception_does_not_break(self):
        """事件回调抛异常不影响推理主流程"""

        llm = ToolStreamingLLM()
        agent = _agent_with(llm, tools=[CalculatorTool()])

        def bad_callback(event):
            raise ValueError("前端渲染出错")

        result = agent.think("计算 2**10", events_callback=bad_callback)
        assert result == "结果是 1024"

    def test_base_chat_stream_carries_tool_calls_for_sync_clients(self):
        """仅实现同步 chat 的客户端：基类流式回退完整携带工具调用"""

        class SyncOnlyLLM(LLMClient):
            def __init__(self):
                self.i = 0

            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                self.i += 1
                if self.i == 1:
                    return ChatResult(
                        content="",
                        tool_calls=[
                            ToolCall(id="t9", name="calculator", arguments={"expression": "1+2"})
                        ],
                        finish_reason="tool_calls",
                        usage=TokenUsage(2, 2, 4),
                    )
                return ChatResult(content="sync最终答复", usage=TokenUsage(1, 1, 2))

        agent = _agent_with(SyncOnlyLLM(), tools=[CalculatorTool()])
        collected = []
        result = agent.think("q", stream_callback=collected.append)
        assert result == "sync最终答复"
        assert "".join(collected) == "sync最终答复"


# ====================================================================
# 结构化规划（HierarchicalProcess）
# ====================================================================


class StructuredPlanLLM(LLMClient):
    """接受 response_format 的规划 mock：返回 JSON 对象并记录收到的 kwargs"""

    def __init__(self, content):
        self.content = content
        self.kwargs_seen = []

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.kwargs_seen.append(kwargs)
        return ChatResult(content=self.content, finish_reason="stop")


class NarrowSignaturePlanLLM(LLMClient):
    """不支持 response_format 的窄签名 mock（触发 TypeError 降级路径）"""

    def __init__(self, content):
        self.content = content
        self.chat_calls = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.chat_calls += 1
        return ChatResult(content=self.content, finish_reason="stop")


def _make_member(agent_id, capability, reply):
    class M(Agent):
        def __init__(self):
            super().__init__(
                agent_id, agent_id, llm_client=_EchoLLM(reply), capabilities=[capability]
            )

        def receive(self, message):
            pass

    return M()


class _EchoLLM(LLMClient):
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        return ChatResult(content=self.reply, usage=TokenUsage(1, 1, 2))


class TestStructuredPlanning:
    def test_response_format_requested_when_supported(self):
        plan = json.dumps(
            {"subtasks": [{"subtask": "查资料", "capability": "search"}]},
            ensure_ascii=False,
        )
        llm = StructuredPlanLLM(plan)

        class L(Agent):
            def __init__(self):
                super().__init__("leader", "经理", llm_client=llm, capabilities=["plan"])

            def receive(self, m):
                pass

        leader = L()
        proc = HierarchicalProcess()
        subtasks = proc._plan("任务", [], leader, SharedContext())
        assert subtasks == [{"subtask": "查资料", "capability": "search"}]
        # 结构化输出约束确实传给了客户端
        assert any(kw.get("response_format") == {"type": "json_object"} for kw in llm.kwargs_seen)

    def test_narrow_client_degrades_gracefully(self):
        """客户端不支持 response_format（TypeError）时降级普通调用，规划仍可用"""
        plan = json.dumps([{"subtask": "干活", "capability": "work"}], ensure_ascii=False)
        llm = NarrowSignaturePlanLLM(plan)

        class L(Agent):
            def __init__(self):
                super().__init__("leader", "经理", llm_client=llm, capabilities=["plan"])

            def receive(self, m):
                pass

        proc = HierarchicalProcess()
        subtasks = proc._plan("任务", [], L(), SharedContext())
        assert subtasks == [{"subtask": "干活", "capability": "work"}]
        assert llm.chat_calls == 1  # 降级后只调了一次

    def test_parse_plan_accepts_legacy_array(self):
        proc = HierarchicalProcess()
        raw = json.dumps([{"subtask": "a", "capability": "b"}], ensure_ascii=False)
        assert proc._parse_plan(raw, []) == [{"subtask": "a", "capability": "b"}]

    def test_parse_plan_accepts_object_with_subtasks(self):
        proc = HierarchicalProcess()
        raw = json.dumps(
            {"subtasks": [{"task": "a", "ability": "b"}, "垃圾项"]}, ensure_ascii=False
        )
        assert proc._parse_plan(raw, []) == [{"subtask": "a", "capability": "b"}]

    def test_parse_plan_embedded_json_in_prose(self):
        proc = HierarchicalProcess()
        raw = '好的，规划如下：\n{"subtasks": [{"subtask": "x", "capability": "y"}]}\n以上。'
        assert proc._parse_plan(raw, []) == [{"subtask": "x", "capability": "y"}]

    def test_parse_plan_garbage_returns_empty(self):
        proc = HierarchicalProcess()
        assert proc._parse_plan("这不是JSON", []) == []

    def test_hierarchical_team_runs_with_structured_plan(self):
        """端到端：结构化规划 → 按能力执行 → Leader 汇总"""
        plan = json.dumps(
            {
                "subtasks": [
                    {"subtask": "检索资料", "capability": "search"},
                    {"subtask": "撰写内容", "capability": "write"},
                ]
            },
            ensure_ascii=False,
        )

        # leader：规划返回结构化 JSON；汇总返回固定文本
        class LeaderLLM(StructuredPlanLLM):
            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                self.kwargs_seen.append(kwargs)
                last = messages[-1]["content"]
                if "项目经理" in last:
                    return ChatResult(content=self.content, finish_reason="stop")
                return ChatResult(content="汇总完成", usage=TokenUsage(1, 1, 2))

        leader_llm = LeaderLLM(plan)

        class Leader(Agent):
            def __init__(self):
                super().__init__("leader", "经理", llm_client=leader_llm, capabilities=["plan"])

            def receive(self, m):
                pass

        researcher = _make_member("researcher", "search", "研究产出")
        writer = _make_member("writer", "write", "写作产出")
        team = Team(
            name="内容组",
            members=[researcher, writer],
            process=HierarchicalProcess(),
            leader=Leader(),
        )
        result = team.run("做一份报告")
        assert result == "汇总完成"
        ctx_outputs = team.task_history[-1]["context_summary"]["output_agents"]
        assert set(ctx_outputs) == {"researcher", "writer"}


# ====================================================================
# 工具参数校验（P0-4）
# ====================================================================


class TestToolArgumentValidation:
    def test_missing_required_raises_with_schema(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        with pytest.raises(ValueError, match="缺少必填参数.*expression"):
            reg.execute("calculator")

    def test_unknown_tool_still_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="not found"):
            reg.execute("ghost")

    def test_numeric_string_coerced_for_number_schema(self):
        from core.tools import Tool

        class ThermometerTool(Tool):
            def __init__(self):
                super().__init__(
                    "thermometer",
                    "测温",
                    parameters={
                        "type": "object",
                        "properties": {"celsius": {"type": "number"}},
                        "required": ["celsius"],
                    },
                )

            def run(self, celsius):
                return {"celsius": celsius, "kind": type(celsius).__name__}

        reg = ToolRegistry()
        reg.register(ThermometerTool())
        # LLM 把数字给成字符串 "36.5" → 矫正回 float 后执行
        r = reg.execute("thermometer", celsius="36.5")
        assert r == {"celsius": 36.5, "kind": "float"}

    def test_integer_coercion(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        # 表达式 schema 声明 string；int 123 被矫正为 "123" 后正常计算
        r = reg.execute("calculator", expression=123)
        assert r["result"] == 123

    def test_string_coercion_for_search_query(self):
        reg = ToolRegistry()
        reg.register(WebSearchTool(mock_mode=True))
        # LLM 把 query 给成 int 42 → 矫正为 "42"，mock 模式正常返回
        r = reg.execute("web_search", query=42)
        assert r["query"] == "42"
        assert "results" in r

    def test_boolean_coercion(self):
        from core.tools import _coerce_to_schema_type

        value, ok = _coerce_to_schema_type("boolean", "true")
        assert ok and value is True
        value, ok = _coerce_to_schema_type("integer", "42")
        assert ok and value == 42 and isinstance(value, int)

    def test_uncorrectable_type_raises(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        with pytest.raises(ValueError, match="参数 'expression' 需要 string 类型"):
            reg.execute("calculator", expression=["2+2"])

    def test_model_self_corrects_after_validation_error(self):
        """校验错误回传给模型后，模型修正参数重试成功"""

        class SelfCorrectingLLM(LLMClient):
            def __init__(self):
                self.i = 0
                self.seen_tool_messages = []

            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                self.i += 1
                if self.i == 1:
                    return ChatResult(
                        content="",
                        tool_calls=[
                            # 缺 expression：触发校验失败并回传
                            ToolCall(id="t1", name="calculator", arguments={})
                        ],
                        finish_reason="tool_calls",
                        usage=TokenUsage(2, 2, 4),
                    )
                self.seen_tool_messages = [m for m in messages if m.get("role") == "tool"]
                return ChatResult(content="修正后完成", usage=TokenUsage(1, 1, 2))

        llm = SelfCorrectingLLM()
        agent = _agent_with(llm, tools=[CalculatorTool()])
        result = agent.think("计算 2*3")
        assert result == "修正后完成"
        # 模型在第二步看到了带 schema 说明的校验错误，具备自纠信息
        error_payload = json.loads(llm.seen_tool_messages[0]["content"])
        assert "参数校验失败" in error_payload["error"]
        assert "expression" in error_payload["error"]


# ====================================================================
# 事件协议与导出
# ====================================================================


class TestStreamEventProtocol:
    def test_event_dataclass_shape(self):
        ev = AgentStreamEvent(type="text", content="你好", step=2)
        assert ev.type == "text"
        assert ev.content == "你好"
        assert ev.step == 2

    def test_exported_from_core(self):
        import core

        assert core.AgentStreamEvent is AgentStreamEvent

    def test_session_passes_events_through(self):
        from core.session import ConversationSession

        llm = _EchoLLM("会话回复")
        agent = _agent_with(llm)
        session = ConversationSession(agent)
        events = []
        reply = session.chat("你好", events_callback=events.append)
        assert reply == "会话回复"
        assert [e.type for e in events] == ["text"]
