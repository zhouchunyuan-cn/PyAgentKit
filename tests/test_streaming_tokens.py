"""
流式输出与 Token 统计单元测试

用 mock 数据（不联网）验证：
- TokenUsage 数据结构与累加
- ChatResult.usage 字段
- GLMClient._extract_usage 从 mock response 提取
- chat_stream 流式 yield 多块
- Agent.think 的 stream_callback 被调用
- Agent token 累计
"""
import pytest
from types import SimpleNamespace

from core.llm import (
    LLMClient, ChatResult, ToolCall, TokenUsage, StreamChunk, GLMClient,
)
from core.agent import Agent
from core.message import Message


# --------------------------------------------------------------------
# TokenUsage
# --------------------------------------------------------------------

class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0

    def test_add(self):
        u1 = TokenUsage(10, 20, 30)
        u2 = TokenUsage(1, 2, 3)
        u3 = u1.add(u2)
        assert u3.prompt_tokens == 11
        assert u3.completion_tokens == 22
        assert u3.total_tokens == 33
        # add 返回新实例，不改原对象
        assert u1.prompt_tokens == 10

    def test_chat_result_default_usage_none(self):
        r = ChatResult(content="x")
        assert r.usage is None


# --------------------------------------------------------------------
# GLMClient usage 提取（不联网，测 _extract_usage 静态方法）
# --------------------------------------------------------------------

class TestUsageExtraction:
    def test_extract_normal_usage(self):
        # 构造 mock response（模拟 zhipuai 响应结构）
        resp = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=15, completion_tokens=25, total_tokens=40
        ))
        u = GLMClient._extract_usage(resp)
        assert u.prompt_tokens == 15
        assert u.completion_tokens == 25
        assert u.total_tokens == 40

    def test_extract_no_usage_returns_none(self):
        resp = SimpleNamespace(usage=None)
        assert GLMClient._extract_usage(resp) is None

    def test_extract_partial_fields(self):
        # 缺失字段视为 0
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5))
        u = GLMClient._extract_usage(resp)
        assert u.prompt_tokens == 5
        assert u.completion_tokens == 0


# --------------------------------------------------------------------
# 流式 chat_stream（用 MockLLM 模拟 SDK 迭代器）
# --------------------------------------------------------------------

class StreamingMockLLM(LLMClient):
    """模拟流式 LLM：chat_stream 产出多个 chunk"""

    def __init__(self, chunks, usage=None):
        self._chunks = chunks
        self._usage = usage
        self.stream_calls = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        # 非流式：拼接所有 chunk 作为完整内容
        full = "".join(c for c in self._chunks if isinstance(c, str))
        return ChatResult(content=full, usage=self._usage)

    def chat_stream(self, messages, tools=None, tool_choice="auto"):
        self.stream_calls += 1
        for i, chunk_text in enumerate(self._chunks):
            is_last = (i == len(self._chunks) - 1)
            yield StreamChunk(
                delta_content=chunk_text,
                finish_reason="stop" if is_last else None,
                usage=self._usage if is_last else None,
            )


class TestStreaming:
    def test_chat_stream_yields_chunks(self):
        llm = StreamingMockLLM(["你", "好", "世界"], usage=TokenUsage(3, 6, 9))
        chunks = list(llm.chat_stream([]))
        assert len(chunks) == 3
        assert chunks[0].delta_content == "你"
        # usage 只在最后一块
        assert chunks[-1].usage is not None
        assert chunks[0].usage is None

    def test_chat_stream_concatenates(self):
        llm = StreamingMockLLM(["A", "B", "C"])
        chunks = list(llm.chat_stream([]))
        assert "".join(c.delta_content for c in chunks) == "ABC"

    def test_default_chat_stream_fallback(self):
        """未覆盖 chat_stream 的 LLMClient，用基类默认实现回退非流式"""
        class SimpleLLM(LLMClient):
            def chat(self, messages, tools=None, tool_choice="auto"):
                return ChatResult(content="完整回复", usage=TokenUsage(1, 2, 3))

        llm = SimpleLLM()
        chunks = list(llm.chat_stream([]))
        assert len(chunks) == 1
        assert chunks[0].delta_content == "完整回复"
        assert chunks[0].usage.total_tokens == 3


# --------------------------------------------------------------------
# Agent.think 流式回调与 token 累计
# --------------------------------------------------------------------

class TestAgentStreamingAndTokens:
    @pytest.fixture
    def agent_with_llm(self):
        llm = StreamingMockLLM(["你好", "世界"], usage=TokenUsage(10, 20, 30))
        class A(Agent):
            def __init__(self):
                super().__init__("a", "助手", system_prompt="你是助手", llm_client=llm)
            def receive(self, m): pass
        return A()

    def test_stream_callback_invoked(self, agent_with_llm):
        collected = []
        result = agent_with_llm.think("hi", stream_callback=lambda s: collected.append(s))
        # 流式回调应被调用，拼出完整内容
        assert "".join(collected) == "你好世界"
        assert result == "你好世界"

    def test_token_usage_accumulated(self, agent_with_llm):
        agent_with_llm.think("hi")
        usage = agent_with_llm.get_token_usage()
        # 至少累计了流式的 usage（10+20+30）
        assert usage.total_tokens >= 30

    def test_token_usage_accumulates_across_calls(self, agent_with_llm):
        agent_with_llm.think("q1")
        first = agent_with_llm.get_token_usage().total_tokens
        agent_with_llm.think("q2")
        second = agent_with_llm.get_token_usage().total_tokens
        assert second >= first  # 累加不减少

    def test_reset_token_usage(self, agent_with_llm):
        agent_with_llm.think("q")
        agent_with_llm.reset_token_usage()
        u = agent_with_llm.get_token_usage()
        assert u.total_tokens == 0

    def test_no_callback_uses_nonstream(self):
        """不传 stream_callback 时走非流式 chat"""
        llm = StreamingMockLLM(["x"], usage=TokenUsage(1, 1, 2))
        class A(Agent):
            def __init__(self):
                super().__init__("a", "a", llm_client=llm)
            def receive(self, m): pass
        agent = A()
        result = agent.think("hi")
        assert result == "x"
        # 非流式 chat 被调用（不是 stream）
        assert llm.stream_calls == 0
