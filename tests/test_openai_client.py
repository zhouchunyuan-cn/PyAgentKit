"""
OpenAICompatClient 单元测试（离线，monkeypatch HTTP 层）

验证 LLM 抽象层的可替换性：OpenAI 兼容客户端与 GLMClient 行为同构——
- 非流式：工具调用解析、reasoning_content 透传、usage 提取、response_format 透传
- 流式：SSE 增量 → StreamChunk（思考增量 + 工具碎片），ReactEngine 流式主循环可用
- 重试与错误分类（4xx 不重试，网络/5xx 重试）
- 端点归一化与鉴权缺失提示
- YAML 配置 provider: openai 一行切换后端
"""

import pytest

from core.agent import Agent
from core.config import ConfigLoader
from core.openai_client import OpenAICompatClient


def _make_client(**kw):
    kwargs = {"base_url": "https://fake.local/v1", "model": "test-model", "api_key": "sk-test"}
    kwargs.update(kw)
    return OpenAICompatClient(**kwargs)


def _chat_response(content="答复", reasoning="", tool_calls=None, usage=None):
    message = {"content": content, "role": "assistant"}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": usage or {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


class TestBasics:
    def test_endpoint_normalization(self):
        c = _make_client(base_url="https://api.x.com/v1/")
        assert c.endpoint == "https://api.x.com/v1/chat/completions"
        c2 = _make_client(base_url="https://api.x.com/v1/chat/completions")
        assert c2.endpoint == "https://api.x.com/v1/chat/completions"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAICompatClient(base_url="https://x/v1", model="m")

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        c = OpenAICompatClient(base_url="https://x/v1", model="m")
        assert c.api_key == "sk-env"
        assert c._headers()["Authorization"] == "Bearer sk-env"


class TestChat:
    def test_plain_chat(self, monkeypatch):
        c = _make_client()
        captured = {}

        def fake_request(payload):
            captured.update(payload)
            return _chat_response(content="你好")

        monkeypatch.setattr(c, "_request_json", fake_request)
        result = c.chat([{"role": "user", "content": "q"}])
        assert result.content == "你好"
        assert result.usage.total_tokens == 8
        assert captured["model"] == "test-model"

    def test_tool_calls_parsed(self, monkeypatch):
        c = _make_client()
        resp = _chat_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "calculator", "arguments": '{"expression": "1+2"}'},
                }
            ],
        )
        monkeypatch.setattr(c, "_request_json", lambda payload: resp)
        result = c.chat([], tools=[{"type": "function", "function": {"name": "calculator"}}])
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "1+2"}
        assert result.tool_calls[0].id == "call_1"
        assert result.finish_reason == "tool_calls"

    def test_reasoning_content_passthrough(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, "_request_json", lambda p: _chat_response(reasoning="内心推理"))
        result = c.chat([])
        assert result.reasoning_content == "内心推理"

    def test_response_format_sent(self, monkeypatch):
        c = _make_client()
        captured = {}
        monkeypatch.setattr(c, "_request_json", lambda p: (captured.update(p), _chat_response())[1])
        c.chat([], response_format={"type": "json_object"})
        assert captured["response_format"] == {"type": "json_object"}

    def test_malformed_tool_arguments_tolerated(self, monkeypatch):
        c = _make_client()
        resp = _chat_response(
            tool_calls=[{"id": "c", "function": {"name": "t", "arguments": "{不是JSON"}}]
        )
        monkeypatch.setattr(c, "_request_json", lambda p: resp)
        result = c.chat([])
        assert result.tool_calls[0].arguments == {}


class TestRetry:
    def test_5xx_retried_then_success(self, monkeypatch):
        """5xx/网络类错误重试（重试逻辑在 _request_json，patch 它下面的 _post_once）"""
        c = _make_client(max_retries=1, retry_base_delay=0)
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("HTTP 503: unavailable")
            return _chat_response(content="重试成功")

        monkeypatch.setattr(c, "_post_once", flaky)
        assert c.chat([]).content == "重试成功"
        assert calls["n"] == 2

    def test_4xx_not_retried(self, monkeypatch):
        c = _make_client(max_retries=3, retry_base_delay=0)
        calls = {"n": 0}

        def denied(payload):
            calls["n"] += 1
            raise RuntimeError("HTTP 401: invalid key")

        monkeypatch.setattr(c, "_post_once", denied)
        with pytest.raises(RuntimeError, match="401"):
            c.chat([])
        assert calls["n"] == 1


class TestStream:
    def test_stream_chunks_with_reasoning_and_tool_fragments(self, monkeypatch):
        c = _make_client()
        deltas = [
            {"choices": [{"delta": {"reasoning_content": "想想"}}]},
            {"choices": [{"delta": {"content": "先算。"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "calc", "arguments": '{"e'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'xpression":"2**10"}'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        monkeypatch.setattr(c, "_stream_deltas", lambda payload: iter(deltas))
        chunks = list(c.chat_stream([]))
        assert chunks[0].delta_reasoning == "想想"
        assert chunks[1].delta_content == "先算。"
        assert chunks[2].delta_tool_calls[0]["name"] == "calc"
        assert chunks[-1].usage.total_tokens == 15
        assert chunks[-1].finish_reason == "tool_calls"

    def test_stream_done_marker_ignored(self, monkeypatch):
        """SSE 的 [DONE] 不会产出行（_stream_deltas 已过滤，chunk 层再验证空行为）"""
        c = _make_client()
        monkeypatch.setattr(
            c,
            "_stream_deltas",
            lambda p: iter(
                [
                    {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                ]
            ),
        )
        chunks = list(c.chat_stream([]))
        assert len(chunks) == 1
        assert chunks[0].delta_content == "ok"


class TestEngineIntegration:
    """证明抽象层：ReactEngine 对 OpenAI 兼容客户端无感工作（流式 + 工具往返）"""

    def test_react_roundtrip_with_compat_client(self, monkeypatch):
        c = _make_client()
        responses = iter(
            [
                _chat_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "t1",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "2**10"}',
                            },
                        }
                    ],
                ),
                _chat_response(content="结果是 1024"),
            ]
        )

        def fake_stream(payload):
            r = next(responses)
            # 模拟把非流式响应拆成两个 SSE 增量（走基类回退同构的逻辑）
            yield {
                "choices": [{"delta": {"content": r["choices"][0]["message"].get("content") or ""}}]
            }
            if r["choices"][0]["message"].get("tool_calls"):
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc["id"],
                                        "function": dict(tc["function"]),
                                    }
                                    for i, tc in enumerate(r["choices"][0]["message"]["tool_calls"])
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": r["usage"],
                }
            else:
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": r["usage"]}

        monkeypatch.setattr(c, "_stream_deltas", fake_stream)

        from core.tools import CalculatorTool

        class A(Agent):
            def __init__(self):
                super().__init__("a", "助手", llm_client=c)
                self.tool_registry.register(CalculatorTool())

            def receive(self, m):
                pass

        agent = A()
        collected = []
        result = agent.think("计算 2**10", stream_callback=collected.append)
        assert result == "结果是 1024"
        assert agent.get_token_usage().total_tokens >= 16  # 两轮 usage 均累计
        assert "".join(collected) == "结果是 1024"


class TestConfigProvider:
    def test_yaml_provider_switch(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        loader = ConfigLoader()
        config = loader.parse(
            {
                "llm": {
                    "provider": "openai",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                },
                "agents": [{"id": "a", "capabilities": ["chat"]}],
            }
        )
        llm, agents, session = loader.build(config)
        assert isinstance(llm, OpenAICompatClient)
        assert llm.endpoint == "https://api.deepseek.com/v1/chat/completions"
        assert agents["a"].llm is llm

    def test_invalid_provider_rejected(self):
        loader = ConfigLoader()
        with pytest.raises(ValueError, match="provider"):
            loader.parse(
                {
                    "llm": {"provider": "nonexistent", "model": "m"},
                    "agents": [{"id": "a"}],
                }
            )
