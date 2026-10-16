"""
OpenAI 兼容客户端（OpenAICompatClient）

LLMClient 抽象层的第二个实现——证明"换后端不改上层代码"：
任何兼容 OpenAI Chat Completions 协议的端点都能接入，
包括 OpenAI、DeepSeek、Moonshot、vLLM、Ollama（OpenAI 兼容模式），
以及智谱自己的兼容端点（https://open.bigmodel.cn/api/paas/v4）。

实现要点：
- 纯标准库 urllib（零新增依赖），与项目 HttpTool 一致的重试策略
- 支持工具调用、response_format 结构化输出
- 兼容 DeepSeek R1 等推理模型的 reasoning_content 透传
- 真流式：解析 SSE 增量，产出与 GLMClient 同构的 StreamChunk
  （含 delta_reasoning 与 delta_tool_calls 碎片），ReactEngine 无感切换

用法：
    from core import OpenAICompatClient

    llm = OpenAICompatClient(
        base_url="https://api.deepseek.com/v1",   # 或 vLLM/Ollama 本地端点
        model="deepseek-chat",
    )   # api_key 从环境变量 OPENAI_API_KEY 读取，也可显式传入
    agent = MyAgent(llm_client=llm)               # 其余代码零改动
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .llm import ChatResult, LLMClient, StreamChunk, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class OpenAICompatClient(LLMClient):
    """
    OpenAI Chat Completions 兼容协议的 LLM 客户端
    """

    # 可重试的异常关键字（网络/限流类）
    _RETRYABLE_KEYWORDS = ("timeout", "connection", "timed out", "429", "500", "502", "503", "504")

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        timeout: float = 60.0,
    ):
        """
        Args:
            base_url: API 根地址（如 "https://api.openai.com/v1"），
                      也接受完整端点（以 /chat/completions 结尾）
            model: 模型名（如 "gpt-4o-mini"、"deepseek-chat"）
            api_key: API Key；未提供则从环境变量 OPENAI_API_KEY 读取
            max_retries: 最大重试次数（不含首次）
            retry_base_delay: 重试基础退避秒数（指数退避基数）
            timeout: 单次请求超时秒数
        """
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "未找到 API Key。请设置环境变量 OPENAI_API_KEY，"
                "或在初始化 OpenAICompatClient 时传入 api_key 参数。"
            )
        base = base_url.rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.model = model
        self.api_key = resolved_key
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout

    # ------------------------------------------------------------------
    # HTTP 层（测试通过替换 _request_json / _stream_deltas 注入假响应）
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """单次 POST 并返回解析后的 JSON（不做重试；异常原样抛出）"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:500] if e.fp else ""
            raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from e

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST 并返回解析后的 JSON（带超时与指数退避重试）"""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_once(payload)
            except Exception as e:
                last_error = e
                # 4xx（鉴权/参数类）重试无意义
                if isinstance(e, RuntimeError) and e.args and str(e.args[0]).startswith("HTTP 4"):
                    raise
            if attempt < self.max_retries and self._is_retryable(last_error):
                time.sleep(self.retry_base_delay * (2**attempt))
                continue
            raise last_error
        raise last_error  # 理论上不可达

    def _stream_deltas(self, payload: dict[str, Any]):
        """
        发起流式请求，逐个 yield 解析后的增量 dict（SSE data 行）

        每个 dict 形如 OpenAI 的 choices[0].delta：
        {"content": str?, "reasoning_content": str?, "tool_calls": [...]}，
        以及 {"finish_reason": ...}；usage 在最后一个 data 行携带。
        流式不重试（难以安全重发），失败即抛出。
        """
        payload = dict(payload, stream=True)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue  # 忽略无法解析的心跳/注释行

    def _is_retryable(self, error: Exception) -> bool:
        msg = str(error).lower()
        return any(k in msg for k in self._RETRYABLE_KEYWORDS)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    # ------------------------------------------------------------------
    # LLMClient 接口实现
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """非流式对话（OpenAI Chat Completions 协议）"""
        data = self._request_json(
            self._build_payload(messages, tools, tool_choice, response_format)
        )
        return self._parse_response(data)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ):
        """
        真流式对话：解析 SSE 增量，产出与 GLMClient 同构的 StreamChunk
        （含思考增量与工具调用碎片），ReactEngine 可无感复用流式主循环。
        """
        payload = self._build_payload(messages, tools, tool_choice, None)
        for item in self._stream_deltas(payload):
            choice = (item.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            finish_reason = choice.get("finish_reason")
            fragments = [
                {
                    "index": tc.get("index", i) or i,
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                }
                for i, tc in enumerate(delta.get("tool_calls") or [])
            ]
            usage = self._extract_usage(item)
            if content or reasoning or finish_reason or usage or fragments:
                yield StreamChunk(
                    delta_content=content or "",
                    finish_reason=finish_reason,
                    usage=usage,
                    delta_reasoning=reasoning or "",
                    delta_tool_calls=fragments,
                )

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------
    def _parse_response(self, data: dict[str, Any]) -> ChatResult:
        """把 OpenAI 风格的响应 dict 解析为 ChatResult"""
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"响应缺少 choices: {str(data)[:200]}")
        message = choices[0].get("message") or {}
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            func = tc.get("function") or {}
            raw_args = func.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=func.get("name", ""), arguments=arguments)
            )
        return ChatResult(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason") or "stop",
            usage=self._extract_usage(data),
            reasoning_content=message.get("reasoning_content") or "",
        )

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> TokenUsage | None:
        """从响应中提取 token 用量，缺失返回 None"""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        try:
            return TokenUsage(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
            )
        except (TypeError, ValueError):
            return None
