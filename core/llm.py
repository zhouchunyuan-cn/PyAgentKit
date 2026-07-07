"""
LLM 客户端抽象层

提供统一的 LLM 调用接口，解耦具体 SDK 实现。
当前提供基于智谱 GLM（zhipuai SDK）的实现，
未来可扩展 OpenAI / Ollama 等其他后端。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import os
import time


@dataclass
class ToolCall:
    """
    工具调用请求（由 LLM 返回）

    Attributes:
        id: 本次调用的唯一标识，回传工具结果时需带上
        name: 要调用的工具名称
        arguments: 工具参数（已反序列化为字典）
    """
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResult:
    """
    一次 LLM 调用的统一返回结果

    Attributes:
        content: 文本回复内容（无工具调用时为最终答复）
        tool_calls: 工具调用列表，为空表示模型已给出最终答复
        finish_reason: 结束原因 ("stop" / "tool_calls" / "length" 等)
        usage: token 用量统计，None 表示模型未返回
    """
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional["TokenUsage"] = None


@dataclass
class TokenUsage:
    """
    一次 LLM 调用的 token 用量

    Attributes:
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        total_tokens: 总 token 数
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        """累加另一份用量，返回新实例（便于累计）"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class StreamChunk:
    """
    流式调用的一块增量

    Attributes:
        delta_content: 本块新增的文本片段（可能为空）
        finish_reason: 本块触发的结束原因；None 表示还在继续
        usage: 仅最后一块可能携带，汇总用量
    """
    delta_content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


class LLMClient(ABC):
    """
    LLM 客户端抽象基类

    所有具体实现（GLM/OpenAI/Ollama）都应实现 chat 方法。
    上层 Agent 只依赖此抽象，不接触具体 SDK 细节。
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> ChatResult:
        """
        发起一次对话请求（非流式）

        Args:
            messages: 对话消息列表，格式遵循 OpenAI 风格
                      [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
            tools: 可用工具的 schema 列表（OpenAI function-calling 格式），可选
            tool_choice: 工具选择策略 ("auto" / "none" / "required")

        Returns:
            ChatResult: 统一的返回结果（含 token 用量）
        """
        raise NotImplementedError

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ):
        """
        发起流式对话请求，逐块 yield StreamChunk

        默认实现回退到非流式 chat（作为单块返回），子类可覆盖以提供真实流式。
        这样未实现流式的 LLMClient 子类仍可用 stream_callback 路径。

        Yields:
            StreamChunk
        """
        result = self.chat(messages, tools=tools, tool_choice=tool_choice)
        yield StreamChunk(
            delta_content=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )


class GLMClient(LLMClient):
    """
    基于智谱 zhipuai SDK 的 GLM 实现

    特性:
    - API Key 默认从环境变量 ZHIPUAI_API_KEY 读取
    - 内置指数退避重试（应对网络波动 / 限流）
    - 将 zhipuai 响应统一封装为 ChatResult，上层无需感知 SDK 结构
    """

    # 可重试的异常关键字（网络/限流类）
    _RETRYABLE_KEYWORDS = ("timeout", "connection", "rate_limit", "429", "500", "502", "503", "504")

    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        """
        初始化 GLM 客户端

        Args:
            model: 模型名称，默认 glm-4-flash（完全免费）；轻量场景可用 glm-4-flash，
                   glm-4.7-flash 同样免费且更强；glm-4.5 等需付费额度
            api_key: API Key，未提供则从环境变量 ZHIPUAI_API_KEY 读取
            max_retries: 最大重试次数
            retry_base_delay: 重试基础退避秒数（指数退避基数）
        """
        resolved_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "未找到 GLM API Key。请设置环境变量 ZHIPUAI_API_KEY，"
                "或在初始化 GLMClient 时传入 api_key 参数。"
                "获取地址：https://open.bigmodel.cn/"
            )

        # 延迟导入，避免未安装 SDK 时影响其他模块
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError(
                "未安装 zhipuai SDK，请运行：pip install zhipuai"
            ) from e

        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.client = ZhipuAI(api_key=resolved_key)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> ChatResult:
        """
        调用 GLM 对话接口

        Args:
            messages: 对话消息列表（OpenAI 风格）
            tools: 可用工具 schema 列表，可选
            tool_choice: 工具选择策略

        Returns:
            ChatResult: 统一返回结果
        """
        # 组装请求参数（仅在有工具时传入 tools / tool_choice）
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # 带重试的请求
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and self._is_retryable(e):
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                # 不可重试或重试次数耗尽，直接抛出
                raise

        # 理论上不会走到这里
        raise RuntimeError(f"GLM 调用失败: {last_error}")

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ):
        """
        流式调用 GLM，逐块 yield StreamChunk

        文本增量通过 delta_content 产出；token 用量通常在最后一个 chunk 携带。
        流式不重试（流中断难以安全重发），失败即抛出。
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        for chunk in response:
            # 每个 chunk 结构同非流式：choices[0].delta
            try:
                choice = chunk.choices[0]
            except (AttributeError, IndexError):
                continue
            delta = getattr(choice, "delta", None)
            delta_content = getattr(delta, "content", None) if delta else None
            finish_reason = getattr(choice, "finish_reason", None)
            # usage 通常只在最后一个 chunk 出现
            usage = self._extract_usage(chunk)
            if delta_content or finish_reason or usage:
                yield StreamChunk(
                    delta_content=delta_content or "",
                    finish_reason=finish_reason,
                    usage=usage,
                )

    def _parse_response(self, response: Any) -> ChatResult:
        """
        将 zhipuai 响应解析为统一的 ChatResult

        Args:
            response: zhipuai SDK 返回的响应对象

        Returns:
            ChatResult: 统一返回结果
        """
        # 兼容 SDK 版本差异：choices[0].message
        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        # 提取文本内容
        content = getattr(message, "content", None) or ""

        # 提取工具调用
        tool_calls: List[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            for tc in raw_tool_calls:
                # SDK 中 tool_call.function.arguments 是 JSON 字符串
                func = getattr(tc, "function", None)
                if not func:
                    continue
                name = getattr(func, "name", "")
                raw_args = getattr(func, "arguments", "{}")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(ToolCall(id=getattr(tc, "id", ""), name=name, arguments=arguments))

        # 提取 token 用量（response.usage，GLM 与 OpenAI 兼容字段）
        usage = self._extract_usage(response)

        return ChatResult(
            content=content, tool_calls=tool_calls,
            finish_reason=finish_reason, usage=usage,
        )

    @staticmethod
    def _extract_usage(response: Any) -> Optional[TokenUsage]:
        """从响应中提取 token 用量，缺失字段视为 0"""
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return None
        try:
            return TokenUsage(
                prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
            )
        except (TypeError, ValueError):
            return None

    def _is_retryable(self, error: Exception) -> bool:
        """
        判断异常是否可重试

        Args:
            error: 捕获的异常

        Returns:
            是否建议重试
        """
        msg = str(error).lower()
        return any(keyword in msg for keyword in self._RETRYABLE_KEYWORDS)
