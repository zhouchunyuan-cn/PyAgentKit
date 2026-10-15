"""
调用追踪（Trace）

记录 Agent / Team 执行过程中的时序、耗时与 token，形成可观测的 trace。
解决之前"复杂协作是黑盒、无 per-step 延迟与 token 归属"的问题。

设计：
- Tracer：单例，基于 contextvars 保存当前 trace；通过 enable/disable 开关，
  默认关闭。contextvars 使追踪上下文可跨 asyncio 任务传播，也可随
  copy_context() 进入执行器线程（AsyncRouter 的 receive 默认实现即如此），
  异步路径下不再丢失 span。
- TraceSpan：一段命名的执行区间（如某次 think、某个子任务），含 start/end/duration
- Trace：一次完整执行的所有 span，可导出为可读文本或 JSON

插桩方式（非侵入）：
- Agent.think 增加可选 trace 钩子，记录每次 LLM 调用的耗时与 token
- Process.execute 记录每个成员的处理 span
- 未启用 tracer 时零开销（早返回）

用法：
    Tracer.enable()
    trace = Tracer.start_trace("团队任务")
    # ... agent.think(...) 或 team.run(...) 内部自动记录 ...
    Tracer.end_trace()
    print(trace.format_text())
"""

import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceSpan:
    """
    一段执行区间

    Attributes:
        name: span 名称（如 "think:助手"、"子任务:搜索"）
        start: 开始时间戳（秒）
        end: 结束时间戳；None 表示未结束
        attributes: 自由属性（token 数、输入摘要、工具名等）
        parent_id: 父 span id（构建嵌套关系，如 think 内的工具调用）
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start: float = 0.0
    end: float | None = None
    name: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    @property
    def duration(self) -> float | None:
        """耗时（秒）；未结束返回 None"""
        if self.end is None:
            return None
        return self.end - self.start

    def finish(self) -> None:
        """标记结束"""
        self.end = time.time()

    def set(self, key: str, value: Any) -> None:
        """设置属性"""
        self.attributes[key] = value


@dataclass
class Trace:
    """
    一次完整执行的所有 span

    Attributes:
        name: trace 名称（如任务描述）
        spans: 所有 span，按记录顺序
        started_at: trace 开始时间
    """

    name: str = ""
    spans: list[TraceSpan] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def add_span(self, span: TraceSpan) -> TraceSpan:
        self.spans.append(span)
        return span

    def total_tokens(self) -> int:
        """汇总所有 span 记录的 token"""
        return sum(
            s.attributes.get("total_tokens", 0)
            for s in self.spans
            if isinstance(s.attributes.get("total_tokens"), int)
        )

    def format_text(self) -> str:
        """格式化为可读文本（类似分段时间线）"""
        lines = [f"=== Trace: {self.name} ==="]
        if self.duration is not None:
            lines.append(f"总耗时: {self.duration:.2f}s")
        tok = self.total_tokens()
        if tok:
            lines.append(f"总 token: {tok}")
        lines.append(f"span 数: {len(self.spans)}")
        lines.append("")
        for s in self.spans:
            dur = f"{s.duration:.2f}s" if s.duration is not None else "?"
            indent = "  "
            attr_str = ""
            if s.attributes:
                attr_str = " " + " ".join(f"{k}={v}" for k, v in s.attributes.items())
            lines.append(f"{indent}{s.name:<30} {dur:>8}{attr_str}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（便于 JSON 导出）"""
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "total_tokens": self.total_tokens(),
            "spans": [
                {
                    "id": s.id,
                    "name": s.name,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "attributes": s.attributes,
                    "parent_id": s.parent_id,
                }
                for s in self.spans
            ],
        }


@dataclass
class _TraceState:
    """某个执行上下文中的活跃追踪状态（trace + span 栈）"""

    trace: Trace
    stack: list[str] = field(default_factory=list)
    warned_cap: bool = False  # span 上限告警只发一次


class Tracer:
    """
    全局追踪器（基于 contextvars）

    通过 enable()/disable() 开关。未启用时所有记录方法零开销（早返回），
    不影响生产性能。

    上下文传播：活跃 trace 保存在 ContextVar 中——同一 asyncio 任务树
    自动共享；执行器线程需通过 contextvars.copy_context() 显式带入
    （Agent.receive_async 默认实现已处理）。
    """

    _enabled = False
    # 单个 trace 的 span 数上限（防长驻任务无限累积）
    MAX_SPANS = 10_000
    _state_var: contextvars.ContextVar[_TraceState | None] = contextvars.ContextVar(
        "pyagentkit_trace_state", default=None
    )

    @classmethod
    def enable(cls) -> None:
        """启用追踪"""
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """禁用追踪"""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def start_trace(cls, name: str) -> Trace | None:
        """开始一个新的 trace（成为当前上下文的活跃 trace）"""
        if not cls._enabled:
            return None
        old = cls._state_var.get()
        if old is not None:
            # 上一条 trace 未结束就开新 trace：先收尾，避免孤儿 trace
            logger.warning("上一条 trace '%s' 未结束，已被新 trace 接替", old.trace.name)
            old.trace.ended_at = time.time()
        trace = Trace(name=name)
        cls._state_var.set(_TraceState(trace=trace))
        return trace

    @classmethod
    def end_trace(cls) -> Trace | None:
        """结束当前 trace 并返回"""
        if not cls._enabled:
            return None
        state = cls._state_var.get()
        if state is not None:
            state.trace.ended_at = time.time()
            cls._state_var.set(None)
            return state.trace
        return None

    @classmethod
    def current_trace(cls) -> Trace | None:
        state = cls._state_var.get()
        return state.trace if state is not None else None

    @classmethod
    def start_span(cls, name: str, **attributes) -> TraceSpan | None:
        """开始一个 span（自动加到当前 trace）"""
        if not cls._enabled:
            return None
        state = cls._state_var.get()
        if state is None:
            return None
        if len(state.trace.spans) >= cls.MAX_SPANS:
            # 超上限后丢弃新 span（只告警一次），防内存无限增长
            if not state.warned_cap:
                state.warned_cap = True
                logger.warning(
                    "trace '%s' 的 span 数达到上限 %d，后续 span 将不被记录",
                    state.trace.name,
                    cls.MAX_SPANS,
                )
            return None
        parent = state.stack[-1] if state.stack else None
        span = TraceSpan(
            name=name,
            start=time.time(),
            parent_id=parent,
            attributes=dict(attributes),
        )
        state.trace.add_span(span)
        state.stack.append(span.id)
        return span

    @classmethod
    def end_span(cls, span: TraceSpan | None, **extra_attributes) -> None:
        """结束一个 span，可补充属性"""
        if not cls._enabled or span is None:
            return
        span.finish()
        for k, v in extra_attributes.items():
            span.set(k, v)
        state = cls._state_var.get()
        if state is not None and span.id in state.stack:
            # 从栈顶方向找第一个同 id 的项移除（正确的嵌套弹栈）
            for i in range(len(state.stack) - 1, -1, -1):
                if state.stack[i] == span.id:
                    del state.stack[i]
                    break

    @classmethod
    def record(cls, name: str, duration: float | None = None, **attributes) -> None:
        """记录一个即时 span（已有耗时，无需 start/end 配对）"""
        if not cls._enabled:
            return
        state = cls._state_var.get()
        if state is None:
            return
        if len(state.trace.spans) >= cls.MAX_SPANS:
            return
        now = time.time()
        span = TraceSpan(
            name=name,
            start=now - (duration or 0),
            end=now,
            attributes=dict(attributes),
        )
        state.trace.add_span(span)
