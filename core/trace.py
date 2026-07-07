"""
调用追踪（Trace）

记录 Agent / Team 执行过程中的时序、耗时与 token，形成可观测的 trace。
解决之前"复杂协作是黑盒、无 per-step 延迟与 token 归属"的问题。

设计：
- Tracer：单例，线程局部存储当前 trace；通过 enable/disable 开关，默认关闭
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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import threading
import time
import uuid


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
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    start: float = 0.0
    end: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
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
    spans: List[TraceSpan] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
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

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（便于 JSON 导出）"""
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "total_tokens": self.total_tokens(),
            "spans": [
                {
                    "id": s.id, "name": s.name,
                    "start": s.start, "end": s.end, "duration": s.duration,
                    "attributes": s.attributes, "parent_id": s.parent_id,
                }
                for s in self.spans
            ],
        }


class Tracer:
    """
    全局追踪器（线程局部）

    通过 enable()/disable() 开关。未启用时所有记录方法零开销（早返回），
    不影响生产性能。
    """

    _enabled = False
    _local = threading.local()

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
    def start_trace(cls, name: str) -> Optional[Trace]:
        """开始一个新的 trace（成为当前线程的活跃 trace）"""
        if not cls._enabled:
            return None
        trace = Trace(name=name)
        cls._local.current_trace = trace
        cls._local.span_stack = []
        return trace

    @classmethod
    def end_trace(cls) -> Optional[Trace]:
        """结束当前 trace 并返回"""
        if not cls._enabled:
            return None
        trace: Optional[Trace] = getattr(cls._local, "current_trace", None)
        if trace is not None:
            trace.ended_at = time.time()
            cls._local.current_trace = None
            cls._local.span_stack = []
        return trace

    @classmethod
    def current_trace(cls) -> Optional[Trace]:
        return getattr(cls._local, "current_trace", None)

    @classmethod
    def start_span(cls, name: str, **attributes) -> Optional[TraceSpan]:
        """开始一个 span（自动加到当前 trace）"""
        if not cls._enabled:
            return None
        trace = cls.current_trace()
        if trace is None:
            return None
        stack: List[str] = getattr(cls._local, "span_stack", [])
        parent = stack[-1] if stack else None
        span = TraceSpan(
            name=name, start=time.time(), parent_id=parent,
            attributes=dict(attributes),
        )
        trace.add_span(span)
        stack.append(span.id)
        cls._local.span_stack = stack
        return span

    @classmethod
    def end_span(cls, span: Optional[TraceSpan], **extra_attributes) -> None:
        """结束一个 span，可补充属性"""
        if not cls._enabled or span is None:
            return
        span.finish()
        for k, v in extra_attributes.items():
            span.set(k, v)
        stack: List[str] = getattr(cls._local, "span_stack", [])
        if span.id in stack:
            stack.remove(span.id)
            cls._local.span_stack = stack

    @classmethod
    def record(cls, name: str, duration: Optional[float] = None, **attributes) -> None:
        """记录一个即时 span（已有耗时，无需 start/end 配对）"""
        if not cls._enabled:
            return
        trace = cls.current_trace()
        if trace is None:
            return
        now = time.time()
        span = TraceSpan(
            name=name,
            start=now - (duration or 0),
            end=now,
            attributes=dict(attributes),
        )
        trace.add_span(span)
