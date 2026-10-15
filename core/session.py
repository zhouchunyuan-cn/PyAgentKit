"""
多轮对话会话（ConversationSession）

将"一发一收"升级为"带上下文的多轮对话"。Session 维护对话历史，
每次提问时把历史作为上下文注入 Agent 的 think()，使 Agent 具备
指代消解、追问、上下文延续等能力。

设计要点：
- 一个 Session 绑定一个 Agent（单 Agent 多轮对话场景）
- 历史按条数限制滑动窗口，避免上下文爆炸；可选开启"摘要压缩"
  （summarize_on_trim）：被挤出窗口的旧对话用 LLM 压缩成摘要，
  以 system 消息形式保留关键信息，而不是直接丢弃
- 历史以 OpenAI messages 格式存储，可直接喂给 Agent.think(context_messages=...)
"""

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# 滚动摘要的最大字符数（超出保留最新部分）
_MAX_SUMMARY_CHARS = 2000


class ConversationSession:
    """
    多轮对话会话

    用法：
        session = ConversationSession(agent)
        r1 = session.chat("我叫小明")        # Agent 记住名字
        r2 = session.chat("我叫什么？")      # 能基于上文回答"小明"
    """

    def __init__(
        self,
        agent,
        session_id: str | None = None,
        max_history: int = 10,
        system_context: str | None = None,
        summarize_on_trim: bool = False,
    ):
        """
        Args:
            agent: 绑定的 Agent（需支持 think(user_input, context_messages)）
            session_id: 会话 ID，None 自动生成
            max_history: 保留的最大对话轮数（每轮含 user+assistant 两条），
                         超出则丢弃最早的，控制上下文长度
            system_context: 额外的系统级上下文提示，注入每轮对话开头
            summarize_on_trim: 超窗时是否用 Agent 的 LLM 把被丢弃的旧对话
                         压缩成滚动摘要（失败自动回退硬截断）
        """
        self.agent = agent
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        self.system_context = system_context
        self.summarize_on_trim = summarize_on_trim
        # 滚动摘要：被挤出窗口的旧对话的压缩版（作为 system 消息注入）
        self._summary: str | None = None

        # 对话历史，OpenAI messages 风格：[{"role","content"}, ...]
        self.history: list[dict[str, str]] = []
        # 创建时间，便于会话管理
        self.created_at = time.time()

    def chat(
        self,
        user_input: str,
        use_tools: bool = True,
        stream_callback=None,
        events_callback=None,
    ) -> str:
        """
        进行一轮对话

        把已有历史作为 context_messages 注入 Agent.think()，
        使其能参考上文进行回应。

        Args:
            user_input: 本轮用户输入
            use_tools: 是否允许 Agent 调用工具
            stream_callback: 流式回调（透传给 Agent.think，文本增量实时回调）
            events_callback: 结构化事件回调（透传给 Agent.think）

        Returns:
            Agent 的本轮回复
        """
        # 把历史作为上下文（不含本轮输入，本轮输入由 think 内部添加）
        context = self._build_context_messages()

        # 调用 Agent 推理（回调仅在显式提供时传入，兼容只认基础签名的 Agent 实现）
        extra: dict = {}
        if stream_callback is not None:
            extra["stream_callback"] = stream_callback
        if events_callback is not None:
            extra["events_callback"] = events_callback
        response = self.agent.think(
            user_input=user_input,
            context_messages=context,
            use_tools=use_tools,
            **extra,
        )

        # 记录本轮对话到历史
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        # 滑动窗口裁剪
        self._trim_history()

        return response

    def _build_context_messages(self) -> list[dict[str, str]]:
        """组装注入 think 的上下文消息（滚动摘要 + 系统上下文 + 历史）"""
        messages: list[dict[str, str]] = []
        if self._summary:
            messages.append({"role": "system", "content": f"此前对话的摘要：\n{self._summary}"})
        if self.system_context:
            messages.append({"role": "system", "content": self.system_context})
        # 复制历史，避免被 think 内部修改影响
        messages.extend([m.copy() for m in self.history])
        return messages

    def _trim_history(self) -> None:
        """按 max_history 裁剪，保留最近的对轮（user+assistant 成对）

        开启 summarize_on_trim 时，被裁掉的旧对话先用 LLM 压缩进滚动摘要
        （失败自动回退纯硬截断），关键信息得以跨窗口保留。
        """
        if len(self.history) <= self.max_history:
            return
        # max_history 以"消息条数"计；从开头丢弃，保证剩余成对
        drop = len(self.history) - self.max_history
        # 若丢弃后开头是 assistant（不成对），多丢一条保持成对
        if drop < len(self.history) and self.history[drop]["role"] == "assistant":
            drop += 1
        if self.summarize_on_trim:
            summary = self._summarize_dropped(self.history[:drop])
            if summary:
                self._summary = (
                    f"{self._summary}\n{summary}"[-_MAX_SUMMARY_CHARS:]
                    if self._summary
                    else summary
                )
        self.history = self.history[drop:]

    def _summarize_dropped(self, dropped: list[dict[str, str]]) -> str | None:
        """
        用 Agent 的 LLM 把被裁剪的旧对话压缩成一段摘要

        直接调 llm.chat（不经过 think），避免记忆/钩子等副作用。
        任何失败（无 LLM、调用异常、空结果）都返回 None，回退硬截断。
        """
        llm = getattr(self.agent, "llm", None)
        if llm is None or not dropped:
            return None
        try:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in dropped)
            prompt = (
                "请把以下已结束的对话片段压缩成一段简短摘要，"
                "保留关键事实、用户偏好、已做的决定与未决问题。直接输出摘要正文：\n" + transcript
            )
            result = llm.chat([{"role": "user", "content": prompt}], tools=None, tool_choice="none")
            return (result.content or "").strip() or None
        except Exception as e:
            logger.warning("会话摘要压缩失败，回退硬截断: %s", e)
            return None

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def get_history(self) -> list[dict[str, str]]:
        """返回对话历史的副本"""
        return [m.copy() for m in self.history]

    def clear_history(self) -> None:
        """清空对话历史与滚动摘要（开启全新话题）"""
        self.history.clear()
        self._summary = None

    def summary(self) -> dict[str, Any]:
        """返回会话摘要信息"""
        return {
            "session_id": self.session_id,
            "agent_id": getattr(self.agent, "agent_id", None),
            "turn_count": len(self.history) // 2,
            "message_count": len(self.history),
            "created_at": self.created_at,
        }


class SessionManager:
    """
    会话管理器

    管理多个并发的 ConversationSession（如多用户场景），
    按 session_id 存取。
    """

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self, agent, **kwargs) -> ConversationSession:
        """创建并登记一个新会话"""
        session = ConversationSession(agent, **kwargs)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """按 ID 获取会话"""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """关闭并移除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID"""
        return list(self._sessions.keys())
