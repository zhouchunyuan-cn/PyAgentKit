"""
统一记忆管理器（MemoryManager）

外观层，把三套割裂的记忆后端组合为一个统一接口：
- Memory（KV：短期 TTL + 长期持久化）
- VectorMemory（语义召回）
- 会话历史（OpenAI messages 风格）

设计目标：Agent 不再直接操作三套系统，只面对 MemoryManager 一个接口。
核心方法 build_context() 把"system_prompt + 长期记忆 + 向量召回 + 会话历史 + 输入"
一次性聚合成 LLM 的 messages，收敛了原本散落在 Agent._build_initial_messages 的逻辑。

三个后端类保持不变，仅被组合——零破坏现有实现。
"""

from typing import Any, Dict, List, Optional
import json
import logging

from .memory import Memory
from .vector_memory import VectorMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    统一记忆管理器

    组合 KV / 向量 / 会话三类记忆后端，对外提供单一接口。
    三个后端均可选：按需启用，缺失时自动降级（不报错）。

    用法：
        brain = MemoryManager(vector_memory=VectorMemory(embedder=...))
        brain.remember("user_name", "小明")           # KV
        brain.learn("Python 是解释型语言")              # 向量
        brain.record_turn("user", "你好")              # 会话
        messages = brain.build_context("你好吗", system_prompt="你是助手")
    """

    def __init__(
        self,
        memory: Optional[Memory] = None,
        vector_memory: Optional[VectorMemory] = None,
        conversation_max_history: int = 10,
    ):
        """
        Args:
            memory: KV 记忆后端；None 时新建默认 Memory
            vector_memory: 向量记忆后端；None 时不启用语义召回
            conversation_max_history: 内置会话历史保留的最大消息条数
        """
        self.memory: Memory = memory if memory is not None else Memory()
        self.vector_memory: Optional[VectorMemory] = vector_memory
        self.conversation_max_history = conversation_max_history
        # 内置会话历史（OpenAI messages 风格）
        self._conversation: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # KV 通道（委托 Memory）
    # ------------------------------------------------------------------
    def remember(self, key: str, value: Any, memory_type: str = "short") -> None:
        """存入 KV 记忆（短期或长期）"""
        self.memory.store(key, value, memory_type)

    def recall(self, key: str, default: Any = None) -> Any:
        """按 key 取 KV 记忆"""
        return self.memory.retrieve(key, default)

    # ------------------------------------------------------------------
    # 向量通道（委托 VectorMemory）
    # ------------------------------------------------------------------
    def learn(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        存入一条语义记忆（供后续语义召回）

        未配置 vector_memory 时静默跳过（降级），返回 None。
        """
        if self.vector_memory is None:
            logger.debug("未配置 vector_memory，learn 被跳过")
            return None
        return self.vector_memory.add(text, metadata)

    def recall_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """
        按语义召回相关记忆文本

        未配置 vector_memory 时返回空列表（降级）。
        """
        if self.vector_memory is None:
            return []
        try:
            return self.vector_memory.recall_semantic(query, top_k=top_k)
        except Exception as e:
            logger.warning("向量召回失败，已跳过: %s", e)
            return []

    # ------------------------------------------------------------------
    # 会话通道（内置历史）
    # ------------------------------------------------------------------
    def record_turn(self, role: str, content: str) -> None:
        """记录一轮对话（追加到会话历史并裁剪）"""
        self._conversation.append({"role": role, "content": content})
        self._trim_conversation()

    def get_conversation(self) -> List[Dict[str, str]]:
        """获取会话历史副本"""
        return [m.copy() for m in self._conversation]

    def clear_conversation(self) -> None:
        """清空会话历史"""
        self._conversation.clear()

    def _trim_conversation(self) -> None:
        """按 max_history 裁剪会话历史，保持 user/assistant 成对"""
        if len(self._conversation) <= self.conversation_max_history:
            return
        drop = len(self._conversation) - self.conversation_max_history
        # 若丢弃后开头是 assistant（不成对），多丢一条保持成对
        if drop < len(self._conversation) and self._conversation[drop]["role"] == "assistant":
            drop += 1
        self._conversation = self._conversation[drop:]

    # ------------------------------------------------------------------
    # 聚合：一次性产出 LLM 的 messages ⭐
    # ------------------------------------------------------------------
    def build_context(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        extra_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        聚合所有记忆，产出 LLM 的完整 messages

        组装顺序：
        1. system_prompt（角色设定）
        2. 长期记忆（KV 长期部分，作为 system 补充）
        3. 向量召回（与 user_input 语义相关的经验）
        4. 会话历史（多轮上下文）
        5. 额外上下文（外部传入的 context_messages）
        6. 当前用户输入

        Args:
            user_input: 当前用户输入
            system_prompt: 系统提示词（角色设定）
            extra_context: 额外注入的消息（如上游 Agent 传来的结构化数据）

        Returns:
            OpenAI 风格的 messages 列表，可直接喂给 LLM
        """
        messages: List[Dict[str, Any]] = []

        # 1. system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. 长期记忆（KV）
        long_term = self.memory.list_long_term_memory()
        if long_term:
            memory_brief = json.dumps(long_term, ensure_ascii=False)
            messages.append({
                "role": "system",
                "content": f"以下是你的长期记忆，供参考：\n{memory_brief}",
            })

        # 3. 向量召回（语义相关经验）
        related = self.recall_relevant(user_input, top_k=3)
        if related:
            recall_text = "\n".join(f"- {t}" for t in related)
            messages.append({
                "role": "system",
                "content": f"以下是与你当前任务相关的历史经验，供参考：\n{recall_text}",
            })

        # 4. 会话历史（多轮上下文）
        messages.extend(self.get_conversation())

        # 5. 额外上下文
        if extra_context:
            messages.extend(extra_context)

        # 6. 当前用户输入
        messages.append({"role": "user", "content": user_input})
        return messages

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """记忆状态摘要"""
        mem_stats = self.memory.get_memory_stats()
        return {
            "kv_short_count": mem_stats.get("short_term_count", 0),
            "kv_long_count": mem_stats.get("long_term_count", 0),
            "vector_enabled": self.vector_memory is not None,
            "vector_count": self.vector_memory.count() if self.vector_memory else 0,
            "conversation_turns": len(self._conversation) // 2,
        }
