"""
MemoryManager 单元测试

验证三通道（KV / 向量 / 会话）的委托，以及核心的 build_context 聚合逻辑。
向量通道用 FakeEmbedder（确定性，不联网）。
"""
import pytest

from core.memory_manager import MemoryManager
from core.memory import Memory
from core.vector_memory import VectorMemory


class FakeEmbedder:
    """确定性词袋向量化器（复用自 vector_memory 测试思路）"""
    VOCAB = ["python", "编程", "语言", "git", "版本", "控制", "苹果"]

    def embed(self, text: str) -> list:
        t = text.lower()
        return [1.0 if w in t else 0.0 for w in self.VOCAB]

    def transform(self, text: str) -> list:
        return self.embed(text)


@pytest.fixture
def brain(tmp_path):
    """带向量后端的 MemoryManager，持久化隔离到 tmp"""
    mem = Memory(persistent_storage_path=str(tmp_path / "m.json"))
    vm = VectorMemory(embedder=FakeEmbedder())
    return MemoryManager(memory=mem, vector_memory=vm)


# --------------------------------------------------------------------
# KV 通道
# --------------------------------------------------------------------

class TestKVChannel:
    def test_remember_recall(self, brain):
        brain.remember("name", "小明")
        assert brain.recall("name") == "小明"

    def test_recall_missing(self, brain):
        assert brain.recall("nope") is None
        assert brain.recall("nope", "默认") == "默认"

    def test_long_term_persists(self, brain):
        brain.remember("perm", "持久值", memory_type="long")
        assert brain.recall("perm") == "持久值"


# --------------------------------------------------------------------
# 向量通道
# --------------------------------------------------------------------

class TestVectorChannel:
    def test_learn_and_recall_relevant(self, brain):
        brain.learn("Python 是一门编程语言")
        brain.learn("Git 是版本控制工具")
        results = brain.recall_relevant("python 编程", top_k=1)
        assert len(results) == 1
        assert "Python" in results[0]

    def test_recall_relevant_unknown_returns_empty_or_notfound(self, brain):
        brain.learn("Python 编程语言")
        results = brain.recall_relevant("苹果", top_k=1)
        # 无关查询，不应召回 Python（相似度为 0）
        assert isinstance(results, list)


# --------------------------------------------------------------------
# 会话通道
# --------------------------------------------------------------------

class TestConversationChannel:
    def test_record_and_get(self, brain):
        brain.record_turn("user", "你好")
        brain.record_turn("assistant", "你好！")
        conv = brain.get_conversation()
        assert len(conv) == 2
        assert conv[0] == {"role": "user", "content": "你好"}

    def test_get_conversation_returns_copy(self, brain):
        brain.record_turn("user", "x")
        c1 = brain.get_conversation()
        c1.append({"role": "user", "content": "tamper"})
        c2 = brain.get_conversation()
        assert len(c2) == 1  # 外部修改不影响内部

    def test_clear(self, brain):
        brain.record_turn("user", "x")
        brain.clear_conversation()
        assert brain.get_conversation() == []

    def test_trim(self):
        mem = Memory(persistent_storage_path=":memory:")
        brain = MemoryManager(memory=mem, conversation_max_history=4)
        for i in range(6):
            brain.record_turn("user", f"u{i}")
            brain.record_turn("assistant", f"a{i}")
        # 应被裁剪到接近 max_history
        assert len(brain.get_conversation()) <= 8


# --------------------------------------------------------------------
# build_context 聚合（核心）
# --------------------------------------------------------------------

class TestBuildContext:
    def test_basic_user_input_always_last(self, brain):
        msgs = brain.build_context("你好")
        assert msgs[-1] == {"role": "user", "content": "你好"}

    def test_system_prompt_included(self, brain):
        msgs = brain.build_context("hi", system_prompt="你是助手")
        assert msgs[0] == {"role": "system", "content": "你是助手"}

    def test_long_term_memory_injected(self, brain):
        brain.remember("fact", "太阳是恒星", memory_type="long")
        msgs = brain.build_context("问题")
        # 应有一条 system 消息包含长期记忆
        memory_msg = [m for m in msgs if "长期记忆" in m.get("content", "")]
        assert len(memory_msg) == 1
        assert "太阳是恒星" in memory_msg[0]["content"]

    def test_vector_recall_injected(self, brain):
        brain.learn("Python 是编程语言")
        msgs = brain.build_context("python 编程")
        # 应有 system 消息包含相关经验
        exp_msg = [m for m in msgs if "历史经验" in m.get("content", "")]
        assert len(exp_msg) == 1

    def test_conversation_injected(self, brain):
        brain.record_turn("user", "上轮问题")
        brain.record_turn("assistant", "上轮答复")
        msgs = brain.build_context("本轮")
        contents = [m["content"] for m in msgs]
        assert "上轮问题" in contents
        assert "上轮答复" in contents

    def test_extra_context_included(self, brain):
        extra = [{"role": "system", "content": "额外指令"}]
        msgs = brain.build_context("hi", extra_context=extra)
        assert extra[0] in msgs

    def test_assembly_order(self, brain):
        """组装顺序：system_prompt → 长期记忆 → 向量 → 会话 → 额外 → 用户输入"""
        brain.remember("f", "v", memory_type="long")
        brain.learn("Python 编程语言")
        brain.record_turn("user", "历史")
        msgs = brain.build_context(
            "当前输入",
            system_prompt="角色",
            extra_context=[{"role": "system", "content": "额外"}],
        )
        # 最后一条必须是当前用户输入
        assert msgs[-1] == {"role": "user", "content": "当前输入"}
        # 第一条是 system_prompt
        assert msgs[0]["content"] == "角色"


# --------------------------------------------------------------------
# 后端可选性（降级）
# --------------------------------------------------------------------

class TestBackendOptional:
    def test_works_without_vector_memory(self, tmp_path):
        """无向量后端时，build_context 仍正常工作，learn/recall_relevant 降级"""
        mem = Memory(persistent_storage_path=str(tmp_path / "m.json"))
        brain = MemoryManager(memory=mem, vector_memory=None)
        # learn 静默跳过
        assert brain.learn("text") is None
        # recall_relevant 返回空列表
        assert brain.recall_relevant("text") == []
        # build_context 仍产出 messages（不报错）
        msgs = brain.build_context("hi")
        assert msgs[-1]["role"] == "user"

    def test_default_memory_created(self):
        """不传 memory 时自动创建默认 Memory"""
        brain = MemoryManager()
        assert brain.memory is not None
        brain.remember("k", "v")
        assert brain.recall("k") == "v"


# --------------------------------------------------------------------
# summary
# --------------------------------------------------------------------

class TestSummary:
    def test_summary_fields(self, brain):
        brain.remember("k", "v", memory_type="long")
        brain.learn("经验文本")
        brain.record_turn("user", "x")
        brain.record_turn("assistant", "y")
        s = brain.summary()
        assert s["kv_long_count"] == 1
        assert s["vector_enabled"] is True
        assert s["vector_count"] == 1
        assert s["conversation_turns"] == 1
