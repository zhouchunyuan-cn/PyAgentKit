"""
Memory 模块单元测试

覆盖：短/长期记忆存取、短期过期、持久化加载/保存、删除/清空、统计
所有持久化测试用 tmp_path 隔离，避免污染真实 memory.json
"""
import os
import time
import pytest
from core.memory import Memory


@pytest.fixture
def mem(tmp_path):
    """每个测试用独立的持久化文件，互不干扰"""
    return Memory(persistent_storage_path=str(tmp_path / "test_memory.json"))


class TestStoreRetrieve:
    """存取基本功能"""

    def test_store_short_retrieve(self, mem):
        mem.store("k1", "v1", memory_type="short")
        assert mem.retrieve("k1") == "v1"

    def test_store_long_retrieve(self, mem):
        mem.store("k1", "v1", memory_type="long")
        assert mem.retrieve("k1") == "v1"

    def test_retrieve_missing_returns_default(self, mem):
        assert mem.retrieve("nope") is None
        assert mem.retrieve("nope", default="fallback") == "fallback"

    def test_store_complex_value(self, mem):
        data = {"list": [1, 2, 3], "nested": {"a": 1}}
        mem.store("complex", data)
        assert mem.retrieve("complex") == data

    def test_short_takes_precedence_over_long(self, mem):
        # 同名 key，短期值应优先返回
        mem.store("k", "long_val", memory_type="long")
        mem.store("k", "short_val", memory_type="short")
        assert mem.retrieve("k") == "short_val"


class TestExpiry:
    """短期记忆过期机制"""

    def test_short_term_expires(self, tmp_path):
        # duration=1 秒，存入后等待过期
        mem = Memory(short_term_duration=1, persistent_storage_path=str(tmp_path / "m.json"))
        mem.store("temp", "data", memory_type="short")
        assert mem.retrieve("temp") == "data"
        time.sleep(1.1)
        # 过期后应返回 None
        assert mem.retrieve("temp") is None

    def test_cleanup_expired(self, tmp_path):
        mem = Memory(short_term_duration=1, persistent_storage_path=str(tmp_path / "m.json"))
        mem.store("temp", "data", memory_type="short")
        time.sleep(1.1)
        mem.cleanup_expired_memory()
        assert mem.get_memory_stats()["short_term_count"] == 0


class TestPersistence:
    """长期记忆持久化（跨实例）"""

    def test_long_term_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "persist.json")
        mem1 = Memory(persistent_storage_path=path)
        mem1.store("permanent", "kept", memory_type="long")

        # 新实例应能加载到上一个实例存的长期记忆
        mem2 = Memory(persistent_storage_path=path)
        assert mem2.retrieve("permanent") == "kept"

    def test_short_term_not_persisted(self, tmp_path):
        path = str(tmp_path / "persist.json")
        mem1 = Memory(persistent_storage_path=path)
        mem1.store("temp", "ephemeral", memory_type="short")

        mem2 = Memory(persistent_storage_path=path)
        # 短期记忆不应持久化
        assert mem2.retrieve("temp") is None


class TestDeleteClear:
    """删除与清空"""

    def test_delete_short(self, mem):
        mem.store("k", "v", memory_type="short")
        assert mem.delete("k", memory_type="short") is True
        assert mem.retrieve("k") is None

    def test_delete_long_persists(self, mem):
        mem.store("k", "v", memory_type="long")
        assert mem.delete("k", memory_type="long") is True
        assert mem.retrieve("k") is None

    def test_delete_missing_returns_false(self, mem):
        assert mem.delete("nope") is False

    def test_clear_short_only(self, mem):
        mem.store("s", 1, memory_type="short")
        mem.store("l", 2, memory_type="long")
        mem.clear(memory_type="short")
        assert mem.retrieve("s") is None
        assert mem.retrieve("l") == 2

    def test_clear_both(self, mem):
        mem.store("s", 1, memory_type="short")
        mem.store("l", 2, memory_type="long")
        mem.clear(memory_type="both")
        assert mem.get_memory_stats()["short_term_count"] == 0
        assert mem.get_memory_stats()["long_term_count"] == 0


class TestStats:
    """统计信息"""

    def test_memory_stats_counts(self, mem):
        mem.store("s1", 1, memory_type="short")
        mem.store("s2", 2, memory_type="short")
        mem.store("l1", 3, memory_type="long")
        stats = mem.get_memory_stats()
        assert stats["short_term_count"] == 2
        assert stats["long_term_count"] == 1
