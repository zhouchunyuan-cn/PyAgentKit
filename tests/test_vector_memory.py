"""
VectorMemory 单元测试

用确定性 FakeEmbedder（基于词袋）避免联网依赖，
同时保证相似度结果可预测、可断言。

覆盖：存入/计数、语义召回排序、相似度阈值、删除、清空、持久化加载
"""
import os
import pytest
import numpy as np
from core.vector_memory import VectorMemory


class FakeEmbedder:
    """
    确定性的词袋向量化器（仅供测试）

    把文本按预设词汇表转为 one-hot 向量，相似度由共现词决定，
    结果确定且可断言，无需联网。
    """

    VOCAB = ["python", "编程", "语言", "git", "版本", "控制", "苹果", "吃", "午餐"]

    def embed(self, text: str) -> list:
        text_lower = text.lower()
        return [1.0 if word in text_lower else 0.0 for word in self.VOCAB]


@pytest.fixture
def vm():
    return VectorMemory(embedder=FakeEmbedder())


class TestStoreAndCount:
    def test_add_returns_id(self, vm):
        iid = vm.add("python 编程 语言")
        assert isinstance(iid, str) and len(iid) > 0

    def test_count(self, vm):
        vm.add("python 编程")
        vm.add("git 版本 控制")
        assert vm.count() == 2

    def test_empty_search_returns_empty(self, vm):
        assert vm.search("任意查询") == []


class TestSemanticRecall:
    """语义召回的核心测试"""

    def test_recall_returns_related_first(self, vm):
        vm.add("python 是一门编程语言")          # 含 python/编程/语言
        vm.add("git 是版本控制工具")              # 含 git/版本/控制
        vm.add("今天午餐吃了苹果")                # 含 午餐/苹果

        # 查询"python 编程"应优先召回第一条
        results = vm.search("python 编程", top_k=1)
        assert len(results) == 1
        assert "python" in results[0]["text"]

    def test_recall_orders_by_similarity(self, vm):
        vm.add("python 编程语言")
        vm.add("git 版本控制")
        vm.add("python 版本")  # 同时含 python 和 版本

        # 查询"python 版本"：同时含两个词的应排最前
        results = vm.search("python 版本", top_k=3)
        # 完全匹配两项的条目应相似度最高
        assert "python" in results[0]["text"] and "版本" in results[0]["text"]

    def test_recall_semantic_returns_texts(self, vm):
        vm.add("python 编程语言")
        texts = vm.recall_semantic("python", top_k=1)
        assert isinstance(texts, list)
        assert len(texts) == 1

    def test_top_k_limit(self, vm):
        for i in range(5):
            vm.add(f"python 编程 {i}")
        results = vm.search("python", top_k=3)
        assert len(results) == 3


class TestSimilarityThreshold:
    def test_min_similarity_filters_results(self, vm):
        vm.add("python 编程语言")
        # 用极高阈值，应过滤掉所有结果
        results = vm.search("git 版本控制", top_k=3, min_similarity=0.99)
        assert results == []


class TestDeleteAndClear:
    def test_delete(self, vm):
        iid = vm.add("python 编程")
        assert vm.delete(iid) is True
        assert vm.count() == 0
        # 删除后召回应为空
        assert vm.search("python") == []

    def test_delete_missing_returns_false(self, vm):
        assert vm.delete("ghost") is False

    def test_clear(self, vm):
        vm.add("python 编程")
        vm.add("git 版本")
        vm.clear()
        assert vm.count() == 0


class TestPersistence:
    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "vm.json")
        # 写入
        vm1 = VectorMemory(embedder=FakeEmbedder(), persistence_path=path)
        vm1.add("python 编程语言")

        # 新实例从同一文件加载
        vm2 = VectorMemory(embedder=FakeEmbedder(), persistence_path=path)
        assert vm2.count() == 1
        # 加载后仍可正常召回
        results = vm2.search("python", top_k=1)
        assert len(results) == 1
        assert "python" in results[0]["text"]
