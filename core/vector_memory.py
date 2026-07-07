"""
向量记忆（语义召回）

基于文本 embedding + 余弦相似度，实现"按语义相关性"召回记忆，
而非只能按 key 精确匹配。

设计要点：
- Embedder 可注入（默认 LocalTfidfEmbedder，零成本离线可用；
  充值后可换 GLMEmbedder 获得更强语义效果）
- 相似度计算用 numpy + sklearn cosine，零新增重依赖（均为已装库）
- 支持持久化（可选）：保存记忆条目到 JSON，向量在加载时重新生成
"""
from typing import Any, Dict, List, Optional, Protocol, Tuple
import json
import os
import logging
import uuid
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """
    文本向量化协议（结构化子类型）

    任何实现 embed(text) -> List[float] 的对象均可作为 embedder。
    """

    def embed(self, text: str) -> List[float]: ...


class GLMEmbedder:
    """
    基于智谱 embedding-3 的文本向量化器

    复用项目已有的 zhipuai SDK 与 API Key。语义效果最佳，但需消耗 embedding 额度。
    """

    def __init__(self, model: str = "embedding-3", api_key: Optional[str] = None):
        resolved_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "未找到 GLM API Key。请设置环境变量 ZHIPUAI_API_KEY "
                "（向量记忆需要 embedding 服务）。"
            )
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError("未安装 zhipuai SDK，请运行：pip install zhipuai") from e

        self.model = model
        self._client = ZhipuAI(api_key=resolved_key)

    def embed(self, text: str) -> List[float]:
        """将文本转为向量"""
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


# 中文常用停用词，提升 TF-IDF 向量质量
_ZH_STOP_WORDS = frozenset([
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "它", "他", "她", "与", "及", "或", "但",
    "而", "则", "于", "对", "为", "以", "可", "能", "把", "被", "使", "让",
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "and", "or", "for",
])


def _tokenize(text: str) -> List[str]:
    """
    轻量分词：中文按字、英文按词。

    不依赖 jieba，保证零额外依赖；对召回质量影响有限但够用。
    """
    tokens = []
    # 英文单词
    tokens.extend(re.findall(r"[a-zA-Z]+", text.lower()))
    # 中文字符（逐字）
    tokens.extend(re.findall(r"[\u4e00-\u9fa5]", text))
    return [t for t in tokens if t not in _ZH_STOP_WORDS]


class LocalTfidfEmbedder:
    """
    基于 sklearn TF-IDF 的纯本地向量化器（默认实现）

    优势：零成本、离线可用、无 API 依赖。
    劣势：语义理解弱于神经网络 embedding（无法识别同义/推理关系），
          仅基于词项重叠。充值 embedding-3 后建议换 GLMEmbedder。

    注意：TF-IDF 需要语料来学习 IDF 权重。本实现采用【增量拟合 + 回退】策略：
    首次 embed 时用该文本自身拟合；后续文本用已学词表转换，新词忽略。
    这是一种工程妥协，保证零依赖下的可用性。
    """

    def __init__(self):
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._corpus: List[str] = []
        self._fitted = False

    def _ensure_fitted(self, texts: List[str]) -> None:
        """用给定文本拟合 TF-IDF（仅在首次或语料扩充时）"""
        all_texts = self._corpus + [t for t in texts if t not in self._corpus]
        self._corpus = all_texts
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,  # 用自定义 tokenizer 时禁用默认 pattern
        )
        try:
            self._vectorizer.fit(all_texts)
            self._fitted = True
        except ValueError:
            # 语料全为空或分词后无有效 token 时，fit 会失败
            self._fitted = False

    def embed(self, text: str) -> List[float]:
        """将文本转为 TF-IDF 向量（会扩展词表，用于存入记忆）"""
        if not self._fitted or text.strip():
            self._ensure_fitted([text])

        if not self._fitted or self._vectorizer is None:
            # 无法拟合（如空文本），返回空向量
            return [0.0]

        vec = self._vectorizer.transform([text])
        return vec.toarray()[0].tolist()

    def transform(self, text: str) -> List[float]:
        """
        用已 fit 的固定词表转换文本（不扩展词表，用于查询）

        关键：查询时不能改变词表，否则查询向量维度与库内向量不一致。
        新词（词表中没有的）会被忽略，保证维度固定。
        """
        if not self._fitted or self._vectorizer is None:
            return [0.0]
        vec = self._vectorizer.transform([text])
        return vec.toarray()[0].tolist()


class VectorMemory:
    """
    向量记忆库

    存储 (id, text, metadata, vector) 条目，支持按语义相似度召回。
    适用于 Agent 的长期经验记忆：存入后能"想起相关的"内容。

    用法：
        # 默认用本地 TF-IDF（零成本）
        vm = VectorMemory(embedder=LocalTfidfEmbedder())
        vm.add("Python 是解释型语言", metadata={"topic": "python"})
        results = vm.search("解释型编程语言", top_k=3)  # 召回最相关的

        # 充值后可换更强的语义 embedding
        vm = VectorMemory(embedder=GLMEmbedder())
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        persistence_path: Optional[str] = None,
        min_similarity: float = 0.0,
    ):
        """
        Args:
            embedder: 文本向量化器；None 时默认用 LocalTfidfEmbedder（零成本）
            persistence_path: 持久化文件路径，None 表示不持久化（纯内存）
            min_similarity: 召回的最小相似度阈值，低于此值不返回
        """
        self.embedder = embedder if embedder is not None else LocalTfidfEmbedder()
        self.persistence_path = persistence_path
        self.min_similarity = min_similarity

        # 条目存储：id -> {"text", "metadata", "vector"}
        self._items: Dict[str, Dict[str, Any]] = {}
        # 缓存向量矩阵与对应 id 顺序，加速相似度计算
        self._ids_in_order: List[str] = []
        self._matrix: Optional[np.ndarray] = None

        if persistence_path:
            self._load()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, item_id: Optional[str] = None) -> str:
        """
        存入一条记忆

        Args:
            text: 记忆文本（召回时基于它的语义）
            metadata: 附加元数据（如来源、时间、topic）
            item_id: 自定义 id，None 则自动生成

        Returns:
            这条记忆的 id
        """
        item_id = item_id or str(uuid.uuid4())

        # 增量型 embedder（如 LocalTfidfEmbedder）的词表会随语料增长而变化，
        # 新增文本后已有条目的向量需重新计算，否则向量空间不一致导致相似度失真。
        # 对固定词表的 embedder（如 GLMEmbedder）重算也无害，统一处理。
        self._items[item_id] = {
            "text": text,
            "metadata": metadata or {},
        }
        self._ids_in_order.append(item_id)
        self._recompute_all_vectors()

        if self.persistence_path:
            self._save()
        return item_id

    def delete(self, item_id: str) -> bool:
        """删除一条记忆"""
        if item_id not in self._items:
            return False
        del self._items[item_id]
        self._ids_in_order = [i for i in self._ids_in_order if i != item_id]
        self._recompute_all_vectors()
        if self.persistence_path:
            self._save()
        return True

    # ------------------------------------------------------------------
    # 语义召回
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        按语义相似度召回最相关的记忆

        Args:
            query: 查询文本
            top_k: 最多返回条数
            min_similarity: 本次查询的最小相似度阈值，None 用实例默认值

        Returns:
            命中条目列表（按相似度降序），每项含 id/text/metadata/similarity
        """
        if not self._items:
            return []

        threshold = self.min_similarity if min_similarity is None else min_similarity
        # 查询向量化：增量型 embedder 必须用固定词表转换，保证维度与库一致
        embedder = self.embedder
        if isinstance(embedder, LocalTfidfEmbedder):
            query_vec = np.array(embedder.transform(query)).reshape(1, -1)
        else:
            query_vec = np.array(embedder.embed(query)).reshape(1, -1)

        # 计算查询向量与所有条目的余弦相似度
        sims = cosine_similarity(query_vec, self._matrix)[0]  # shape: (n,)

        # 组装结果并按相似度降序
        scored: List[Tuple[float, str]] = [
            (float(sims[idx]), self._ids_in_order[idx])
            for idx in range(len(self._ids_in_order))
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, item_id in scored[:top_k]:
            if sim < threshold:
                continue
            item = self._items[item_id]
            results.append({
                "id": item_id,
                "text": item["text"],
                "metadata": item["metadata"],
                "similarity": sim,
            })
        return results

    def recall_semantic(self, query: str, top_k: int = 3) -> List[str]:
        """
        便捷方法：仅返回最相关记忆的文本（常见于 Agent 注入上下文）

        Args:
            query: 查询文本
            top_k: 最多返回条数

        Returns:
            文本列表（按相似度降序）
        """
        return [item["text"] for item in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------
    # 维护与持久化
    # ------------------------------------------------------------------
    def count(self) -> int:
        """返回记忆条目数"""
        return len(self._items)

    def clear(self) -> None:
        """清空所有记忆"""
        self._items.clear()
        self._ids_in_order.clear()
        self._matrix = None
        if self.persistence_path:
            self._save()

    def _recompute_all_vectors(self) -> None:
        """
        重新计算所有条目的向量并重建矩阵

        增量型 embedder（TF-IDF）的词表随语料变化，每次 add/delete 后
        必须重算全部向量，保证向量空间一致。对固定词表 embedder（GLM）同样适用。
        """
        if not self._items:
            self._matrix = None
            return
        # 对增量型 embedder，先让它"见到"全部当前语料再逐条向量化
        all_texts = [self._items[i]["text"] for i in self._ids_in_order]
        embedder = self.embedder
        if isinstance(embedder, LocalTfidfEmbedder):
            embedder._ensure_fitted(all_texts)
        vectors = [embedder.embed(self._items[i]["text"]) for i in self._ids_in_order]
        # 缓存向量到条目（供持久化之外的快速访问）
        for iid, vec in zip(self._ids_in_order, vectors):
            self._items[iid]["vector"] = vec
        self._matrix = np.array(vectors)

    def _save(self) -> None:
        """持久化文本与元数据（向量不存，加载时按 embedder 重新生成）"""
        try:
            data = [
                {"id": iid, "text": it["text"], "metadata": it["metadata"]}
                for iid, it in self._items.items()
            ]
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("向量记忆持久化失败: %s", e)

    def _load(self) -> None:
        """从持久化文件加载（重新计算向量）"""
        if not (self.persistence_path and os.path.exists(self.persistence_path)):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                item_id = entry.get("id") or str(uuid.uuid4())
                self._items[item_id] = {
                    "text": entry["text"],
                    "metadata": entry.get("metadata", {}),
                }
                self._ids_in_order.append(item_id)
            # 统一重算所有向量（增量型 embedder 需在完整语料上拟合）
            self._recompute_all_vectors()
            logger.info("向量记忆已加载 %d 条", len(self._items))
        except Exception as e:
            logger.warning("向量记忆加载失败: %s", e)
