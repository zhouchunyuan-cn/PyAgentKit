"""
双层记忆（Memory）

短期记忆基于时间的缓存，长期记忆持久化到磁盘 JSON。
为 Agent 提供跨交互的上下文保持能力。
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class Memory:
    """
    PyAgentKit中的记忆模块

    功能:
    - 短期记忆：基于时间的上下文缓存
    - 长期记忆：数据库/向量存储（简化版实现）
    """

    def __init__(
        self, short_term_duration: int = 3600, persistent_storage_path: str = "memory.json"
    ):
        """
        初始化记忆模块

        Args:
            short_term_duration: 短期记忆持续时间（秒），默认1小时
            persistent_storage_path: 长期记忆持久化存储路径
        """
        # 短期记忆存储在内存中
        self.short_term_memory: dict[str, dict[str, Any]] = {}
        self.long_term_memory: dict[str, Any] = {}
        self.short_term_duration = short_term_duration
        self.persistent_storage_path = persistent_storage_path

        # 从持久化存储加载长期记忆
        self._load_persistent_memory()

    def store(self, key: str, value: Any, memory_type: str = "short") -> None:
        """
        存储记忆

        Args:
            key: 记忆键
            value: 记忆值
            memory_type: 记忆类型 ("short" 或 "long")
        """
        timestamp = datetime.now()

        if memory_type == "short":
            self.short_term_memory[key] = {"value": value, "timestamp": timestamp}
        elif memory_type == "long":
            self.long_term_memory[key] = value
            # 持久化长期记忆
            self._save_persistent_memory()

    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        检索记忆

        Args:
            key: 记忆键
            default: 默认值

        Returns:
            记忆值或默认值
        """
        # 首先检查短期记忆
        if key in self.short_term_memory:
            memory_entry = self.short_term_memory[key]
            # 检查是否过期
            if datetime.now() - memory_entry["timestamp"] <= timedelta(
                seconds=self.short_term_duration
            ):
                return memory_entry["value"]
            else:
                # 过期则删除
                del self.short_term_memory[key]

        # 然后检查长期记忆
        if key in self.long_term_memory:
            return self.long_term_memory[key]

        return default

    def _load_persistent_memory(self) -> None:
        """
        从持久化存储加载长期记忆
        """
        try:
            if os.path.exists(self.persistent_storage_path):
                with open(self.persistent_storage_path, encoding="utf-8") as f:
                    self.long_term_memory = json.load(f)
        except Exception as e:
            logger.warning("无法加载持久化记忆: %s", e)

    def _save_persistent_memory(self) -> None:
        """
        将长期记忆保存到持久化存储
        """
        try:
            with open(self.persistent_storage_path, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("无法保存持久化记忆: %s", e)

    def cleanup_expired_memory(self) -> None:
        """
        清理过期的短期记忆
        """
        expired_keys = []
        current_time = datetime.now()

        for key, entry in self.short_term_memory.items():
            if current_time - entry["timestamp"] > timedelta(seconds=self.short_term_duration):
                expired_keys.append(key)

        for key in expired_keys:
            del self.short_term_memory[key]

    def list_short_term_memory(self) -> dict[str, Any]:
        """
        列出所有未过期的短期记忆

        Returns:
            未过期的短期记忆字典
        """
        self.cleanup_expired_memory()
        return {k: v["value"] for k, v in self.short_term_memory.items()}

    def list_long_term_memory(self) -> dict[str, Any]:
        """
        列出所有长期记忆

        Returns:
            长期记忆字典
        """
        return self.long_term_memory.copy()

    def delete(self, key: str, memory_type: str = "short") -> bool:
        """
        删除记忆

        Args:
            key: 记忆键
            memory_type: 记忆类型 ("short" 或 "long")

        Returns:
            是否成功删除
        """
        deleted = False
        if memory_type == "short" and key in self.short_term_memory:
            del self.short_term_memory[key]
            deleted = True
        elif memory_type == "long" and key in self.long_term_memory:
            del self.long_term_memory[key]
            deleted = True
            # 更新持久化存储
            self._save_persistent_memory()
        return deleted

    def clear(self, memory_type: str = "both") -> None:
        """
        清空记忆

        Args:
            memory_type: 记忆类型 ("short", "long" 或 "both")
        """
        if memory_type in ["short", "both"]:
            self.short_term_memory.clear()
        if memory_type in ["long", "both"]:
            self.long_term_memory.clear()
            # 更新持久化存储
            self._save_persistent_memory()

    def get_memory_stats(self) -> dict[str, int]:
        """
        获取记忆统计信息

        Returns:
            包含短期和长期记忆数量的字典
        """
        self.cleanup_expired_memory()
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
        }

    def get_memory_size(self, memory_type: str = "both") -> int:
        """
        获取记忆大小（估计）

        Args:
            memory_type: 记忆类型 ("short", "long" 或 "both")

        Returns:
            记忆大小（字节）
        """
        size = 0
        if memory_type in ["short", "both"]:
            for key, entry in self.short_term_memory.items():
                size += len(key) + len(str(entry.get("value", "")))
        if memory_type in ["long", "both"]:
            for key, value in self.long_term_memory.items():
                size += len(key) + len(str(value))
        return size
