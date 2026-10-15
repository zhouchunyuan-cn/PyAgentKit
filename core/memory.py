"""
双层记忆（Memory）

短期记忆基于时间的缓存，长期记忆持久化到磁盘 JSON。
为 Agent 提供跨交互的上下文保持能力。

工程要点：
- 持久化写入是原子操作（先写临时文件再 os.replace），且序列化在截断
  之前完成——写入失败不会破坏已有数据文件。
- 所有公开方法加锁，多线程（如 AsyncRouter 的执行器线程）并发读写安全。
- 短期记忆的过期判定基于 time.monotonic()，不受系统墙钟校时/夏令时影响。
"""

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class Memory:
    """
    PyAgentKit中的记忆模块

    功能:
    - 短期记忆：基于时间的上下文缓存
    - 长期记忆：持久化到 JSON 文件（原子写入）
    """

    def __init__(self, short_term_duration: int = 3600, persistent_storage_path: str | None = None):
        """
        初始化记忆模块

        Args:
            short_term_duration: 短期记忆持续时间（秒），默认1小时
            persistent_storage_path: 长期记忆持久化存储路径；
                                      None（默认）表示不持久化（纯内存，重启丢失）
        """
        # 短期记忆存储在内存中；timestamp 用单调时钟，防墙钟跳变
        self.short_term_memory: dict[str, dict[str, Any]] = {}
        self.long_term_memory: dict[str, Any] = {}
        self.short_term_duration = short_term_duration
        self.persistent_storage_path = persistent_storage_path
        # 保护两块记忆的并发访问（retrieve 触发的惰性删除、cleanup 的遍历删除
        # 与 store 的写入之间 otherwise 会竞态）
        self._lock = threading.RLock()

        # 从持久化存储加载长期记忆
        self._load_persistent_memory()

    def store(self, key: str, value: Any, memory_type: str = "short") -> None:
        """
        存储记忆

        Args:
            key: 记忆键
            value: 记忆值
            memory_type: 记忆类型 ("short" 或 "long")

        Raises:
            ValueError: memory_type 不是 "short"/"long"
            TypeError: 长期记忆值无法 JSON 序列化（持久化场景，写入前校验）
        """
        if memory_type not in ("short", "long"):
            raise ValueError(f"无效的 memory_type: {memory_type!r}，可选 'short' / 'long'")

        with self._lock:
            if memory_type == "short":
                self.short_term_memory[key] = {"value": value, "timestamp": time.monotonic()}
            else:
                # 长期记忆需要可序列化（build_context 会 json.dumps，持久化也依赖），
                # 写入前校验，避免坏值污染存储或事后炸掉上下文组装
                if self.persistent_storage_path:
                    try:
                        json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError) as e:
                        raise TypeError(
                            f"长期记忆值无法 JSON 序列化（key={key!r}）: {e}。"
                            "请存入可序列化的数据，或改用短期记忆。"
                        ) from e
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
        with self._lock:
            # 首先检查短期记忆
            if key in self.short_term_memory:
                memory_entry = self.short_term_memory[key]
                # 检查是否过期（单调时钟）
                if time.monotonic() - memory_entry["timestamp"] <= self.short_term_duration:
                    return memory_entry["value"]
                # 过期则删除
                del self.short_term_memory[key]

            # 然后检查长期记忆
            if key in self.long_term_memory:
                return self.long_term_memory[key]

            return default

    def _load_persistent_memory(self) -> None:
        """
        从持久化存储加载长期记忆

        文件损坏或结构非法时：记录警告并保持空记忆（不抛出，避免阻断启动），
        原文件保留在现场供人工排查。
        """
        try:
            if self.persistent_storage_path and os.path.exists(self.persistent_storage_path):
                with open(self.persistent_storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"持久化记忆格式非法（应为 JSON 对象，实际 {type(data).__name__}）"
                    )
                self.long_term_memory = data
        except Exception as e:
            logger.warning("无法加载持久化记忆: %s", e)
            self.long_term_memory = {}

    def _save_persistent_memory(self) -> None:
        """
        将长期记忆保存到持久化存储（原子写入；path 为 None 时跳过）

        先在内存中完成序列化，再写临时文件并 os.replace 替换目标文件：
        序列化失败或写入中途异常都不会破坏已有文件。
        """
        if not self.persistent_storage_path:
            return
        try:
            # 序列化前置：失败在此抛出，目标文件尚未被触碰
            payload = json.dumps(self.long_term_memory, ensure_ascii=False, indent=2)
            # 写同目录临时文件后原子替换（同目录保证同一文件系统，replace 原子生效）
            dir_name = os.path.dirname(os.path.abspath(self.persistent_storage_path))
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".memory_", suffix=".tmp", dir=dir_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_path, self.persistent_storage_path)
            except BaseException:
                # 清理未完成替换的临时文件
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning("无法保存持久化记忆: %s", e)

    def cleanup_expired_memory(self) -> None:
        """
        清理过期的短期记忆
        """
        with self._lock:
            expired_keys = [
                key
                for key, entry in self.short_term_memory.items()
                if time.monotonic() - entry["timestamp"] > self.short_term_duration
            ]
            for key in expired_keys:
                del self.short_term_memory[key]

    def list_short_term_memory(self) -> dict[str, Any]:
        """
        列出所有未过期的短期记忆

        Returns:
            未过期的短期记忆字典
        """
        self.cleanup_expired_memory()
        with self._lock:
            return {k: v["value"] for k, v in self.short_term_memory.items()}

    def list_long_term_memory(self) -> dict[str, Any]:
        """
        列出所有长期记忆

        Returns:
            长期记忆字典
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            return {
                "short_term_count": len(self.short_term_memory),
                "long_term_count": len(self.long_term_memory),
            }

    def get_memory_size(self, memory_type: str = "both") -> int:
        """
        获取记忆大小（估计，以字符数计）

        Args:
            memory_type: 记忆类型 ("short", "long" 或 "both")

        Returns:
            记忆大小的粗略估计（key + 值字符串化的字符数总和）
        """
        with self._lock:
            size = 0
            if memory_type in ["short", "both"]:
                for key, entry in self.short_term_memory.items():
                    size += len(key) + len(str(entry.get("value", "")))
            if memory_type in ["long", "both"]:
                for key, value in self.long_term_memory.items():
                    size += len(key) + len(str(value))
            return size
