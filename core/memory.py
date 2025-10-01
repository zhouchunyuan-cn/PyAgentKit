from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import json


class Memory:
    """
    PyAgentKit中的记忆模块
    
    功能:
    - 短期记忆：基于时间的上下文缓存
    - 长期记忆：数据库/向量存储（简化版实现）
    """

    def __init__(self, short_term_duration: int = 3600):
        """
        初始化记忆模块
        
        Args:
            short_term_duration: 短期记忆持续时间（秒），默认1小时
        """
        # 短期记忆存储在内存中
        self.short_term_memory: Dict[str, Dict[str, Any]] = {}
        self.long_term_memory: Dict[str, Any] = {}
        self.short_term_duration = short_term_duration

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
            self.short_term_memory[key] = {
                "value": value,
                "timestamp": timestamp
            }
        elif memory_type == "long":
            self.long_term_memory[key] = value

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
            if datetime.now() - memory_entry["timestamp"] <= timedelta(seconds=self.short_term_duration):
                return memory_entry["value"]
            else:
                # 过期则删除
                del self.short_term_memory[key]
        
        # 然后检查长期记忆
        if key in self.long_term_memory:
            return self.long_term_memory[key]
            
        return default

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

    def list_short_term_memory(self) -> Dict[str, Any]:
        """
        列出所有未过期的短期记忆
        
        Returns:
            未过期的短期记忆字典
        """
        self.cleanup_expired_memory()
        return {k: v["value"] for k, v in self.short_term_memory.items()}

    def list_long_term_memory(self) -> Dict[str, Any]:
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
        if memory_type == "short" and key in self.short_term_memory:
            del self.short_term_memory[key]
            return True
        elif memory_type == "long" and key in self.long_term_memory:
            del self.long_term_memory[key]
            return True
        return False

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

    def get_memory_stats(self) -> Dict[str, int]:
        """
        获取记忆统计信息
        
        Returns:
            包含短期和长期记忆数量的字典
        """
        self.cleanup_expired_memory()
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory)
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