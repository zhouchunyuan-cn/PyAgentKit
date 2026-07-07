"""
工具工厂

按名称实例化内置工具，供 YAML 配置驱动使用。
这样配置文件只需写工具名（如 "web_search"），无需写 Python 代码。

也支持注册自定义工具，扩展可配置的工具集。
"""
from typing import Any, Callable, Dict, List, Optional, Type
import logging

from .tools import (
    Tool,
    WebSearchTool,
    CalculatorTool,
    FileReadTool,
    DatabaseTool,
)

logger = logging.getLogger(__name__)


class ToolNotFoundError(KeyError):
    """请求的工具未注册时抛出"""


class ToolFactory:
    """
    工具工厂

    维护"工具名 -> 构造器"映射，按名称产出工具实例。
    内置常用工具，并允许注册自定义工具。
    """

    def __init__(self):
        # name -> 构造器（无参 callable，返回 Tool 实例）
        self._registry: Dict[str, Callable[[], Tool]] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        """注册内置工具的默认构造器"""
        self.register("web_search", lambda: WebSearchTool(mock_mode=False))
        self.register("web_search_mock", lambda: WebSearchTool(mock_mode=True))
        self.register("calculator", CalculatorTool)
        self.register("file_read", FileReadTool)
        self.register("database", DatabaseTool)

    def register(self, name: str, constructor: Callable[[], Tool]) -> None:
        """
        注册一个工具构造器

        Args:
            name: 工具名（配置文件中引用）
            constructor: 无参 callable，返回 Tool 实例
        """
        self._registry[name] = constructor

    def register_class(self, name: str, tool_class: Type[Tool]) -> None:
        """注册一个工具类（便捷方法）"""
        self.register(name, tool_class)

    def create(self, name: str) -> Tool:
        """
        按名称创建工具实例

        Args:
            name: 工具名

        Returns:
            Tool 实例

        Raises:
            ToolNotFoundError: 工具未注册
        """
        constructor = self._registry.get(name)
        if constructor is None:
            raise ToolNotFoundError(
                f"未知工具 '{name}'。可用工具: {self.list_tools()}"
            )
        return constructor()

    def create_many(self, names: List[str]) -> List[Tool]:
        """批量创建工具实例"""
        return [self.create(name) for name in names]

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名"""
        return sorted(self._registry.keys())


# 模块级默认工厂单例，供配置加载复用
default_factory = ToolFactory()
