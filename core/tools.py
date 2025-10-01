from typing import Dict, Any, Optional, Callable
import requests
import json
import os
import re
import operator
import ast
from .mcp_tools import MCPToolRegistry, ConcreteMCPIntegrationTool


class Tool:
    """
    PyAgentKit中的工具基类
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, **kwargs) -> Any:
        """
        执行工具
        """
        raise NotImplementedError


class WebSearchTool(Tool):
    """
    网络搜索工具
    """

    def __init__(self):
        super().__init__("web_search", "通过网络搜索获取信息")

    def run(self, query: str) -> Dict[str, Any]:
        """
        模拟网络搜索结果
        在实际实现中，这里会调用真实的搜索引擎API
        """
        # 模拟搜索结果
        mock_results = {
            "人工智能": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "机器学习": "机器学习是人工智能的一个子领域，专注于让计算机通过数据学习和改进。",
            "深度学习": "深度学习是机器学习的一个分支，使用神经网络来模拟人脑处理信息的方式。"
        }
        
        result = mock_results.get(query, f"未找到关于'{query}'的具体信息。")
        
        return {
            "query": query,
            "result": result,
            "source": "MockSearchEngine"
        }


class CalculatorTool(Tool):
    """
    计算器工具 - 使用安全的表达式解析器
    """

    def __init__(self):
        super().__init__("calculator", "执行数学计算")
        # 支持的操作符
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

    def eval_expr(self, expr: str) -> float:
        """
        安全地计算数学表达式
        
        Args:
            expr: 数学表达式字符串
            
        Returns:
            计算结果
        """
        # 预处理表达式，移除空格
        expr = re.sub(r'\s+', '', expr)
        
        # 验证表达式只包含允许的字符
        if not re.match(r'^[0-9+\-*/().% ]+$', expr):
            raise ValueError("表达式包含不允许的字符")
            
        # 解析并计算表达式
        node = ast.parse(expr, mode='eval')
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        """
        递归计算AST节点的值
        
        Args:
            node: AST节点
            
        Returns:
            节点值
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的操作符: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元操作符: {type(node.op)}")
            return op(operand)
        else:
            raise ValueError(f"不支持的表达式节点类型: {type(node)}")

    def run(self, expression: str) -> Dict[str, Any]:
        """
        执行数学计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果字典
        """
        try:
            result = self.eval_expr(expression)
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }


class FileReadTool(Tool):
    """
    文件读取工具
    """

    def __init__(self):
        super().__init__("file_read", "读取文件内容")

    def run(self, path: str) -> Dict[str, Any]:
        """
        读取文件内容
        
        Args:
            path: 文件路径
            
        Returns:
            文件内容或错误信息
        """
        try:
            # 安全检查：确保路径在当前目录下
            if '..' in path or path.startswith('/'):
                raise ValueError("不允许访问该路径")
                
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "path": path,
                "content": content
            }
        except Exception as e:
            return {
                "path": path,
                "error": str(e)
            }


class ToolRegistry:
    """
    工具注册表
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        # 添加对MCP工具的支持
        self.mcp_registry = MCPToolRegistry()
        self.mcp_integration_tool = ConcreteMCPIntegrationTool(self.mcp_registry)
        self.register(self.mcp_integration_tool)

    def register(self, tool: Tool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具对象
        """
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具对象或None
        """
        return self.tools.get(name)

    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 工具未找到时抛出异常
        """
        tool = self.get(tool_name)
        if tool:
            return tool.run(**kwargs)
        else:
            raise ValueError(f"Tool '{tool_name}' not found")

    def list_tools(self) -> Dict[str, str]:
        """
        列出所有注册的工具
        
        Returns:
            工具名称和描述的字典
        """
        tools = {name: tool.description for name, tool in self.tools.items()}
        # 添加MCP工具
        mcp_tools = self.mcp_registry.list_tools()
        tools.update({f"mcp:{name}": desc for name, desc in mcp_tools.items()})
        return tools

    def register_mcp_tool(self, mcp_tool) -> bool:
        """
        注册MCP工具
        
        Args:
            mcp_tool: MCP工具实例
            
        Returns:
            是否注册成功
        """
        return self.mcp_registry.register_tool(mcp_tool)

    def execute_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行MCP工具
        
        Args:
            tool_name: MCP工具名称
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        return self.mcp_registry.execute_tool(tool_name, **kwargs)