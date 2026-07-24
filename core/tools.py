"""
工具系统（Tools）

定义 Agent 可调用的工具：Tool 基类、HttpTool（带超时重试）、
内置工具（calculator/web_search/file_read/database）、ToolRegistry。
工具自带 JSON Schema，供 LLM function-calling 使用。
"""

import ast
import json
import logging
import operator
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from .mcp_tools import ConcreteMCPIntegrationTool, MCPToolRegistry

logger = logging.getLogger(__name__)


class Tool:
    """
    PyAgentKit中的工具基类

    Attributes:
        name: 工具名称（唯一标识）
        description: 工具功能描述（供 LLM 理解何时使用）
        parameters: 入参的 JSON Schema，供 function-calling 使用
    """

    def __init__(self, name: str, description: str, parameters: dict[str, Any] | None = None):
        """
        初始化工具

        Args:
            name: 工具名称
            description: 工具描述
            parameters: 入参 JSON Schema（描述 run() 接受的参数）
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    def run(self, **kwargs) -> Any:
        """
        执行工具
        """
        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        """
        转换为 GLM/OpenAI function-calling 要求的工具 schema 格式

        Returns:
            {"type": "function", "function": {"name", "description", "parameters"}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class HttpTool(Tool):
    """
    需要 HTTP 访问的工具基类

    统一管理超时与指数退避重试，WebSearchTool / MCP 工具可复用。
    使用标准库 urllib，不引入额外依赖。
    """

    # 可重试的异常关键字（网络/限流类）
    _RETRYABLE_KEYWORDS = (
        "timeout",
        "connection",
        "urlopen",
        "timed out",
        "429",
        "500",
        "502",
        "503",
        "504",
    )

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        timeout: float = 8.0,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
    ):
        """
        Args:
            name / description / parameters: 同 Tool
            timeout: 单次请求超时秒数
            max_retries: 最大重试次数（不含首次）
            retry_base_delay: 重试退避基数（指数退避：base * 2^attempt）
        """
        super().__init__(name, description, parameters)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    def _http_get(self, url: str, headers: dict[str, str] | None = None) -> str:
        """
        带 timeout + 指数退避重试的 HTTP GET

        Args:
            url: 请求 URL
            headers: 请求头

        Returns:
            响应正文文本

        Raises:
            最后一次仍失败时抛出原始异常
        """
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "PyAgentKit/1.0"})
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and self._is_retryable(e):
                    delay = self.retry_base_delay * (2**attempt)
                    logger.debug(
                        "HTTP 请求失败（%s），%.1fs 后重试 (%d/%d)",
                        e,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise
        raise last_error  # 理论上不会走到

    def _is_retryable(self, error: Exception) -> bool:
        """判断异常是否值得重试"""
        msg = str(error).lower()
        return any(k in msg for k in self._RETRYABLE_KEYWORDS)


class WebSearchTool(HttpTool):
    """
    网络搜索工具（基于 DuckDuckGo，多源聚合召回）

    召回策略（按顺序尝试，命中即用）：
    1. DuckDuckGo Instant Answer API —— 有结构化摘要时直接用
    2. DuckDuckGo HTML 结果页解析 —— 兜底拿多条网页结果摘要

    相比早期仅依赖 Instant Answer（对中文召回极差），现在能返回多条结果，
    显著提升中文等场景的可用性。继承 HttpTool，复用超时与重试机制。
    """

    # mock 模式下使用的内置示例数据（仅用于无网络/测试场景）
    _MOCK_RESULTS = {
        "人工智能": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
        "机器学习": "机器学习是人工智能的一个子领域，专注于让计算机通过数据学习和改进。",
        "深度学习": "深度学习是机器学习的一个分支，使用神经网络来模拟人脑处理信息的方式。",
    }

    def __init__(
        self,
        mock_mode: bool = False,
        timeout: float = 8.0,
        max_results: int = 5,
        max_retries: int = 2,
    ):
        """
        初始化网络搜索工具

        Args:
            mock_mode: 是否启用 mock 模式。
                       True 时直接返回内置示例数据，不发起网络请求（用于测试/离线）；
                       False（默认）时发起真实网络请求，失败则返回明确的 error。
            timeout: 网络请求超时秒数
            max_results: 最多返回的结果条数
            max_retries: 网络失败重试次数（0 表示不重试，测试场景常用）
        """
        super().__init__(
            name="web_search",
            description="通过网络搜索获取信息，输入一个查询词，返回相关的多条搜索结果摘要。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询关键词",
                    }
                },
                "required": ["query"],
            },
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=0.1,  # 搜索重试退避短一些
        )
        self.mock_mode = mock_mode
        self.max_results = max_results

    def run(self, query: str) -> dict[str, Any]:
        """
        通过网络搜索获取信息

        Args:
            query: 搜索查询

        Returns:
            成功：{query, results: [{title, snippet, url}], source}
            失败：{query, error, source}（绝不静默返回伪造数据）
        """
        # mock 模式：直接返回内置数据（来源明确标注，便于区分）
        if self.mock_mode:
            snippet = self._MOCK_RESULTS.get(query, f"未找到关于'{query}'的具体信息。")
            return {
                "query": query,
                "results": [{"title": query, "snippet": snippet, "url": ""}],
                "result": snippet,  # 向后兼容字段
                "source": "MockSearchEngine",
            }

        results: list[dict[str, str]] = []

        # 路径1：DuckDuckGo Instant Answer API（结构化摘要）
        try:
            ia = self._search_instant_answer(query)
            if ia:
                results.append(ia)
        except Exception as e:
            logger.debug("Instant Answer 路径失败: %s", e)

        # 路径2：HTML 解析（多结果兜底）
        try:
            html_results = self._search_html(query)
            # 去重：HTML 结果中已存在相同 URL 则不重复加
            existing_urls = {r.get("url") for r in results if r.get("url")}
            for r in html_results:
                if r.get("url") not in existing_urls:
                    results.append(r)
                    if len(results) >= self.max_results:
                        break
        except Exception as e:
            logger.debug("HTML 解析路径失败: %s", e)

        if not results:
            return {
                "query": query,
                "error": "搜索完成但未获取到结果（可能是网络受限或被限流）",
                "source": "DuckDuckGo",
            }

        # 为兼容旧调用方（只看 result 字段），同时提供拼接的文本摘要
        summary = "\n".join(f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results)
        return {
            "query": query,
            "results": results[: self.max_results],
            "result": summary,  # 向后兼容字段
            "source": "DuckDuckGo",
        }

    def _search_instant_answer(self, query: str) -> dict[str, str] | None:
        """DuckDuckGo Instant Answer API（结构化摘要）"""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disamb=1"
        text_body = self._http_get(url, headers={"User-Agent": "PyAgentKit/1.0"})
        data = json.loads(text_body)

        text = ""
        if data.get("AbstractText"):
            text = data["AbstractText"]
        elif data.get("RelatedTopics"):
            first = data["RelatedTopics"][0]
            if isinstance(first, dict) and first.get("Text"):
                text = first["Text"]
        if not text:
            return None
        return {
            "title": data.get("Heading") or query,
            "snippet": text,
            "url": data.get("AbstractURL") or "",
        }

    def _search_html(self, query: str) -> list[dict[str, str]]:
        """
        解析 DuckDuckGo Lite 结果页，提取多条结果

        用 DuckDuckGo Lite 版本（lite.duckduckgo.com），它的结果是纯 HTML 表格，
        结构稳定且对非浏览器 UA 友好。用正则提取链接与文本，无需额外 HTML 解析库。
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
        html = self._http_get(url, headers={"User-Agent": "Mozilla/5.0 (PyAgentKit web search)"})

        results: list[dict[str, str]] = []
        # Lite 页结构：<a rel="nofollow" href="//duckduckgo.com/l/?uddg=<真实URL>">标题文本</a>
        # 后面同行的 td 含摘要文本。用正则逐个结果块提取。
        link_re = re.compile(
            r'href="//duckduckgo\.com/l/\?uddg=([^"&]+)[^"]*"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        for m in link_re.finditer(html):
            encoded_url, raw_title = m.group(1), m.group(2)
            title = self._strip_html(raw_title)
            # uddg 参数值是 URL 编码的真实地址
            clean_url = urllib.parse.unquote(encoded_url)
            # 跳过 DuckDuckGo 自己的广告/追踪链接（y.js 结尾的是广告）
            if clean_url.endswith("y.js") or "duckduckgo.com/y.js" in clean_url:
                continue
            if title:
                results.append({"title": title, "snippet": title, "url": clean_url})

        return results

    @staticmethod
    def _strip_html(text: str) -> str:
        """去除 HTML 标签和常见实体（含十进制/十六进制形式），压平空白"""
        text = re.sub(r"<[^>]+>", "", text)
        # 命名实体 & 十进制 &#NN; & 十六进制 &#xHH;
        text = re.sub(r"&[a-zA-Z]+;|&#\d+;|&#[xX][0-9a-fA-F]+;", " ", text)
        return re.sub(r"\s+", " ", text).strip()


class CalculatorTool(Tool):
    """
    计算器工具 - 使用安全的表达式解析器
    """

    def __init__(self):
        super().__init__(
            "calculator",
            "执行数学计算，支持加减乘除、取模、幂运算，例如 '2 * 3 + 5'。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，仅支持数字和 + - * / % ( ) ** 运算符",
                    }
                },
                "required": ["expression"],
            },
        )
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
        expr = re.sub(r"\s+", "", expr)

        # 验证表达式只包含允许的字符
        if not re.match(r"^[0-9+\-*/().% ]+$", expr):
            raise ValueError("表达式包含不允许的字符")

        # 解析并计算表达式
        node = ast.parse(expr, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float:
        """
        递归计算AST节点的值

        Args:
            node: AST节点

        Returns:
            节点值
        """
        if isinstance(node, ast.Constant):  # Python 3.8+ 统一处理所有字面量
            return node.value
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

    def run(self, expression: str) -> dict[str, Any]:
        """
        执行数学计算

        Args:
            expression: 数学表达式

        Returns:
            计算结果字典
        """
        try:
            result = self.eval_expr(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}


class FileReadTool(Tool):
    """
    文件读取工具
    """

    def __init__(self, allowed_dir: str = "."):
        """
        初始化文件读取工具

        Args:
            allowed_dir: 允许读取的根目录（默认当前目录），路径越界将被拒绝
        """
        super().__init__(
            "file_read",
            "读取指定文本文件的内容，仅允许访问工作目录内的文件。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件相对路径",
                    }
                },
                "required": ["path"],
            },
        )
        self.allowed_dir = os.path.realpath(allowed_dir)

    def run(self, path: str) -> dict[str, Any]:
        """
        读取文件内容

        Args:
            path: 文件路径

        Returns:
            文件内容或错误信息
        """
        try:
            # 安全检查：解析真实路径后，确保在允许的工作目录内
            real_path = os.path.realpath(path)
            if not (
                real_path == self.allowed_dir or real_path.startswith(self.allowed_dir + os.sep)
            ):
                raise ValueError("不允许访问该路径（超出工作目录范围）")

            with open(real_path, encoding="utf-8") as f:
                content = f.read()
            return {"path": path, "content": content}
        except Exception as e:
            return {"path": path, "error": str(e)}


class DatabaseTool(Tool):
    """
    数据库访问工具（基于 sqlite3，真实可用）

    支持：
    - 自动区分查询（SELECT）与执行（INSERT/UPDATE/DELETE/CREATE...）
    - 可选只读模式（防止 LLM 误改数据）
    - 连接已有 .db 文件，或使用内存库
    """

    # SELECT 开头的关键词，用于判断是查询还是执行
    _READ_KEYWORDS = ("select", "with", "explain", "pragma")

    def __init__(self, db_path: str = ":memory:", readonly: bool = False):
        """
        Args:
            db_path: SQLite 数据库文件路径；":memory:" 为内存库（默认）
            readonly: 只读模式，True 时拒绝任何非查询语句
        """
        super().__init__(
            "database",
            "执行 SQLite 数据库操作。SELECT 返回查询结果，其他 SQL 返回影响行数。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL 语句（SQLite 方言）",
                    }
                },
                "required": ["query"],
            },
        )
        self.db_path = db_path
        self.readonly = readonly
        # 复用同一连接：内存库的表只在该连接生命周期内存在，
        # 若每次 run 都新建连接，建表后插入会找不到表。
        import sqlite3

        # check_same_thread=False 允许在 Orchestrator 的线程池中复用
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = __import__("threading").Lock()

    def run(self, query: str) -> dict[str, Any]:
        """
        执行 SQL 语句

        Args:
            query: SQL 语句

        Returns:
            查询：{query, rows, row_count, columns}
            执行：{query, rowcount}
            出错：{query, error}
        """
        try:
            normalized = query.strip().lower()
            is_query = any(normalized.startswith(k) for k in self._READ_KEYWORDS)

            # 加锁保证并发安全（同一连接不支持真正的并发写入）
            with self._lock:
                if is_query:
                    cur = self._conn.execute(query)
                    rows = [dict(r) for r in cur.fetchall()]
                    columns = [d[0] for d in cur.description] if cur.description else []
                    return {
                        "query": query,
                        "rows": rows,
                        "columns": columns,
                        "row_count": len(rows),
                    }
                else:
                    if self.readonly:
                        return {
                            "query": query,
                            "error": "只读模式下不允许执行非查询语句",
                        }
                    cur = self._conn.execute(query)
                    self._conn.commit()
                    return {
                        "query": query,
                        "rowcount": cur.rowcount,
                    }
        except Exception as e:
            logger.warning("DatabaseTool 执行失败: %s", e)
            return {"query": query, "error": str(e)}

    def close(self) -> None:
        """关闭数据库连接"""
        import contextlib

        with contextlib.suppress(Exception):
            self._conn.close()


class ToolRegistry:
    """
    工具注册表
    """

    def __init__(self):
        self.tools: dict[str, Tool] = {}
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

    def get(self, name: str) -> Tool | None:
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

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """
        导出全部工具的 OpenAI/GLM function-calling schema 列表

        用于在 LLM 调用时作为 tools 参数传入，让模型感知可用工具。

        Returns:
            工具 schema 列表
        """
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def list_tools(self) -> dict[str, str]:
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

    def execute_mcp_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        执行MCP工具

        Args:
            tool_name: MCP工具名称
            **kwargs: 执行参数

        Returns:
            执行结果
        """
        return self.mcp_registry.execute_tool(tool_name, **kwargs)
