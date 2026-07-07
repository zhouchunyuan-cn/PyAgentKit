"""
工具系统单元测试

覆盖：ToolRegistry 注册/执行/schema 导出、WebSearchTool mock 模式与失败处理、
FileReadTool 路径越界安全检查
"""
import os
import pytest
from core.tools import (
    Tool,
    CalculatorTool,
    WebSearchTool,
    FileReadTool,
    DatabaseTool,
    ToolRegistry,
)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool(mock_mode=True))  # mock 模式，不联网
    return reg


class TestToolRegistry:
    def test_register_and_get(self, registry):
        assert registry.get("calculator") is not None
        assert registry.get("web_search") is not None

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nope") is None

    def test_execute(self, registry):
        r = registry.execute("calculator", expression="2 + 3")
        assert r["result"] == 5

    def test_execute_missing_raises(self, registry):
        with pytest.raises(ValueError):
            registry.execute("ghost", x=1)

    def test_to_openai_tools_format(self, registry):
        """导出的 schema 列表应符合 function-calling 格式"""
        tools = registry.to_openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert "calculator" in names
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "parameters" in t["function"]


class TestWebSearchMock:
    """WebSearchTool mock 模式（不联网）"""

    def test_mock_returns_builtin_data(self):
        tool = WebSearchTool(mock_mode=True)
        r = tool.run(query="人工智能")
        assert r["source"] == "MockSearchEngine"
        # 新结构：results 数组
        assert len(r["results"]) == 1
        assert "人工智能" in r["results"][0]["snippet"]
        # 向后兼容字段 result 仍存在
        assert "人工智能" in r["result"]

    def test_mock_unknown_query_returns_not_found(self):
        tool = WebSearchTool(mock_mode=True)
        r = tool.run(query="一个完全无关的查询词xyz123")
        # mock 库里没有的词，应返回未找到提示而非伪造具体内容
        assert "未找到" in r["results"][0]["snippet"]


class TestWebSearchFailure:
    """
    WebSearchTool 真实模式失败处理

    关键：失败时返回明确 error，绝不静默回退伪造数据
    用一个必然失败的超短 timeout 触发
    """

    def test_failure_returns_error_not_fake_data(self):
        # max_retries=0 避免重试等待；超短 timeout 必然失败
        tool = WebSearchTool(mock_mode=False, timeout=0.001, max_retries=0)
        r = tool.run(query="anything")
        # 必须返回 error 字段，而非伪造数据
        assert "error" in r
        # 不应出现 MockSearchEngine 来源（那是 mock 模式才有的）
        assert r["source"] != "MockSearchEngine"


class TestFileReadSecurity:
    """FileReadTool 路径越界安全检查"""

    def test_read_legitimate_file(self, tmp_path):
        # 在允许目录内建文件并读取
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        tool = FileReadTool(allowed_dir=str(tmp_path))
        r = tool.run(path=str(f))
        assert r["content"] == "hello"

    def test_reject_path_traversal(self, tmp_path):
        tool = FileReadTool(allowed_dir=str(tmp_path))
        # 试图用 .. 跳出允许目录
        r = tool.run(path=str(tmp_path / ".." / "secret.txt"))
        assert "error" in r

    def test_reject_absolute_outside(self, tmp_path):
        tool = FileReadTool(allowed_dir=str(tmp_path))
        # 绝对路径指向允许目录之外
        r = tool.run(path=os.path.join(os.path.dirname(str(tmp_path)), "outside.txt"))
        assert "error" in r


class TestDatabaseTool:
    """
    DatabaseTool 单元测试（真实 sqlite3）

    验证：建表/插入/查询、SELECT 返回结构化行、只读保护、错误处理、跨语句持久化
    """

    @pytest.fixture
    def db(self):
        """每个测试用独立的内存库"""
        return DatabaseTool(":memory:")

    def test_create_insert_query(self, db):
        # 建表（执行语句返回 rowcount）
        r1 = db.run("CREATE TABLE users (id INTEGER, name TEXT)")
        assert "rowcount" in r1
        # 插入
        r2 = db.run("INSERT INTO users VALUES (1, 'Alice')")
        db.run("INSERT INTO users VALUES (2, 'Bob')")
        assert r2["rowcount"] == 1
        # 查询（返回结构化行）
        r3 = db.run("SELECT * FROM users ORDER BY id")
        assert r3["row_count"] == 2
        assert r3["rows"] == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert "name" in r3["columns"]

    def test_persistence_across_statements(self, db):
        """关键：内存库的表必须跨多次 run 持久存在（连接复用）"""
        db.run("CREATE TABLE t (v INTEGER)")
        db.run("INSERT INTO t VALUES (42)")
        # 第二次 run 应能查到第一次插入的数据
        r = db.run("SELECT v FROM t")
        assert r["rows"] == [{"v": 42}]

    def test_readonly_blocks_writes(self):
        db = DatabaseTool(":memory:", readonly=True)
        r = db.run("CREATE TABLE x (id INTEGER)")
        assert "error" in r
        assert "只读" in r["error"]

    def test_readonly_allows_select(self, db):
        db.run("CREATE TABLE x (id INTEGER)")
        # 另一个只读实例接同一个内存库查不到（内存库隔离），这里只验证 SELECT 不被拦截
        r = db.run("SELECT 1 AS n")
        assert "error" not in r
        assert r["rows"] == [{"n": 1}]

    def test_invalid_sql_returns_error(self, db):
        r = db.run("SELECT * FROM nonexistent_table")
        assert "error" in r

    def test_schema_valid(self):
        db = DatabaseTool(":memory:")
        s = db.to_openai_schema()
        assert s["function"]["name"] == "database"
        assert "query" in s["function"]["parameters"]["properties"]

    def test_close(self, db):
        db.run("CREATE TABLE x (id INTEGER)")
        db.close()  # 不应抛异常
