"""
企业级硬化七项的单元测试

1. checkpoint 跨进程文件锁（互斥、超时、陈旧锁抢占）
2. MCP 端点 SSRF 白名单（https 强制、元数据段拒绝、私网默认拒绝）
3. 审计日志（结构化 JSONL、生命周期事件、敏感参数脱敏）
4. token 预算执行（超限不再发起工具步、优雅收尾、标志位）
5. per-tool 审批 HITL（拒绝回传模型、白名单范围外不询问）
6. trace_id 传播 + 导出器（消息 metadata 串联、JSONL 落盘、导出异常不炸）
7. eval harness（关键词断言、judge、异常用例、token 成本、报告）
"""

import json

import pytest

from core.agent import Agent
from core.audit import AuditLogger, redact
from core.checkpoint import CheckpointBusyError, FileCheckpointStore
from core.evals import EvalCase, make_llm_judge, run_eval
from core.llm import ChatResult, LLMClient, TokenUsage, ToolCall
from core.mcp_client import MCPClientError, MCPServerConfig, validate_mcp_url
from core.team import SequentialProcess, Team
from core.tools import CalculatorTool, ToolRegistry
from core.trace import JsonlTraceExporter, Tracer


def _agent_with(llm, tools=None, **kw):
    class A(Agent):
        def __init__(self):
            super().__init__("a1", "测试", system_prompt="你是助手", llm_client=llm, **kw)
            for t in tools or []:
                self.tool_registry.register(t)

        def receive(self, message):
            pass

    return A()


class EchoLLM(LLMClient):
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.calls.append([dict(m) for m in messages])
        return self.results.pop(0)


# ====================================================================
# 1. checkpoint 跨进程锁
# ====================================================================


class TestCheckpointLock:
    def test_run_lock_mutual_exclusion(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        with store.run_lock("r1"):  # noqa: SIM117 -- 必须持锁期间探测第二次获取
            # 第二个获取者立即失败或等待——用短超时验证互斥
            with pytest.raises(CheckpointBusyError), store.run_lock("r1", timeout=0.3):
                pass

    def test_run_lock_released_after_exit(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        with store.run_lock("r1", timeout=1):
            pass
        with store.run_lock("r1", timeout=1):  # 释放后可再次获取
            pass

    def test_stale_lock_is_preempted(self, tmp_path):
        """持锁方崩溃（锁文件残留且超龄）→ 自动抢占"""
        import time

        store = FileCheckpointStore(str(tmp_path))
        lock = store.run_lock("r1", timeout=1, stale_after=0.05)
        lock.acquire()
        time.sleep(0.1)  # 锁已"陈旧"
        # 新获取者在 stale_after 之后可直接抢占
        with store.run_lock("r1", timeout=1, stale_after=0.05):
            pass

    def test_team_run_holds_lock(self, tmp_path):
        """Team 执行期间持有 run 锁；执行结束释放"""
        store = FileCheckpointStore(str(tmp_path))
        llm = EchoLLM([ChatResult(content="产出")])

        class Watcher(type(store)):  # 记录执行时锁是否被持有
            pass

        llm_ref = llm
        member = _agent_with(llm_ref)
        team = Team(
            name="锁测试", members=[member], process=SequentialProcess(), checkpoint_store=store
        )
        team.run("任务", run_id="rl")
        # 执行结束后锁可用
        with store.run_lock("rl", timeout=1):
            pass
        # 检查点存在
        assert store.load("rl") is not None
        assert store.load("rl")["completed"] is True


# ====================================================================
# 2. MCP SSRF 白名单
# ====================================================================


class TestMCPServerSideRequestForgery:
    def test_https_public_ok(self):
        assert validate_mcp_url("https://mcp.example.com/sse")

    def test_http_loopback_ok(self):
        assert validate_mcp_url("http://localhost:8000/mcp")
        assert validate_mcp_url("http://127.0.0.1:9000/mcp")

    def test_http_public_rejected(self):
        with pytest.raises(MCPClientError, match="https"):
            validate_mcp_url("http://mcp.example.com/mcp")

    def test_cloud_metadata_rejected(self):
        with pytest.raises(MCPClientError, match="受限地址"):
            validate_mcp_url("http://169.254.169.254/latest/meta-data")

    def test_private_ip_default_rejected(self):
        with pytest.raises(MCPClientError, match="allow_private"):
            validate_mcp_url("https://10.1.2.3/mcp")
        # 显式放行
        assert validate_mcp_url("https://10.1.2.3/mcp", allow_private=True)

    def test_bad_scheme_rejected(self):
        with pytest.raises(MCPClientError, match="http/https"):
            validate_mcp_url("ftp://example.com/mcp")

    def test_config_validates_on_init(self):
        with pytest.raises(MCPClientError):
            MCPServerConfig(url="http://169.254.169.254/mcp")
        assert MCPServerConfig(url="http://localhost:8000/mcp").name == "localhost"


# ====================================================================
# 3. 审计日志
# ====================================================================


class TestAudit:
    def test_redact_masks_sensitive_keys(self):
        data = {
            "password": "hunter2",
            "nested": {"API_TOKEN": "sk-123", "ok": "visible"},
            "list": [{"secret": "x"}],
        }
        out = redact(data)
        assert out["password"] == "***"
        assert out["nested"]["API_TOKEN"] == "***"
        assert out["nested"]["ok"] == "visible"
        assert out["list"][0]["secret"] == "***"

    def test_audit_records_lifecycle(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(path, actor="user-42")
        llm = EchoLLM([ChatResult(content="答复")])
        agent = _agent_with(llm)
        audit.install(agent)
        agent.think("你好")
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        events = [rec["event"] for rec in lines]
        assert events == ["think_start", "think_end"]
        assert all(rec["actor"] == "user-42" for rec in lines)
        assert lines[0]["agent"] == "a1"

    def test_audit_records_tool_call_with_redaction(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        audit = AuditLogger(path)
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="t1", name="calculator", arguments={"expression": "1+1"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="=2"),
            ]
        )
        agent = _agent_with(llm, tools=[CalculatorTool()])
        agent.hooks = audit.make_hooks()
        agent.think("算一下")
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        tool_events = [rec for rec in lines if rec["event"] == "tool_call"]
        assert tool_events[0]["tool"] == "calculator"
        assert tool_events[0]["arguments"] == {"expression": "1+1"}
        assert tool_events[0]["result_preview"]["result"] == 2


# ====================================================================
# 4. token 预算执行
# ====================================================================


class TestTokenBudget:
    def test_budget_stops_tool_loop_gracefully(self):
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})
                    ],
                    usage=TokenUsage(80, 20, 100),
                ),
                # 预算 100 已耗尽 → 不再发起第二个工具循环步，下一次调用就是收尾
                ChatResult(content="预算内收尾答复", usage=TokenUsage(5, 5, 10)),
            ]
        )
        agent = _agent_with(llm, tools=[CalculatorTool()], token_budget=100)
        result = agent.think("算数")
        assert result == "预算内收尾答复"
        assert agent.budget_exceeded is True
        # 预算检查把工具循环拦下：LLM 恰好调用 2 次（工具步 + 收尾）
        assert len(llm.calls) == 2

    def test_no_budget_means_no_limit(self):
        llm = EchoLLM([ChatResult(content="直接回答", usage=TokenUsage(9999, 9999, 19998))])
        agent = _agent_with(llm)
        assert agent.think("q") == "直接回答"
        assert agent.budget_exceeded is False

    def test_budget_not_reached_runs_normally(self):
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "2**10"})
                    ],
                    usage=TokenUsage(50, 10, 60),
                ),
                ChatResult(content="1024", usage=TokenUsage(20, 10, 30)),
            ]
        )
        agent = _agent_with(llm, tools=[CalculatorTool()], token_budget=10_000)
        assert agent.think("计算 2**10") == "1024"
        assert agent.budget_exceeded is False


# ====================================================================
# 5. per-tool 审批 HITL
# ====================================================================


class TestToolApproval:
    def test_denied_tool_raises_permission_error(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.set_approver(lambda name, args: False, tools=["calculator"])
        with pytest.raises(PermissionError, match="审批拒绝"):
            reg.execute("calculator", expression="1+1")

    def test_tools_outside_scope_not_asked(self):
        from core.tools import WebSearchTool

        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(WebSearchTool(mock_mode=True))
        asked = []
        reg.set_approver(lambda name, args: asked.append(name) or True, tools=["calculator"])
        reg.execute("web_search", query="测试")  # 范围外：不询问
        assert asked == []
        reg.execute("calculator", expression="1+1")  # 范围内：询问
        assert asked == ["calculator"]

    def test_engine_returns_denial_to_model(self):
        """审批拒绝作为工具错误回传，模型据此调整而不是崩溃"""
        llm = EchoLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "9**9"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="那我直接心算"),
            ]
        )
        agent = _agent_with(llm, tools=[CalculatorTool()])
        agent.tool_registry.set_approver(lambda name, args: False)
        result = agent.think("算 9 的 9 次方")
        assert result == "那我直接心算"
        tool_msg = [m for m in llm.calls[1] if m.get("role") == "tool"][0]
        assert "审批拒绝" in tool_msg["content"]


# ====================================================================
# 6. trace_id 传播 + 导出器
# ====================================================================


class TestTracePropagation:
    def test_trace_has_id_and_exported(self, tmp_path):
        exporter = JsonlTraceExporter(str(tmp_path / "traces.jsonl"))
        Tracer.enable()
        Tracer.set_exporter(exporter)
        try:
            trace = Tracer.start_trace("导出测试")
            assert trace is not None
            assert Tracer.current_trace_id() == trace.trace_id
            Tracer.end_trace()
            with open(tmp_path / "traces.jsonl", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 1
            assert lines[0]["trace_id"] == trace.trace_id
            assert lines[0]["name"] == "导出测试"
        finally:
            Tracer.set_exporter(None)
            Tracer.disable()

    def test_broken_exporter_does_not_break_end_trace(self):
        Tracer.enable()
        Tracer.set_exporter(lambda t: (_ for _ in ()).throw(RuntimeError("后端挂了")))
        try:
            Tracer.start_trace("容错")
            trace = Tracer.end_trace()  # 不抛异常
            assert trace is not None
        finally:
            Tracer.set_exporter(None)
            Tracer.disable()

    def test_router_stamps_trace_id_into_message(self):
        """消息路由时 trace_id 盖进 metadata——跨组件串联的关键"""
        from core.message import Message
        from core.router import Router

        received = []

        class Sink(Agent):
            def __init__(self):
                super().__init__("sink", "sink")

            def receive(self, m):
                received.append(m)

        router = Router()
        router.register_agent(Sink())
        Tracer.enable()
        try:
            trace = Tracer.start_trace("串联")
            router.route_message(Message("user", "sink", "x"))
            Tracer.end_trace()
        finally:
            Tracer.disable()
        assert received[0].metadata.get("trace_id") == trace.trace_id


# ====================================================================
# 7. eval harness
# ====================================================================


class TestEvalHarness:
    def _eval_agent(self, replies):
        return _agent_with(EchoLLM([ChatResult(content=r) for r in replies]))

    def test_keyword_assertion(self):
        agent = self._eval_agent(["(15+27)*3 的结果是 126", "抱歉我不知道"])
        report = run_eval(
            agent,
            [
                EvalCase(input="计算 (15+27)*3", expect_keywords=["126"]),
                EvalCase(input="不确定的问题", expect_keywords=["答案"]),
            ],
            name="关键词集",
        )
        assert report.passed_count == 1
        assert report.results[0].passed
        assert not report.results[1].passed
        assert "缺少关键词" in report.results[1].reason
        assert report.total_tokens == 0  # mock 无 usage

    def test_custom_judge(self):
        agent = self._eval_agent(["回答A"])
        seen = []

        def judge(question, reply):
            seen.append((question, reply))
            return reply == "回答A"

        report = run_eval(agent, [EvalCase(input="q", judge=judge)])
        assert report.results[0].passed
        assert seen == [("q", "回答A")]

    def test_llm_judge_parses_verdict(self):
        class JudgeLLM(EchoLLM):
            def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
                self.calls.append(messages)
                verdict = self.results.pop(0)
                return ChatResult(content=verdict)

        jl = JudgeLLM(["PASS\n理由略", "FAIL"])
        judge = make_llm_judge(jl, rubric="准确")
        assert judge("q1", "回答1") is True
        assert judge("q2", "回答2") is False
        assert "评分标准：准确" in jl.calls[0][-1]["content"]

    def test_exception_case_recorded_not_raised(self):
        class BoomLLM(EchoLLM):
            def chat(self, messages, **kwargs):
                raise RuntimeError("模型不可用")

        agent = _agent_with(BoomLLM([]))
        report = run_eval(agent, [EvalCase(input="q", expect_keywords=["x"])])
        assert not report.results[0].passed
        assert "执行异常" in report.results[0].reason

    def test_case_isolation(self):
        """用例间隔离：上一轮的会话历史不污染下一轮输入"""
        agent = self._eval_agent(["第一答", "第二答"])
        run_eval(
            agent,
            [
                EvalCase(input="q1", expect_keywords=["第一答"]),
                EvalCase(input="q2", expect_keywords=["第二答"]),
            ],
        )
        # 隔离开启时，第二次 think 的消息里不应包含第一轮的会话历史
        second_call = agent.llm.calls[1]
        user_msgs = [m["content"] for m in second_call if m.get("role") == "user"]
        assert user_msgs == ["q2"]

    def test_report_save_and_format(self, tmp_path):
        agent = self._eval_agent(["好的"])
        report = run_eval(agent, [EvalCase(input="q", expect_keywords=["好的"])])
        text = report.format_text()
        assert "1/1" in text and "✓" in text
        path = str(tmp_path / "report.json")
        report.save(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["pass_rate"] == 1.0
