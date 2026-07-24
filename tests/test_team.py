"""
Team 模块单元测试

用 MockLLM（确定性，不联网）验证：
- SharedContext 存取与产出拼接
- SequentialProcess 顺序接力、产出累计、最终取末位成员
- HierarchicalProcess Leader 规划 → 能力匹配 → 汇总，含 JSON 解析失败回退
- Team.run 串联流程、summary、边界错误
"""

import json

import pytest

from core.agent import Agent
from core.llm import ChatResult, LLMClient
from core.message import Message
from core.team import (
    HierarchicalProcess,
    SequentialProcess,
    SharedContext,
    Task,
    Team,
)

# --------------------------------------------------------------------
# 测试夹具：MockLLM + 可配置的 MockAgent
# --------------------------------------------------------------------


class MockLLM(LLMClient):
    """确定性 Mock LLM，按预设规则产出，不联网"""

    def __init__(self, responses=None, plan=None):
        """
        Args:
            responses: 默认回复（每次 think 都返回它）
            plan: HierarchicalProcess 用，Leader 规划时返回这个 JSON 字符串
        """
        self.responses = responses or "mock回复"
        self.plan = plan
        self.calls = []  # 记录所有 chat 调用的 messages

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append(messages)
        # 如果是规划请求（prompt 里含"项目经理"），返回 plan
        last_content = ""
        if messages:
            last_content = messages[-1].get("content", "")
        if self.plan and "项目经理" in last_content:
            return ChatResult(content=self.plan, finish_reason="stop")
        if self.plan and "综合所有产出" in last_content:
            return ChatResult(content="最终汇总结果", finish_reason="stop")
        return ChatResult(content=self.responses, finish_reason="stop")


def make_agent(agent_id, name, capabilities, llm, tools=None):
    """快速构造一个测试用 Agent"""

    class _A(Agent):
        def __init__(self):
            super().__init__(
                agent_id=agent_id,
                name=name,
                llm_client=llm,
                capabilities=capabilities,
            )

        def receive(self, message: Message):
            pass  # Team 不走 receive，走 think

    a = _A()
    return a


@pytest.fixture
def llm():
    return MockLLM(responses="OK")


# --------------------------------------------------------------------
# SharedContext
# --------------------------------------------------------------------


class TestSharedContext:
    def test_add_and_get_output(self):
        ctx = SharedContext(task="t")
        ctx.add_output("a1", "产出1")
        assert ctx.get_output("a1") == "产出1"
        assert ctx.get_output("missing") is None

    def test_get_all_outputs_concatenates(self):
        ctx = SharedContext(task="t")
        ctx.add_output("a1", "产出A")
        ctx.add_output("a2", "产出B")
        all_out = ctx.get_all_outputs()
        assert "产出A" in all_out
        assert "产出B" in all_out
        assert "a1" in all_out and "a2" in all_out

    def test_get_all_outputs_empty(self):
        assert SharedContext().get_all_outputs() == ""

    def test_intermediate(self):
        ctx = SharedContext()
        ctx.add_intermediate("plan", [{"subtask": "x"}])
        assert ctx.get_intermediate("plan") == [{"subtask": "x"}]
        assert ctx.get_intermediate("missing", "default") == "default"

    def test_history_recorded(self):
        ctx = SharedContext()
        ctx.add_output("a1", "x")
        ctx.record_step("custom_step")
        assert len(ctx.history) >= 2

    def test_summary(self):
        ctx = SharedContext(task="t")
        ctx.add_output("a1", "x")
        s = ctx.summary()
        assert s["task"] == "t"
        assert s["output_count"] == 1
        assert "a1" in s["output_agents"]


# --------------------------------------------------------------------
# SequentialProcess
# --------------------------------------------------------------------


class TestSequentialProcess:
    def test_members_execute_in_order(self, llm):
        # 用不同的 MockLLM 让每个成员产出不同内容，验证顺序与累计
        members = [
            make_agent("a1", "一", ["x"], MockLLM(responses="第一步产出")),
            make_agent("a2", "二", ["x"], MockLLM(responses="第二步产出")),
            make_agent("a3", "三", ["x"], MockLLM(responses="最终产出")),
        ]
        task = Task(description="任务", context=SharedContext("任务"))
        result = SequentialProcess().execute(task, members)

        # 最终结果取最后成员产出
        assert result == "最终产出"
        # 每个成员都有产出记录
        assert len(task.context.outputs) == 3

    def test_prior_outputs_passed_to_next_member(self):
        """第二个成员的输入应包含第一个成员的产出"""
        llm1 = MockLLM(responses="A产出")
        # 第二个成员的 LLM 记录调用，便于断言
        llm2 = MockLLM(responses="B产出")
        members = [
            make_agent("a1", "一", ["x"], llm1),
            make_agent("a2", "二", ["x"], llm2),
        ]
        task = Task(description="任务", context=SharedContext("任务"))
        SequentialProcess().execute(task, members)

        # a2 的 think 输入应包含 a1 的产出
        a2_input = llm2.calls[-1][-1]["content"]
        assert "A产出" in a2_input

    def test_first_member_gets_bare_task(self):
        llm1 = MockLLM(responses="首产出")
        member = make_agent("a1", "一", ["x"], llm1)
        task = Task(description="纯任务", context=SharedContext("纯任务"))
        SequentialProcess().execute(task, [member])
        # 第一个成员不应看到"此前产出"字样
        assert "此前产出" not in llm1.calls[-1][-1]["content"]

    def test_empty_members_raises(self):
        with pytest.raises(ValueError, match="至少一个成员"):
            SequentialProcess().execute(Task("t", SharedContext()), [])

    def test_member_without_llm_raises(self):
        # 构造一个无 LLM 的 agent
        class _NoLLM(Agent):
            def __init__(self):
                super().__init__("a", "a", llm_client=None)

            def receive(self, m):
                pass

        with pytest.raises(RuntimeError, match="未配置 LLM"):
            SequentialProcess().execute(Task("t", SharedContext()), [_NoLLM()])


# --------------------------------------------------------------------
# HierarchicalProcess
# --------------------------------------------------------------------


class TestHierarchicalProcess:
    def test_leader_plan_executed_by_capability(self):
        """
        Leader 输出 JSON 规划，按能力匹配成员执行，最后汇总。
        """
        # Leader 规划：两个子任务，分别需要 search 和 write 能力
        plan = json.dumps(
            [
                {"subtask": "搜索资料", "capability": "search"},
                {"subtask": "撰写报告", "capability": "write"},
            ]
        )
        leader_llm = MockLLM(plan=plan)
        leader = make_agent("leader", "经理", ["plan"], leader_llm)

        searcher = make_agent("s", "搜索员", ["search"], MockLLM(responses="搜索结果"))
        writer = make_agent("w", "作者", ["write"], MockLLM(responses="报告内容"))

        task = Task(description="做研究", context=SharedContext("做研究"))
        result = HierarchicalProcess().execute(task, [searcher, writer], leader)

        # 最终应是 Leader 汇总结果
        assert result == "最终汇总结果"
        # 两个成员都应被调用（按能力匹配）
        assert "s" in task.context.outputs
        assert "w" in task.context.outputs

    def test_no_leader_raises(self):
        with pytest.raises(ValueError, match="leader"):
            HierarchicalProcess().execute(
                Task("t", SharedContext()),
                [make_agent("a", "a", ["x"], MockLLM())],
                leader=None,
            )

    def test_json_parse_failure_falls_back(self):
        """
        Leader 输出无法解析的 JSON 时，应回退为全员顺序处理而非崩溃
        """
        # Leader 返回乱码（非 JSON）
        leader_llm = MockLLM(plan="这不是合法JSON")
        leader = make_agent("leader", "经理", ["plan"], leader_llm)
        member = make_agent("m", "成员", ["work"], MockLLM(responses="成员产出"))

        task = Task(description="任务", context=SharedContext("任务"))
        result = HierarchicalProcess().execute(task, [member], leader)

        # 回退后至少有产出，且 Leader 仍会汇总
        assert "最终汇总结果" in result or len(task.context.outputs) >= 1

    def test_no_matching_capability_skipped(self):
        """
        子任务要求的能力无人具备时，应跳过该子任务而非崩溃，
        最终 Leader 仍能基于已有产出汇总。
        """
        plan = json.dumps(
            [
                {"subtask": "子任务A", "capability": "missing_cap"},  # 无人具备
                {"subtask": "子任务B", "capability": "real_cap"},
            ]
        )
        leader_llm = MockLLM(plan=plan)
        leader = make_agent("leader", "经理", ["plan"], leader_llm)
        member = make_agent("m", "成员", ["real_cap"], MockLLM(responses="实际产出"))

        task = Task(description="任务", context=SharedContext("任务"))
        HierarchicalProcess().execute(task, [member], leader)
        # real_cap 的成员被调用，missing_cap 被跳过
        assert "m" in task.context.outputs


# --------------------------------------------------------------------
# Team
# --------------------------------------------------------------------


class TestTeam:
    def test_sequential_team_run(self, llm):
        members = [
            make_agent("a1", "一", ["x"], MockLLM(responses="R1")),
            make_agent("a2", "二", ["x"], MockLLM(responses="最终")),
        ]
        team = Team(name="T", members=members, process=SequentialProcess())
        result = team.run("任务")
        assert result == "最终"
        assert len(team.task_history) == 1

    def test_default_process_is_sequential(self):
        team = Team(name="T")
        assert isinstance(team.process, SequentialProcess)

    def test_add_remove_member(self):
        team = Team(name="T")
        a = make_agent("a", "a", ["x"], MockLLM())
        team.add_member(a)
        assert len(team.members) == 1
        assert team.remove_member("a") is True
        assert len(team.members) == 0
        assert team.remove_member("ghost") is False

    def test_run_empty_team_raises(self):
        team = Team(name="T")
        with pytest.raises(RuntimeError, match="没有成员"):
            team.run("任务")

    def test_hierarchical_without_leader_raises(self):
        member = make_agent("a", "a", ["x"], MockLLM())
        team = Team(name="T", members=[member], process=HierarchicalProcess())
        with pytest.raises(RuntimeError, match="leader"):
            team.run("任务")

    def test_hierarchical_team_run(self):
        plan = json.dumps([{"subtask": "干活", "capability": "work"}])
        leader = make_agent("L", "经理", ["plan"], MockLLM(plan=plan))
        worker = make_agent("w", "工人", ["work"], MockLLM(responses="工作成果"))
        team = Team(name="T", members=[worker], process=HierarchicalProcess(), leader=leader)
        result = team.run("任务")
        assert result == "最终汇总结果"

    def test_summary(self):
        leader = make_agent("L", "经理", ["plan"], MockLLM())
        m = make_agent("m", "成员", ["work"], MockLLM())
        team = Team(name="测试团队", members=[m], process=SequentialProcess(), leader=leader)
        team.run("任务1")
        s = team.summary()
        assert s["name"] == "测试团队"
        assert s["member_count"] == 1
        assert s["has_leader"] is True
        assert s["process"] == "sequential"
        assert s["task_count"] == 1
