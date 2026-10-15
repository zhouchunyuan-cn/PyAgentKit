"""
P2 检查点（断点续跑）单元测试

覆盖：
- FileCheckpointStore：原子写/读/删、损坏容错、任务不匹配丢弃、run_id 安全化
- SharedContext 序列化往返
- Sequential 断点续跑：step_limit 暂停 → 续跑，已执行成员不再重跑（LLM 不重复付费）
- 崩溃恢复：run(run_id=...) 从检查点继续
- 幂等：已完成的 run 重跑直接返回缓存结果
- Hierarchical 断点：规划复用（Leader 不重新规划）、已完成子任务跳过
- 暂停条件校验（step_limit 需要 run_id + store）
"""

import json

import pytest

from core.agent import Agent
from core.checkpoint import FileCheckpointStore, RunOutcome
from core.llm import ChatResult, LLMClient
from core.team import HierarchicalProcess, SequentialProcess, SharedContext, Team


class CountingLLM(LLMClient):
    """按序返回预置结果并统计调用次数（区分规划/执行/汇总）"""

    def __init__(self, results):
        self.results = list(results)
        self.chat_calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        self.chat_calls += 1
        return ChatResult(content=self.results.pop(0))


class PlanLeaderLLM(LLMClient):
    """规划返回固定 JSON；汇总返回固定文本；统计规划调用次数"""

    def __init__(self, plan):
        self.plan = plan
        self.plan_calls = 0
        self.synthesis_calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        last = messages[-1]["content"]
        if "项目经理" in last:
            self.plan_calls += 1
            return ChatResult(content=self.plan)
        if "综合所有产出" in last:
            self.synthesis_calls += 1
            return ChatResult(content="汇总结果")
        return ChatResult(content="leader-其它")


def _member(agent_id, llm):
    class M(Agent):
        def __init__(self):
            super().__init__(agent_id, agent_id, llm_client=llm)

        def receive(self, message):
            pass

    return M()


def _leader(plan):
    class L(Agent):
        def __init__(self):
            super().__init__(
                "leader", "经理", llm_client=PlanLeaderLLM(plan), capabilities=["plan"]
            )

        def receive(self, message):
            pass

    return L()


# ====================================================================
# FileCheckpointStore
# ====================================================================


class TestFileCheckpointStore:
    def test_save_load_roundtrip(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        state = {"task": "t", "process": "sequential", "next_index": 2, "context": {"a": 1}}
        store.save("run1", state)
        assert store.load("run1") == state

    def test_load_missing_returns_none(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        assert store.load("nope") is None

    def test_corrupt_file_returns_none(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        (tmp_path / "bad.json").write_text("{ 半截", encoding="utf-8")
        assert store.load("bad") is None

    def test_non_serializable_state_keeps_old_file(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        store.save("run1", {"ok": 1})
        with pytest.raises((TypeError, ValueError)):
            store.save("run1", {"bad": object()})
        # 旧检查点未被破坏
        assert store.load("run1") == {"ok": 1}

    def test_delete(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        store.save("run1", {"a": 1})
        assert store.delete("run1") is True
        assert store.load("run1") is None
        assert store.delete("run1") is False

    def test_unsafe_run_id_sanitized(self, tmp_path):
        """含路径分隔符的 run_id 被哈希替代，不越出检查点目录"""
        store = FileCheckpointStore(str(tmp_path))
        evil = "../../escape"
        store.save(evil, {"a": 1})
        assert store.load(evil) == {"a": 1}
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert ".." not in files[0].name


# ====================================================================
# SharedContext 序列化
# ====================================================================


class TestSharedContextSerialization:
    def test_roundtrip(self):
        ctx = SharedContext(task="原始任务")
        ctx.add_output("a", "产出1")
        ctx.add_output("b", "产出2")
        ctx.add_output("a", "产出3")
        ctx.add_intermediate("plan", [{"subtask": "s", "capability": "c"}])
        ctx.record_step("x", 1)

        restored = SharedContext.from_dict(ctx.to_dict(), task="原始任务")
        assert restored.get_output("a") == "产出3"
        assert restored.get_output("b") == "产出2"
        assert restored.get_all_outputs().count("产出1") == 1  # 日志完整保留
        assert restored.get_intermediate("plan") == [{"subtask": "s", "capability": "c"}]
        assert restored.summary()["output_count"] == 3
        # 恢复后可继续写入
        restored.add_output("c", "续跑产出")
        assert "续跑产出" in restored.get_all_outputs()


# ====================================================================
# Sequential 断点续跑
# ====================================================================


class TestSequentialCheckpoint:
    def _team(self, store, outputs):
        llms = [CountingLLM([out]) for out in outputs]
        members = [_member(f"m{i}", llm) for i, llm in enumerate(llms)]
        return Team(
            name="流水线", members=members, process=SequentialProcess(), checkpoint_store=store
        ), llms

    def test_pause_and_resume_without_rerun(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        team, llms = self._team(store, ["产出A", "产出B", "产出C"])

        # 第一次：只跑 1 步即暂停
        o1 = team.run_resumable("写报告", run_id="r1", step_limit=1)
        assert isinstance(o1, RunOutcome)
        assert o1.completed is False
        assert o1.result is None
        assert o1.steps_done == 1
        assert o1.run_id == "r1"
        assert llms[0].chat_calls == 1
        assert llms[1].chat_calls == 0

        # 第二次：跑到结束；成员 0 不再重跑（LLM 调用数不增加）
        o2 = team.run_resumable("写报告", run_id="r1")
        assert o2.completed is True
        assert o2.result == "产出C"
        assert llms[0].chat_calls == 1  # 关键：断点前的步骤没有重复付费
        assert llms[1].chat_calls == 1
        assert llms[2].chat_calls == 1

    def test_crash_recovery_via_run(self, tmp_path):
        """run(run_id=...) 在崩溃重跑场景下自动从断点继续"""
        store = FileCheckpointStore(str(tmp_path))
        team, llms = self._team(store, ["A", "B"])
        team.run_resumable("任务X", run_id="rx", step_limit=1)

        # 模拟"新进程"：用新的 Team 实例（同 store）继续
        team2, llms2 = self._team(store, ["A", "B"])
        result = team2.run("任务X", run_id="rx")
        assert result == "B"
        assert llms2[0].chat_calls == 0  # 第一个成员直接跳过

    def test_completed_run_is_idempotent(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        team, llms = self._team(store, ["A", "B"])
        assert team.run("任务Y", run_id="ry") == "B"
        # 同 run_id 重跑：直接返回缓存结果，不再调 LLM
        team2, llms2 = self._team(store, ["A", "B"])
        assert team2.run("任务Y", run_id="ry") == "B"
        assert all(llm.chat_calls == 0 for llm in llms2)

    def test_task_mismatch_ignores_checkpoint(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        team, llms = self._team(store, ["A", "B"])
        team.run_resumable("任务一", run_id="rm", step_limit=1)

        team2, llms2 = self._team(store, ["A", "B"])
        team2.run("任务二", run_id="rm")  # 任务不同：忽略旧检查点重新开始
        assert llms2[0].chat_calls == 1  # 从头跑
        assert llms2[1].chat_calls == 1

    def test_step_limit_requires_run_id_and_store(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        team, _ = self._team(store, ["A"])
        with pytest.raises(ValueError, match="step_limit"):
            team.run_resumable("任务", run_id=None, step_limit=1)

        team_no_store = Team(name="无存储", members=team.members, process=SequentialProcess())
        with pytest.raises(ValueError, match="step_limit"):
            team_no_store.run_resumable("任务", run_id="r", step_limit=1)

    def test_without_store_behaves_unchanged(self):
        llm = CountingLLM(["A"])
        team = Team(name="普通", members=[_member("m0", llm)], process=SequentialProcess())
        assert team.run("任务") == "A"


# ====================================================================
# Hierarchical 断点续跑
# ====================================================================


class TestHierarchicalCheckpoint:
    def _team(self, store, plan, member_outputs):
        plan_json = json.dumps(plan, ensure_ascii=False)
        members = []
        llms = []
        for i, (cap, outs) in enumerate(member_outputs):
            llm = CountingLLM(list(outs))
            llms.append(llm)
            members.append(_member_with_cap(f"w{i}", cap, llm))
        return (
            Team(
                name="层级组",
                members=members,
                process=HierarchicalProcess(),
                leader=_leader(plan_json),
                checkpoint_store=store,
            ),
            llms,
        )

    def test_resume_reuses_plan_and_skips_done(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        plan = {
            "subtasks": [
                {"subtask": "检索", "capability": "search"},
                {"subtask": "写作", "capability": "write"},
            ]
        }
        team, llms = self._team(store, plan, [("search", ["研究产出"]), ("write", ["写作产出"])])

        # 第一次：完成 1 个子任务后暂停
        o1 = team.run_resumable("做报告", run_id="h1", step_limit=1)
        assert o1.completed is False
        assert o1.steps_done == 1

        # 第二次：续跑 —— Leader 不重新规划（plan_calls 仍为 1），已完成的子任务跳过
        o2 = team.run_resumable("做报告", run_id="h1")
        assert o2.completed is True
        assert o2.result == "汇总结果"
        leader_llm = team.leader.llm
        assert leader_llm.plan_calls == 1  # 规划被复用，没有二次规划
        assert leader_llm.synthesis_calls == 1
        # search 成员只在第一次被调用过一次；write 只在续跑时调用
        assert llms[0].chat_calls == 1
        assert llms[1].chat_calls == 1

    def test_completed_returns_cached_result(self, tmp_path):
        store = FileCheckpointStore(str(tmp_path))
        plan = {"subtasks": [{"subtask": "干活", "capability": "work"}]}
        team, _ = self._team(store, plan, [("work", ["产出"])])
        assert team.run("任务", run_id="h2") == "汇总结果"

        # 重跑：全部命中缓存，无任何 LLM 调用
        team2, llms2 = self._team(store, plan, [("work", ["产出"])])
        assert team2.run("任务", run_id="h2") == "汇总结果"
        assert llms2[0].chat_calls == 0
        assert team2.leader.llm.plan_calls == 0


def _member_with_cap(agent_id, capability, llm):
    class M(Agent):
        def __init__(self):
            super().__init__(agent_id, agent_id, llm_client=llm, capabilities=[capability])

        def receive(self, message):
            pass

    return M()
