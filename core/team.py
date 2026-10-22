"""
团队协作（Team）

在多智能体（multi-agent）消息路由之上，提供"有分工、有流程、有共同目标"的
团队协作模式。一个 Team 由若干成员（Agent）和一个流程（Process）组成，
通过共享上下文（SharedContext）传递中间产物。

与现有 multi-agent 机制的区别：
- Router 是"消息总线"（谁发给谁），Team 是"流程编排"（按什么顺序、谁做什么）
- Team 成员间通过 SharedContext 共享产出，而非各自独立的 memory
- 流程由 Process 决定，与 Agent 解耦——换个 Process 就换种协作方式，无需改 Agent

两种内置流程：
- SequentialProcess：成员按声明顺序接力（确定性流水线）
- HierarchicalProcess：Leader 用 LLM 自动拆解任务并按能力分配成员

用法：
    team = Team(name="内容组", members=[研究, 写作, 审核],
                process=SequentialProcess())
    result = team.run("写一篇关于量子计算的科普文章")
"""

import contextlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .trace import Tracer

logger = logging.getLogger(__name__)


class SharedContext:
    """
    团队共享上下文（黑板模式）

    一个 Task 执行期间，所有成员通过这里共享中间产物与各自的产出，
    替代"各自独立 memory"。成员 think 前，Process 把这里的相关内容
    作为上下文注入给它。
    """

    def __init__(self, task: str = ""):
        self.task: str = task
        # 各成员的最新产出：agent_id -> output（get_output 语义：取最新）
        self.outputs: dict[str, str] = {}
        # 完整产出日志（按时间顺序）：同一成员多次执行时每次都保留，
        # 供后续成员与 Leader 汇总看到全部中间产出
        self.output_log: list[dict[str, str]] = []
        # 任意中间产物（子任务、规划、统计等）
        self.intermediate: dict[str, Any] = {}
        # 执行步骤记录，便于追溯
        self.history: list[dict[str, Any]] = []
        # 并行模式下多个成员线程同时写黑板，加锁保证一致
        self._lock = threading.Lock()

    def add_output(self, agent_id: str, output: str) -> None:
        """记录某成员的产出（同一成员多次产出会全部保留在日志中；线程安全）"""
        with self._lock:
            self.outputs[agent_id] = output
            self.output_log.append({"agent": agent_id, "output": output})
            self.history.append({"agent": agent_id, "action": "output", "detail": output[:200]})

    def get_output(self, agent_id: str) -> str | None:
        """获取某成员的最新产出"""
        return self.outputs.get(agent_id)

    def get_all_outputs(self) -> str:
        """
        拼接所有成员的产出（按产出顺序），作为下一个成员的上下文。

        同一成员的多次产出（Hierarchical 中多个子任务命中同一成员）都会
        出现，格式清晰标注每个产出来自谁，便于 LLM 理解。
        """
        if not self.output_log:
            return ""
        lines = []
        for entry in self.output_log:
            lines.append(f"【{entry['agent']} 的产出】\n{entry['output']}")
        return "\n\n".join(lines)

    def add_intermediate(self, key: str, value: Any) -> None:
        """存入中间产物（如 Leader 的任务规划）"""
        self.intermediate[key] = value
        self.history.append({"action": "intermediate", "key": key})

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        """获取中间产物"""
        return self.intermediate.get(key, default)

    def record_step(self, step: str, detail: Any = None) -> None:
        """记录一个执行步骤（便于调试/展示）"""
        self.history.append({"step": step, "detail": detail})

    def summary(self) -> dict[str, Any]:
        """上下文摘要"""
        return {
            "task": self.task,
            "output_count": len(self.output_log),
            "output_agents": list(self.outputs.keys()),
            "intermediate_keys": list(self.intermediate.keys()),
            "history_steps": len(self.history),
        }

    # ------------------------------------------------------------------
    # 序列化（检查点用）
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典（检查点落盘）"""
        return {
            "task": self.task,
            "outputs": dict(self.outputs),
            "output_log": [dict(e) for e in self.output_log],
            "intermediate": self.intermediate,
            "history": [h if not isinstance(h, dict) else dict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], task: str | None = None) -> "SharedContext":
        """从检查点恢复（to_dict 的逆操作）"""
        ctx = cls(task=task if task is not None else data.get("task", ""))
        ctx.outputs = {k: v for k, v in (data.get("outputs") or {}).items()}
        ctx.output_log = [dict(e) for e in (data.get("output_log") or [])]
        ctx.intermediate = dict(data.get("intermediate") or {})
        ctx.history = list(data.get("history") or [])
        return ctx


@dataclass
class Task:
    """
    团队任务

    Attributes:
        description: 任务描述（团队共同目标）
        context: 执行期间创建的共享上下文
    """

    description: str
    context: SharedContext = field(default_factory=SharedContext)


class Process(ABC):
    """
    流程策略基类

    定义团队成员如何协作完成任务。不同 Process 实现 Sequential、
    Hierarchical 等不同协作模式。

    断点续跑支持（可选参数，None 表示未启用）：
    - checkpoint_state：上次执行保存的流程状态（从断点继续）
    - save_checkpoint：每完成一步后调用的落盘回调（传入该步后的流程状态）
    - step_limit：本次执行最多完成的步骤数；达到即暂停并返回 None
    """

    @abstractmethod
    def execute(
        self,
        task: Task,
        members: list[Agent],
        leader: Agent | None = None,
        stream_callback: Callable[[str], None] | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        step_limit: int | None = None,
    ) -> str | None:
        """
        执行团队任务

        Args:
            task: 团队任务（含共享上下文）
            members: 团队成员列表
            leader: 团队领导（Hierarchical 流程必需，Sequential 可为 None）
            stream_callback: 流式回调；传入后最终结果流式产出
            checkpoint_state: 断点续跑：上次保存的流程状态
            save_checkpoint: 断点续跑：步骤完成后的落盘回调
            step_limit: 本次执行的步数预算；达到后暂停（返回 None）

        Returns:
            最终结果文本；step_limit 触发暂停时返回 None（进度已落盘）
        """
        raise NotImplementedError


class SequentialProcess(Process):
    """
    顺序接力流程

    成员按声明顺序依次执行。每个成员收到：
      - 任务描述
      - 之前所有成员的累计产出（来自 SharedContext）
    并用 think() 处理，产出存入 SharedContext。
    最后一个成员的产出作为最终结果。

    适用场景：固定的确定性流水线（研究 → 写作 → 审核）。
    """

    name = "sequential"

    def execute(
        self,
        task: Task,
        members: list[Agent],
        leader: Agent | None = None,
        stream_callback: Callable[[str], None] | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        step_limit: int | None = None,
    ) -> str | None:
        if not members:
            raise ValueError("SequentialProcess 需要至少一个成员")

        ctx = task.context
        # 断点续跑：从上次完成的位置继续（已执行成员不再重跑）
        start_index = 0
        if checkpoint_state and checkpoint_state.get("process") == "sequential":
            start_index = int(checkpoint_state.get("next_index", 0) or 0)
            start_index = max(0, min(start_index, len(members)))
            if start_index > 0:
                logger.info("[Sequential] 从断点继续：跳过前 %d 个成员", start_index)
        ctx.record_step("sequential_start", f"成员顺序: {[m.agent_id for m in members]}")

        executed = 0
        last_idx = len(members) - 1
        for idx, member in enumerate(members):
            if idx < start_index:
                continue
            if step_limit is not None and executed >= step_limit:
                logger.info("[Sequential] 达到本次步数上限 %d，暂停（进度已落盘）", step_limit)
                return None
            self._ensure_can_think(member)

            # 组装上下文：任务 + 之前成员的累计产出
            prior = ctx.get_all_outputs()
            user_input = self._build_input(task.description, prior, is_first=(idx == 0))

            logger.info("[Sequential] 成员 %s (%d/%d) 开始处理", member.name, idx + 1, len(members))
            span = Tracer.start_span(
                f"sequential:{member.agent_id}", member=member.agent_id, step=idx + 1
            )
            # 最后一个成员的产出是最终结果，传入 stream_callback 流式输出
            cb = stream_callback if idx == last_idx else None
            output = member.think(user_input, stream_callback=cb)
            Tracer.end_span(span)
            ctx.add_output(member.agent_id, output)
            ctx.record_step("member_done", {"agent": member.agent_id, "index": idx})
            executed += 1
            # 每完成一个成员即落盘（崩溃最多损失当前进行中的这一步）
            if save_checkpoint:
                save_checkpoint({"process": "sequential", "next_index": idx + 1})

        # 最终结果取最后一个成员的产出
        final = ctx.get_output(members[-1].agent_id)
        ctx.record_step("sequential_done")
        return final or ""

    @staticmethod
    def _build_input(task_desc: str, prior_outputs: str, is_first: bool) -> str:
        """组装交给当前成员的输入"""
        if is_first or not prior_outputs:
            return task_desc
        return (
            f"团队任务：{task_desc}\n\n"
            f"以下是团队成员此前的产出，请在此基础上继续推进：\n{prior_outputs}"
        )

    @staticmethod
    def _ensure_can_think(agent: Agent) -> None:
        if agent.llm is None:
            raise RuntimeError(
                f"成员 '{agent.name}' 未配置 LLM，无法执行 think()。请为团队成员传入 llm_client。"
            )


class HierarchicalProcess(Process):
    """
    层级流程（Leader 自动编排）⭐

    与 Sequential 的关键区别：由 Leader 用 LLM 动态分析任务，输出结构化的
    子任务规划（每个子任务指定所需能力），再按能力匹配成员执行，最后由
    Leader 汇总所有产出。

    流程：
    1. Leader 用 LLM 分析任务 → 输出 JSON 子任务列表 [{subtask, capability}]
    2. 按子任务顺序：找到具备所需能力的成员 → 把「子任务 + 已完成产出」交给它 think
    3. 全部子任务完成后，Leader 汇总为最终结果

    适用场景：复杂任务、不确定该谁做什么时，让 Leader 智能决策。
    """

    name = "hierarchical"

    # Leader 规划任务的 prompt 模板
    # 注意：配合 response_format={"type": "json_object"} 使用，模型被约束为
    # 输出合法 JSON 对象；对不支持该参数的客户端自动回退普通文本解析
    _PLAN_PROMPT = (
        "你是一个团队的项目经理。团队任务如下：\n{task}\n\n"
        "团队成员及其能力：\n{members}\n\n"
        "请把任务拆解为 2-{max_subtasks} 个子任务，每个子任务指定由哪种能力来完成。"
        "只输出 JSON 对象，不要任何其他文字。格式：\n"
        '{{"subtasks": [{{"subtask": "子任务描述", "capability": "能力名"}}]}}\n'
        "capability 必须从上述成员能力中选择。"
    )

    # Leader 汇总的 prompt 模板
    _SYNTHESIS_PROMPT = (
        "团队任务：{task}\n\n"
        "以下是各成员的产出：\n{outputs}\n\n"
        "请综合所有产出，给出一份完整、连贯的最终结果。直接输出最终内容。"
    )

    def __init__(
        self,
        max_subtasks: int = 5,
        max_steps: int = 10,
        parallel_subtasks: bool = False,
    ):
        """
        Args:
            max_subtasks: Leader 最多拆出的子任务数上限（防止过度拆解）
            max_steps: 总执行步数上限（防死循环）
            parallel_subtasks: 子任务并行执行（fan-out）。按执行成员分组，
                        组间并行、组内串行（同一成员的会话状态不并发写）。
                        注意：并行模式下各子任务拿不到彼此的产出（同时进行），
                        依赖前序产出的任务请保持默认的顺序模式。
        """
        self.max_subtasks = max_subtasks
        self.max_steps = max_steps
        self.parallel_subtasks = parallel_subtasks

    def execute(
        self,
        task: Task,
        members: list[Agent],
        leader: Agent | None = None,
        stream_callback: Callable[[str], None] | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        step_limit: int | None = None,
    ) -> str | None:
        if leader is None:
            raise ValueError("HierarchicalProcess 需要 leader")
        self._ensure_can_think(leader)

        ctx = task.context
        ctx.record_step("hierarchical_start", f"leader={leader.agent_id}")

        # 1. Leader 规划子任务（断点续跑时直接复用上次的规划，省一次 LLM 调用）
        resume_state = (
            checkpoint_state
            if checkpoint_state and checkpoint_state.get("process") == "hierarchical"
            else None
        )
        if resume_state and isinstance(resume_state.get("plan"), list) and resume_state["plan"]:
            subtasks = [dict(st) for st in resume_state["plan"]][: self.max_subtasks]
            ctx.add_intermediate("plan", subtasks)
            ctx.record_step("plan_restored", f"复用断点规划：{len(subtasks)} 个子任务")
        else:
            subtasks = self._plan(task.description, members, leader, ctx)
            ctx.add_intermediate("plan", subtasks)
            ctx.record_step("plan_done", f"{len(subtasks)} 个子任务")
            if save_checkpoint:
                save_checkpoint(
                    {"process": "hierarchical", "plan": subtasks, "completed_subtasks": []}
                )

        # 2. 执行子任务：并行（按成员分组 fan-out）或逐个顺序执行；跳过断点中已完成的
        done: set[int] = (
            set(resume_state.get("completed_subtasks") or []) if resume_state else set()
        )
        if self.parallel_subtasks and len(subtasks) > 1:
            self._execute_subtasks_parallel(task, members, subtasks, ctx, done, save_checkpoint)
        else:
            paused = self._execute_subtasks_sequential(
                task, members, subtasks, ctx, done, save_checkpoint, step_limit
            )
            if paused:
                return None

        # 3. Leader 汇总
        if not ctx.outputs:
            # 所有子任务都没找到匹配成员，回退：让每个成员都处理一次
            logger.warning("[Hierarchical] 无任何产出，回退为全员顺序处理")
            for member in members:
                if member.llm is None:
                    continue
                out = member.think(
                    f"团队任务：{task.description}\n\n此前产出：\n{ctx.get_all_outputs()}"
                )
                ctx.add_output(member.agent_id, out)

        logger.info("[Hierarchical] Leader 汇总最终结果")
        synthesis_input = self._SYNTHESIS_PROMPT.format(
            task=task.description, outputs=ctx.get_all_outputs()
        )
        final = leader.think(synthesis_input, use_tools=False, stream_callback=stream_callback)
        ctx.record_step("synthesis_done")
        return final

    def _execute_subtasks_sequential(
        self,
        task: Task,
        members: list[Agent],
        subtasks: list[dict[str, str]],
        ctx: SharedContext,
        done: set[int] | None = None,
        save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        step_limit: int | None = None,
    ) -> bool:
        """
        顺序执行子任务：每个子任务都能看到此前所有产出

        Returns:
            True 表示因 step_limit 暂停（进度已落盘）；False 表示全部执行完
        """
        done = done if done is not None else set()
        steps = 0
        for i, st in enumerate(subtasks):
            if i in done:
                continue
            if steps >= self.max_steps:
                logger.warning("[Hierarchical] 达到最大步数 %d，终止", self.max_steps)
                break
            if step_limit is not None and steps >= step_limit:
                logger.info("[Hierarchical] 达到本次步数上限 %d，暂停（进度已落盘）", step_limit)
                return True

            member = self._find_member_by_capability(members, st["capability"])
            if member is None:
                # 没人具备该能力，记录并跳过（不中断整体流程）
                ctx.record_step("no_matching_member", st)
                logger.warning("[Hierarchical] 无成员具备能力 '%s'，跳过子任务", st["capability"])
                continue

            self._ensure_can_think(member)
            prior = ctx.get_all_outputs()
            user_input = f"子任务：{st['subtask']}\n（团队整体目标：{task.description}）" + (
                f"\n\n此前产出：\n{prior}" if prior else ""
            )

            logger.info("[Hierarchical] 子任务交给 %s（能力 %s）", member.name, st["capability"])
            span = Tracer.start_span(
                f"hierarchical:{member.agent_id}",
                member=member.agent_id,
                subtask=st["subtask"][:40],
                capability=st["capability"],
            )
            output = member.think(user_input)
            Tracer.end_span(span)
            ctx.add_output(member.agent_id, output)
            steps += 1
            done.add(i)
            if save_checkpoint:
                save_checkpoint(
                    {
                        "process": "hierarchical",
                        "plan": subtasks,
                        "completed_subtasks": sorted(done),
                    }
                )
        return False

    def _execute_subtasks_parallel(
        self,
        task: Task,
        members: list[Agent],
        subtasks: list[dict[str, str]],
        ctx: SharedContext,
        done: set[int] | None = None,
        save_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """
        并行执行子任务（fan-out）

        按执行成员分组：组间并行、组内串行——同一成员的 brain/会话状态
        不会被并发写入，不同成员之间互不阻塞。子任务数受 max_subtasks
        约束，组数即并行度。检查点粒度为整轮 fan-out（完成后统一落盘）。
        """
        done = done if done is not None else set()
        groups: dict[str, tuple[Agent, list[tuple[int, dict[str, str]]]]] = {}
        for i, st in enumerate(subtasks):
            if i in done:
                continue
            member = self._find_member_by_capability(members, st["capability"])
            if member is None:
                ctx.record_step("no_matching_member", st)
                logger.warning("[Hierarchical] 无成员具备能力 '%s'，跳过子任务", st["capability"])
                continue
            self._ensure_can_think(member)
            entry = groups.setdefault(member.agent_id, (member, []))
            entry[1].append((i, st))

        if not groups:
            return

        def run_group(member: Agent, group_subtasks: list[tuple[int, dict[str, str]]]) -> int:
            count = 0
            for i, st in group_subtasks[: self.max_steps]:
                user_input = f"子任务：{st['subtask']}\n（团队整体目标：{task.description}）"
                logger.info(
                    "[Hierarchical·并行] 子任务交给 %s（能力 %s）", member.name, st["capability"]
                )
                span = Tracer.start_span(
                    f"hierarchical:{member.agent_id}",
                    member=member.agent_id,
                    subtask=st["subtask"][:40],
                    capability=st["capability"],
                )
                output = member.think(user_input)
                Tracer.end_span(span)
                ctx.add_output(member.agent_id, output)
                done.add(i)
                count += 1
            return count

        ctx.record_step("parallel_fan_out", f"{len(groups)} 个成员组并行")
        with ThreadPoolExecutor(
            max_workers=min(4, len(groups)), thread_name_prefix="subtask"
        ) as pool:
            futures = [pool.submit(run_group, m, sts) for m, sts in groups.values()]
            executed = sum(f.result() for f in futures)
        ctx.record_step("parallel_done", f"共执行 {executed} 个子任务")
        if save_checkpoint and executed:
            save_checkpoint(
                {
                    "process": "hierarchical",
                    "plan": subtasks,
                    "completed_subtasks": sorted(done),
                }
            )

    def _plan(
        self,
        task_desc: str,
        members: list[Agent],
        leader: Agent,
        ctx: SharedContext,
    ) -> list[dict[str, str]]:
        """
        让 Leader 规划子任务

        优先用 response_format={"type": "json_object"} 约束模型输出合法 JSON
        （可靠性远高于自由文本 + 正则抠取）；客户端不支持该参数时自动降级
        为普通调用，解析层再兜底。

        Returns:
            子任务列表 [{"subtask": str, "capability": str}]
            解析失败时回退为"每个成员一个子任务"
        """
        members_desc = "\n".join(
            f"- {m.name}（id={m.agent_id}，能力={m.capabilities}）" for m in members
        )
        prompt = self._PLAN_PROMPT.format(
            task=task_desc, members=members_desc, max_subtasks=self.max_subtasks
        )

        try:
            try:
                raw = leader.think(prompt, use_tools=False, response_format={"type": "json_object"})
            except TypeError:
                # LLM 客户端不支持 response_format 参数（自定义/旧实现）：回退
                raw = leader.think(prompt, use_tools=False)
            subtasks = self._parse_plan(raw, members)
            if subtasks:
                return subtasks[: self.max_subtasks]
        except Exception as e:
            logger.warning("[Hierarchical] Leader 规划失败，回退: %s", e)

        # 回退：每个成员一个子任务（用其第一个能力）
        return self._fallback_plan(members)

    def _parse_plan(self, raw: str, members: list[Agent]) -> list[dict[str, str]]:
        """
        解析 Leader 输出的子任务规划（多层容错）

        依次尝试：整体 JSON 解析（json_object 模式的正常产物）→ 从自由文本中
        截取首个 JSON 对象/数组（旧模型/不支持约束的客户端的兜底）→ 校验字段。
        """
        text = raw.strip()

        # 候选解析结果：整体解析优先，截取解析兜底
        candidates: list[Any] = []
        if text.startswith("{") or text.startswith("["):
            with contextlib.suppress(json.JSONDecodeError):
                candidates.append(json.loads(text))
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start, end = text.find(open_ch), text.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    candidates.append(json.loads(text[start : end + 1]))
                except json.JSONDecodeError:
                    continue

        for data in candidates:
            # 新格式：{"subtasks": [...]}；兼容旧格式：直接是数组
            items = data.get("subtasks") if isinstance(data, dict) else data
            if not isinstance(items, list):
                continue
            result = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                subtask = item.get("subtask") or item.get("task")
                capability = item.get("capability") or item.get("ability")
                if subtask and capability:
                    result.append({"subtask": str(subtask), "capability": str(capability)})
            if result:
                return result
        return []

    @staticmethod
    def _fallback_plan(members: list[Agent]) -> list[dict[str, str]]:
        """回退规划：每个有能力的成员处理一个通用子任务"""
        plan = []
        for m in members:
            if not m.capabilities:
                continue
            plan.append(
                {
                    "subtask": "请用你的专长推进团队任务",
                    "capability": m.capabilities[0],
                }
            )
        return plan

    @staticmethod
    def _find_member_by_capability(members: list[Agent], capability: str) -> Agent | None:
        """按能力找成员（返回第一个具备该能力的）"""
        for m in members:
            if m.has_capability(capability):
                return m
        return None

    @staticmethod
    def _ensure_can_think(agent: Agent) -> None:
        if agent.llm is None:
            raise RuntimeError(f"成员 '{agent.name}' 未配置 LLM，无法执行 think()。")


class Team:
    """
    团队

    组合若干成员与一个流程，协同完成共同任务。

    用法：
        team = Team("内容组", members=[a, b, c], process=SequentialProcess())
        result = team.run("写一篇科普文章")
    """

    def __init__(
        self,
        name: str,
        members: list[Agent] | None = None,
        process: Process | None = None,
        leader: Agent | None = None,
        checkpoint_store: Any | None = None,
    ):
        """
        Args:
            name: 团队名称
            members: 成员列表
            process: 流程策略；None 默认 SequentialProcess
            leader: 团队领导；HierarchicalProcess 时必需
            checkpoint_store: 检查点存储（如 FileCheckpointStore）；
                        配合 run(run_id=...) / run_resumable(...) 实现断点续跑
        """
        self.name = name
        self.members: list[Agent] = list(members) if members else []
        self.process: Process = process or SequentialProcess()
        self.leader: Agent | None = leader
        self.checkpoint_store = checkpoint_store
        # 历次任务记录（便于 summary / 调试）
        self.task_history: list[dict[str, Any]] = []

    def add_member(self, agent: Agent) -> None:
        """添加成员"""
        self.members.append(agent)

    def remove_member(self, agent_id: str) -> bool:
        """移除成员"""
        before = len(self.members)
        self.members = [m for m in self.members if m.agent_id != agent_id]
        return len(self.members) < before

    def run(
        self,
        task_description: str,
        stream_callback: Callable[[str], None] | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        执行团队任务

        创建共享上下文，按流程驱动成员协作，返回最终结果。
        传入 run_id（且配置了 checkpoint_store）时：若存在同 run_id 的
        未完成检查点则自动从断点继续（已完成的步骤不再重跑）；
        已完成的检查点直接返回缓存结果。

        Args:
            task_description: 任务描述（团队共同目标）
            stream_callback: 流式回调；传入后最终结果流式产出（Leader 汇总
                             或 Sequential 最后一个成员的输出实时回调）
            run_id: 检查点标识（配合 checkpoint_store 断点续跑）

        Returns:
            最终结果文本
        """
        outcome = self._run_impl(task_description, stream_callback, run_id, step_limit=None)
        if outcome.result is None:
            raise RuntimeError("团队执行未产生结果（不应发生：run 不设步数上限）")
        return outcome.result

    def run_resumable(
        self,
        task_description: str,
        run_id: str,
        stream_callback: Callable[[str], None] | None = None,
        step_limit: int | None = None,
    ):
        """
        可暂停/可续跑的执行（人工审批、分段预算场景）

        达到 step_limit 后停止执行（进度已落盘），返回 completed=False 的
        RunOutcome；再次调用同 run_id 从断点继续。

        Args:
            task_description: 任务描述（团队共同目标）
            run_id: 检查点标识（必填；需要团队配置 checkpoint_store）
            stream_callback: 流式回调
            step_limit: 本次执行最多完成的成员步骤数；None 表示跑到结束

        Returns:
            RunOutcome(result, completed, steps_done, run_id)
        """
        return self._run_impl(task_description, stream_callback, run_id, step_limit)

    def _run_impl(self, task_description, stream_callback, run_id, step_limit):

        # 跨进程执行锁：两个进程同时续跑同一 run_id 时，后者等待/报忙，
        # 而不是竞写检查点（陈旧锁在持锁方崩溃后自动抢占）
        store = self.checkpoint_store
        lock = store.run_lock(run_id) if (run_id and store is not None) else None
        if lock is not None:
            lock.acquire()
        try:
            return self._run_locked(task_description, stream_callback, run_id, step_limit)
        finally:
            if lock is not None:
                lock.release()

    def _run_locked(self, task_description, stream_callback, run_id, step_limit):
        from .checkpoint import RunOutcome

        store = self.checkpoint_store
        if not self.members:
            raise RuntimeError(f"团队 '{self.name}' 没有成员")

        if isinstance(self.process, HierarchicalProcess) and self.leader is None:
            raise RuntimeError("HierarchicalProcess 需要指定 leader")

        store = self.checkpoint_store
        if run_id and store is None:
            logger.warning("传入了 run_id 但团队未配置 checkpoint_store，断点续跑未启用")
        if step_limit is not None and (not run_id or store is None):
            raise ValueError("step_limit 需要 run_id 与 checkpoint_store（暂停后才能续跑）")

        # 读取检查点：任务不匹配则丢弃重来；已完成则直接返回缓存结果
        checkpoint_state: dict[str, Any] | None = None
        if run_id and store is not None:
            loaded = store.load(run_id)
            if loaded is not None:
                if loaded.get("task") != task_description:
                    logger.warning("检查点 %s 的任务与本次不一致，忽略旧检查点并重新开始", run_id)
                    store.delete(run_id)
                elif loaded.get("completed"):
                    logger.info("检查点 %s 已完成，直接返回缓存结果", run_id)
                    return RunOutcome(
                        result=loaded.get("result"),
                        completed=True,
                        steps_done=int(loaded.get("steps", 0) or 0),
                        run_id=run_id,
                    )
                else:
                    checkpoint_state = loaded.get("process_state")

        # 重建任务与共享上下文（断点恢复时还原黑板）
        if checkpoint_state and isinstance(checkpoint_state.get("context"), dict):
            ctx = SharedContext.from_dict(checkpoint_state["context"], task=task_description)
        else:
            ctx = SharedContext(task=task_description)
        task = Task(description=task_description, context=ctx)

        logger.info(
            "[Team:%s] 启动任务（流程=%s，成员=%d%s）",
            self.name,
            self.process.name,
            len(self.members),
            "，断点续跑" if checkpoint_state else "",
        )

        # 落盘回调：流程每完成一步即保存（含共享上下文，崩溃最多损失进行中的一步）
        def save_hook(process_state: dict[str, Any]) -> None:
            if run_id and store is not None:
                store.save(
                    run_id,
                    {
                        "task": task_description,
                        "team": self.name,
                        "process": self.process.name,
                        "process_state": process_state,
                        "steps": ctx.summary()["output_count"],
                        "context": ctx.to_dict(),
                    },
                )

        # 若 tracer 已启用，自动开启一个 trace 覆盖整个团队任务
        _started_trace = None
        if Tracer.is_enabled() and Tracer.current_trace() is None:
            _started_trace = Tracer.start_trace(f"team:{self.name}:{task_description[:30]}")

        try:
            extra: dict[str, Any] = {}
            if checkpoint_state is not None:
                extra["checkpoint_state"] = checkpoint_state
            if run_id and store is not None:
                extra["save_checkpoint"] = save_hook
            if step_limit is not None:
                extra["step_limit"] = step_limit
            result = self.process.execute(task, self.members, self.leader, stream_callback, **extra)
        finally:
            if _started_trace is not None:
                Tracer.end_trace()

        steps_done = ctx.summary()["output_count"]
        if result is None:
            # 暂停：进度已在每步落盘
            return RunOutcome(result=None, completed=False, steps_done=steps_done, run_id=run_id)

        # 完成：写入终态检查点（同 run_id 重跑直接返回缓存结果）
        if run_id and store is not None:
            store.save(
                run_id,
                {
                    "task": task_description,
                    "team": self.name,
                    "process": self.process.name,
                    "completed": True,
                    "result": result,
                    "steps": steps_done,
                    "context": ctx.to_dict(),
                },
            )
        self.task_history.append(
            {
                "task": task_description,
                "process": self.process.name,
                "context_summary": task.context.summary(),
            }
        )
        return RunOutcome(result=result, completed=True, steps_done=steps_done, run_id=run_id)

    def summary(self) -> dict[str, Any]:
        """团队摘要"""
        return {
            "name": self.name,
            "member_count": len(self.members),
            "members": [
                {"id": m.agent_id, "name": m.name, "capabilities": m.capabilities}
                for m in self.members
            ],
            "process": self.process.name,
            "has_leader": self.leader is not None,
            "task_count": len(self.task_history),
            "recent_tasks": self.task_history[-3:],
        }
