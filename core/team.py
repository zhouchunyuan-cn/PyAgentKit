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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging

from .agent import Agent
from .llm import LLMClient
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
        # 各成员的产出：agent_id -> output
        self.outputs: Dict[str, str] = {}
        # 任意中间产物（子任务、规划、统计等）
        self.intermediate: Dict[str, Any] = {}
        # 执行步骤记录，便于追溯
        self.history: List[Dict[str, Any]] = []

    def add_output(self, agent_id: str, output: str) -> None:
        """记录某成员的产出"""
        self.outputs[agent_id] = output
        self.history.append({"agent": agent_id, "action": "output", "detail": output[:200]})

    def get_output(self, agent_id: str) -> Optional[str]:
        """获取某成员的产出"""
        return self.outputs.get(agent_id)

    def get_all_outputs(self) -> str:
        """
        拼接所有成员的产出（按记录顺序），作为下一个成员的上下文。

        格式清晰标注每个产出来自谁，便于 LLM 理解。
        """
        if not self.outputs:
            return ""
        lines = []
        for agent_id, output in self.outputs.items():
            lines.append(f"【{agent_id} 的产出】\n{output}")
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

    def summary(self) -> Dict[str, Any]:
        """上下文摘要"""
        return {
            "task": self.task,
            "output_count": len(self.outputs),
            "output_agents": list(self.outputs.keys()),
            "intermediate_keys": list(self.intermediate.keys()),
            "history_steps": len(self.history),
        }


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
    """

    @abstractmethod
    def execute(
        self,
        task: Task,
        members: List[Agent],
        leader: Optional[Agent] = None,
    ) -> str:
        """
        执行团队任务

        Args:
            task: 团队任务（含共享上下文）
            members: 团队成员列表
            leader: 团队领导（Hierarchical 流程必需，Sequential 可为 None）

        Returns:
            最终结果文本
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

    def execute(self, task: Task, members: List[Agent], leader: Optional[Agent] = None) -> str:
        if not members:
            raise ValueError("SequentialProcess 需要至少一个成员")

        ctx = task.context
        ctx.record_step("sequential_start", f"成员顺序: {[m.agent_id for m in members]}")

        for idx, member in enumerate(members):
            self._ensure_can_think(member)

            # 组装上下文：任务 + 之前成员的累计产出
            prior = ctx.get_all_outputs()
            user_input = self._build_input(task.description, prior, is_first=(idx == 0))

            logger.info("[Sequential] 成员 %s (%d/%d) 开始处理", member.name, idx + 1, len(members))
            span = Tracer.start_span(f"sequential:{member.agent_id}", member=member.agent_id, step=idx + 1)
            output = member.think(user_input)
            Tracer.end_span(span)
            ctx.add_output(member.agent_id, output)
            ctx.record_step("member_done", {"agent": member.agent_id, "index": idx})

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
                f"成员 '{agent.name}' 未配置 LLM，无法执行 think()。"
                "请为团队成员传入 llm_client。"
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
    _PLAN_PROMPT = (
        "你是一个团队的项目经理。团队任务如下：\n{task}\n\n"
        "团队成员及其能力：\n{members}\n\n"
        "请把任务拆解为 2-{max_subtasks} 个子任务，每个子任务指定由哪种能力来完成。"
        "只输出 JSON 数组，不要任何其他文字。格式：\n"
        '[{{"subtask": "子任务描述", "capability": "能力名"}}]\n'
        "capability 必须从上述成员能力中选择。"
    )

    # Leader 汇总的 prompt 模板
    _SYNTHESIS_PROMPT = (
        "团队任务：{task}\n\n"
        "以下是各成员的产出：\n{outputs}\n\n"
        "请综合所有产出，给出一份完整、连贯的最终结果。直接输出最终内容。"
    )

    def __init__(self, max_subtasks: int = 5, max_steps: int = 10):
        """
        Args:
            max_subtasks: Leader 最多拆出的子任务数上限（防止过度拆解）
            max_steps: 总执行步数上限（防死循环）
        """
        self.max_subtasks = max_subtasks
        self.max_steps = max_steps

    def execute(self, task: Task, members: List[Agent], leader: Optional[Agent] = None) -> str:
        if leader is None:
            raise ValueError("HierarchicalProcess 需要 leader")
        self._ensure_can_think(leader)

        ctx = task.context
        ctx.record_step("hierarchical_start", f"leader={leader.agent_id}")

        # 1. Leader 规划子任务
        subtasks = self._plan(task.description, members, leader, ctx)
        ctx.add_intermediate("plan", subtasks)
        ctx.record_step("plan_done", f"{len(subtasks)} 个子任务")

        # 2. 逐个子任务：按能力匹配成员并执行
        steps = 0
        for st in subtasks:
            if steps >= self.max_steps:
                logger.warning("[Hierarchical] 达到最大步数 %d，终止", self.max_steps)
                break

            member = self._find_member_by_capability(members, st["capability"])
            if member is None:
                # 没人具备该能力，记录并跳过（不中断整体流程）
                ctx.record_step("no_matching_member", st)
                logger.warning("[Hierarchical] 无成员具备能力 '%s'，跳过子任务", st["capability"])
                continue

            self._ensure_can_think(member)
            prior = ctx.get_all_outputs()
            user_input = (
                f"子任务：{st['subtask']}\n"
                f"（团队整体目标：{task.description}）"
                + (f"\n\n此前产出：\n{prior}" if prior else "")
            )

            logger.info("[Hierarchical] 子任务交给 %s（能力 %s）",
                        member.name, st["capability"])
            span = Tracer.start_span(
                f"hierarchical:{member.agent_id}",
                member=member.agent_id, subtask=st["subtask"][:40], capability=st["capability"],
            )
            output = member.think(user_input)
            Tracer.end_span(span)
            ctx.add_output(member.agent_id, output)
            steps += 1

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
        final = leader.think(synthesis_input, use_tools=False)
        ctx.record_step("synthesis_done")
        return final

    def _plan(
        self,
        task_desc: str,
        members: List[Agent],
        leader: Agent,
        ctx: SharedContext,
    ) -> List[Dict[str, str]]:
        """
        让 Leader 规划子任务

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
            raw = leader.think(prompt, use_tools=False)
            subtasks = self._parse_plan(raw, members)
            if subtasks:
                return subtasks[: self.max_subtasks]
        except Exception as e:
            logger.warning("[Hierarchical] Leader 规划失败，回退: %s", e)

        # 回退：每个成员一个子任务（用其第一个能力）
        return self._fallback_plan(members)

    def _parse_plan(self, raw: str, members: List[Agent]) -> List[Dict[str, str]]:
        """
        解析 Leader 输出的 JSON 子任务规划

        容错：提取首个 JSON 数组、校验字段、过滤无效项。
        """
        # 提取首个 [ ... ] 片段（Leader 可能混入其他文字）
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []

        valid_caps = {c for m in members for c in m.capabilities}
        result = []
        for item in data:
            if not isinstance(item, dict):
                continue
            subtask = item.get("subtask") or item.get("task")
            capability = item.get("capability") or item.get("ability")
            if subtask and capability:
                result.append({"subtask": str(subtask), "capability": str(capability)})
        return result

    @staticmethod
    def _fallback_plan(members: List[Agent]) -> List[Dict[str, str]]:
        """回退规划：每个有能力的成员处理一个通用子任务"""
        plan = []
        for m in members:
            if not m.capabilities:
                continue
            plan.append({
                "subtask": f"请用你的专长推进团队任务",
                "capability": m.capabilities[0],
            })
        return plan

    @staticmethod
    def _find_member_by_capability(members: List[Agent], capability: str) -> Optional[Agent]:
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
        members: Optional[List[Agent]] = None,
        process: Optional[Process] = None,
        leader: Optional[Agent] = None,
    ):
        """
        Args:
            name: 团队名称
            members: 成员列表
            process: 流程策略；None 默认 SequentialProcess
            leader: 团队领导；HierarchicalProcess 时必需
        """
        self.name = name
        self.members: List[Agent] = list(members) if members else []
        self.process: Process = process or SequentialProcess()
        self.leader: Optional[Agent] = leader
        # 历次任务记录（便于 summary / 调试）
        self.task_history: List[Dict[str, Any]] = []

    def add_member(self, agent: Agent) -> None:
        """添加成员"""
        self.members.append(agent)

    def remove_member(self, agent_id: str) -> bool:
        """移除成员"""
        before = len(self.members)
        self.members = [m for m in self.members if m.agent_id != agent_id]
        return len(self.members) < before

    def run(self, task_description: str) -> str:
        """
        执行团队任务

        创建共享上下文，按流程驱动成员协作，返回最终结果。

        Args:
            task_description: 任务描述（团队共同目标）

        Returns:
            最终结果文本
        """
        if not self.members:
            raise RuntimeError(f"团队 '{self.name}' 没有成员")

        if isinstance(self.process, HierarchicalProcess) and self.leader is None:
            raise RuntimeError("HierarchicalProcess 需要指定 leader")

        task = Task(description=task_description, context=SharedContext(task=task_description))

        logger.info("[Team:%s] 启动任务（流程=%s，成员=%d）",
                    self.name, self.process.name, len(self.members))

        # 若 tracer 已启用，自动开启一个 trace 覆盖整个团队任务
        _started_trace = None
        if Tracer.is_enabled() and Tracer.current_trace() is None:
            _started_trace = Tracer.start_trace(f"team:{self.name}:{task_description[:30]}")

        try:
            result = self.process.execute(task, self.members, self.leader)
        finally:
            if _started_trace is not None:
                Tracer.end_trace()

        self.task_history.append({
            "task": task_description,
            "process": self.process.name,
            "context_summary": task.context.summary(),
        })
        return result

    def summary(self) -> Dict[str, Any]:
        """团队摘要"""
        return {
            "name": self.name,
            "member_count": len(self.members),
            "members": [{"id": m.agent_id, "name": m.name, "capabilities": m.capabilities} for m in self.members],
            "process": self.process.name,
            "has_leader": self.leader is not None,
            "task_count": len(self.task_history),
            "recent_tasks": self.task_history[-3:],
        }
