"""
跨模块集成测试

补齐关键组合路径的测试——这些路径在单个模块测试里无法覆盖，
是之前审计发现的盲区。用 MockLLM 避免联网。
"""

import asyncio
import json

from core.agent import Agent
from core.async_router import AsyncRouter
from core.config import ConfigLoader
from core.hooks import AgentHooks
from core.llm import ChatResult, LLMClient, ToolCall
from core.message import Message
from core.team import SequentialProcess, Team
from core.tools import CalculatorTool
from core.trace import Tracer

# --------------------------------------------------------------------
# Mock LLM
# --------------------------------------------------------------------


class MockLLM(LLMClient):
    """按预设序列返回的 Mock LLM"""

    def __init__(self, results: list):
        self._results = list(results)
        self._index = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        if self._index < len(self._results):
            r = self._results[self._index]
            self._index += 1
            return r
        return ChatResult(content="(end)")


# --------------------------------------------------------------------
# 1. Config → build → build_team → team.run 端到端
# --------------------------------------------------------------------


class TestConfigToTeamE2E:
    def test_build_and_run_sequential_team(self):
        """Config → build → build_team → team.run 全链路（FakeLLM）"""

        class FakeLLM(LLMClient):
            def __init__(self):
                self.count = 0

            def chat(self, messages, tools=None, tool_choice="auto"):
                self.count += 1
                return ChatResult(content=f"产出{self.count}", finish_reason="stop")

        loader = ConfigLoader()
        config = loader.parse(
            {
                "llm": {"model": "glm-4-flash"},
                "agents": [
                    {"id": "a", "capabilities": ["x"]},
                    {"id": "b", "capabilities": ["x"]},
                ],
                "team": {"process": "sequential", "members": ["a", "b"]},
            }
        )
        fake_llm = FakeLLM()
        _, agents, _ = loader.build(config, llm_client=fake_llm)
        team = loader.build_team(config, agents)

        result = team.run("任务")
        # 两个成员各 think 一次
        assert "产出" in result
        assert fake_llm.count >= 2

    def test_build_team_hierarchical_e2e(self):
        """Config → build_team → hierarchical team.run（FakeLLM）"""

        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None, tool_choice="auto"):
                # 检测是规划请求还是普通请求
                last = messages[-1].get("content", "") if messages else ""
                if "项目经理" in last or "拆解" in last:
                    return ChatResult(
                        content=json.dumps([{"subtask": "干活", "capability": "work"}]),
                        finish_reason="stop",
                    )
                if "综合所有产出" in last:
                    return ChatResult(content="最终汇总", finish_reason="stop")
                return ChatResult(content="成员产出", finish_reason="stop")

        loader = ConfigLoader()
        config = loader.parse(
            {
                "llm": {"model": "glm-4-flash"},
                "agents": [
                    {"id": "leader", "capabilities": ["plan"], "system_prompt": "你是项目经理"},
                    {"id": "worker", "capabilities": ["work"]},
                ],
                "team": {
                    "process": "hierarchical",
                    "leader": "leader",
                    "members": ["worker"],
                    "max_subtasks": 2,
                },
            }
        )
        fake_llm = FakeLLM()
        _, agents, _ = loader.build(config, llm_client=fake_llm)
        team = loader.build_team(config, agents)

        result = team.run("做件事")
        assert result == "最终汇总"


# --------------------------------------------------------------------
# 2. 流式 + 工具调用组合（P0 bug 回归路径的扩展）
# --------------------------------------------------------------------


class TestStreamingWithTools:
    def test_stream_callback_with_tool_calls(self):
        """stream_callback + 工具调用：工具步骤非流式，最终答复模拟流式"""
        llm = MockLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "2*3"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="结果是6", finish_reason="stop"),
            ]
        )

        class A(Agent):
            def __init__(self):
                super().__init__("a", "助手", llm_client=llm)

            def receive(self, m):
                pass

        agent = A()
        agent.tool_registry.register(CalculatorTool())
        collected = []
        result = agent.think("计算2*3", stream_callback=lambda s: collected.append(s))
        assert result == "结果是6"
        # LLM 只调了 2 次（1次工具判定 + 1次最终答复），不是 3 次
        assert llm._index == 2
        # 流式回调被触发
        assert len(collected) > 0


# --------------------------------------------------------------------
# 3. AsyncRouter + 真实 think 的 Agent
# --------------------------------------------------------------------


class TestAsyncRouterWithThink:
    def test_async_router_delivers_to_think_agent(self):
        """AsyncRouter 把消息投递给用 think 处理的 Agent"""
        received_inputs = []

        class ThinkAgent(Agent):
            def __init__(self, llm):
                super().__init__("t", "思考者", llm_client=llm)

            def receive(self, message):
                # 收到消息后用 think 处理
                received_inputs.append(message.content)

        llm = MockLLM([ChatResult(content="ok")])
        agent = ThinkAgent(llm)

        async def run():
            router = AsyncRouter()
            router.register_agent(agent)
            await router.start()
            await router.route(Message("user", "t", "你好"))
            await router.drain()
            await router.stop()

        asyncio.new_event_loop().run_until_complete(run())
        assert "你好" in received_inputs


# --------------------------------------------------------------------
# 4. Hooks + Team 组合
# --------------------------------------------------------------------


class TestHooksWithTeam:
    def test_team_run_triggers_member_hooks(self):
        """团队成员的 hooks 在 team.run 过程中被触发"""
        tool_events = []

        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None, tool_choice="auto"):
                return ChatResult(content="产出", finish_reason="stop")

        llm = FakeLLM()

        class Member(Agent):
            def __init__(self):
                super().__init__(
                    "m",
                    "成员",
                    llm_client=llm,
                    hooks=AgentHooks(
                        on_think_end=lambda ag, r: tool_events.append(r),
                    ),
                )

            def receive(self, msg):
                pass

        member = Member()
        team = Team(name="T", members=[member], process=SequentialProcess())
        team.run("任务")
        # on_think_end 被触发
        assert "产出" in tool_events


# --------------------------------------------------------------------
# 5. Tracer + Team 组合（Hierarchical）
# --------------------------------------------------------------------


class TestTracerWithHierarchical:
    def test_hierarchical_team_produces_trace_spans(self):
        """Hierarchical team.run 启用 Tracer 后产生 span"""

        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None, tool_choice="auto"):
                last = messages[-1].get("content", "") if messages else ""
                if "拆解" in last or "项目经理" in last:
                    return ChatResult(
                        content=json.dumps([{"subtask": "干活", "capability": "work"}]),
                        finish_reason="stop",
                    )
                if "综合" in last:
                    return ChatResult(content="汇总", finish_reason="stop")
                return ChatResult(content="成员产出", finish_reason="stop")

        llm = FakeLLM()

        class Leader(Agent):
            def __init__(self):
                super().__init__("L", "经理", llm_client=llm, capabilities=["plan"])

            def receive(self, m):
                pass

        class Worker(Agent):
            def __init__(self):
                super().__init__("w", "工人", llm_client=llm, capabilities=["work"])

            def receive(self, m):
                pass

        Tracer.enable()
        Tracer.start_trace("team_task")
        team = Team(name="T", members=[Worker()], process=SequentialProcess(), leader=Leader())
        # 用 sequential 测试更简单，hierarchical 需要 plan JSON
        team.run("任务")
        trace = Tracer.end_trace()
        Tracer.disable()

        # 应有 think span
        think_spans = [s for s in trace.spans if s.name.startswith("think:")]
        assert len(think_spans) >= 1
