"""
Agent 基类核心测试

重点覆盖 think() 的 ReAct 循环各路径：
- 无 LLM 时报错
- 直接答复（无工具调用）
- 工具调用循环（调工具→回传→再思考→答复）
- max_steps 兜底
- 交互记忆写入
- capabilities 能力声明
- send 通信（无 router 时返回 Message）
- token 统计

用 MockLLM（确定性，不联网）。
"""

import pytest

from core.agent import Agent
from core.llm import ChatResult, LLMClient, TokenUsage, ToolCall
from core.message import Message
from core.tools import CalculatorTool

# --------------------------------------------------------------------
# Mock LLM：可编排返回序列，模拟 ReAct 各步骤
# --------------------------------------------------------------------


class ScriptedLLM(LLMClient):
    """按预设序列返回结果的 Mock LLM"""

    def __init__(self, results: list):
        """results: 依次返回的 ChatResult 列表"""
        self._results = list(results)
        self._index = 0
        self.chat_count = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.chat_count += 1
        if self._index < len(self._results):
            r = self._results[self._index]
            self._index += 1
            return r
        # 序列耗尽，返回一个最终答复防止死循环
        return ChatResult(content="(结束)", finish_reason="stop")


def make_agent(llm, capabilities=None, max_steps=5):
    """构造测试用 Agent"""

    class _TestAgent(Agent):
        def __init__(self):
            super().__init__(
                agent_id="test",
                name="测试Agent",
                system_prompt="你是测试助手",
                llm_client=llm,
                capabilities=capabilities,
                max_steps=max_steps,
            )

        def receive(self, message: Message):
            pass

    return _TestAgent()


# --------------------------------------------------------------------
# think() 前置检查
# --------------------------------------------------------------------


class TestThinkPreconditions:
    def test_no_llm_raises(self):
        class _NoLLM(Agent):
            def __init__(self):
                super().__init__("a", "a", llm_client=None)

            def receive(self, m):
                pass

        agent = _NoLLM()
        with pytest.raises(RuntimeError, match="未配置 LLM"):
            agent.think("hello")


# --------------------------------------------------------------------
# think() 直接答复路径
# --------------------------------------------------------------------


class TestThinkDirectAnswer:
    def test_returns_content_when_no_tool_calls(self):
        llm = ScriptedLLM([ChatResult(content="你好！", finish_reason="stop")])
        agent = make_agent(llm)
        result = agent.think("你好")
        assert result == "你好！"
        assert llm.chat_count == 1

    def test_interaction_remembered(self):
        """think 完成后，交互应写入 brain 的会话通道"""
        llm = ScriptedLLM([ChatResult(content="答复", finish_reason="stop")])
        agent = make_agent(llm)
        agent.think("问题")
        conv = agent.brain.get_conversation()
        # 应有 user + assistant 两条
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "问题"
        assert conv[1]["role"] == "assistant"
        assert conv[1]["content"] == "答复"


# --------------------------------------------------------------------
# think() 工具调用循环
# --------------------------------------------------------------------


class TestThinkToolLoop:
    def test_tool_call_then_answer(self):
        """
        经典 ReAct 两步：
        1. LLM 请求调 calculator → 执行 → 结果回传
        2. LLM 基于结果给出最终答复
        """
        llm = ScriptedLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "2+3"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="结果是 5", finish_reason="stop"),
            ]
        )
        agent = make_agent(llm)
        agent.tool_registry.register(CalculatorTool())

        result = agent.think("计算 2+3")
        assert result == "结果是 5"
        assert llm.chat_count == 2  # 两次 LLM 调用

    def test_tool_result_in_short_term_memory(self):
        """工具执行结果应存入短期记忆"""
        llm = ScriptedLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="done", finish_reason="stop"),
            ]
        )
        agent = make_agent(llm)
        agent.tool_registry.register(CalculatorTool())
        agent.think("计算")

        # 短期记忆里应有 tool:calculator:c1 的记录
        short = agent.memory.list_short_term_memory()
        tool_keys = [k for k in short if k.startswith("tool:calculator")]
        assert len(tool_keys) >= 1

    def test_tool_failure_does_not_break_loop(self):
        """工具执行失败不应中断 think 循环"""
        llm = ScriptedLLM(
            [
                ChatResult(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="nonexistent", arguments={})],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="工具失败了，但我还能回答", finish_reason="stop"),
            ]
        )
        agent = make_agent(llm)
        # 不注册 nonexistent 工具 → 执行会失败
        result = agent.think("test")
        # 循环不中断，最终仍返回答复
        assert "回答" in result


# --------------------------------------------------------------------
# max_steps 兜底
# --------------------------------------------------------------------


class TestMaxStepsFallback:
    def test_exceeding_max_steps_does_fallback_call(self):
        """
        当模型持续请求工具（不给出最终答复），达到 max_steps 后
        应做一次不带工具的收尾调用。
        """
        # 每次都请求工具，永不停止
        always_tool = ChatResult(
            content="",
            tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "1"})],
            finish_reason="tool_calls",
        )
        final = ChatResult(content="兜底答复", finish_reason="stop")
        # max_steps=2 → 2轮工具 + 1轮兜底
        llm = ScriptedLLM([always_tool, always_tool, final])
        agent = make_agent(llm, max_steps=2)
        agent.tool_registry.register(CalculatorTool())

        result = agent.think("无限循环")
        assert result == "兜底答复"


# --------------------------------------------------------------------
# token 统计
# --------------------------------------------------------------------


class TestTokenTracking:
    def test_usage_accumulated(self):
        llm = ScriptedLLM(
            [
                ChatResult(content="ok", finish_reason="stop", usage=TokenUsage(10, 20, 30)),
            ]
        )
        agent = make_agent(llm)
        agent.think("hi")
        usage = agent.get_token_usage()
        assert usage.total_tokens == 30

    def test_reset_usage(self):
        llm = ScriptedLLM([ChatResult(content="ok", usage=TokenUsage(5, 5, 10))])
        agent = make_agent(llm)
        agent.think("hi")
        agent.reset_token_usage()
        assert agent.get_token_usage().total_tokens == 0


# --------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------


class TestCapabilities:
    def test_has_capability(self):
        agent = make_agent(ScriptedLLM([]), capabilities=["search", "write"])
        assert agent.has_capability("search")
        assert not agent.has_capability("missing")

    def test_has_any_capability(self):
        agent = make_agent(ScriptedLLM([]), capabilities=["search"])
        assert agent.has_any_capability(["search", "write"])
        assert not agent.has_any_capability(["write", "analysis"])

    def test_no_capabilities_by_default(self):
        agent = make_agent(ScriptedLLM([]))
        assert agent.capabilities == []


# --------------------------------------------------------------------
# send 通信（无 router）
# --------------------------------------------------------------------


class TestSend:
    def test_send_without_router_returns_message(self):
        agent = make_agent(ScriptedLLM([]))
        msg = agent.send("target", "hello", "text")
        # 无 router 时 send 返回 Message 对象（而非 None）
        assert isinstance(msg, Message)
        assert msg.sender == "test"
        assert msg.receiver == "target"
        assert msg.content == "hello"

    def test_broadcast_without_router_raises(self):
        agent = make_agent(ScriptedLLM([]))
        with pytest.raises(RuntimeError, match="Router"):
            agent.broadcast("hello")


# --------------------------------------------------------------------
# 工具管理
# --------------------------------------------------------------------


class TestToolManagement:
    def test_add_and_list_tools(self):
        agent = make_agent(ScriptedLLM([]))
        calc = CalculatorTool()
        agent.add_tool("calc", calc)  # 注册时用 tool.name（"calculator"）
        assert "calculator" in agent.list_tools()

    def test_get_tool(self):
        agent = make_agent(ScriptedLLM([]))
        tool = CalculatorTool()
        agent.add_tool("calc", tool)
        assert agent.get_tool("calculator") is tool
        assert agent.get_tool("missing") is None
