"""
Agent 生命周期钩子

提供在 ReAct 推理过程的关键节点注入自定义逻辑的能力，
无需子类化 Agent。适用于日志、监控、审计、人机协作（HITL）等场景。

用法：
    hooks = AgentHooks(
        on_tool_call=lambda agent, name, args, result: print(f"调用了 {name}"),
        on_error=lambda agent, error, ctx: alert(error),
    )
    agent.hooks = hooks   # 或在 Agent.__init__ 传 hooks=hooks
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentHooks:
    """
    生命周期钩子容器

    每个字段是一个可选的回调函数。未设置（None）的钩子不会触发任何调用。
    所有钩子都是"通知型"的——返回值被忽略，不影响主流程（除了 on_think_start
    可返回 False 中止，用于 human-in-the-loop 审批）。

    钩子签名：
        on_think_start(agent, user_input) -> bool | None
            think 开始时触发。返回 False 可中止本次 think（用于审批拒绝）。
        on_llm_response(agent, result, step) -> None
            每次 LLM 调用返回后触发。result 是 ChatResult，step 是循环步数。
        on_tool_call(agent, tool_name, arguments, result) -> None
            工具执行后触发。result 是工具返回值。
        on_think_end(agent, response) -> None
            think 正常结束时触发。response 是最终文本。
        on_error(agent, error, context) -> None
            think 出错时触发。context 是描述性字符串。
    """

    on_think_start: Callable[..., Any] | None = None
    on_llm_response: Callable[..., Any] | None = None
    on_tool_call: Callable[..., Any] | None = None
    on_think_end: Callable[..., Any] | None = None
    on_error: Callable[..., Any] | None = None
