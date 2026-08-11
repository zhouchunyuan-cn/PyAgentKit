"""
ReAct 推理引擎（ReactEngine）

从 Agent 基类抽出的推理逻辑，集中管理 ReAct 循环、流式输出、工具执行、
交互记忆与 token 统计。Agent 通过委托（`self._engine.think(...)`）调用，
对外接口完全不变。

设计目的：
- 让 Agent 基类聚焦"身份 + 通信 + 工具"，推理细节内聚到 engine
- 便于后续在此处扩展错误恢复、重试策略、生命周期钩子
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from .llm import TokenUsage
from .trace import Tracer

logger = logging.getLogger(__name__)


def _emit_via_callback(text: str, callback: Callable[[str], None], chunk_size: int = 8) -> None:
    """
    将已有文本按固定大小切分，通过回调"模拟流式"逐段输出

    用于 think() 已通过非流式拿到最终答复后，仍能以流式体验呈现给用户，
    而无需重复调用 LLM（避免双调用的 token 浪费）。

    Args:
        text: 已有的完整文本
        callback: 每段文本的回调
        chunk_size: 每段字符数（默认 8，平衡流畅度与调用开销）
    """
    for i in range(0, len(text), chunk_size):
        callback(text[i : i + chunk_size])


class ReactEngine:
    """
    Agent 的 ReAct 推理引擎

    通过 `self.agent` 弱引用宿主 Agent，访问其 llm / tool_registry / brain /
    memory / system_prompt / name / agent_id / max_steps 等属性。
    token 统计由 engine 内部管理，Agent 的 get_token_usage() 委托读取。
    """

    def __init__(self, agent: Any) -> None:
        """
        Args:
            agent: 宿主 Agent 实例（弱引用，不拥有）
        """
        self.agent = agent
        # token 用量累计（每次 think 累加），供 Agent.get_token_usage() 读取
        self._token_usage = TokenUsage()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def think(
        self,
        user_input: str,
        context_messages: list[dict[str, Any]] | None = None,
        use_tools: bool = True,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        """
        ReAct 推理主循环

        流程：
        1. 组装对话：system_prompt + 相关长期记忆 + 历史 + 当前输入
        2. 循环调用 LLM：
           - 若模型请求工具：执行工具，把结果以 role="tool" 回传，继续循环
           - 若模型给出最终答复：返回内容
        3. 把本次交互摘要存入记忆

        Args:
            user_input: 当前用户/上游输入文本
            context_messages: 额外的上下文消息（如上游 Agent 传来的结构化数据），可选
            use_tools: 是否允许模型调用工具
            stream_callback: 流式回调；传入后，最终答复以流式方式产出。

        Returns:
            模型的最终文本答复
        """
        agent = self.agent
        if agent.llm is None:
            raise RuntimeError(
                f"Agent '{agent.name}' 未配置 LLM 客户端，无法执行 think()。"
                "请在初始化时传入 llm_client。"
            )

        # on_think_start 钩子（返回 False 可中止，用于 HITL 审批）
        if agent.hooks.on_think_start is not None:
            try:
                if agent.hooks.on_think_start(agent, user_input) is False:
                    logger.info("[%s] think 被 on_think_start 钩子中止", agent.name)
                    return ""
            except Exception as e:
                logger.warning("[%s] on_think_start 钩子异常: %s", agent.name, e)

        # 1. 组装初始消息列表
        messages = self._build_initial_messages(user_input, context_messages)

        # 工具 schema（仅在启用工具且有工具时提供）
        tools = (
            agent.tool_registry.to_openai_tools()
            if (use_tools and agent.tool_registry.tools)
            else None
        )

        # trace 插桩（未启用时零开销）
        _span = Tracer.start_span(
            f"think:{agent.name}", agent=agent.agent_id, input_preview=user_input[:50]
        )

        # 2. ReAct 循环
        for step in range(agent.max_steps):
            try:
                result = agent.llm.chat(
                    messages, tools=tools, tool_choice="auto" if tools else "none"
                )
            except Exception as e:
                # LLM 调用失败（GLMClient 内部重试已耗尽）：尝试优雅降级
                logger.warning("[%s] 第 %d 步 LLM 调用失败: %s", agent.name, step + 1, e)
                # on_error 钩子
                self._fire_hook("on_error", agent, e, f"第{step + 1}步 LLM 调用失败")
                partial = self._recover_from_llm_error(
                    messages, user_input, stream_callback, _span, step, tools
                )
                if partial is not None:
                    return partial
                # 第一步就失败且无法降级，抛出带上下文的错误
                raise RuntimeError(
                    f"Agent '{agent.name}' think() 失败（第 {step + 1} 步）：{e}"
                ) from e
            # 累计 token 用量
            if result.usage is not None:
                self._token_usage = self._token_usage.add(result.usage)

            # on_llm_response 钩子
            self._fire_hook("on_llm_response", agent, result, step + 1)

            if not result.tool_calls:
                # 模型给出最终答复（直接用本次结果，不再重复调用 LLM）
                logger.debug("[%s] ReAct 完成，步数=%d", agent.name, step + 1)
                final_text = result.content
                if stream_callback is not None and final_text:
                    _emit_via_callback(final_text, stream_callback)
                self._remember_interaction(
                    user_input, final_text, caller_manages_history=context_messages is not None
                )
                Tracer.end_span(_span, steps=step + 1, total_tokens=self._token_usage.total_tokens)
                # on_think_end 钩子
                self._fire_hook("on_think_end", agent, final_text)
                return final_text

            # 把助手的工具调用请求加入对话历史
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in result.tool_calls
                    ],
                }
            )

            # 逐个执行工具，把结果作为 tool 消息回传
            for tc in result.tool_calls:
                tool_result = self._execute_tool_safe(tc.name, tc.arguments)
                # on_tool_call 钩子
                self._fire_hook("on_tool_call", agent, tc.name, tc.arguments, tool_result)
                # 工具结果存入短期记忆，便于后续步骤引用
                agent.remember(f"tool:{tc.name}:{tc.id}", tool_result, memory_type="short")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        # 超过最大步数仍未结束，做一次不带工具的收尾调用以获取最终答复
        logger.warning("[%s] 达到最大步数 %d，进行收尾调用", agent.name, agent.max_steps)
        if stream_callback is not None:
            final_text = self._stream_fallback(messages, stream_callback)
        else:
            final = agent.llm.chat(messages, tools=None, tool_choice="none")
            if final.usage is not None:
                self._token_usage = self._token_usage.add(final.usage)
            final_text = final.content
        self._remember_interaction(
            user_input, final_text, caller_manages_history=context_messages is not None
        )
        Tracer.end_span(_span, steps=agent.max_steps, total_tokens=self._token_usage.total_tokens)
        # on_think_end 钩子
        self._fire_hook("on_think_end", agent, final_text)
        return final_text

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fire_hook(self, hook_name: str, *args, **kwargs) -> None:
        """
        安全触发生命周期钩子

        钩子未设置或抛异常都不影响主流程。
        """
        agent = self.agent
        hook = getattr(agent.hooks, hook_name, None)
        if hook is None:
            return
        try:
            hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s 钩子异常: %s", agent.name, hook_name, e)

    def _recover_from_llm_error(
        self,
        messages: list[dict[str, Any]],
        user_input: str,
        stream_callback: Callable[[str], None] | None,
        _span: Any,
        failed_step: int,
        tools: Any,
    ) -> str | None:
        """
        LLM 调用失败时的优雅降级

        策略：如果之前已有工具执行结果（messages 里含 tool 消息），
        尝试做一次无工具的收尾调用，基于已有信息给出"尽力而为"的答复。
        如果第一步就失败（无任何上下文），返回 None 让调用方抛错。

        Returns:
            降级答复文本；None 表示无法降级
        """
        agent = self.agent
        has_context = any(m.get("role") == "tool" for m in messages)
        if not has_context:
            return None

        logger.info("[%s] 尝试基于已有上下文降级回复", agent.name)
        try:
            final = agent.llm.chat(messages, tools=None, tool_choice="none")
            if final.usage is not None:
                self._token_usage = self._token_usage.add(final.usage)
            final_text = final.content
            if stream_callback is not None and final_text:
                _emit_via_callback(final_text, stream_callback)
            self._remember_interaction(user_input, final_text)
            Tracer.end_span(
                _span, steps=failed_step + 1, total_tokens=self._token_usage.total_tokens, recovered=True
            )
            return final_text
        except Exception:
            # 降级调用也失败，放弃
            return None

    def _stream_fallback(
        self,
        messages: list[dict[str, Any]],
        stream_callback: Callable[[str], None],
    ) -> str:
        """
        收尾流式调用（单次，不重复）：用 chat_stream 实时产出最终答复

        仅在 max_steps 耗尽后的收尾调用中使用——此时只调一次 LLM，
        不存在双调用问题。流式失败则回退非流式。
        """
        agent = self.agent
        collected: list[str] = []
        try:
            for chunk in agent.llm.chat_stream(messages, tools=None, tool_choice="none"):
                if chunk.delta_content:
                    collected.append(chunk.delta_content)
                    stream_callback(chunk.delta_content)
                if chunk.usage is not None:
                    self._token_usage = self._token_usage.add(chunk.usage)
            return "".join(collected)
        except Exception as e:
            logger.warning("[%s] 收尾流式失败，回退非流式: %s", agent.name, e)
            result = agent.llm.chat(messages, tools=None, tool_choice="none")
            if result.usage is not None:
                self._token_usage = self._token_usage.add(result.usage)
            stream_callback(result.content)
            return result.content

    def _build_initial_messages(
        self,
        user_input: str,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        组装初始对话消息（委托给统一记忆管理器）

        把 system_prompt + 长期记忆 + 向量召回 + 会话历史 + 输入 聚合的逻辑
        收敛到 MemoryManager.build_context。
        """
        agent = self.agent
        return agent.brain.build_context(
            user_input=user_input,
            system_prompt=agent.system_prompt,
            extra_context=context_messages,
        )

    def _execute_tool_safe(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        安全执行工具调用，捕获异常避免中断循环

        Returns:
            工具执行结果；出错时返回 {"error": ...}
        """
        agent = self.agent
        try:
            return agent.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            logger.warning("[%s] 工具 '%s' 执行失败: %s", agent.name, tool_name, e)
            return {"error": f"工具 '{tool_name}' 执行失败: {str(e)}"}

    def _remember_interaction(
        self, user_input: str, response: str, caller_manages_history: bool = False
    ) -> None:
        """
        将一次完整交互存入记忆

        - 若 caller_manages_history=False（think 直接调用，无 Session）：
          记录到 brain 的会话通道，使后续 think 能延续上下文
        - 若 caller_manages_history=True（Session 调用，已通过 context_messages 传入历史）：
          跳过会话通道，避免历史重复
        - 两种情况都保留 KV 短期记忆快照（向后兼容）
        """
        agent = self.agent
        if not caller_manages_history:
            # 会话通道：记录本轮（仅在无外部 Session 管理历史时）
            agent.brain.record_turn("user", user_input)
            agent.brain.record_turn("assistant", response)
        # KV 通道：保留交互快照（向后兼容）
        agent.brain.remember(
            f"interaction:{len(agent.memory.list_short_term_memory())}",
            {"input": user_input, "response": response},
            memory_type="short",
        )

    # ------------------------------------------------------------------
    # Token 统计（由 Agent 委托调用）
    # ------------------------------------------------------------------

    def get_token_usage(self) -> TokenUsage:
        """获取累计的 token 用量"""
        return self._token_usage

    def reset_token_usage(self) -> None:
        """重置 token 用量统计"""
        self._token_usage = TokenUsage()
