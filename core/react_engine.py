"""
ReAct 推理引擎（ReactEngine）

从 Agent 基类抽出的推理逻辑，集中管理 ReAct 循环、流式输出、工具执行、
交互记忆与 token 统计。Agent 通过委托（`self._engine.think(...)`）调用，
对外接口完全不变。

设计目的：
- 让 Agent 基类聚焦"身份 + 通信 + 工具"，推理细节内聚到 engine
- 便于后续在此处扩展错误恢复、重试策略、生命周期钩子

流式（2026 重构）：
- 只要调用方给了 stream_callback / events_callback，主循环的每一步 LLM 调用
  都走 chat_stream 真流式：文本/思考增量实时产出，首 token 延迟≈模型出字时间。
- events_callback 接收结构化 AgentStreamEvent（reasoning/text/tool_call/tool_result），
  前端可据此渲染"正在思考/正在调用工具"等状态。
- stream_callback（兼容接口）实时接收助手文本增量。
- 混合推理模型的 reasoning_content 会随 assistant 消息回传，保持多轮推理连贯。
"""

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .llm import ChatResult, TokenUsage, ToolCall
from .trace import Tracer

logger = logging.getLogger(__name__)


@dataclass
class AgentStreamEvent:
    """
    Agent 执行过程的流式事件（结构化，供前端实时渲染）

    type 及 content 含义：
        "reasoning"   content: str，思考过程增量（推理模型）
        "text"        content: str，助手文本增量（含工具调用前的叙述性文字）
        "tool_call"   content: {"name": str, "arguments": dict}
        "tool_result" content: {"name": str, "result": Any}
    """

    type: str
    content: Any
    step: int = 0


def _emit_via_callback(text: str, callback: Callable[[str], None], chunk_size: int = 8) -> None:
    """
    将已有文本按固定大小切分，通过回调"模拟流式"逐段输出

    仅用于降级路径（LLM 调用失败后基于已有上下文的无工具收尾回复）——
    此时答复来自非流式调用，仍以流式体验呈现，无需重复调用 LLM。

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

    # 同一步内多个工具调用的最大并发数
    MAX_PARALLEL_TOOLS = 4

    def __init__(self, agent: Any) -> None:
        """
        Args:
            agent: 宿主 Agent 实例（弱引用，不拥有）
        """
        self.agent = agent
        # token 用量累计（每次 think 累加），供 Agent.get_token_usage() 读取
        self._token_usage = TokenUsage()
        # 交互序号：KV 快照 key 用递增序号而非"当前条数"，
        # 避免短期记忆过期收缩后 key 碰撞互相覆盖
        self._interaction_seq = 0
        # 最近一次 think 是否因 token 预算耗尽而提前收尾（供成本监控消费）
        self.last_run_budget_exceeded = False

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def think(
        self,
        user_input: str,
        context_messages: list[dict[str, Any]] | None = None,
        use_tools: bool = True,
        stream_callback: Callable[[str], None] | None = None,
        events_callback: Callable[[AgentStreamEvent], None] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """
        ReAct 推理主循环

        流程：
        1. 组装对话：system_prompt + 相关长期记忆 + 历史 + 当前输入
        2. 循环调用 LLM（有回调时走真流式）：
           - 若模型请求工具：执行工具，把结果以 role="tool" 回传，继续循环
           - 若模型给出最终答复：返回内容
        3. 把本次交互摘要存入记忆

        Args:
            user_input: 当前用户/上游输入文本
            context_messages: 额外的上下文消息（如上游 Agent 传来的结构化数据），可选
            use_tools: 是否允许模型调用工具
            stream_callback: 流式回调（兼容接口）：助手文本增量实时回调。
                             注意文本可能在工具调用前就有叙述性输出，属正常流式语义
            events_callback: 结构化事件回调：接收 AgentStreamEvent
                             （reasoning/text/tool_call/tool_result），两者可同时使用
            response_format: 结构化输出约束（如 {"type": "json_object"}），
                             仅在需要确定性 JSON 输出时使用（非流式路径生效）

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
        # 本次 think 的 token 差值基线（trace 里记录本次用量而非历史累计值）
        _tokens_at_start = self._token_usage.total_tokens

        self.last_run_budget_exceeded = False

        # 只要有任一回调，就走真流式主路径
        streaming = stream_callback is not None or events_callback is not None

        def emit_event(ev_type: str, content: Any, step: int = 0) -> None:
            if events_callback is None:
                return
            try:
                events_callback(AgentStreamEvent(type=ev_type, content=content, step=step))
            except Exception as e:
                # 事件回调异常不干扰推理主流程
                logger.warning("[%s] events_callback 异常: %s", agent.name, e)

        # 2. ReAct 循环（token 预算耗尽则提前收尾，不再发起新的工具循环步）
        for step in range(agent.max_steps):
            if self._budget_exceeded():
                logger.warning(
                    "[%s] 累计 token %d 已达预算上限 %d，提前收尾",
                    agent.name,
                    self._token_usage.total_tokens,
                    agent.token_budget,
                )
                self.last_run_budget_exceeded = True
                break
            try:
                if streaming:
                    result = self._chat_streaming(
                        messages, tools, step, stream_callback, emit_event
                    )
                elif response_format is not None:
                    # 结构化输出：仅在显式要求时传该参数（兼容不支持它的实现）
                    result = agent.llm.chat(
                        messages,
                        tools=tools,
                        tool_choice="auto" if tools else "none",
                        response_format=response_format,
                    )
                else:
                    result = agent.llm.chat(
                        messages, tools=tools, tool_choice="auto" if tools else "none"
                    )
            except TypeError:
                # 接口不兼容（如客户端不支持 response_format）：重试无意义，直接上抛，
                # 供调用方（如 HierarchicalProcess._plan）按客户端能力降级
                raise
            except Exception as e:
                # LLM 调用失败（GLMClient 内部重试已耗尽）：尝试优雅降级
                logger.warning("[%s] 第 %d 步 LLM 调用失败: %s", agent.name, step + 1, e)
                # on_error 钩子
                self._fire_hook("on_error", agent, e, f"第{step + 1}步 LLM 调用失败")
                partial = self._recover_from_llm_error(
                    messages,
                    user_input,
                    stream_callback,
                    _span,
                    step,
                    tools,
                    caller_manages_history=context_messages is not None,
                    tokens_at_start=_tokens_at_start,
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
                # 模型给出最终答复（流式路径已在产出时实时回调，无需再补发）
                logger.debug("[%s] ReAct 完成，步数=%d", agent.name, step + 1)
                final_text = result.content
                self._remember_interaction(
                    user_input, final_text, caller_manages_history=context_messages is not None
                )
                Tracer.end_span(
                    _span,
                    steps=step + 1,
                    total_tokens=self._token_usage.total_tokens - _tokens_at_start,
                )
                # on_think_end 钩子
                self._fire_hook("on_think_end", agent, final_text)
                return final_text

            # 把助手的工具调用请求加入对话历史
            assistant_msg: dict[str, Any] = {
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
            # 混合推理模型：回传思考内容，保持多轮推理连贯（Preserved Thinking）
            if result.reasoning_content:
                assistant_msg["reasoning_content"] = result.reasoning_content
            messages.append(assistant_msg)

            # 执行工具，把结果作为 tool 消息回传（同一批多个工具并发执行）
            tool_results = self._execute_tool_calls(result.tool_calls, step, emit_event)
            for tc, tool_result in zip(result.tool_calls, tool_results, strict=True):
                emit_event("tool_result", {"name": tc.name, "result": tool_result}, step + 1)
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

        # 循环耗尽（最大步数或 token 预算）：做一次不带工具的收尾调用以获取最终答复
        reason = "预算耗尽" if self.last_run_budget_exceeded else f"达到最大步数 {agent.max_steps}"
        logger.warning("[%s] %s，进行收尾调用", agent.name, reason)
        if streaming:
            final_result = self._chat_streaming(
                messages, None, agent.max_steps, stream_callback, emit_event
            )
            final_text = final_result.content
        else:
            final = agent.llm.chat(messages, tools=None, tool_choice="none")
            if final.usage is not None:
                self._token_usage = self._token_usage.add(final.usage)
            final_text = final.content
        self._remember_interaction(
            user_input, final_text, caller_manages_history=context_messages is not None
        )
        span_attrs: dict[str, Any] = {
            "steps": agent.max_steps,
            "total_tokens": self._token_usage.total_tokens - _tokens_at_start,
        }
        if self.last_run_budget_exceeded:
            span_attrs["budget_exceeded"] = True
        Tracer.end_span(_span, **span_attrs)
        # on_think_end 钩子
        self._fire_hook("on_think_end", agent, final_text)
        return final_text

    # ------------------------------------------------------------------
    # 流式调用（主路径）
    # ------------------------------------------------------------------
    def _chat_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        step: int,
        stream_callback: Callable[[str], None] | None,
        emit_event: Callable[[str, Any, int], None],
    ) -> ChatResult:
        """
        流式执行一步 LLM 调用，边收边回调，最终拼装为 ChatResult

        - 文本增量 → stream_callback + "text" 事件
        - 思考增量 → "reasoning" 事件
        - 工具调用碎片按 index 累积拼装为 ToolCall 列表
        - 流在未产出任何内容时失败 → 安全回退一次非流式调用；
          已产出内容后失败 → 返回已产出部分（不重复调用避免内容重复）
        """
        agent = self.agent
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_acc: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None
        finish = "stop"
        emitted = False

        try:
            for chunk in agent.llm.chat_stream(
                messages, tools=tools, tool_choice="auto" if tools else "none"
            ):
                if chunk.delta_reasoning:
                    reasoning_parts.append(chunk.delta_reasoning)
                    emit_event("reasoning", chunk.delta_reasoning, step + 1)
                    emitted = True
                if chunk.delta_content:
                    text_parts.append(chunk.delta_content)
                    emitted = True
                    if stream_callback is not None:
                        stream_callback(chunk.delta_content)
                    emit_event("text", chunk.delta_content, step + 1)
                for frag in getattr(chunk, "delta_tool_calls", None) or []:
                    idx = frag.get("index", 0) or 0
                    slot = tc_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if frag.get("id"):
                        slot["id"] = frag["id"]
                    if frag.get("name"):
                        slot["name"] = frag["name"]
                    if frag.get("arguments"):
                        slot["arguments"] += frag["arguments"]
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.finish_reason:
                    finish = chunk.finish_reason
        except Exception as e:
            if not emitted:
                logger.warning("[%s] 流式调用失败（未产出内容），回退非流式: %s", agent.name, e)
                return agent.llm.chat(
                    messages, tools=tools, tool_choice="auto" if tools else "none"
                )
            logger.warning("[%s] 流式中途失败，返回已产出内容: %s", agent.name, e)
            finish = "error_stream"

        # 拼装工具调用（arguments 是分片送达的 JSON 字符串）
        tool_calls: list[ToolCall] = []
        for idx in sorted(tc_acc):
            slot = tc_acc[idx]
            raw_args = slot["arguments"] or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(id=slot["id"] or "", name=slot["name"] or "", arguments=arguments)
            )
        return ChatResult(
            content="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            reasoning_content="".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _budget_exceeded(self) -> bool:
        """token 预算检查（未设置预算视为不限制；按实例生命周期累计用量）"""
        budget = getattr(self.agent, "token_budget", None)
        return budget is not None and self._token_usage.total_tokens >= budget

    def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        step: int,
        emit_event: Callable[[str, Any, int], None],
    ) -> list[Any]:
        """
        执行一批工具调用：单个直接执行；多个用线程池并发执行

        结果按提交顺序返回（对话协议里 tool 消息与 tool_calls 一一对应，
        顺序确定性不受并发影响）。tool_call 事件按序先行宣告，
        tool_result 事件在结果收集后按序补发。

        Args:
            tool_calls: 本步的全部工具调用请求
            step: 当前 ReAct 步数（0 基）
            emit_event: 结构化事件发射器

        Returns:
            与 tool_calls 等长、按序对应的结果列表
        """
        if not tool_calls:
            return []
        for tc in tool_calls:
            emit_event("tool_call", {"name": tc.name, "arguments": tc.arguments}, step + 1)

        if len(tool_calls) == 1:
            return [self._execute_tool_safe(tool_calls[0].name, tool_calls[0].arguments)]

        max_workers = min(self.MAX_PARALLEL_TOOLS, len(tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool") as pool:
            futures = [
                pool.submit(self._execute_tool_safe, tc.name, tc.arguments) for tc in tool_calls
            ]
            return [f.result() for f in futures]

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
        caller_manages_history: bool = False,
        tokens_at_start: int = 0,
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
            # 与正常路径一致：外部 Session 管理历史时不再写入会话通道，避免重复
            self._remember_interaction(user_input, final_text, caller_manages_history)
            Tracer.end_span(
                _span,
                steps=failed_step + 1,
                total_tokens=self._token_usage.total_tokens - tokens_at_start,
                recovered=True,
            )
            return final_text
        except Exception:
            # 降级调用也失败，放弃
            return None

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
        # KV 通道：保留交互快照（向后兼容）；key 用递增序号防碰撞
        self._interaction_seq += 1
        agent.brain.remember(
            f"interaction:{self._interaction_seq}",
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
