"""
Agent 智能体基类

定义 PyAgentKit 中所有 Agent 的核心抽象：角色设定、记忆、工具、通信，
以及核心的 think() ReAct 推理循环（思考→调工具→观察→再思考）。
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .llm import ChatResult, LLMClient, TokenUsage
from .memory import Memory
from .memory_manager import MemoryManager
from .message import Message
from .tools import ToolRegistry
from .trace import Tracer

if TYPE_CHECKING:
    from .router import Router

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    PyAgentKit中的智能体(Agent)基类

    核心功能:
    - 角色设定（system prompt）
    - 记忆（memory）
    - 工具调用（tools）
    - 通信接口（send/receive）
    - LLM 推理（think / ReAct 循环）
    """

    # ReAct 循环最大步数，防止工具调用死循环
    DEFAULT_MAX_STEPS = 5

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str | None = None,
        llm_client: LLMClient | None = None,
        model: str = "glm-4-flash",
        max_steps: int = DEFAULT_MAX_STEPS,
        capabilities: list[str] | None = None,
        tags: list[str] | None = None,
    ):
        """
        初始化Agent

        Args:
            agent_id: Agent的唯一标识符
            name: Agent名称
            system_prompt: 系统提示词，用于角色设定
            llm_client: LLM 客户端实例，传入后 think() 才能进行推理；不传则该 Agent 不走 LLM
            model: 模型名称（保留字段，实际模型由 llm_client 决定）
            max_steps: ReAct 循环最大步数
            capabilities: 能力标签列表（如 ["search","write"]），供协作策略做能力匹配，
                          替代硬编码的 Agent 名称判断
            tags: 自由标签列表，用于更灵活的分类/筛选
        """
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.llm: LLMClient | None = llm_client
        self.model = model
        self.max_steps = max_steps
        # 能力声明：协作系统据此匹配 Agent，而非依赖硬编码 name
        self.capabilities: list[str] = list(capabilities) if capabilities else []
        self.tags: list[str] = list(tags) if tags else []
        self.memory = Memory()
        # 向量记忆（语义召回），可选注入；默认 None
        self.vector_memory = None
        # 统一记忆管理器（外观）：组合 KV + 向量 + 会话，对外一个接口
        # 现有 self.memory / self.vector_memory 作为其后端，保持向后兼容
        self.brain = MemoryManager(memory=self.memory, vector_memory=self.vector_memory)
        # token 用量累计（每次 think 累加）
        self._token_usage = TokenUsage()
        self.router: Router | None = None
        # 统一使用 ToolRegistry 管理 LLM 可调用的工具
        self.tool_registry = ToolRegistry()
        # 保留旧字段以兼容外部访问
        self.tools: dict[str, Any] = self.tool_registry.tools

    def has_capability(self, capability: str) -> bool:
        """
        判断 Agent 是否具备某项能力

        Args:
            capability: 能力名称

        Returns:
            是否具备
        """
        return capability in self.capabilities

    def has_any_capability(self, capabilities: list[str]) -> bool:
        """
        判断 Agent 是否具备给定能力中的任意一项

        Args:
            capabilities: 能力名称列表

        Returns:
            是否至少具备一项
        """
        return any(c in self.capabilities for c in capabilities)

    def set_router(self, router: "Router") -> None:
        """
        设置路由器

        Args:
            router: 路由器实例
        """
        self.router = router

    def set_vector_memory(self, vector_memory) -> None:
        """
        注入向量记忆，启用语义召回

        注入后，think() 在决策前会按当前输入语义召回相关记忆并注入上下文，
        使 Agent 能够"想起"相关的历史经验。

        Args:
            vector_memory: VectorMemory 实例
        """
        self.vector_memory = vector_memory
        # 同步更新 brain 的后端，使 build_context 能用上新的向量记忆
        self.brain.vector_memory = vector_memory

    @abstractmethod
    def receive(self, message: Message) -> None:
        """
        接收消息的抽象方法，子类必须实现

        Args:
            message: 接收到的消息
        """
        pass

    # ------------------------------------------------------------------
    # LLM 推理（ReAct 循环）
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
           - 若模型请求工具（finish_reason=="tool_calls"）：执行工具，把结果以 role="tool" 回传，继续循环
           - 若模型给出最终答复（finish_reason=="stop"）：返回内容
        3. 把本次交互摘要存入记忆

        Args:
            user_input: 当前用户/上游输入文本
            context_messages: 额外的上下文消息（如上游 Agent 传来的结构化数据），可选
            use_tools: 是否允许模型调用工具
            stream_callback: 流式回调；传入后，最终答复以流式方式产出，
                             每收到一段文本片段就回调一次（如 CLI 实时打印）。
                             仅最终答复流式，工具调用步骤仍用非流式。

        Returns:
            模型的最终文本答复
        """
        if self.llm is None:
            raise RuntimeError(
                f"Agent '{self.name}' 未配置 LLM 客户端，无法执行 think()。"
                "请在初始化时传入 llm_client。"
            )

        # 1. 组装初始消息列表
        messages: list[dict[str, Any]] = self._build_initial_messages(user_input, context_messages)

        # 工具 schema（仅在启用工具且有工具时提供）
        tools = (
            self.tool_registry.to_openai_tools()
            if (use_tools and self.tool_registry.tools)
            else None
        )

        # trace 插桩（未启用时零开销）
        _span = Tracer.start_span(
            f"think:{self.name}", agent=self.agent_id, input_preview=user_input[:50]
        )

        # 2. ReAct 循环
        for step in range(self.max_steps):
            result = self.llm.chat(messages, tools=tools, tool_choice="auto" if tools else "none")
            # 累计 token 用量
            if result.usage is not None:
                self._token_usage = self._token_usage.add(result.usage)

            if not result.tool_calls:
                # 模型给出最终答复
                logger.debug("[%s] ReAct 完成，步数=%d", self.name, step + 1)
                if stream_callback is not None:
                    # 流式重新产出最终答复，实时回调
                    final_text = self._stream_final_answer(
                        messages, tools, stream_callback, base=result
                    )
                else:
                    final_text = result.content
                self._remember_interaction(user_input, final_text)
                Tracer.end_span(_span, steps=step + 1, total_tokens=self._token_usage.total_tokens)
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
                # 工具结果存入短期记忆，便于后续步骤引用
                self.remember(f"tool:{tc.name}:{tc.id}", tool_result, memory_type="short")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        # 超过最大步数仍未结束，做一次不带工具的收尾调用以获取最终答复
        logger.warning("[%s] 达到最大步数 %d，进行收尾调用", self.name, self.max_steps)
        if stream_callback is not None:
            final_text = self._stream_final_answer(messages, None, stream_callback)
        else:
            final = self.llm.chat(messages, tools=None, tool_choice="none")
            if final.usage is not None:
                self._token_usage = self._token_usage.add(final.usage)
            final_text = final.content
        self._remember_interaction(user_input, final_text)
        Tracer.end_span(_span, steps=self.max_steps, total_tokens=self._token_usage.total_tokens)
        return final_text

    def _stream_final_answer(
        self,
        messages: list[dict[str, Any]],
        tools,
        stream_callback: Callable[[str], None],
        base: ChatResult | None = None,
    ) -> str:
        """
        流式产出最终答复，实时回调文本片段

        Args:
            messages: 当前对话消息
            tools: 工具 schema（收尾时传 None）
            stream_callback: 每段文本片段的回调
            base: 已有的非流式结果（若 LLM 不支持流式，回退用它的 content）

        Returns:
            完整的最终答复文本
        """
        collected: list[str] = []
        try:
            for chunk in self.llm.chat_stream(
                messages, tools=tools, tool_choice="auto" if tools else "none"
            ):
                if chunk.delta_content:
                    collected.append(chunk.delta_content)
                    stream_callback(chunk.delta_content)
                if chunk.usage is not None:
                    self._token_usage = self._token_usage.add(chunk.usage)
            return "".join(collected)
        except Exception as e:
            logger.warning("[%s] 流式调用失败，回退非流式: %s", self.name, e)
            if base is not None:
                stream_callback(base.content)
                return base.content
            result = self.llm.chat(messages, tools=tools, tool_choice="auto" if tools else "none")
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
        收敛到 MemoryManager.build_context，Agent 不再直接操作三套记忆。

        Args:
            user_input: 当前输入文本
            context_messages: 额外上下文消息

        Returns:
            OpenAI 风格的消息列表
        """
        return self.brain.build_context(
            user_input=user_input,
            system_prompt=self.system_prompt,
            extra_context=context_messages,
        )

    def _execute_tool_safe(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        安全执行工具调用，捕获异常避免中断循环

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果；出错时返回 {"error": ...}
        """
        try:
            return self.tool_registry.execute(tool_name, **arguments)
        except Exception as e:
            logger.warning("[%s] 工具 '%s' 执行失败: %s", self.name, tool_name, e)
            return {"error": f"工具 '{tool_name}' 执行失败: {str(e)}"}

    def _remember_interaction(self, user_input: str, response: str) -> None:
        """
        将一次完整交互存入记忆

        - 记录到会话历史（brain 的会话通道），使后续多轮对话能延续上下文
        - 同时保留一份 KV 短期记忆（向后兼容）

        Args:
            user_input: 用户输入
            response: Agent 回复
        """
        # 会话通道：记录本轮，使 build_context 下次能带上
        self.brain.record_turn("user", user_input)
        self.brain.record_turn("assistant", response)
        # KV 通道：保留交互快照（向后兼容）
        self.brain.remember(
            f"interaction:{len(self.memory.list_short_term_memory())}",
            {"input": user_input, "response": response},
            memory_type="short",
        )

    # ------------------------------------------------------------------
    # 通信（保留原有接口）
    # ------------------------------------------------------------------

    def send(
        self, receiver_id: str, content: Any, msg_type: str = "text", metadata: dict | None = None
    ) -> None:
        """
        发送消息给其他Agent

        Args:
            receiver_id: 接收者ID
            content: 消息内容
            msg_type: 消息类型
            metadata: 消息元数据
        """
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )

        # 如果设置了路由器，则自动路由消息
        if self.router:
            self.router.route_message(message)
        else:
            # 否则返回消息对象以便手动路由
            return message

    async def async_send(
        self, receiver_id: str, content: Any, msg_type: str = "text", metadata: dict | None = None
    ) -> None:
        """
        异步发送消息（AsyncRouter 场景使用）

        与 send() 平行，但把消息投递给 AsyncRouter（await route），
        使接收方在消费协程中处理，打破同步递归。

        Args:
            receiver_id: 接收者ID
            content: 消息内容
            msg_type: 消息类型
            metadata: 消息元数据
        """
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )
        if self.router is not None and hasattr(self.router, "route"):
            # AsyncRouter 提供 async route
            await self.router.route(message)
        else:
            # 无异步路由器，回退同步发送
            self.send(receiver_id, content, msg_type, metadata)

    def broadcast(self, content: Any, msg_type: str = "text", metadata: dict | None = None) -> None:
        """
        广播消息给所有Agent

        Args:
            content: 消息内容
            msg_type: 消息类型
            metadata: 消息元数据
        """
        if self.router:
            message = Message(
                sender=self.agent_id,
                receiver="all",  # 特殊接收者表示广播
                content=content,
                msg_type=msg_type,
                metadata=metadata,
            )
            self.router.broadcast(message)
        else:
            raise RuntimeError("Router not set. Cannot broadcast without a router.")

    # ------------------------------------------------------------------
    # 工具与记忆（保留并增强原有接口）
    # ------------------------------------------------------------------

    def add_tool(self, name: str, tool: Any) -> None:
        """
        添加工具到Agent

        Args:
            name: 工具名称
            tool: 工具对象
        """
        # 统一通过 ToolRegistry 注册，保留同名兼容
        self.tool_registry.register(tool)
        self.tools = self.tool_registry.tools

    def get_tool(self, name: str) -> Any:
        """
        获取工具

        Args:
            name: 工具名称

        Returns:
            工具对象
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """
        列出所有可用工具

        Returns:
            工具名称列表
        """
        return list(self.tools.keys())

    def remember(self, key: str, value: Any, memory_type: str = "short") -> None:
        """
        将信息存入记忆

        Args:
            key: 记忆键
            value: 记忆值
            memory_type: 记忆类型 ("short" 或 "long")
        """
        self.memory.store(key, value, memory_type)

    def recall(self, key: str) -> Any:
        """
        从记忆中检索信息

        Args:
            key: 记忆键

        Returns:
            记忆值
        """
        return self.memory.retrieve(key)

    # ------------------------------------------------------------------
    # Token 统计与异步支持
    # ------------------------------------------------------------------

    def get_token_usage(self) -> TokenUsage:
        """
        获取本 Agent 累计的 token 用量（跨多次 think 调用）

        Returns:
            TokenUsage（prompt/completion/total 累计值）
        """
        return self._token_usage

    def reset_token_usage(self) -> None:
        """重置 token 用量统计"""
        self._token_usage = TokenUsage()

    async def receive_async(self, message: Message) -> None:
        """
        异步接收消息（AsyncRouter 优先调用此方法）

        默认实现回退到同步 receive()，在阻塞线程池中执行。
        子类可覆盖为真正的 async 实现，以获得并发收益。
        """
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.receive, message)
