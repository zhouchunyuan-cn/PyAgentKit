"""
Agent 智能体基类

定义 PyAgentKit 中所有 Agent 的核心抽象：角色设定、记忆、工具、通信。
ReAct 推理循环委托给 ReactEngine（core/react_engine.py），保持对外接口不变。

职责划分：
- Agent：身份（id/name/capabilities）、通信（send/receive）、工具管理、记忆存取
- ReactEngine：ReAct 推理循环（think）、流式输出、token 统计
"""

import asyncio
import contextvars
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .hooks import AgentHooks
from .llm import LLMClient, TokenUsage
from .memory import Memory
from .memory_manager import MemoryManager
from .message import Message
from .react_engine import ReactEngine
from .tools import ToolRegistry

if TYPE_CHECKING:
    from .router import Router

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    PyAgentKit中的智能体(Agent)基类

    核心功能:
    - 角色设定（system prompt）
    - 记忆（memory / brain）
    - 工具调用（tools）
    - 通信接口（send/receive）
    - LLM 推理（think，委托给 ReactEngine）
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
        hooks: AgentHooks | None = None,
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
        # KV 记忆：默认不落盘（纯内存），需持久化时调 enable_persistence()
        self.memory = Memory()
        self.vector_memory = None
        # 统一记忆管理器（外观）：组合 KV + 向量 + 会话，对外一个接口
        # 现有 self.memory / self.vector_memory 作为其后端，保持向后兼容
        self.brain = MemoryManager(memory=self.memory, vector_memory=self.vector_memory)
        self.router: Router | None = None
        # 统一使用 ToolRegistry 管理 LLM 可调用的工具
        self.tool_registry = ToolRegistry()
        # 保留旧字段以兼容外部访问（monitor、测试等读此属性）
        self.tools: dict[str, Any] = self.tool_registry.tools
        # ReAct 推理引擎（委托模式：think 等推理逻辑内聚于此）
        self._engine = ReactEngine(self)
        # 生命周期钩子（可选，默认无钩子）
        self.hooks: AgentHooks = hooks or AgentHooks()

    # ------------------------------------------------------------------
    # 身份与能力
    # ------------------------------------------------------------------

    def has_capability(self, capability: str) -> bool:
        """判断 Agent 是否具备某项能力"""
        return capability in self.capabilities

    def has_any_capability(self, capabilities: list[str]) -> bool:
        """判断 Agent 是否具备给定能力中的任意一项"""
        return any(c in self.capabilities for c in capabilities)

    # ------------------------------------------------------------------
    # 注入
    # ------------------------------------------------------------------

    def set_router(self, router: "Router") -> None:
        """设置路由器"""
        self.router = router

    def set_vector_memory(self, vector_memory) -> None:
        """
        注入向量记忆，启用语义召回

        注入后同步更新 brain 的后端，使 build_context 能用上新的向量记忆。
        """
        self.vector_memory = vector_memory
        self.brain.vector_memory = vector_memory

    def enable_persistence(self, path: str | None = None) -> None:
        """
        启用 KV 记忆的持久化（落盘）

        默认 Agent 的记忆是纯内存的（重启丢失）。调用此方法后，
        长期记忆会写入按 agent_id 命名的 JSON 文件，重启后恢复。

        Args:
            path: 自定义文件路径；None 则用 memory_{agent_id}.json
        """
        filepath = path or f"memory_{self.agent_id}.json"
        self.memory.persistent_storage_path = filepath
        self.memory._load_persistent_memory()  # 加载已有记忆

    # ------------------------------------------------------------------
    # 消息接收（子类契约）
    # ------------------------------------------------------------------

    @abstractmethod
    def receive(self, message: Message) -> None:
        """接收消息的抽象方法，子类必须实现"""
        pass

    # ------------------------------------------------------------------
    # LLM 推理（委托给 ReactEngine）
    # ------------------------------------------------------------------

    def think(
        self,
        user_input: str,
        context_messages: list[dict[str, Any]] | None = None,
        use_tools: bool = True,
        stream_callback: Any = None,
    ) -> str:
        """
        ReAct 推理主循环（委托给 ReactEngine）

        详见 ReactEngine.think() 的实现。对外接口保持不变。

        Args:
            user_input: 当前用户/上游输入文本
            context_messages: 额外的上下文消息，可选
            use_tools: 是否允许模型调用工具
            stream_callback: 流式回调；传入后最终答复以流式方式产出

        Returns:
            模型的最终文本答复
        """
        return self._engine.think(user_input, context_messages, use_tools, stream_callback)

    # ------------------------------------------------------------------
    # 通信
    # ------------------------------------------------------------------

    def send(
        self, receiver_id: str, content: Any, msg_type: str = "text", metadata: dict | None = None
    ) -> Message | None:
        """发送消息给其他 Agent（通过 Router 路由）

        Returns:
            已路由时返回 None（由路由器负责投递）；未设置 Router 时
            返回构建好的 Message 交由调用方自行处理
        """
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )
        if self.router:
            self.router.route_message(message)
        else:
            return message

    async def async_send(
        self, receiver_id: str, content: Any, msg_type: str = "text", metadata: dict | None = None
    ) -> None:
        """异步发送消息（AsyncRouter 场景使用）"""
        message = Message(
            sender=self.agent_id,
            receiver=receiver_id,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )
        if self.router is not None and hasattr(self.router, "route"):
            await self.router.route(message)
        else:
            self.send(receiver_id, content, msg_type, metadata)

    def broadcast(self, content: Any, msg_type: str = "text", metadata: dict | None = None) -> None:
        """广播消息给所有 Agent"""
        if self.router:
            message = Message(
                sender=self.agent_id,
                receiver="all",
                content=content,
                msg_type=msg_type,
                metadata=metadata,
            )
            self.router.broadcast(message)
        else:
            raise RuntimeError("Router not set. Cannot broadcast without a router.")

    # ------------------------------------------------------------------
    # 工具管理
    # ------------------------------------------------------------------

    def add_tool(self, name: str, tool: Any) -> None:
        """添加工具到 Agent"""
        self.tool_registry.register(tool)
        self.tools = self.tool_registry.tools

    def get_tool(self, name: str) -> Any:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有可用工具名称"""
        return list(self.tools.keys())

    # ------------------------------------------------------------------
    # 记忆存取（委托给 KV 记忆后端）
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, memory_type: str = "short") -> None:
        """将信息存入 KV 记忆"""
        self.memory.store(key, value, memory_type)

    def recall(self, key: str) -> Any:
        """从 KV 记忆中检索信息"""
        return self.memory.retrieve(key)

    # ------------------------------------------------------------------
    # Token 统计（委托给 ReactEngine）
    # ------------------------------------------------------------------

    def get_token_usage(self) -> TokenUsage:
        """获取本 Agent 累计的 token 用量（跨多次 think 调用）"""
        return self._engine.get_token_usage()

    def reset_token_usage(self) -> None:
        """重置 token 用量统计"""
        self._engine.reset_token_usage()

    # ------------------------------------------------------------------
    # 异步支持
    # ------------------------------------------------------------------

    async def receive_async(self, message: Message) -> None:
        """
        异步接收消息（AsyncRouter 优先调用此方法）

        默认实现回退到同步 receive()，在阻塞线程池中执行。
        执行前复制当前上下文（contextvars），使消息跳数继承与 Tracer
        追踪在执行器线程中依然生效。
        子类可覆盖为真正的 async 实现，以获得并发收益。
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        await loop.run_in_executor(None, lambda: ctx.run(self.receive, message))
