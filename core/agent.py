from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING, List
from .message import Message
from .memory import Memory

if TYPE_CHECKING:
    from .router import Router


class Agent(ABC):
    """
    PyAgentKit中的智能体(Agent)基类
    
    核心功能:
    - 角色设定（system prompt）
    - 记忆（memory）
    - 工具调用（tools）
    - 通信接口（send/receive）
    """

    def __init__(self, agent_id: str, name: str, system_prompt: Optional[str] = None):
        """
        初始化Agent
        
        Args:
            agent_id: Agent的唯一标识符
            name: Agent名称
            system_prompt: 系统提示词，用于角色设定
        """
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.memory = Memory()
        self.router: Optional['Router'] = None
        self.tools: Dict[str, Any] = {}

    def set_router(self, router: 'Router') -> None:
        """
        设置路由器
        
        Args:
            router: 路由器实例
        """
        self.router = router

    @abstractmethod
    def receive(self, message: Message) -> None:
        """
        接收消息的抽象方法，子类必须实现
        
        Args:
            message: 接收到的消息
        """
        pass

    def send(self, receiver_id: str, content: Any, msg_type: str = "text", metadata: Optional[Dict] = None) -> None:
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
            metadata=metadata
        )
        
        # 如果设置了路由器，则自动路由消息
        if self.router:
            self.router.route_message(message)
        else:
            # 否则返回消息对象以便手动路由
            return message

    def broadcast(self, content: Any, msg_type: str = "text", metadata: Optional[Dict] = None) -> None:
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
                metadata=metadata
            )
            self.router.broadcast(message)
        else:
            raise RuntimeError("Router not set. Cannot broadcast without a router.")

    def add_tool(self, name: str, tool: Any) -> None:
        """
        添加工具到Agent
        
        Args:
            name: 工具名称
            tool: 工具对象
        """
        self.tools[name] = tool

    def get_tool(self, name: str) -> Any:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具对象
        """
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
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