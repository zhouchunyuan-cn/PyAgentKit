"""
消息协议（Message）

定义 Agent 间通信的标准消息格式：sender/receiver/content/type/metadata。
是 Router 路由与 Team 协作的数据载体。

转发链防护：消息携带 hops/max_hops。Agent 在处理消息期间调用 send() 创建的
新消息会继承当前链路深度（通过 ContextVar 传递），路由器在投递前检查跳数，
超过 max_hops 的消息被丢弃——根治 Agent 互发回复造成的无限循环。
"""

import contextvars
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# 当前转发链状态（Agent.receive 执行期间有效；Agent.send 读取它继承链路深度与预算）
# hops: 链路深度；max_hops: 该链路允许的最大跳数（由链首消息决定）
@dataclass
class _ChainState:
    hops: int
    max_hops: int


_chain_var: contextvars.ContextVar[_ChainState | None] = contextvars.ContextVar(
    "pyagentkit_message_chain", default=None
)

# 消息默认最大转发跳数（超过即丢弃，防止 A→B→A→… 无限循环）
DEFAULT_MAX_HOPS = 128


def current_message_hops() -> int:
    """读取当前转发链深度（仅在 Agent.receive 执行期间有意义）"""
    state = _chain_var.get()
    return state.hops if state is not None else 0


def set_message_hops(hops: int, max_hops: int | None = None) -> object:
    """设置当前转发链状态，返回用于恢复的 token"""
    return _chain_var.set(_ChainState(hops=hops, max_hops=max_hops or DEFAULT_MAX_HOPS))


def reset_message_hops(token: object) -> None:
    """恢复转发链状态（配合 set_message_hops 的返回值使用）"""
    _chain_var.reset(token)


class Message:
    """
    PyAgentKit中的消息类

    核心属性:
    - sender: 发送者
    - receiver: 接收者
    - content: 内容（text / action / result）
    - metadata: 元数据（timestamp, session_id 等）
    """

    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Any,
        msg_type: str = "text",
        metadata: dict | None = None,
        max_hops: int | None = None,
    ):
        """
        初始化消息

        Args:
            sender: 发送者ID
            receiver: 接收者ID
            content: 消息内容
            msg_type: 消息类型 ("text", "action", "result" 等)
            metadata: 元数据
            max_hops: 最大转发跳数上限；消息在 Agent 间每转发一次 hops 加一，
                      超过上限后路由器丢弃该消息（防无限循环）。
                      None（默认）时：若处于某条转发链中（处理消息期间创建）
                      继承该链路的预算，否则用 DEFAULT_MAX_HOPS
        """
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = msg_type
        self.timestamp = datetime.now()
        # 避免可变默认值，同时不共享调用方传入的 dict
        self.metadata = dict(metadata) if metadata else {}
        # 转发链状态：若本消息是在处理另一条消息期间创建（Agent.receive 内 send），
        # 自动继承该链路深度 +1 与跳数预算；顶层消息为 0
        state = _chain_var.get()
        self.hops = state.hops if state is not None else 0
        self.max_hops = (
            max_hops
            if max_hops is not None
            else (state.max_hops if state is not None else DEFAULT_MAX_HOPS)
        )

    def add_metadata(self, key: str, value: Any) -> None:
        """
        添加元数据

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """
        将消息转换为字典格式

        Returns:
            消息的字典表示
        """
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def get_content_size(self) -> int:
        """
        获取消息内容的大小（字节）

        Returns:
            消息内容的大小
        """
        try:
            content_str = str(self.content)
            return len(content_str.encode("utf-8"))
        except (TypeError, UnicodeEncodeError):
            return 0

    def __repr__(self) -> str:
        """
        返回消息的字符串表示
        """
        return f"Message(id={self.id[:8]}, sender={self.sender}, receiver={self.receiver}, type={self.type})"

    def __str__(self) -> str:
        """
        返回消息的详细字符串表示
        """
        return (
            f"Message {{id: {self.id[:8]}, sender: {self.sender}, receiver: {self.receiver}, "
            f"type: {self.type}, timestamp: {self.timestamp}, content_size: {self.get_content_size()} bytes}}"
        )
