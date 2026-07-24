"""
消息协议（Message）

定义 Agent 间通信的标准消息格式：sender/receiver/content/type/metadata。
是 Router 路由与 Team 协作的数据载体。
"""

import uuid
from datetime import datetime
from typing import Any


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
    ):
        """
        初始化消息

        Args:
            sender: 发送者ID
            receiver: 接收者ID
            content: 消息内容
            msg_type: 消息类型 ("text", "action", "result" 等)
            metadata: 元数据
        """
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = msg_type
        self.timestamp = datetime.now()
        self.metadata = metadata or {}

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
