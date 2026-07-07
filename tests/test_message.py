"""
Message 模块单元测试

覆盖：属性、唯一 ID、metadata、序列化、大小估算、字符串表示
"""
import pytest
from core.message import Message


@pytest.fixture
def msg():
    return Message(
        sender="alice",
        receiver="bob",
        content="hello world",
        msg_type="text",
        metadata={"session": "s1"},
    )


class TestMessageAttributes:
    def test_basic_attributes(self, msg):
        assert msg.sender == "alice"
        assert msg.receiver == "bob"
        assert msg.content == "hello world"
        assert msg.type == "text"

    def test_unique_id(self):
        m1 = Message("a", "b", "c")
        m2 = Message("a", "b", "c")
        assert m1.id != m2.id

    def test_default_metadata_is_empty_dict(self):
        m = Message("a", "b", "c")
        assert m.metadata == {}

    def test_add_metadata(self, msg):
        msg.add_metadata("key", "value")
        assert msg.metadata["key"] == "value"

    def test_metadata_isolation(self):
        # 默认 None 不应共享可变默认值
        m1 = Message("a", "b", "c")
        m2 = Message("a", "b", "c")
        m1.add_metadata("x", 1)
        assert "x" not in m2.metadata


class TestMessageSerialization:
    def test_to_dict_contains_all_fields(self, msg):
        d = msg.to_dict()
        assert d["id"] == msg.id
        assert d["sender"] == "alice"
        assert d["receiver"] == "bob"
        assert d["content"] == "hello world"
        assert d["type"] == "text"
        assert d["metadata"] == {"session": "s1"}
        assert "timestamp" in d

    def test_to_dict_is_json_serializable(self, msg):
        """to_dict 结果必须可 JSON 序列化（含 timestamp）"""
        import json
        # 不应抛异常
        json.dumps(msg.to_dict())


class TestMessageSize:
    def test_get_content_size_string(self, msg):
        # "hello world" UTF-8 为 11 字节
        assert msg.get_content_size() == 11

    def test_get_content_size_non_ascii(self):
        # 中文每个字符 UTF-8 为 3 字节
        m = Message("a", "b", "你好")
        assert m.get_content_size() == 6

    def test_get_content_size_dict_content(self):
        # content 为 dict 时应基于 str() 估算，不报错
        m = Message("a", "b", {"k": "v"})
        assert m.get_content_size() > 0


class TestMessageRepr:
    def test_repr_contains_id_and_sender(self, msg):
        r = repr(msg)
        assert "alice" in r
        assert "bob" in r

    def test_str_contains_content_size(self, msg):
        s = str(msg)
        assert "11 bytes" in s
