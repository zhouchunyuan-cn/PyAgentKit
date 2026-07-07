"""
ConversationSession / SessionManager 单元测试

用 MockAgent 记录每次 think 收到的 context_messages，
验证多轮对话的上下文注入与滑动窗口裁剪，无需联网。
"""
import pytest
from core.session import ConversationSession, SessionManager


class MockAgent:
    """
    模拟 Agent，记录每次 think 收到的 user_input 与 context，
    并返回固定回复。用于验证 Session 的上下文管理逻辑。
    """

    def __init__(self, agent_id="mock"):
        self.agent_id = agent_id
        self.calls = []  # 记录 (user_input, context_roles)

    def think(self, user_input, context_messages=None, use_tools=True):
        context = context_messages or []
        self.calls.append((user_input, [m["role"] for m in context]))
        return f"回复:{user_input}"


@pytest.fixture
def agent():
    return MockAgent()


@pytest.fixture
def session(agent):
    return ConversationSession(agent)


class TestSingleTurn:
    def test_chat_returns_response(self, session, agent):
        r = session.chat("你好")
        assert r == "回复:你好"

    def test_first_turn_has_no_history_context(self, session, agent):
        session.chat("你好")
        # 第一轮：context 应为空（没有历史）
        _, context_roles = agent.calls[0]
        assert context_roles == []

    def test_history_recorded(self, session):
        session.chat("你好")
        history = session.get_history()
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"


class TestMultiTurnContext:
    def test_second_turn_includes_first_turn_history(self, session, agent):
        session.chat("我叫小明")
        session.chat("我叫什么")
        # 第二轮：context 应包含第一轮的 user+assistant
        _, context_roles = agent.calls[1]
        assert context_roles == ["user", "assistant"]

    def test_history_grows_with_turns(self, session):
        for i in range(3):
            session.chat(f"消息{i}")
        # 3 轮 = 6 条消息
        assert len(session.get_history()) == 6

    def test_clear_history(self, session, agent):
        session.chat("话题A")
        session.clear_history()
        assert len(session.get_history()) == 0
        # 清空后下一轮 context 又应为空
        session.chat("新话题")
        _, context_roles = agent.calls[1]
        assert context_roles == []


class TestSlidingWindow:
    """验证历史超出上限时的滑动窗口裁剪"""

    def test_history_trimmed_to_max(self, agent):
        session = ConversationSession(agent, max_history=4)  # 保留最近 4 条
        for i in range(5):
            session.chat(f"消息{i}")
        # 应裁剪到 max_history 附近（成对裁剪）
        assert len(session.get_history()) <= 6

    def test_recent_history_preserved_after_trim(self, agent):
        session = ConversationSession(agent, max_history=4)
        for i in range(5):
            session.chat(f"消息{i}")
        history = session.get_history()
        # 最近一轮的内容应保留
        contents = [m["content"] for m in history]
        assert "回复:消息4" in contents


class TestSystemContext:
    def test_system_context_injected(self, agent):
        session = ConversationSession(agent, system_context="你是助手")
        session.chat("你好")
        _, context_roles = agent.calls[0]
        # 即使首轮，system_context 也应作为 system 消息注入
        assert "system" in context_roles


class TestSessionSummary:
    def test_summary_fields(self, session):
        session.chat("你好")
        session.chat("再来")
        s = session.summary()
        assert s["turn_count"] == 2
        assert s["message_count"] == 4
        assert s["agent_id"] == "mock"
        assert "session_id" in s


class TestSessionManager:
    def test_create_and_get(self, agent):
        mgr = SessionManager()
        s = mgr.create_session(agent)
        assert mgr.get_session(s.session_id) is s

    def test_close(self, agent):
        mgr = SessionManager()
        s = mgr.create_session(agent)
        assert mgr.close_session(s.session_id) is True
        assert mgr.get_session(s.session_id) is None

    def test_list_sessions(self, agent):
        mgr = SessionManager()
        mgr.create_session(agent)
        mgr.create_session(agent)
        assert len(mgr.list_sessions()) == 2
