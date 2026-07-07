"""
Router 模块单元测试

覆盖：Agent 注册/注销、点对点路由、广播、消息历史、统计，
以及关键的【异常边界】——单个 Agent 抛异常不应中断消息流
"""
import pytest
from core.router import Router
from core.message import Message
from core.agent import Agent


class StubAgent(Agent):
    """用于测试的桩 Agent，把收到的消息记录到 received 列表"""

    def __init__(self, agent_id, name="stub"):
        super().__init__(agent_id, name)
        self.received = []

    def receive(self, message):
        self.received.append(message)


class CrashingAgent(Agent):
    """receive 永远抛异常，用于测试异常边界"""

    def __init__(self, agent_id, name="crash"):
        super().__init__(agent_id, name)

    def receive(self, message):
        raise RuntimeError("故意崩溃")


@pytest.fixture
def router():
    return Router()


class TestRegistration:
    def test_register_sets_router_back_reference(self, router):
        a = StubAgent("a1")
        router.register_agent(a)
        assert a.router is router
        assert "a1" in router.list_agents()

    def test_unregister_clears_router(self, router):
        a = StubAgent("a1")
        router.register_agent(a)
        router.unregister_agent("a1")
        assert "a1" not in router.list_agents()
        assert a.router is None

    def test_get_agent(self, router):
        a = StubAgent("a1")
        router.register_agent(a)
        assert router.get_agent("a1") is a
        assert router.get_agent("missing") is None


class TestRouting:
    def test_point_to_point_delivery(self, router):
        sender = StubAgent("s")
        receiver = StubAgent("r")
        router.register_agent(sender)
        router.register_agent(receiver)

        msg = Message(sender="s", receiver="r", content="hi")
        ok = router.route_message(msg)

        assert ok is True
        assert len(receiver.received) == 1
        assert receiver.received[0].content == "hi"

    def test_route_to_missing_agent(self, router):
        a = StubAgent("a")
        router.register_agent(a)
        msg = Message(sender="a", receiver="ghost", content="x")
        ok = router.route_message(msg)
        # 不存在的接收者应返回 False 并计入错误
        assert ok is False
        assert router.get_router_stats()["stats"]["routing_errors"] >= 1

    def test_message_history_recorded(self, router):
        s = StubAgent("s")
        r = StubAgent("r")
        router.register_agent(s)
        router.register_agent(r)
        router.route_message(Message("s", "r", "m1"))
        router.route_message(Message("s", "r", "m2"))
        history = router.get_message_history()
        assert len(history) == 2

    def test_stats_counters(self, router):
        s = StubAgent("s")
        r = StubAgent("r")
        router.register_agent(s)
        router.register_agent(r)
        router.route_message(Message("s", "r", "m"))
        stats = router.get_router_stats()["stats"]
        assert stats["messages_sent"] == 1
        assert stats["messages_received"] == 1


class TestBroadcast:
    def test_broadcast_reaches_all_others(self, router):
        a = StubAgent("a")
        b = StubAgent("b")
        c = StubAgent("c")
        for x in (a, b, c):
            router.register_agent(x)

        results = router.broadcast(Message("a", "all", "news"))

        # a 是发送者应被排除，b/c 收到
        assert results["b"] is True
        assert results["c"] is True
        assert results["a"] is False
        assert len(b.received) == 1
        assert len(c.received) == 1
        assert len(a.received) == 0

    def test_broadcast_includes_sender_when_not_excluded(self, router):
        a = StubAgent("a")
        b = StubAgent("b")
        router.register_agent(a)
        router.register_agent(b)
        router.broadcast(Message("a", "all", "x"), exclude_sender=False)
        assert len(a.received) == 1


class TestExceptionBoundary:
    """
    关键测试：Router 异常边界

    单个 Agent 的 receive 抛异常时：
    - route_message 不应向上抛出（不应中断整个流程）
    - 应返回 False 并计入 routing_errors
    """

    def test_crashing_agent_does_not_propagate(self, router):
        crash = CrashingAgent("crash")
        normal = StubAgent("normal")
        router.register_agent(crash)
        router.register_agent(normal)

        # 路由到崩溃 Agent，不应抛异常
        msg = Message("normal", "crash", "trigger")
        # 不应 raise
        ok = router.route_message(msg)
        assert ok is False
        assert router.get_router_stats()["stats"]["routing_errors"] >= 1

    def test_broadcast_continues_past_crash(self, router):
        # 广播时一个 Agent 崩溃，其他 Agent 仍应收到
        crash = CrashingAgent("crash")
        ok1 = StubAgent("ok1")
        ok2 = StubAgent("ok2")
        router.register_agent(crash)
        router.register_agent(ok1)
        router.register_agent(ok2)

        results = router.broadcast(Message("external", "all", "m"))
        # crash 标记失败，其余成功，且 broadcast 不抛异常
        assert results["crash"] is False
        assert results["ok1"] is True
        assert results["ok2"] is True
        assert len(ok1.received) == 1
        assert len(ok2.received) == 1
