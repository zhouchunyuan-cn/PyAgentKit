"""
AsyncRouter 单元测试

用 asyncio + 桩 Agent 验证：
- 异步路由投递
- drain 等待
- broadcast 异步
- 关键：长链路不栈溢出（同步 Router 会 RecursionError 的场景）
- receive_async 被优先调用
"""
import asyncio
import pytest

from core.async_router import AsyncRouter
from core.agent import Agent
from core.message import Message


def run_async(coro):
    """运行异步测试的辅助函数"""
    return asyncio.new_event_loop().run_until_complete(coro)


class StubAgent(Agent):
    """同步桩 Agent，记录收到的消息"""

    def __init__(self, agent_id, name="stub"):
        super().__init__(agent_id, name)
        self.received = []

    def receive(self, message):
        self.received.append(message)


class ForwardingAgent(Agent):
    """收到后转发给目标，制造长调用链（同步会栈溢出）"""

    def __init__(self, agent_id, target_id):
        super().__init__(agent_id, agent_id)
        self.target_id = target_id
        self.count = 0

    def receive(self, message):
        self.count += 1
        if self.count < 50:  # 制造长链路
            self.send(self.target_id, message.content)


class AsyncStubAgent(Agent):
    """带 receive_async 的桩 Agent，验证异步路径被优先调用"""

    def __init__(self, agent_id):
        super().__init__(agent_id, agent_id)
        self.async_received = []
        self.sync_received = []

    async def receive_async(self, message):
        self.async_received.append(message)

    def receive(self, message):
        self.sync_received.append(message)


# --------------------------------------------------------------------
# 基本异步路由
# --------------------------------------------------------------------

class TestAsyncRouting:
    def test_message_delivered_async(self):
        async def run():
            r = AsyncRouter()
            a = StubAgent("a")
            r.register_agent(a)
            await r.start()
            await r.route(Message("sender", "a", "hello"))
            await r.drain()
            await r.stop()
            assert len(a.received) == 1
            assert a.received[0].content == "hello"
        run_async(run())

    def test_unknown_agent_counted_as_error(self):
        async def run():
            r = AsyncRouter()
            await r.start()
            await r.route(Message("s", "ghost", "x"))
            await r.drain()
            await r.stop()
            assert r.stats["routing_errors"] >= 1
        run_async(run())

    def test_stats_counters(self):
        async def run():
            r = AsyncRouter()
            a = StubAgent("a")
            r.register_agent(a)
            await r.start()
            await r.route(Message("s", "a", "m"))
            await r.drain()
            await r.stop()
            stats = r.get_router_stats()["stats"]
            assert stats["messages_sent"] == 1
            assert stats["messages_received"] == 1
        run_async(run())


# --------------------------------------------------------------------
# 关键：长链路不栈溢出
# --------------------------------------------------------------------

class TestNoStackOverflow:
    """
    验证 AsyncRouter 不再栈溢出

    同步 Router 下，A→B→A→B... 的转发链会 RecursionError。
    AsyncRouter 下，每跳入队再处理，调用栈不增长。
    """

    def test_long_chain_completes(self):
        async def run():
            r = AsyncRouter()
            a = ForwardingAgent("a", "b")
            b = ForwardingAgent("b", "a")
            r.register_agent(a)
            r.register_agent(b)
            await r.start()
            await r.route(Message("ext", "a", "ping"))
            await r.drain()
            await r.stop()
            # 应正常完成 50 跳（不 RecursionError）
            assert a.count + b.count >= 50
        run_async(run())


# --------------------------------------------------------------------
# broadcast
# --------------------------------------------------------------------

class TestAsyncBroadcast:
    def test_broadcast_reaches_all(self):
        async def run():
            r = AsyncRouter()
            a = StubAgent("a")
            b = StubAgent("b")
            c = StubAgent("c")
            r.register_agent(a)
            r.register_agent(b)
            r.register_agent(c)
            await r.start()
            await r.broadcast(Message("a", "all", "news"))
            await r.drain()
            await r.stop()
            # a 是发送者被排除，b/c 收到
            assert len(b.received) == 1
            assert len(c.received) == 1
            assert len(a.received) == 0
        run_async(run())


# --------------------------------------------------------------------
# receive_async 优先
# --------------------------------------------------------------------

class TestReceiveAsync:
    def test_receive_async_preferred(self):
        async def run():
            r = AsyncRouter()
            a = AsyncStubAgent("a")
            r.register_agent(a)
            await r.start()
            await r.route(Message("s", "a", "x"))
            await r.drain()
            await r.stop()
            # 异步路径被调用，同步路径不被调用
            assert len(a.async_received) == 1
            assert len(a.sync_received) == 0
        run_async(run())


# --------------------------------------------------------------------
# 异常边界
# --------------------------------------------------------------------

class TestExceptionBoundary:
    def test_crashing_agent_does_not_break_router(self):
        async def run():
            class CrashAgent(Agent):
                def __init__(self):
                    super().__init__("crash", "crash")
                def receive(self, message):
                    raise RuntimeError("故意崩溃")

            class OkAgent(Agent):
                def __init__(self):
                    super().__init__("ok", "ok")
                    self.got = []
                def receive(self, message):
                    self.got.append(message)

            r = AsyncRouter()
            crash = CrashAgent()
            ok = OkAgent()
            r.register_agent(crash)
            r.register_agent(ok)
            await r.start()
            await r.route(Message("s", "crash", "x"))
            await r.route(Message("s", "ok", "y"))
            await r.drain()
            await r.stop()
            # 崩溃 agent 不影响后续消息处理
            assert r.stats["routing_errors"] >= 1
            assert len(ok.got) == 1
        run_async(run())
