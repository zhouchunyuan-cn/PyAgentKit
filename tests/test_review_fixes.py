"""
代码评审修复项的回归测试

覆盖 2026-10 评审发现的缺陷修复：
- Memory：原子持久化、类型校验、损坏文件容错、并发安全
- VectorMemory：重复 id 覆盖、损坏条目容错、数字分词
- Router：互发环路终止（hop 上限）、迭代泵不栈溢出、历史上限
- AsyncRouter：stop() 不死锁、drain() 覆盖同步桥在途消息
- Tools：数据库只读绕过、计算器幂上限、file_read 敏感文件
- Tracer：contextvars 跨执行器线程传播
- ReactEngine：降级路径不重复记录会话历史
- Team：同一成员多次产出保留
- Collaboration：投递失败如实返回
- CLI：--tools 逗号语法与错误提示
"""

import asyncio
import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.agent import Agent
from core.async_router import AsyncRouter
from core.collaboration import DynamicCollaborationManager
from core.llm import ChatResult, LLMClient, TokenUsage, ToolCall
from core.memory import Memory
from core.message import DEFAULT_MAX_HOPS, Message
from core.router import MAX_MESSAGE_HISTORY, Router
from core.team import SharedContext
from core.tools import CalculatorTool, DatabaseTool, FileReadTool
from core.trace import Tracer
from core.vector_memory import LocalTfidfEmbedder, VectorMemory, _tokenize


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ====================================================================
# Memory：原子持久化 + 类型校验 + 容错
# ====================================================================


class TestMemoryPersistence:
    def test_non_serializable_long_term_rejected_and_file_intact(self, tmp_path):
        """不可序列化的长期记忆被拒绝，且既有文件不被破坏（原子写）"""
        path = str(tmp_path / "mem.json")
        mem = Memory(persistent_storage_path=path)
        mem.store("good", "value", memory_type="long")

        with pytest.raises(TypeError):
            mem.store("bad", object(), memory_type="long")

        # 原文件仍是合法 JSON 且内容完整
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"good": "value"}

    def test_invalid_memory_type_raises(self):
        mem = Memory()
        with pytest.raises(ValueError):
            mem.store("k", "v", memory_type="Short")  # 拼写错误不再静默丢弃

    def test_corrupted_file_loads_empty(self, tmp_path):
        """文件损坏时保持空记忆，不抛异常"""
        path = tmp_path / "mem.json"
        path.write_text("{ 半截 JSON", encoding="utf-8")
        mem = Memory(persistent_storage_path=str(path))
        assert mem.list_long_term_memory() == {}
        # 且能继续正常写入（写完仍是合法 JSON）
        mem.store("k", "v", memory_type="long")
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"k": "v"}

    def test_non_dict_file_rejected(self, tmp_path):
        path = tmp_path / "mem.json"
        path.write_text('["a", "b"]', encoding="utf-8")
        mem = Memory(persistent_storage_path=str(path))
        assert mem.list_long_term_memory() == {}

    def test_concurrent_access_no_crash(self):
        """多线程并发 store/retrieve/cleanup 不抛 KeyError/RuntimeError"""
        mem = Memory(short_term_duration=100)
        errors = []

        def worker(seed: int):
            try:
                for i in range(200):
                    key = f"k{(seed + i) % 20}"
                    mem.store(key, i, memory_type="short")
                    mem.retrieve(key)
                    mem.cleanup_expired_memory()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ====================================================================
# VectorMemory：重复 id / 容错 / 数字分词
# ====================================================================


class TestVectorMemory:
    def test_duplicate_id_overwrites_cleanly(self):
        """重复 item_id 覆盖更新：count 与矩阵一致，搜索不重复返回"""
        vm = VectorMemory(embedder=LocalTfidfEmbedder())
        vm.add("苹果是一种水果", item_id="x")
        vm.add("香蕉是黄色的水果", item_id="x")  # 覆盖
        assert vm.count() == 1
        results = vm.search("香蕉", top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "x"
        assert results[0]["text"] == "香蕉是黄色的水果"

    def test_load_skips_corrupt_entries(self, tmp_path):
        """单条损坏只跳过该条，其余条目可用且 search 不崩"""
        path = tmp_path / "vm.json"
        data = [
            {"id": "a", "text": "Python 是解释型语言", "metadata": {}},
            {"id": "b", "metadata": {}},  # 缺 text
            "not-a-dict",  # 结构非法
            {"id": "a", "text": "重复 id 应被跳过", "metadata": {}},
        ]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        vm = VectorMemory(embedder=LocalTfidfEmbedder(), persistence_path=str(path))
        assert vm.count() == 1
        results = vm.search("解释型", top_k=3)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_save_is_valid_json_after_many_adds(self, tmp_path):
        path = str(tmp_path / "vm.json")
        vm = VectorMemory(embedder=LocalTfidfEmbedder(), persistence_path=path)
        for i in range(20):
            vm.add(f"第 {i} 条向量记忆")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 20

    def test_digits_are_tokenized(self):
        """数字参与分词：仅数字不同的文本向量可区分"""
        assert "404" in _tokenize("错误码 404")
        assert "500" in _tokenize("错误码 500")
        # 两条记忆召回时不会完全混淆
        vm = VectorMemory(embedder=LocalTfidfEmbedder())
        vm.add("服务器错误码 404 表示未找到", item_id="e404")
        vm.add("服务器错误码 500 表示内部错误", item_id="e500")
        results = vm.search("错误码 404", top_k=1)
        assert results[0]["id"] == "e404"

    def test_add_performance_not_quadratic_per_embed(self):
        """每次 add 只做一次语料拟合：embedding 调用次数应为 O(条目数)"""
        calls = {"fit": 0}

        class CountingEmbedder(LocalTfidfEmbedder):
            def _refit(self):
                calls["fit"] += 1
                super()._refit()

        vm = VectorMemory(embedder=CountingEmbedder())
        for i in range(30):
            vm.add(f"记忆条目编号 {i} 内容各不相同")
        # 修复前：每次 embed 都重拟合（30 条 ≈ 数百次 fit）；修复后 = 每条 1 次
        assert calls["fit"] <= 31


# ====================================================================
# Router：环路终止 / 迭代泵 / 历史上限
# ====================================================================


class LoopAgent(Agent):
    """无条件把消息转发回对方（制造无限环路）"""

    def __init__(self, agent_id, target_id):
        super().__init__(agent_id, agent_id)
        self.target_id = target_id
        self.n = 0

    def receive(self, message):
        self.n += 1
        self.send(self.target_id, "loop")


class ChainAgent(Agent):
    """有界转发链（可超过默认递归深度对应的栈量）"""

    def __init__(self, agent_id, target_id, limit):
        super().__init__(agent_id, agent_id)
        self.target_id = target_id
        self.limit = limit
        self.n = 0

    def receive(self, message):
        self.n += 1
        if self.n < self.limit:
            self.send(self.target_id, "next")


class TestRouterLoopProtection:
    def test_mutual_forwarding_terminates(self):
        """A↔B 互发不再无限递归：hop 上限触发丢弃"""
        r = Router()
        a, b = LoopAgent("a", "b"), LoopAgent("b", "a")
        r.register_agent(a)
        r.register_agent(b)
        # 修复前：RecursionError；修复后：正常返回
        r.route_message(Message("user", "a", "start"))
        assert r.stats["hop_limit_drops"] == 1
        assert a.n + b.n == DEFAULT_MAX_HOPS  # 每跳正好投递一次

    def test_long_chain_no_stack_overflow(self):
        """200 跳长链（超过默认递归栈承受力）在同步 Router 上也安全"""
        r = Router()
        a, b = ChainAgent("a", "b", 200), ChainAgent("b", "a", 200)
        r.register_agent(a)
        r.register_agent(b)
        r.route_message(Message("user", "a", "go", max_hops=500))
        # a 转发 200 次，b 接住最后一棒：总投递 2*200-1
        assert a.n == 200
        assert b.n == 199

    def test_message_history_capped(self):
        r = Router()

        class Sink(Agent):
            def __init__(self):
                super().__init__("sink", "sink")

            def receive(self, message):
                pass

        r.register_agent(Sink())
        for i in range(MAX_MESSAGE_HISTORY + 500):
            r.route_message(Message("user", "sink", f"m{i}"))
        assert len(r.message_history) == MAX_MESSAGE_HISTORY
        recent = r.get_message_history(limit=3)
        assert len(recent) == 3


# ====================================================================
# AsyncRouter：stop 不死锁 / drain 覆盖同步桥
# ====================================================================


class TestAsyncRouterShutdown:
    def _forwarders(self, hops):
        return ChainAgent("a", "b", hops), ChainAgent("b", "a", hops)

    def test_stop_without_drain_does_not_deadlock(self):
        """消息链进行中直接 stop()：优雅排空并退出（修复前永久挂起）"""

        async def run():
            r = AsyncRouter()
            a, b = self._forwarders(30)
            r.register_agent(a)
            r.register_agent(b)
            await r.start()
            await r.route(Message("ext", "a", "ping"))
            # 不调用 drain()，直接 stop —— 修复前 queue.join() 死锁
            await asyncio.wait_for(r.stop(), timeout=15)
            assert a.n + b.n == 2 * 30 - 1  # 链上消息全部送达，无丢失

        run_async(run())

    def test_drain_covers_sync_bridge_inflight(self):
        """同步桥在途消息计入 drain：30 跳全部送达（修复前 10/10 丢消息）"""

        async def run():
            r = AsyncRouter()
            a, b = self._forwarders(30)
            r.register_agent(a)
            r.register_agent(b)
            await r.start()
            await r.route(Message("ext", "a", "ping"))
            await asyncio.wait_for(r.drain(), timeout=15)
            assert a.n + b.n == 2 * 30 - 1  # 30 跳全部送达（修复前第 1 跳后就返回）
            await r.stop()

        run_async(run())

    def test_route_message_unknown_receiver_returns_false(self):
        """同步桥 route_message 对未知接收者返回 False（与同步 Router 一致）"""

        async def run():
            r = AsyncRouter()
            await r.start()
            ok = r.route_message(Message("s", "ghost", "x"))
            assert ok is False
            assert r.stats["routing_errors"] >= 1
            await r.stop()

        run_async(run())

    def test_broadcast_returns_per_target_result(self):
        """异步 broadcast 返回与同步版一致的 {agent_id: bool}"""

        async def run():
            r = AsyncRouter()

            class Sink(Agent):
                def __init__(self, aid):
                    super().__init__(aid, aid)

                def receive(self, message):
                    pass

            a, b = Sink("a"), Sink("b")
            r.register_agent(a)
            r.register_agent(b)
            await r.start()
            results = await r.broadcast(Message("a", "all", "news"))
            await r.drain()
            await r.stop()
            assert results == {"a": False, "b": True}  # 发送者被排除

        run_async(run())


# ====================================================================
# Tools：数据库只读 / 计算器上限 / file_read 敏感文件
# ====================================================================


class TestToolHardening:
    def test_readonly_blocks_with_delete(self, tmp_path):
        """只读模式拦截 WITH ... DELETE（修复前可绕过）"""
        db_path = str(tmp_path / "test.db")
        writable = DatabaseTool(db_path)
        writable.run("CREATE TABLE users (id INTEGER)")
        writable.run("INSERT INTO users VALUES (1)")

        ro = DatabaseTool(db_path, readonly=True)
        r = ro.run("WITH t AS (SELECT 1) DELETE FROM users")
        assert "error" in r
        assert "只读" in r["error"]

        # 数据未被破坏
        rows = writable.run("SELECT COUNT(*) AS c FROM users")
        assert rows["rows"][0]["c"] == 1
        writable.close()
        ro.close()

    def test_readonly_blocks_pragma_write(self, tmp_path):
        ro = DatabaseTool(str(tmp_path / "t.db"), readonly=True)
        r = ro.run("PRAGMA writable_schema = ON")
        assert "error" in r
        ro.close()

    def test_calculator_rejects_huge_exponent(self):
        calc = CalculatorTool()
        r = calc.run("9**9**9")  # 修复前：挂死进程
        assert "error" in r
        assert "指数过大" in r["error"]

    def test_calculator_allows_normal_power(self):
        calc = CalculatorTool()
        assert calc.run("2**10")["result"] == 1024

    def test_calculator_rejects_overlong_expression(self):
        calc = CalculatorTool()
        r = calc.run("1+" * 600 + "1")
        assert "error" in r

    def test_file_read_rejects_env(self, tmp_path):
        """默认沙箱拒绝 .env（防 LLM 读走 API Key）"""
        env_file = tmp_path / ".env"
        env_file.write_text("ZHIPUAI_API_KEY=secret", encoding="utf-8")
        tool = FileReadTool(allowed_dir=str(tmp_path))
        r = tool.run(path=str(env_file))
        assert "error" in r
        assert "敏感" in r["error"]

    def test_file_read_rejects_key_files(self, tmp_path):
        pem = tmp_path / "server.pem"
        pem.write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
        tool = FileReadTool(allowed_dir=str(tmp_path))
        assert "error" in tool.run(path=str(pem))

    def test_file_read_allows_normal_files(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        tool = FileReadTool(allowed_dir=str(tmp_path))
        assert tool.run(path=str(f))["content"] == "hello"

    def test_file_read_sensitive_optin(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("K=V", encoding="utf-8")
        tool = FileReadTool(allowed_dir=str(tmp_path), allow_sensitive=True)
        assert tool.run(path=str(env_file))["content"] == "K=V"


# ====================================================================
# Tracer：contextvars 跨线程传播
# ====================================================================


class TestTracerContextPropagation:
    def test_span_recorded_in_executor_thread(self):
        """copy_context 进入执行器线程后 span 不再丢失"""
        Tracer.enable()
        try:
            trace = Tracer.start_trace("exec-test")
            assert trace is not None

            def work():
                span = Tracer.start_span("in_executor")
                Tracer.end_span(span, value=42)

            ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(lambda: ctx.run(work)).result()

            spans = [s.name for s in trace.spans]
            assert "in_executor" in spans
            recorded = [s for s in trace.spans if s.name == "in_executor"][0]
            assert recorded.attributes["value"] == 42
            Tracer.end_trace()
        finally:
            Tracer.disable()

    def test_span_cap(self):
        Tracer.enable()
        try:
            trace = Tracer.start_trace("cap-test")
            for _ in range(Tracer.MAX_SPANS + 100):
                s = Tracer.start_span("s")
                if s is not None:
                    Tracer.end_span(s)
            assert len(trace.spans) == Tracer.MAX_SPANS
            Tracer.end_trace()
        finally:
            Tracer.disable()


# ====================================================================
# ReactEngine：降级路径不重复记录历史
# ====================================================================


class FlakyThenRecoverLLM(LLMClient):
    """第 1 次返回工具调用，第 2 次抛错，第 3 次（降级收尾）成功"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                content="",
                tool_calls=[ToolCall(id="t1", name="calculator", arguments={"expression": "1+1"})],
                finish_reason="tool_calls",
                usage=TokenUsage(10, 5, 15),
            )
        if self.calls == 2:
            raise RuntimeError("模拟 LLM 故障")
        return ChatResult(content="降级答复", usage=TokenUsage(8, 4, 12))


class _PlainAgent(Agent):
    def __init__(self, llm):
        super().__init__("a1", "测试", llm_client=llm)
        self.tool_registry.register(CalculatorTool())

    def receive(self, message):
        pass


class TestReactEngineRecovery:
    def test_degraded_path_respects_session_history_ownership(self):
        """Session 场景下降级回复不写入 brain 会话通道（修复前重复记录）"""
        llm = FlakyThenRecoverLLM()
        agent = _PlainAgent(llm)
        result = agent.think(
            "算一下", context_messages=[{"role": "system", "content": "会话上下文"}]
        )
        assert result == "降级答复"
        # 会话历史由外部 Session 管理：brain 不应记录（修复前会重复写入）
        assert agent.brain.get_conversation() == []
        # KV 快照仍然保留
        assert any(k.startswith("interaction:") for k in agent.memory.list_short_term_memory())

    def test_degraded_path_records_history_when_unmanaged(self):
        """无外部 Session 时降级回复仍记录会话通道（保持记忆延续）"""
        llm = FlakyThenRecoverLLM()
        agent = _PlainAgent(llm)
        result = agent.think("算一下")
        assert result == "降级答复"
        turns = agent.brain.get_conversation()
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert turns[1]["content"] == "降级答复"

    def test_interaction_keys_do_not_collide(self):
        """多次 think 的 KV 快照 key 不因短期记忆条数变化而碰撞"""
        llm = FlakyThenRecoverLLM()
        agent = _PlainAgent(llm)
        agent.think("第一问")
        agent.think("第二问")  # 第 1 次调用又是工具调用路径
        snapshots = [
            k for k in agent.memory.list_short_term_memory() if k.startswith("interaction:")
        ]
        assert len(snapshots) == 2


# ====================================================================
# Team SharedContext：同成员多次产出
# ====================================================================


class TestSharedContext:
    def test_multiple_outputs_same_agent_preserved(self):
        ctx = SharedContext(task="t")
        ctx.add_output("writer", "第一稿")
        ctx.add_output("writer", "第二稿")
        # 最新产出
        assert ctx.get_output("writer") == "第二稿"
        # 汇总上下文包含两稿（修复前只剩最后一稿）
        all_outputs = ctx.get_all_outputs()
        assert "第一稿" in all_outputs
        assert "第二稿" in all_outputs
        assert ctx.summary()["output_count"] == 2


# ====================================================================
# Collaboration：投递结果如实返回
# ====================================================================


class TestCollaboration:
    def test_initiate_returns_false_when_entry_agent_crashes(self):
        r = Router()

        class CrashSearcher(Agent):
            def __init__(self):
                super().__init__("crash", "crash", capabilities=["search"])

            def receive(self, message):
                raise RuntimeError("入口崩溃")

        r.register_agent(CrashSearcher())
        mgr = DynamicCollaborationManager(r)
        assert mgr.initiate_collaboration("研究一下量子计算") is False

    def test_stats_returns_copy(self):
        r = Router()
        mgr = DynamicCollaborationManager(r)
        stats = mgr.get_collaboration_stats()
        stats["recent_tasks"].append({"injected": True})
        assert mgr.get_collaboration_stats()["recent_tasks"] == []


# ====================================================================
# Message：hop 继承
# ====================================================================


class TestMessageHops:
    def test_hops_inherited_from_context(self):
        from core.message import current_message_hops, reset_message_hops, set_message_hops

        token = set_message_hops(5)
        try:
            msg = Message("a", "b", "x")
            assert msg.hops == 5
            assert current_message_hops() == 5
        finally:
            reset_message_hops(token)
        assert Message("a", "b", "x").hops == 0


# ====================================================================
# CLI：--tools 参数
# ====================================================================


class TestCLIParsing:
    def test_comma_separated_tools_with_leading_option(self):
        """`ask --tools=a,b 问题` 可用（绕开多值参数贪婪问题）"""
        from cli import build_parser

        args = build_parser().parse_args(["ask", "--tools=web_search,calculator", "什么是AI"])
        assert args.question == "什么是AI"
        assert args.tools == ["web_search,calculator"]

    def test_question_before_tools_still_works(self):
        from cli import build_parser

        args = build_parser().parse_args(["ask", "什么是AI", "--tools", "web_search", "calculator"])
        assert args.question == "什么是AI"
        assert args.tools == ["web_search", "calculator"]

    def test_greedy_order_failure_gives_hint(self, capsys):
        """`ask --tools x 问题` 报错时给出定向提示与退出码 2"""
        from cli import build_parser

        with pytest.raises(SystemExit) as ei:
            build_parser().parse_args(["ask", "--tools", "web_search", "什么是AI"])
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "提示" in err
        assert "--tools=web_search,calculator" in err

    def test_split_tool_names(self):
        from cli import _split_tool_names

        assert _split_tool_names(["a,b", "c"]) == ["a", "b", "c"]
        assert _split_tool_names(None) == []
        assert _split_tool_names(["a"]) == ["a"]


# ====================================================================
# logging_config：级别校验
# ====================================================================


class TestLoggingConfig:
    def test_invalid_level_raises_with_message(self):
        from core.logging_config import setup_logging

        with pytest.raises(ValueError, match="无效的日志级别"):
            setup_logging("verbose")

    def test_valid_levels_accepted(self):
        from core.logging_config import setup_logging

        setup_logging("debug")  # 大小写不敏感
        setup_logging("WARNING")
