"""
Agent 能力声明 & 协作策略单元测试

验证 P2 核心改造：协作匹配基于能力声明而非硬编码 Agent 名称。
所有测试用桩 Agent，不依赖 LLM 或网络。
"""

import pytest

from core.agent import Agent
from core.collaboration import (
    CapabilityCollaborationStrategy,
    DynamicCollaborationManager,
    ResearchCollaborationStrategy,
)
from core.router import Router


class StubAgent(Agent):
    """测试用桩 Agent，可声明任意能力"""

    def __init__(self, agent_id, name, capabilities=None):
        super().__init__(agent_id, name, capabilities=capabilities)
        self.received = []

    def receive(self, message):
        self.received.append(message)


# --------------------------------------------------------------------
# Agent 能力声明
# --------------------------------------------------------------------
class TestAgentCapabilities:
    def test_has_capability(self):
        a = StubAgent("a1", "A", capabilities=["search", "write"])
        assert a.has_capability("search") is True
        assert a.has_capability("missing") is False

    def test_has_any_capability(self):
        a = StubAgent("a1", "A", capabilities=["search"])
        assert a.has_any_capability(["search", "write"]) is True
        assert a.has_any_capability(["write", "analysis"]) is False

    def test_no_capabilities_by_default(self):
        a = StubAgent("a1", "A")
        assert a.capabilities == []
        assert a.has_capability("anything") is False


# --------------------------------------------------------------------
# 能力驱动的协作策略
# --------------------------------------------------------------------
class TestCapabilityStrategy:
    def test_matches_by_capability_not_name(self):
        """
        关键测试：Agent 名字无关，只看能力声明
        即使名字叫 RandomName，只要有 search 能力就应被选中
        """
        strategy = CapabilityCollaborationStrategy(
            name="test",
            description="t",
            required_capabilities=["search", "write"],
        )
        # 名字完全看不出功能，但声明了正确能力
        a = StubAgent("x", "RandomName", capabilities=["search"])
        b = StubAgent("y", "AnotherName", capabilities=["write"])
        c = StubAgent("z", "NoCap", capabilities=[])  # 无能力，不应被选中

        matched = strategy.should_collaborate({}, [a, b, c])
        matched_ids = {m.agent_id for m in matched}
        assert matched_ids == {"x", "y"}
        assert "z" not in matched_ids

    def test_assign_roles_by_capability(self):
        strategy = CapabilityCollaborationStrategy(
            name="test",
            description="t",
            required_capabilities=["search", "write"],
        )
        a = StubAgent("x", "X", capabilities=["search"])
        b = StubAgent("y", "Y", capabilities=["write"])
        roles = strategy.assign_roles({}, [a, b])
        assert roles["x"] == "search"
        assert roles["y"] == "write"

    def test_entry_agent_by_capability(self):
        strategy = CapabilityCollaborationStrategy(
            name="test",
            description="t",
            required_capabilities=["search", "write"],
            entry_capability="search",
        )
        a = StubAgent("x", "X", capabilities=["write"])
        b = StubAgent("y", "Y", capabilities=["search"])
        # 入口应选具备 search 能力的 b，即使它在列表中靠后
        entry = strategy.get_entry_agent([a, b])
        assert entry.agent_id == "y"

    def test_default_strategy_compatibility(self):
        """旧策略类应仍可用，且按能力匹配"""
        s = ResearchCollaborationStrategy()
        a = StubAgent("r", "Researcher", capabilities=["search"])
        matched = s.should_collaborate({}, [a])
        assert len(matched) == 1


# --------------------------------------------------------------------
# 协作管理器集成
# --------------------------------------------------------------------
class TestCollaborationManager:
    @pytest.fixture
    def setup(self):
        router = Router()
        # 名字故意不起 Researcher/Analyzer，验证靠能力
        search_agent = StubAgent("s1", "Finder", capabilities=["search"])
        write_agent = StubAgent("w1", "Scribe", capabilities=["write"])
        analysis_agent = StubAgent("a1", "NumberCruncher", capabilities=["analysis"])
        router.register_agent(search_agent)
        router.register_agent(write_agent)
        router.register_agent(analysis_agent)
        mgr = DynamicCollaborationManager(router)
        return router, mgr, search_agent, write_agent, analysis_agent

    def test_research_collaboration_routes_to_search_agent(self, setup):
        """研究任务应路由到具备 search 能力的 Agent（不靠名字）"""
        router, mgr, search_agent, write_agent, analysis_agent = setup
        ok = mgr.initiate_collaboration("研究量子计算")
        assert ok is True
        # 入口是 search_agent，应收到消息
        assert len(search_agent.received) == 1

    def test_analysis_collaboration_routes_to_analysis_agent(self, setup):
        """分析任务应路由到具备 analysis 能力的 Agent"""
        router, mgr, search_agent, write_agent, analysis_agent = setup
        ok = mgr.initiate_collaboration("计算并分析这组数据")
        assert ok is True
        assert len(analysis_agent.received) == 1
        # search/write 不应被卷入分析任务
        assert len(search_agent.received) == 0

    def test_no_matching_capability_returns_false(self):
        """没有 Agent 具备所需能力时，应返回 False 而非报错"""
        router = Router()
        router.register_agent(StubAgent("x", "X", capabilities=["unrelated"]))
        mgr = DynamicCollaborationManager(router)
        ok = mgr.initiate_collaboration("研究量子计算")
        assert ok is False

    def test_task_history_recorded(self, setup):
        router, mgr, *_ = setup
        mgr.initiate_collaboration("研究量子计算")
        stats = mgr.get_collaboration_stats()
        assert stats["task_history_count"] == 1
