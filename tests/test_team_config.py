"""
Team 配置接入单元测试

验证：
- YAML team 块解析（sequential / hierarchical）
- 配置校验（非法 process、hierarchical 无 leader、未知 member）
- build_team 构建 Team（含 Process、leader、members）
- 示例配置文件 team_example.yaml 可加载
"""

import pytest

from core.config import ConfigError, ConfigLoader
from core.llm import ChatResult, LLMClient
from core.team import HierarchicalProcess, SequentialProcess


class FakeLLM(LLMClient):
    def chat(self, messages, tools=None, tool_choice="auto"):
        return ChatResult(content="fake")


def make_loader():
    return ConfigLoader()


# --------------------------------------------------------------------
# team 配置解析
# --------------------------------------------------------------------


class TestTeamParse:
    def test_parse_sequential_team(self):
        loader = make_loader()
        cfg = loader.parse(
            {
                "agents": [{"id": "a", "capabilities": ["x"]}, {"id": "b"}],
                "team": {"name": "组", "process": "sequential", "members": ["a", "b"]},
            }
        )
        assert cfg.team is not None
        assert cfg.team.process == "sequential"
        assert cfg.team.members == ["a", "b"]

    def test_parse_hierarchical_team(self):
        loader = make_loader()
        cfg = loader.parse(
            {
                "agents": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                "team": {
                    "process": "hierarchical",
                    "leader": "c",
                    "members": ["a", "b"],
                    "max_subtasks": 3,
                },
            }
        )
        assert cfg.team.process == "hierarchical"
        assert cfg.team.leader == "c"
        assert cfg.team.max_subtasks == 3

    def test_no_team_block_returns_none(self):
        loader = make_loader()
        cfg = loader.parse({"agents": [{"id": "a"}]})
        assert cfg.team is None

    def test_default_process_is_sequential(self):
        loader = make_loader()
        cfg = loader.parse({"agents": [{"id": "a"}], "team": {}})
        assert cfg.team.process == "sequential"

    def test_invalid_process_raises(self):
        loader = make_loader()
        with pytest.raises(ConfigError, match="非法"):
            loader.parse({"agents": [{"id": "a"}], "team": {"process": "bogus"}})

    def test_hierarchical_without_leader_raises(self):
        loader = make_loader()
        with pytest.raises(ConfigError, match="leader"):
            loader.parse({"agents": [{"id": "a"}], "team": {"process": "hierarchical"}})

    def test_unknown_leader_raises(self):
        loader = make_loader()
        with pytest.raises(ConfigError, match="未在 agents"):
            loader.parse(
                {
                    "agents": [{"id": "a"}],
                    "team": {"process": "hierarchical", "leader": "ghost"},
                }
            )

    def test_unknown_member_raises(self):
        loader = make_loader()
        with pytest.raises(ConfigError, match="未在 agents"):
            loader.parse(
                {
                    "agents": [{"id": "a"}],
                    "team": {"members": ["ghost"]},
                }
            )


# --------------------------------------------------------------------
# build_team 构建
# --------------------------------------------------------------------


class TestBuildTeam:
    def test_build_team_none_when_no_team_block(self):
        loader = make_loader()
        cfg = loader.parse({"agents": [{"id": "a"}]})
        assert loader.build_team(cfg, {}) is None

    def test_build_sequential_team(self):
        loader = make_loader()
        cfg = loader.parse(
            {
                "agents": [{"id": "a"}, {"id": "b"}],
                "team": {"process": "sequential", "members": ["a", "b"]},
            }
        )
        _, agents, _ = loader.build(cfg, llm_client=FakeLLM())
        team = loader.build_team(cfg, agents)
        assert team is not None
        assert isinstance(team.process, SequentialProcess)
        assert len(team.members) == 2
        assert team.leader is None

    def test_build_hierarchical_team(self):
        loader = make_loader()
        cfg = loader.parse(
            {
                "agents": [{"id": "a"}, {"id": "b"}, {"id": "leader"}],
                "team": {"process": "hierarchical", "leader": "leader", "members": ["a", "b"]},
            }
        )
        _, agents, _ = loader.build(cfg, llm_client=FakeLLM())
        team = loader.build_team(cfg, agents)
        assert isinstance(team.process, HierarchicalProcess)
        assert team.leader is agents["leader"]

    def test_members_defaults_to_all_agents_when_empty(self):
        loader = make_loader()
        cfg = loader.parse(
            {
                "agents": [{"id": "a"}, {"id": "b"}],
                "team": {"process": "sequential"},  # 不指定 members
            }
        )
        _, agents, _ = loader.build(cfg, llm_client=FakeLLM())
        team = loader.build_team(cfg, agents)
        assert len(team.members) == 2  # 全部 agents 作为成员


# --------------------------------------------------------------------
# 示例配置文件集成
# --------------------------------------------------------------------


class TestExampleConfigFile:
    def test_team_example_yaml_loads(self):
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "team_example.yaml",
        )
        loader = make_loader()
        cfg = loader.load_file(path)
        assert cfg.team is not None
        assert cfg.team.process == "hierarchical"
        assert cfg.team.leader == "leader"
        assert "researcher" in cfg.team.members
