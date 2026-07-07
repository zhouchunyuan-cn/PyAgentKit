"""
配置加载与工具工厂单元测试

覆盖：
- ToolFactory 注册/创建/未知工具报错
- ConfigLoader 解析正确配置
- 配置错误的提前检测（缺 id、未知工具、未知 session agent、重复 id）
- build() 构建运行时对象（用 fake LLM 避免联网）
- 示例配置文件 configs/example.yaml 可被正确加载
"""
import os
import pytest

from core.tool_factory import ToolFactory, default_factory, ToolNotFoundError
from core.config import ConfigLoader, AppConfig, AgentSpec, ConfigError
from core.tools import CalculatorTool, WebSearchTool
from core.llm import LLMClient, ChatResult


# --------------------------------------------------------------------
# Fake LLM，避免 build 时真正联网创建 GLMClient
# --------------------------------------------------------------------
class FakeLLM(LLMClient):
    def chat(self, messages, tools=None, tool_choice="auto"):
        return ChatResult(content="fake reply", finish_reason="stop")


@pytest.fixture
def loader():
    return ConfigLoader()


# --------------------------------------------------------------------
# 工具工厂
# --------------------------------------------------------------------
class TestToolFactory:
    def test_create_builtin(self):
        tool = default_factory.create("calculator")
        assert isinstance(tool, CalculatorTool)
        assert tool.name == "calculator"

    def test_create_web_search(self):
        tool = default_factory.create("web_search")
        assert isinstance(tool, WebSearchTool)

    def test_unknown_tool_raises(self):
        factory = ToolFactory()
        with pytest.raises(ToolNotFoundError):
            factory.create("ghost_tool")

    def test_register_custom(self):
        factory = ToolFactory()

        class MyTool:
            def __init__(self):
                self.name = "my_tool"

        factory.register_class("my_tool", MyTool)
        tool = factory.create("my_tool")
        assert tool.name == "my_tool"

    def test_list_tools(self):
        names = default_factory.list_tools()
        assert "calculator" in names
        assert "web_search" in names


# --------------------------------------------------------------------
# 配置解析
# --------------------------------------------------------------------
class TestConfigParse:
    def test_parse_valid_minimal(self, loader):
        raw = {
            "agents": [
                {"id": "a1", "name": "助手", "system_prompt": "你好"}
            ]
        }
        config = loader.parse(raw)
        assert config.model == "glm-4-flash"  # 默认
        assert len(config.agents) == 1
        assert config.agents[0].id == "a1"
        # session 默认绑定第一个 agent
        assert config.session_agent == "a1"

    def test_parse_full_config(self, loader):
        raw = {
            "llm": {"model": "glm-4.7-flash"},
            "agents": [
                {
                    "id": "bot",
                    "name": "Bot",
                    "system_prompt": "助手",
                    "capabilities": ["chat", "search"],
                    "tools": ["calculator", "web_search"],
                    "max_steps": 3,
                }
            ],
            "session": {"agent": "bot", "max_history": 5, "system_context": "中文"},
        }
        config = loader.parse(raw)
        assert config.model == "glm-4.7-flash"
        spec = config.agents[0]
        assert spec.capabilities == ["chat", "search"]
        assert spec.tools == ["calculator", "web_search"]
        assert spec.max_steps == 3
        assert config.session_max_history == 5
        assert config.session_system_context == "中文"

    def test_missing_id_raises(self, loader):
        with pytest.raises(ConfigError, match="id"):
            loader.parse({"agents": [{"name": "no-id"}]})

    def test_no_agents_raises(self, loader):
        with pytest.raises(ConfigError, match="至少声明一个"):
            loader.parse({})

    def test_unknown_tool_raises(self, loader):
        with pytest.raises(ConfigError, match="未知工具"):
            loader.parse({"agents": [{"id": "a", "tools": ["ghost"]}]})

    def test_unknown_session_agent_raises(self, loader):
        with pytest.raises(ConfigError, match="session.agent"):
            loader.parse({
                "agents": [{"id": "a"}],
                "session": {"agent": "missing"},
            })

    def test_duplicate_agent_id_raises(self, loader):
        with pytest.raises(ConfigError, match="重复"):
            loader.parse({
                "agents": [{"id": "a"}, {"id": "a"}],
            })


# --------------------------------------------------------------------
# build() 构建运行时
# --------------------------------------------------------------------
class TestConfigBuild:
    def test_build_creates_agents_and_session(self, loader):
        config = loader.parse({
            "agents": [
                {"id": "a1", "tools": ["calculator"]},
            ],
            "session": {"agent": "a1"},
        })
        llm, agents, session = loader.build(config, llm_client=FakeLLM())

        assert "a1" in agents
        agent = agents["a1"]
        # 工具应被挂载
        assert "calculator" in agent.tool_registry.tools
        # 会话绑定到 a1
        assert session.agent is agent

    def test_build_with_multiple_agents(self, loader):
        config = loader.parse({
            "agents": [
                {"id": "a1", "capabilities": ["chat"]},
                {"id": "a2", "capabilities": ["analysis"], "tools": ["calculator"]},
            ],
            "session": {"agent": "a2"},
        })
        llm, agents, session = loader.build(config, llm_client=FakeLLM())
        assert set(agents.keys()) == {"a1", "a2"}
        # 会话绑定 a2
        assert session.agent.agent_id == "a2"
        # a2 有工具，a1 没有
        assert "calculator" in agents["a2"].tool_registry.tools
        # 能力声明生效
        assert agents["a1"].has_capability("chat")
        assert agents["a2"].has_capability("analysis")


# --------------------------------------------------------------------
# 示例配置文件集成测试
# --------------------------------------------------------------------
class TestExampleConfigFile:
    def test_example_yaml_loads(self, loader):
        """项目自带的示例配置应能被正确加载"""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "example.yaml",
        )
        assert os.path.exists(path), f"示例配置不存在: {path}"
        config = loader.load_file(path)
        assert config.model == "glm-4-flash"
        assert any(a.id == "assistant" for a in config.agents)
        assert config.session_agent == "assistant"
        # assistant 应有 web_search 和 calculator 工具
        assistant = next(a for a in config.agents if a.id == "assistant")
        assert "web_search" in assistant.tools
        assert "calculator" in assistant.tools
