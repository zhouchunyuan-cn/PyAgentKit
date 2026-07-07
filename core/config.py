"""
YAML 配置驱动

用声明式 YAML 编排 Agent / 工具 / 会话，无需写 Python 代码。

配置模式（示例）：
    llm:
      model: glm-4-flash
    agents:
      - id: assistant
        name: 助手
        system_prompt: "你是一个通用助手"
        capabilities: [chat]
        tools: [web_search, calculator]
    session:
      agent: assistant          # 默认交互的 agent
      system_context: "..."

ConfigLoader 负责解析 YAML 并构建运行时对象（LLM、Agent 列表、Session）。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import logging

from .agent import Agent
from .llm import GLMClient, LLMClient
from .session import ConversationSession
from .tool_factory import ToolFactory, default_factory

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """配置错误（结构非法、缺字段、引用不存在等）"""


@dataclass
class AgentSpec:
    """单个 Agent 的配置规格"""
    id: str
    name: str
    system_prompt: str = ""
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    max_steps: int = Agent.DEFAULT_MAX_STEPS


@dataclass
class TeamSpec:
    """团队的配置规格"""
    name: str = "team"
    process: str = "sequential"          # "sequential" 或 "hierarchical"
    leader: Optional[str] = None          # hierarchical 模式下的 leader agent id
    members: List[str] = field(default_factory=list)  # 成员 agent id 列表；为空时用全部 agents
    max_subtasks: int = 5                 # hierarchical 用
    max_steps: int = 10                   # hierarchical 用


@dataclass
class AppConfig:
    """整份配置解析后的规格"""
    model: str = "glm-4-flash"
    agents: List[AgentSpec] = field(default_factory=list)
    session_agent: Optional[str] = None
    session_system_context: Optional[str] = None
    session_max_history: int = 10
    team: Optional[TeamSpec] = None       # 声明 team 块后启用团队模式


class ConfigLoader:
    """
    YAML 配置加载器

    解析 YAML -> AppConfig -> 构建运行时对象（LLM、Agent、Session）。
    """

    def __init__(self, tool_factory: Optional[ToolFactory] = None):
        self.tool_factory = tool_factory or default_factory

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def load_file(self, path: str) -> AppConfig:
        """从 YAML 文件加载配置"""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("未安装 pyyaml，请运行：pip install pyyaml") from e

        if not os.path.exists(path):
            raise ConfigError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError("配置根节点必须是字典")
        return self.parse(raw)

    def parse(self, raw: Dict[str, Any]) -> AppConfig:
        """
        解析已加载的 YAML 字典为 AppConfig

        Args:
            raw: YAML 解析后的字典

        Returns:
            AppConfig

        Raises:
            ConfigError: 配置非法
        """
        llm_section = raw.get("llm", {}) or {}
        model = llm_section.get("model", "glm-4-flash")

        agents_raw = raw.get("agents", []) or []
        if not agents_raw:
            raise ConfigError("配置必须至少声明一个 agent")
        if not isinstance(agents_raw, list):
            raise ConfigError("agents 必须是列表")

        agent_specs: List[AgentSpec] = []
        agent_ids = set()
        for i, a in enumerate(agents_raw):
            spec = self._parse_agent(a, i)
            if spec.id in agent_ids:
                raise ConfigError(f"重复的 agent id: {spec.id}")
            agent_ids.add(spec.id)
            agent_specs.append(spec)

        session_section = raw.get("session", {}) or {}
        session_agent = session_section.get("agent")
        # 校验 session 引用的 agent 存在
        if session_agent and session_agent not in agent_ids:
            raise ConfigError(
                f"session.agent '{session_agent}' 未在 agents 中声明，"
                f"可用: {sorted(agent_ids)}"
            )

        # 解析 team 块（可选）：区分"无 team 键"与"team 为空字典"
        team_section = raw.get("team")
        team_spec: Optional[TeamSpec] = None
        if team_section is not None:
            team_spec = self._parse_team(team_section, agent_ids)

        return AppConfig(
            model=model,
            agents=agent_specs,
            session_agent=session_agent or agent_specs[0].id,
            session_system_context=session_section.get("system_context"),
            session_max_history=session_section.get("max_history", 10),
            team=team_spec,
        )

    def _parse_agent(self, raw: Dict[str, Any], index: int) -> AgentSpec:
        """解析单个 agent 配置块"""
        if not isinstance(raw, dict):
            raise ConfigError(f"agents[{index}] 必须是字典")

        agent_id = raw.get("id")
        if not agent_id:
            raise ConfigError(f"agents[{index}] 缺少必填字段 'id'")
        name = raw.get("name", agent_id)

        # 校验引用的工具都存在（提前失败，给出清晰错误）
        tools = raw.get("tools", []) or []
        if not isinstance(tools, list):
            raise ConfigError(f"agent '{agent_id}' 的 tools 必须是列表")
        available = set(self.tool_factory.list_tools())
        unknown = [t for t in tools if t not in available]
        if unknown:
            raise ConfigError(
                f"agent '{agent_id}' 引用了未知工具 {unknown}，"
                f"可用工具: {sorted(available)}"
            )

        return AgentSpec(
            id=str(agent_id),
            name=str(name),
            system_prompt=raw.get("system_prompt", ""),
            capabilities=list(raw.get("capabilities", []) or []),
            tags=list(raw.get("tags", []) or []),
            tools=[str(t) for t in tools],
            max_steps=int(raw.get("max_steps", Agent.DEFAULT_MAX_STEPS)),
        )

    def _parse_team(self, raw: Dict[str, Any], agent_ids: set) -> TeamSpec:
        """解析 team 配置块"""
        if not isinstance(raw, dict):
            raise ConfigError("team 必须是字典")

        process = str(raw.get("process", "sequential")).lower()
        if process not in ("sequential", "hierarchical"):
            raise ConfigError(
                f"team.process '{process}' 非法，可选: sequential / hierarchical"
            )

        leader = raw.get("leader")
        if process == "hierarchical" and not leader:
            raise ConfigError("hierarchical 流程必须指定 team.leader（agent id）")
        if leader and leader not in agent_ids:
            raise ConfigError(
                f"team.leader '{leader}' 未在 agents 中声明，可用: {sorted(agent_ids)}"
            )

        members = list(raw.get("members", []) or [])
        for m in members:
            if m not in agent_ids:
                raise ConfigError(
                    f"team.members 中的 '{m}' 未在 agents 中声明，可用: {sorted(agent_ids)}"
                )

        return TeamSpec(
            name=str(raw.get("name", "team")),
            process=process,
            leader=leader,
            members=[str(m) for m in members],
            max_subtasks=int(raw.get("max_subtasks", 5)),
            max_steps=int(raw.get("max_steps", 10)),
        )

    # ------------------------------------------------------------------
    # 构建运行时
    # ------------------------------------------------------------------
    def build(self, config: AppConfig, llm_client: Optional[LLMClient] = None):
        """
        由 AppConfig 构建运行时对象

        Args:
            config: 已解析的 AppConfig
            llm_client: 可选的 LLM 客户端（None 则按 config.model 新建 GLMClient）

        Returns:
            (llm_client, agents_dict, session)
            - agents_dict: {agent_id: Agent}
            - session: 绑定到 session_agent 的 ConversationSession
        """
        llm = llm_client or GLMClient(model=config.model)

        agents: Dict[str, Agent] = {}
        for spec in config.agents:
            agents[spec.id] = self._build_agent(spec, llm)

        # 构建会话
        target_agent = agents[config.session_agent]
        session = ConversationSession(
            agent=target_agent,
            max_history=config.session_max_history,
            system_context=config.session_system_context,
        )
        return llm, agents, session

    def build_team(self, config: AppConfig, agents: Dict[str, Agent]) -> Optional["Team"]:
        """
        由配置构建 Team（若配置含 team 块）

        Args:
            config: 已解析的 AppConfig
            agents: build() 产出的 agents 字典

        Returns:
            Team 实例；配置无 team 块时返回 None
        """
        if not config.team:
            return None

        # 延迟导入，避免循环依赖（team.py 不依赖 config.py）
        from .team import Team, SequentialProcess, HierarchicalProcess

        spec = config.team
        members = [agents[mid] for mid in spec.members] if spec.members else list(agents.values())
        if not members:
            raise ConfigError("team 没有可用成员")

        leader = agents.get(spec.leader) if spec.leader else None
        if spec.process == "hierarchical":
            process = HierarchicalProcess(max_subtasks=spec.max_subtasks, max_steps=spec.max_steps)
        else:
            process = SequentialProcess()

        return Team(
            name=spec.name,
            members=members,
            process=process,
            leader=leader,
        )

    def _build_agent(self, spec: AgentSpec, llm: LLMClient) -> Agent:
        """构建单个通用 Agent 并挂载工具"""

        class _ConfigurableAgent(Agent):
            """由配置驱动的通用 Agent"""

            def __init__(self, spec, llm_client):
                super().__init__(
                    agent_id=spec.id,
                    name=spec.name,
                    system_prompt=spec.system_prompt or None,
                    llm_client=llm_client,
                    capabilities=spec.capabilities,
                    tags=spec.tags,
                    max_steps=spec.max_steps,
                )

            def receive(self, message):
                # 配置驱动的 agent 默认走 think 并打印
                try:
                    reply = self.think(message.content)
                    print(f"[{self.name}] {reply}")
                except Exception as e:
                    print(f"[{self.name}] 处理失败: {e}")

        agent = _ConfigurableAgent(spec, llm)

        # 挂载声明的工具
        for tool_name in spec.tools:
            tool = self.tool_factory.create(tool_name)
            agent.tool_registry.register(tool)

        logger.debug("构建 agent '%s'，工具=%s，能力=%s",
                     spec.id, spec.tools, spec.capabilities)
        return agent
