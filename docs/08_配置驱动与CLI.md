# 第 8 章：配置驱动与 CLI

> 目标：理解如何用 YAML 声明 Agent 与团队，以及 CLI 如何把框架变成可用工具。

## 8.1 这一章解决什么问题

前面几章都是写 Python 代码来构建 Agent。但很多时候我们只想"配置一下就跑起来"——换个 system_prompt、加个工具、组个团队，不想每次都改代码。

PyAgentKit 提供 **YAML 声明式编排** + **命令行工具**，让你用配置文件而非代码来定义 Agent 系统。这是"产品化"的关键一步。

## 8.2 ToolFactory——工具按名实例化（`core/tool_factory.py`）

配置文件里只能写字符串，但工具是 Python 对象。ToolFactory 是桥梁：把"工具名"映射到"构造器"。

```python
class ToolFactory:
    def __init__(self):
        self._registry = {
            "web_search": lambda: WebSearchTool(),
            "calculator": CalculatorTool,
            "file_read": FileReadTool,
            "database": DatabaseTool,
            ...
        }

    def create(self, name) -> Tool:
        """按名称创建工具实例"""
        return self._registry[name]()
```

配置里写 `tools: [web_search, calculator]`，ToolFactory 把它们实例化成对象。也支持 `register()` 注册自定义工具。

## 8.3 YAML 配置模式

### 单 Agent 配置

```yaml
# configs/example.yaml
llm:
  model: glm-4-flash

agents:
  - id: assistant
    name: 智能助手
    system_prompt: "你是一个专业的中文助手。"
    capabilities: [chat, search]
    tools: [web_search, calculator]

session:
  agent: assistant          # 交互时绑定哪个 agent
  max_history: 10
  system_context: "请用中文回复。"
```

### 团队配置

```yaml
# configs/team_example.yaml
llm:
  model: glm-4-flash

agents:
  - id: leader
    capabilities: [plan]
  - id: researcher
    capabilities: [search]
    tools: [web_search]
  - id: writer
    capabilities: [write]

team:                        # 声明 team 块即启用团队模式
  name: 内容组
  process: hierarchical      # 或 sequential
  leader: leader             # hierarchical 必填
  members: [researcher, writer]
  max_subtasks: 4
```

**关键区别**：有 `team:` 块 → 团队模式（Leader 编排）；无 → 单 Agent 会话模式。

## 8.4 ConfigLoader——解析与构建（`core/config.py`）

ConfigLoader 把 YAML 变成运行时对象，分两步：**解析（parse）** 和 **构建（build）**。

### 解析阶段（严格校验，提前失败）

```python
loader = ConfigLoader()
config = loader.load_file("configs/example.yaml")
# config 是 AppConfig 数据类
```

解析时做严格校验，错误立即报错（不让问题留到运行时）：
- 缺 `id` → 报错
- 引用不存在的工具 → 报错并列出可用工具
- `session.agent` 未声明 → 报错
- `team.process` 不是 sequential/hierarchical → 报错
- hierarchical 模式缺 leader → 报错
- leader/member 引用不存在的 agent → 报错

### 构建阶段

```python
llm, agents, session = loader.build(config)
# llm: GLMClient 实例
# agents: {agent_id: Agent} 字典
# session: 绑定到 session_agent 的 ConversationSession

team = loader.build_team(config, agents)  # 有 team 块时返回 Team，否则 None
```

`build` 内部：
1. 创建 `GLMClient(model=config.model)`
2. 为每个 AgentSpec 构造一个通用 Agent（`_ConfigurableAgent`），挂载声明的工具
3. 构建 ConversationSession

## 8.5 CLI——四个子命令（`cli.py`）

`pyagentkit` 命令基于 Python 标准库 argparse（零额外依赖）：

### chat——交互式对话

```bash
pyagentkit chat --tools web_search calculator
```

构建默认通用助手 Agent，包裹在 ConversationSession 里，进入交互循环。支持 `:q` 退出、`:clear` 清空上下文。

### ask——单次提问

```bash
pyagentkit ask "计算 (15+27)*3" --tools calculator
```

提一个问题，得到答复后退出。适合脚本化调用。

### run——加载 YAML 配置

```bash
pyagentkit run configs/example.yaml          # 单 Agent 会话模式
pyagentkit run configs/team_example.yaml     # 团队协作模式
```

`cmd_run` 的逻辑：
1. `ConfigLoader.load_file()` 解析配置
2. `build()` 得到 llm/agents/session
3. `build_team()` 尝试构建团队
4. **有团队** → 进入团队交互循环（`task>` 提示符，输入即触发 `team.run`）
5. **无团队** → 进入会话交互循环

团队模式额外支持 `:summary` 查看团队摘要。

### tools——列出可用工具

```bash
pyagentkit tools
# 可用工具：
#   - calculator
#   - database
#   - file_read
#   - web_search
#   - web_search_mock
```

## 8.6 从配置到执行的完整链路

以 `pyagentkit run configs/team_example.yaml` 为例：

```
team_example.yaml
   │
   ├─ ConfigLoader.load_file()
   │    └─ yaml.safe_load → parse() → AppConfig（含 TeamSpec）
   │
   ├─ loader.build(config)
   │    └─ GLMClient + 3 个 Agent（leader/researcher/writer）+ Session
   │
   ├─ loader.build_team(config, agents)
   │    └─ HierarchicalProcess + Team(members, leader)
   │
   └─ _team_interactive_loop(team)
        └─ 用户输入任务 → team.run(task)
             └─ Leader 规划 → 成员执行 → Leader 汇总 → 输出
```

## 8.7 自定义：注册自己的工具到工厂

```python
from core.tool_factory import default_factory

class MyTool:
    def __init__(self):
        self.name = "my_tool"
        ...

default_factory.register("my_tool", MyTool)
```

注册后就能在 YAML 里用 `tools: [my_tool]`。

## 8.8 设计权衡：为何用代码生成 Agent 而非反射

`_build_agent` 在 `build` 时动态定义一个 `_ConfigurableAgent(Agent)` 内部类，而非用反射加载用户写的类。这样配置是"声明式的"（用户只填字段），框架提供统一的 Agent 实现，降低使用门槛。代价是配置驱动的 Agent 行为固定（receive 就是 think+print）；要复杂行为仍需自己写 Python 类。

## 8.9 与其他模块的关系

- ToolFactory 实例化工具（第 3 章）
- build 构造的 Agent 复用 think/brain/tool_registry（第 2-4 章）
- build_team 构造 Team（第 6 章）
- CLI 是所有能力的统一入口

---

## 教程完结

🎉 恭喜读完全部 8 章！你现在掌握了 PyAgentKit 的完整设计：

- **Agent + ReAct**：思考与行动的循环（第 2 章）
- **工具系统**：安全的"手"（第 3 章）
- **记忆体系**：KV/向量/会话三层统一（第 4 章）
- **消息路由**：同步与异步（第 5 章）
- **团队协作**：Sequential/Hierarchical（第 6 章）
- **可观测性**：流式/Token/Trace（第 7 章）
- **产品化**：YAML + CLI（第 8 章）

下一步建议：
- 读 `agents.py` 和各 `*_demo.py` 看真实用法
- 用 `pyagentkit run configs/team_example.yaml` 跑通团队
- 基于 `core/` 开始构建你自己的 Agent 应用

← [上一章：异步流式与可观测](07_异步流式与可观测.md) | [回到教程首页 →](README.md)
