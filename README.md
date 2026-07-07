# PyAgentKit

> 轻量级、可扩展的多智能体框架 —— 基于智谱 GLM，开箱即用的 ReAct Agent、团队协作与异步路由。

PyAgentKit 让你用最少的代码构建能**思考、调用工具、记忆上下文、彼此协作、团队编排**的 AI Agent。支持命令行交互、YAML 声明式编排、流式输出、Token 统计与执行追踪，以及完整的多智能体编程接口。

```bash
# 一行命令开始对话（自带网络搜索 + 计算）
pyagentkit chat --tools web_search calculator

# 声明一个团队，Leader 自动拆任务分配成员
pyagentkit run configs/team_example.yaml
```

## 核心能力

| 能力 | 说明 |
|------|------|
| 🧠 **ReAct 推理循环** | Agent 自主"思考 → 调用工具 → 观察结果 → 再思考"，直到给出答案 |
| 🤝 **团队协作（Team）** | Sequential 顺序接力 / Hierarchical Leader 自动编排两种流程 |
| 🔄 **同步 + 异步路由** | Router 同步消息总线 + AsyncRouter 队列解耦（根治递归栈溢出） |
| 🌊 **流式输出** | chat_stream 逐块产出，stream_callback 实时回调，体验流畅 |
| 📊 **Token 统计** | 每次 think 累计 prompt/completion/total token，成本可控 |
| 🔍 **执行追踪（Trace）** | 记录每步耗时与 token，复杂协作不再黑盒 |
| 💾 **统一记忆** | MemoryManager 外观聚合 KV / 向量 / 会话三类记忆，一个接口 |
| 🔌 **LLM 抽象层** | 内置 GLM 实现，可扩展其他模型；重试、Key 校验内置 |
| 🛠️ **插件化工具** | calculator / web_search / file_read / database，支持 function-calling |
| 🔍 **向量记忆** | 语义召回相关经验（本地 TF-IDF 零成本，可切 GLM embedding） |
| 💬 **多轮对话** | Session 维护上下文，支持追问与指代消解 |
| 🤝 **能力驱动协作** | Agent 声明能力，协作策略按能力匹配（不依赖硬编码名字） |
| 📋 **YAML 配置** | 声明式编排 Agent / 工具 / 会话 / 团队，零代码定制 |
| 🖥️ **命令行工具** | `pyagentkit` 命令：交互对话 / 单次提问 / 跑配置 / 列工具 |
| 🔒 **工程化** | 统一日志、异常边界、217 个单元测试守护 |

## 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLI / YAML 配置 / Trace                         │
│        cli.py   core/config.py   core/trace.py   core/tool_factory │
├──────────────────────────────────────────────────────────────────┤
│  Team ── Process(Sequential/Hierarchical) ── SharedContext         │
│    │                                                               │
│  ConversationSession ──> Agent.think()  (ReAct + 流式 + Token)     │
│                            │                                       │
│              ┌─────────────┼─────────────┐                         │
│              ▼             ▼             ▼                         │
│         LLMClient     ToolRegistry   MemoryManager                 │
│        (chat/stream)  (calculator…)   (KV/向量/会话)                │
├──────────────────────────────────────────────────────────────────┤
│  Router ◄──同步──► Agents      AsyncRouter ◄──异步队列──► Agents   │
│  Collaboration(能力匹配)   Orchestrator(调度)   Monitor(监控)       │
└──────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装

```bash
git clone <repo-url> && cd PyAgentKit
pip install -e .          # 安装依赖并注册 pyagentkit 命令
```

### 2. 配置 API Key

```bash
cp .env.example .env       # 复制模板
# 编辑 .env，填入你的 Key
```

```
ZHIPUAI_API_KEY=your_api_key_here
```

> API Key 获取：[智谱AI开放平台](https://open.bigmodel.cn/)（注册赠送免费额度）
> 默认用免费的 `glm-4-flash`；`glm-4.7-flash` 同样免费且更强。

### 3. 用起来

```bash
# 列出可用工具
pyagentkit tools

# 单次提问
pyagentkit ask "用三句话介绍量子计算"

# 交互式对话（启用工具 + 流式输出）
pyagentkit chat --tools web_search calculator

# 加载单 Agent 配置
pyagentkit run configs/example.yaml

# 加载团队配置（Leader 自动编排成员协作）
pyagentkit run configs/team_example.yaml
```

交互对话中：`:q` 退出，`:clear` 清空上下文。

## 用代码构建 Agent

```python
from core import GLMClient, Agent

# 1. 创建 LLM 客户端（Key 从环境变量读取）
llm = GLMClient(model="glm-4-flash")

# 2. 定义一个会算数的 Agent
class MathAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="math",
            name="数学助手",
            system_prompt="你是数学专家，需要计算时调用 calculator 工具。",
            llm_client=llm,
            capabilities=["calculate"],
        )
        from core import CalculatorTool
        self.tool_registry.register(CalculatorTool())

    def receive(self, message):
        reply = self.think(message.content)   # 进入 ReAct 循环
        print(f"[{self.name}] {reply}")

agent = MathAgent()
print(agent.think("计算 (15 + 27) * 3"))      # -> 126

# Token 统计（每次 think 自动累计）
print(agent.get_token_usage())  # TokenUsage(prompt_tokens=.., completion_tokens=.., total_tokens=..)
```

### 流式输出

```python
# 传入 stream_callback，最终答复实时逐块回调
chunks = []
agent.think("介绍量子纠缠", stream_callback=lambda s: chunks.append(s))
print("".join(chunks))  # 拼出完整答复
```

### 让多个 Agent 协作

```python
from core import Router, Message

router = Router()
router.register_agent(researcher)    # capabilities=["search"]
router.register_agent(writer)        # capabilities=["write"]

# researcher 处理后会把结果转发给 writer
router.route_message(Message("user", "researcher", "研究量子计算"))
```

### 团队协作（Team）

```python
from core import Team, SequentialProcess, HierarchicalProcess

# 方式一：Sequential 顺序接力（研究→写作→审核）
team = Team(
    name="内容组",
    members=[researcher, writer, reviewer],
    process=SequentialProcess(),
)
result = team.run("写一篇关于量子计算的科普文章")

# 方式二：Hierarchical Leader 自动编排
team = Team(
    name="项目组",
    members=[researcher, writer],
    process=HierarchicalProcess(),
    leader=manager,   # Leader 用 LLM 拆解任务、按能力分配成员
)
result = team.run("完成一份AI行业分析")
```

### 统一记忆 + 向量召回

```python
from core import ConversationSession, MemoryManager, VectorMemory, LocalTfidfEmbedder

# Agent 内置 brain（MemoryManager），自动聚合 KV/向量/会话
agent.brain.remember("user_name", "小明")          # KV
agent.brain.learn("Python 是解释型语言")            # 向量记忆
agent.brain.record_turn("user", "你好")             # 会话
# build_context 一次性聚合三者产出 LLM 的 messages
```

### 执行追踪（Trace）

```python
from core import Tracer

Tracer.enable()
Tracer.start_trace("任务")
agent.think("复杂任务")
trace = Tracer.end_trace()
print(trace.format_text())
# 输出每步耗时与 token：
#   think:助手    2.3s  steps=2 total_tokens=420
```

## YAML 配置驱动

无需写 Python，用配置声明一切。**单 Agent 模式**：

```yaml
# my_assistant.yaml
llm:
  model: glm-4-flash
agents:
  - id: assistant
    system_prompt: "你是一个专业的中文助手，需要时调用工具。"
    capabilities: [chat, search, calculate]
    tools: [web_search, calculator]
session:
  agent: assistant
  max_history: 10
```

**团队模式**（Leader 自动编排）：

```yaml
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
team:
  name: 内容组
  process: hierarchical      # 或 sequential
  leader: leader
  members: [researcher, writer]
  max_subtasks: 4
```

```bash
pyagentkit run my_assistant.yaml        # 单 Agent 会话模式
pyagentkit run configs/team_example.yaml # 团队协作模式
```

完整示例见 [`configs/example.yaml`](configs/example.yaml) 与 [`configs/team_example.yaml`](configs/team_example.yaml)。

## 异步路由（AsyncRouter）

同步 Router 在长链路转发时会递归栈溢出。AsyncRouter 用 asyncio.Queue 解耦：

```python
import asyncio
from core import AsyncRouter, Message

async def main():
    router = AsyncRouter()
    router.register_agent(agent_a)
    router.register_agent(agent_b)
    await router.start()
    await router.route(Message("user", "a", "ping"))
    await router.drain()   # 等待所有消息处理完
    await router.stop()

asyncio.run(main())
```

## 模块清单

| 模块 | 职责 |
|------|------|
| `core/agent.py` | ⭐ Agent 基类 + `think()` ReAct 循环（项目核心） |
| `core/llm.py` | LLM 抽象层 + GLM 实现（含流式 chat_stream、Token 统计） |
| `core/tools.py` | 工具基类 + 内置工具 + HttpTool + function-calling schema |
| `core/memory.py` | 双层记忆（短期/长期 + 持久化） |
| `core/memory_manager.py` | 统一记忆外观（聚合 KV/向量/会话） |
| `core/vector_memory.py` | 向量记忆（语义召回） |
| `core/session.py` | 多轮对话会话管理 |
| `core/router.py` | 同步消息路由（点对点/广播/异常边界） |
| `core/async_router.py` | 异步路由（asyncio.Queue 解耦，根治栈溢出） |
| `core/orchestrator.py` | 任务调度（顺序/并发） |
| `core/collaboration.py` | 能力驱动的协作策略 |
| `core/team.py` | 团队协作（Sequential / Hierarchical 流程） |
| `core/trace.py` | 执行追踪（每步耗时与 token） |
| `core/config.py` | YAML 配置解析与运行时构建（含 team） |
| `core/tool_factory.py` | 按名称实例化工具 |
| `core/message.py` | 消息协议 |
| `core/monitor.py` | 系统监控与报告导出 |
| `core/mcp_tools.py` | MCP 协议工具集成 |
| `core/logging_config.py` | 统一日志配置 |

## 技术栈

- **LLM**: 智谱 GLM（`glm-4-flash` 免费 / `glm-4.7-flash`）
- **SDK**: zhipuai、python-dotenv、pyyaml
- **向量记忆**: scikit-learn (TF-IDF) + numpy（本地零成本，可切 GLM embedding）
- **异步**: asyncio（AsyncRouter）
- **测试**: pytest（217 个用例）
- **Python**: ≥ 3.10

## 测试

```bash
pytest tests/                   # 运行全部 217 个测试
pytest tests/test_team.py -v    # 运行单个模块
```

测试覆盖：calculator 安全解析、记忆过期与持久化、消息路由与异常边界、
向量召回、多轮对话、配置解析、能力协作、团队编排、流式与 token、
异步路由（含 100 跳不栈溢出）、执行追踪等。

## 演示脚本

```bash
python main.py                  # 综合演示（GLM 驱动的 ReAct Agent）
python team_demo.py             # 团队协作（Sequential + Hierarchical）
python async_demo.py            # 异步路由 + 流式输出 + Token 统计
python p2_demo.py               # 能力协作 + 向量记忆 + 多轮对话
python multi_agent_demo.py      # 多 Agent 交互
python mcp_demo.py              # MCP 工具集成
```

## 推荐阅读顺序

第一次接触本项目？按这个顺序读最快上手：

1. **`README.md`**（本文件）— 建立全局认知
2. `core/message.py` → `core/memory.py` → `core/llm.py` — 三大原语
3. **`core/agent.py`** ⭐ — ReAct 循环（读透这个，项目懂八成）
4. `core/tools.py` → `agents.py` — 工具系统与实战范例
5. `core/router.py` → `core/team.py` — 同步路由与团队协作
6. `core/memory_manager.py` → `core/trace.py` — 统一记忆与追踪
7. `core/config.py` → `cli.py` — 配置驱动与命令行

时间有限只读 4 个：`core/agent.py`、`core/llm.py`、`agents.py`、`cli.py`。

## 设计理念

1. **简单性** — 简洁的 API，降低上手门槛
2. **可扩展性** — 模块化设计，LLM / 工具 / embedder / 流程 均可注入替换
3. **解耦** — 抽象层屏蔽具体实现；同步 Router 与异步 AsyncRouter 各司其职
4. **可观测** — Trace 让每步耗时与 token 清晰可见，复杂协作不黑盒
5. **安全** — AST 解析替代 eval、文件路径越界检测、异常边界
6. **可测试** — 核心逻辑有 217 个单元测试守护，依赖可 mock

## 应用场景

智能客服、研究助手、数据分析平台、自动化办公、教育辅导、工具集成平台、团队协作开发。

## License

MIT（见 [LICENSE](LICENSE)）
