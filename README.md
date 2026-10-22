# PyAgentKit

[![CI](https://github.com/zhouchunyuan-cn/PyAgentKit/actions/workflows/ci.yml/badge.svg)](https://github.com/zhouchunyuan-cn/PyAgentKit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-426%20passed-brightgreen)](tests/)
[![Version](https://img.shields.io/badge/version-0.5.0-orange)](CHANGELOG.md)

> 轻量级、可扩展的多智能体框架 —— 基于智谱 GLM，开箱即用的 ReAct Agent、团队协作与异步路由。

> **定位说明**：这是一个**教学参考实现**（约 7000 行核心代码 + 8 章教程）——为了可通读性而刻意保持小巧，
> 覆盖现代 Agent 框架的完整概念（ReAct / 工具 / 记忆 / 编排 / MCP / 检查点 / 可观测），
> 但**不是生产框架**：单机单进程、文件存储、无限流与鉴权。想理解原理、借鉴架构 → 非常合适；
> 想直接部署业务 → 请用工业级框架。

PyAgentKit 让你用最少的代码构建能**思考、调用工具、记忆上下文、彼此协作、团队编排**的 AI Agent。支持命令行交互、YAML 声明式编排、流式输出、Token 统计与执行追踪，以及完整的多智能体编程接口。

📚 **[开发教程（8 章 + 实战）](docs/README.md)** — 从 ReAct 原理到团队协作，循序渐进讲透每个模块的设计与实现。新手建议从 [快速开始](docs/01_快速开始.md) 读起。

```bash
# 一行命令开始对话（自带网络搜索 + 计算）
pyagentkit chat --tools web_search calculator

# 声明一个团队，Leader 自动拆任务分配成员
pyagentkit run configs/team_example.yaml
```

## 效果预览

`pyagentkit chat --tools web_search calculator` 的交互形态（示意转录，思考过程暗色、工具调用单行提示、答复逐字流出）：

```text
你> 用 2026 年的数据估算：如果量子计算算力每年翻倍，3 年后相对现在的倍数是多少？
· 调用工具 web_search({'query': 'quantum computing power growth 2026'})
· 调用工具 calculator({'expression': '2**3'})
助手: 根据搜索到的行业报告，量子计算的有效算力近年保持约每年翻倍的增速。
按翻倍规律 3 年后算力是现在的 2**3 = 8 倍……

task> 写一份AI行业分析报告            # 团队模式：断点续跑（崩溃/暂停后同 run_id 续跑）
[团队执行中...]
  ├─ Leader 规划：3 个子任务（search → analysis → write）
  ├─ researcher 执行… ✓ 已落盘（report-101 检查点）
  └─ 进程中断/暂停 → 再次执行 team.run("写一份AI行业分析报告", run_id="report-101")
     → 从 writer 断点继续，已跑步骤不重复计费
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
| 🔌 **LLM 抽象层** | GLM 原生 + OpenAI 兼容双实现（DeepSeek/vLLM/Ollama 即插即用）；重试、Key 校验内置 |
| 🛠️ **插件化工具** | calculator / web_search / file_read / database，支持 function-calling |
| 🔍 **向量记忆** | 语义召回相关经验（本地 TF-IDF 零成本，可切 GLM embedding） |
| 💬 **多轮对话** | Session 维护上下文，支持追问与指代消解 |
| 🤝 **能力驱动协作** | Agent 声明能力，协作策略按能力匹配（不依赖硬编码名字） |
| ⚡ **并行执行** | 一步内多个工具调用并发执行；Hierarchical 子任务可按成员分组 fan-out（`parallel: true`） |
| 🌐 **真 MCP 客户端** | 官方协议（initialize/tools/list/call_tool），Streamable HTTP 与 stdio 双传输，工具自动注册 |
| 💾 **断点续跑（Checkpoint）** | 每步原子落盘；崩溃/暂停后同 run_id 从断点继续，已跑步骤不重复付费；完成态幂等 |
| 🏢 **企业级关切（教学版）** | 审计日志（含脱敏）、token 预算熔断、工具调用审批（HITL）、SSRF 白名单、检查点跨进程锁、eval 回归基准 |
| 🔗 **异步接口** | `think_async` / `achat` / `achat_stream`，asyncio 程序与多 Agent 并发场景即插即用 |
| 📋 **YAML 配置** | 声明式编排 Agent / 工具 / 会话 / 团队，零代码定制 |
| 🖥️ **命令行工具** | `pyagentkit` 命令：交互对话 / 单次提问 / 跑配置 / 列工具 |
| 🔒 **工程化** | 统一日志、异常边界、消息跳数防环、426 个单元测试守护 |

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
│  Collaboration(能力匹配)   Monitor(监控)                              │
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

### 流式输出（真流式 + 结构化事件）

```python
# stream_callback：助手文本增量实时回调（贯穿 ReAct 主循环，首字延迟≈模型出字时间）
chunks = []
agent.think("介绍量子纠缠", stream_callback=lambda s: chunks.append(s))
print("".join(chunks))

# events_callback：结构化事件流（推理模型思考过程 / 文本 / 工具调用全程可见）
from core import AgentStreamEvent

def on_event(ev: AgentStreamEvent):
    if ev.type == "reasoning":  ...  # 思考增量（GLM-4.5+ 混合推理模型）
    elif ev.type == "text":     ...  # 文本增量
    elif ev.type == "tool_call":...  # {"name", "arguments"}

agent.think("搜索量子纠缠最新进展", events_callback=on_event)

# 混合推理模型：GLMClient 可显式开关思考（默认由模型决定）
llm = GLMClient(model="glm-4.7-flash", thinking=True)   # 输出含 reasoning_content
llm = GLMClient(model="glm-4.7-flash", thinking=False)  # 关闭思考省 token
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

同步 Router 的投递采用迭代泵（消息排队逐条投递），长转发链不会递归栈溢出；
消息还带 hops/max_hops 跳数预算，Agent 互发回复形成环路时自动截断。
AsyncRouter 则用 asyncio.Queue 解耦投递与处理，天然适合并发场景：

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

## 企业级关切（教学版实现）

生产框架的安全与治理关切，以最小可读的实现呈现：

```python
# 1) 审计日志：谁让 Agent 干了什么（敏感参数自动脱敏），JSONL 落盘
from core import AuditLogger
audit = AuditLogger("audit.jsonl", actor="user-123")
audit.install(agent)

# 2) token 预算：超限后不再发起工具循环步，优雅收尾
agent = MyAgent(llm_client=llm, token_budget=100_000)
agent.budget_exceeded   # 最近一次是否因预算提前收尾

# 3) 工具调用审批（human-in-the-loop）：拒绝会作为错误回传模型自纠
agent.tool_registry.set_approver(
    lambda name, args: input(f"允许 {name}({args})? [y/N] ").lower() == "y",
    tools=["database", "file_read"],
)

# 4) MCP 端点 SSRF 防护：非本机必须 https；云元数据/链路本地段永远拒绝
MCPServerConfig(url="https://mcp.example.com/mcp")            # ✓
MCPServerConfig(url="http://169.254.169.254/latest")          # ✗ 拒绝

# 5) 检查点跨进程锁：两个进程同续跑同一 run_id，后者等待/报忙而非竞写

# 6) trace 导出与串联：trace_id 随消息路由传播，完成时导出（可桥接 OTel）
from core import JsonlTraceExporter, Tracer
Tracer.set_exporter(JsonlTraceExporter("traces.jsonl"))

# 7) eval 回归基准：改 prompt/换模型前后跑同一组用例对比通过率
from core import EvalCase, run_eval, make_llm_judge
report = run_eval(agent, [EvalCase(input="计算 15*3", expect_keywords=["45"])])
print(report.format_text())
```

## 更换模型后端（OpenAI 兼容端点）

`LLMClient` 抽象层有两个实现——GLM 原生客户端，以及任何 OpenAI Chat Completions 兼容端点
（OpenAI、DeepSeek、Moonshot、vLLM、Ollama 兼容模式，零新增依赖）。上层代码一行不改：

```python
from core import OpenAICompatClient

llm = OpenAICompatClient(base_url="https://api.deepseek.com/v1", model="deepseek-chat")
agent = MyAgent(llm_client=llm)   # 流式/工具调用/思考透传全部照常工作
```

YAML 一行切换（Key 从环境变量 `OPENAI_API_KEY` 读取）：

```yaml
llm:
  provider: openai
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat
```

## 真 MCP 客户端（官方协议）

接入任何标准 MCP 服务器（需可选依赖：`pip install -e ".[mcp]"`），工具自动发现并注册为本地 Tool：

```python
from core import MCPServerConfig

# Streamable HTTP 服务器
names = agent.tool_registry.register_mcp_server(
    MCPServerConfig(url="http://localhost:8000/mcp")
)
# 或 stdio 子进程服务器
names = agent.tool_registry.register_mcp_server(
    MCPServerConfig(command="python", args=["my_server.py"])
)
# names = 服务器上发现的工具名列表，schema 随迁，LLM 可直接调用
```

会话懒建立并复用（stdio 服务器不因调用重启）；`close_mcp_servers()` 统一断开。

## 并行与异步

```python
# 一步内多个工具调用自动并发执行（上限 4）
agent.think("同时查这三个城市的天气")   # 模型返回多个 tool_calls 时并行跑

# 团队子任务 fan-out：按成员分组并行、组内串行
team = Team(name="并行组", members=[...], process=HierarchicalProcess(parallel_subtasks=True), leader=mgr)

# asyncio 场景
reply = await agent.think_async("你好")
async for chunk in llm.achat_stream(messages): ...
```

YAML 团队配置加一行即可开启子任务并行：`team: { process: hierarchical, parallel: true }`。

## 断点续跑（Checkpoint）

Team 任务每完成一个成员/子任务即原子落盘；进程崩溃、或主动暂停后，同 run_id 再执行会从断点继续——已完成的步骤（含 Leader 的规划）不会重跑，LLM 不重复付费：

```python
from core import FileCheckpointStore

team = Team(name="内容组", members=[researcher, writer, reviewer],
            process=SequentialProcess(),
            checkpoint_store=FileCheckpointStore(".checkpoints"))

# 崩溃恢复：同 run_id 重跑自动从断点继续；已完成的 run 直接返回缓存结果
result = team.run("写一份行业分析", run_id="report-101")

# 分步执行（人工审批 / 分段预算）
outcome = team.run_resumable("写一份行业分析", run_id="report-101", step_limit=1)
if not outcome.completed:
    ...  # 检查中间产出，择时续跑
    outcome = team.run_resumable("写一份行业分析", run_id="report-101")
print(outcome.result, outcome.steps_done)
```

Hierarchical 模式续跑会直接复用断点中的任务规划（省一次 Leader 规划调用）；
任务描述变化时旧检查点自动作废重新开始。

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
| `core/collaboration.py` | 能力驱动的协作策略 |
| `core/team.py` | 团队协作（Sequential / Hierarchical 流程） |
| `core/trace.py` | 执行追踪（每步耗时与 token） |
| `core/config.py` | YAML 配置解析与运行时构建（含 team） |
| `core/tool_factory.py` | 按名称实例化工具 |
| `core/message.py` | 消息协议 |
| `core/monitor.py` | 系统监控与报告导出 |
| `core/mcp_tools.py` | MCP 协议工具集成 |
| `core/mcp_client.py` | 真 MCP 客户端（官方协议，HTTP/stdio，自动发现注册） |
| `core/checkpoint.py` | 检查点存储与 RunOutcome（Team 断点续跑） |
| `core/logging_config.py` | 统一日志配置 |

## 技术栈

- **LLM**: 智谱 GLM（`glm-4-flash` 免费 / `glm-4.7-flash`）
- **SDK**: zhipuai、python-dotenv、pyyaml
- **向量记忆**: scikit-learn (TF-IDF) + numpy（本地零成本，可切 GLM embedding）
- **异步**: asyncio（AsyncRouter）
- **测试**: pytest（426 个用例）
- **Python**: ≥ 3.10

## 测试

```bash
pytest tests/                   # 运行全部 426 个测试
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
5. **安全** — AST 解析替代 eval、文件路径越界检测、敏感文件默认拒读、
   数据库只读白名单、消息跳数防环、异常边界
6. **可测试** — 核心逻辑有 426 个单元测试守护，依赖可 mock

## 应用场景

智能客服、研究助手、数据分析平台、自动化办公、教育辅导、工具集成平台、团队协作开发。

## License

MIT（见 [LICENSE](LICENSE)）
