# 更新日志（Changelog）

本项目的显著变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循语义化版本。历史版本的精确日期以 [git 历史](https://github.com/zhouchunyuan-cn/PyAgentKit/commits/main) 为准。

## [0.4.0] - 2026-10-16

一次聚焦"接入 2026 年主流实践"的架构升级，同时完成一轮全面的代码评审修复。

### 新增

- **真流式 + 事件流**：`stream_callback` 贯穿 ReAct 主循环（首字延迟≈模型出字时间）；
  新增 `AgentStreamEvent` 结构化事件（reasoning / text / tool_call / tool_result）；
  流式工具调用碎片累积拼装；流失败安全回退；CLI chat 实时渲染
- **混合推理模型适配**（GLM-4.5+）：`reasoning_content` 全链路透传、多轮回传
  （Preserved Thinking）、`GLMClient(thinking=...)` 开关
- **结构化输出**：`chat(response_format=...)`；HierarchicalProcess 规划走 JSON 模式，
  多级解析回退
- **工具参数校验**：按工具 JSON Schema 做必填检查与类型矫正，错误带 schema 回传模型自纠
- **异步接口**：`LLMClient.achat / achat_stream`、`Agent.think_async`
  （线程桥接，contextvars 传播使追踪与消息链深度不丢失）
- **并行执行**：ReAct 循环内多工具并发（结果按序回填）；HierarchicalProcess
  `parallel_subtasks` 按成员分组 fan-out；YAML `team.parallel`
- **上下文工程**：长期记忆注入受条数/字符预算约束；`ConversationSession(summarize_on_trim=True)`
  会话滚动摘要压缩（失败回退硬截断）
- **真 MCP 客户端**（`core/mcp_client.py`）：官方协议（initialize → tools/list → call_tool），
  Streamable HTTP 与 stdio 双传输；`ToolRegistry.register_mcp_server()` 一步接入；
  可选依赖 `pip install -e ".[mcp]"`
- **断点续跑（Checkpoint）**（`core/checkpoint.py`）：每步原子落盘；
  `run(run_id=...)` 崩溃恢复、`run_resumable(step_limit=...)` 分段执行、
  完成态幂等、Hierarchical 规划复用
- **OpenAI 兼容客户端**（`core/openai_client.py`）：任何 OpenAI Chat Completions
  兼容端点（OpenAI/DeepSeek/Moonshot/vLLM/Ollama）即插即用，零新增依赖；
  YAML `llm.provider: openai` 一行切换——LLM 抽象层双实现落地
- 工程配套：README 徽章与教学定位声明、CHANGELOG、实战篇升级为 0.4 毕业项目

### 修复（代码评审）

- Memory/VectorMemory 持久化改原子写（先序列化再临时文件 + replace），坏值不再破坏既有数据
- Message 增加 hops/max_hops 链路预算，双 Router 强制防环路；同步 Router 改迭代泵投递，
  长转发链不再递归栈溢出
- AsyncRouter `stop()` 哨兵式分轮排空（修复永久死锁）；`drain()` 覆盖同步桥在途消息
  （修复消息链提前返回丢消息）
- `file_read` 默认拒绝隐藏/敏感文件（`.env`、密钥），加单文件大小上限
- DatabaseTool 只读白名单（拦截 `WITH ... DELETE` / 写向 PRAGMA）；CalculatorTool
  幂指数与表达式长度上限；web_search 提取真实摘要
- Tracer 改 contextvars（异步路径不再丢 span）+ span 上限；ReactEngine 降级路径
  不再重复记录会话历史；trace token 改本次差值
- SharedContext 保留同成员多次产出；Router 消息历史封顶；collaboration 如实返回投递结果；
  Memory 并发锁与 TTL 单调时钟；VectorMemory 拟合降为每 add 一次、重复 id 覆盖清理

### 变更

- 日志统一走 stderr（stdout 留给程序输出）；CLI `--tools=a,b` 逗号语法与参数顺序提示、
  无子命令退出码 2
- `Memory.store` 对非法类型/不可序列化值快速失败（原为静默）
- 依赖版本加上界（numpy<3 / scikit-learn<2）；pytest 移回 dev extra；
  git 清理误跟踪的 `.pyc`/`.idea`，`.gitignore` 收紧
- 测试 271 → 397；版本 0.3.0 → 0.4.0

## [0.3.0] 及更早

按主题归纳（详细提交见 git 历史）：

- **核心框架建成**：Agent/ReAct 循环、工具系统（AST 计算器、文件沙箱、SQLite）、
  KV/向量/会话三层记忆与 MemoryManager、Router/AsyncRouter、Team
  （Sequential/Hierarchical）、YAML 配置与 CLI、Tracer/Token 统计
- **架构演进**：拆出 ReactEngine（推理内聚）；移除 Orchestrator，
  确立 Team 与 Router 为并行的协作模型（ADR-0001）；生命周期钩子系统（AgentHooks）
- **健壮性**：think 循环优雅降级；Memory 默认不落盘 + `enable_persistence()`；
  Team 流式回调传播；AsyncRouter 同步桥线程安全；多轮历史重复发送修复（省 50% token）；
  MCP 集成工具懒注册；流式双调用 bug 修复
- **教程与工程**：8 章开发教程 + 实战篇；CI/ruff/pre-commit 脚手架；
  跨模块集成测试补齐；教程与实现多轮同步
