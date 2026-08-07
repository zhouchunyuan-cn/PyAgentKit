# ADR-0001：Team 与 Router 是并行的协作模型，不是分层关系

- **状态**：已采纳
- **日期**：2026-08-07
- **背景**：多 Agent 协作机制设计

## 背景

PyAgentKit 有两套多 Agent 协作机制：

1. **Router / AsyncRouter**（消息总线）：Agent 通过 `send()`/`receive()` 互发消息。
   - `ResearchAgent.receive()` → `self.think()` → `self.send("writer", ...)` → `Router.route_message` → `WriterAgent.receive()`
   - 特点：Agent 自主决定发给谁，松散对等

2. **Team / Process**（流程编排）：Leader 或固定流程通过 `SharedContext` 传递产出。
   - `SequentialProcess.execute()` → `member.think(user_input)` → 产出存入 `SharedContext` → 下一个成员拿到累计产出
   - 特点：流程由 Process 决定，成员间不直接通信

代码分析证实：**team.py 中零 Router/Message 引用**，Team 完全通过 `member.think()` 驱动，不产生任何消息。两套系统从未交互。

## 决策

**Team 和 Router 是两种并行的多 Agent 协作模型，不是分层关系。** 用户根据场景二选一，而非叠加使用。

### Router 模型——消息驱动

- **机制**：Agent 间通过消息互发，谁发给谁由 Agent 自己的 `receive()` 逻辑决定
- **适用**：松散对等通信、动态路由、Agent 自主决策转发
- **典型场景**：客服系统按问题类型转接、多 Agent 协商

### Team 模型——流程驱动

- **机制**：Process（Sequential/Hierarchical）按固定或动态流程编排成员，通过 SharedContext 共享产出
- **适用**：有共同目标、有明确分工的任务
- **典型场景**：研究报告（研究→分析→写作）、软件开发（设计→编码→测试）

### 选择指南

| 维度 | Router | Team |
|------|--------|------|
| 通信方式 | Agent 互发消息 | Process 通过 SharedContext 传递 |
| 流程控制 | 分散（每个 Agent 决定） | 集中（Process 决定） |
| 共享状态 | 各自独立 memory | SharedContext 共享黑板 |
| 适合 | 动态、对等、不可预测 | 确定、有分工、有目标 |
| 可观测 | Router 消息历史 + 统计 | Trace 执行追踪 |

## Orchestrator 的处置

**已移除。** Orchestrator 原本是 Router 之上的一个调度层，但其所有能力都已被更专注的类取代：

| Orchestrator 方法 | 取代者 |
|-------------------|--------|
| `sequential_execution` | `Team` + `SequentialProcess` |
| `async_send_message` | `AsyncRouter.route` |
| `run_agent_in_thread` | `AsyncRouter`（max_concurrent） |
| `broadcast_message` | `Router.broadcast` |
| `add_agent` | `Router.register_agent` |

移除理由：它是纯冗余薄包装（8 个方法中 2 个零调用、3 个仅测试调用、3 个是对 Router 的 1 行委托），留着会让用户困惑"该用 Orchestrator 还是 Router"。

## 后果

- 用户需在 Router 和 Team 之间做明确选择，不能混用
- 需要消息路由 → 用 Router/AsyncRouter
- 需要流程编排 → 用 Team
- 文档（教程第 5、6 章）需明确这个选择
