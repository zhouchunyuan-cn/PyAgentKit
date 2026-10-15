# 第 2 章：Agent 与 ReAct 循环

> 目标：彻底理解 `think()` 方法——这是整个框架的心脏。
> 读透这一章，项目就懂了八成。

## 2.1 这一章解决什么问题

一个"聊天机器人"只能对话。一个"智能体（Agent）"能**自主决定何时调用工具、调用哪个工具、怎么用工具结果继续推理**。这种"边思考边行动"的能力，就是 ReAct（Reasoning + Acting）。

PyAgentKit 的 `Agent` 基类把 LLM、工具、记忆三者编织进一个 `think()` 循环，让任何继承它的子类自动获得这种能力。

## 2.2 Agent 的结构

`Agent` 是抽象基类（`core/agent.py`），定义在 `__init__` 里：

```python
class Agent(ABC):
    def __init__(self, agent_id, name, system_prompt=None,
                 llm_client=None, model="glm-4-flash",
                 max_steps=5, capabilities=None, tags=None):
        self.agent_id = agent_id          # 唯一标识
        self.name = name                  # 显示名
        self.system_prompt = system_prompt  # 角色设定（喂给 LLM）
        self.llm = llm_client             # LLM 客户端
        self.capabilities = capabilities  # 能力声明（团队协作用）
        self.memory = Memory()            # KV 记忆
        self.brain = MemoryManager(...)   # 统一记忆管理器（聚合 KV/向量/会话）
        self.tool_registry = ToolRegistry()  # 工具集
```

子类必须实现一个抽象方法：

```python
@abstractmethod
def receive(self, message: Message) -> None:
    """收到消息时怎么做（子类决定）"""
```

`receive` 是消息驱动的入口（用于多 Agent 协作）；而 `think` 是推理的核心。

## 2.3 think() 的 ReAct 循环——逐步拆解

`think(user_input, context_messages, use_tools, stream_callback) -> str` 的执行流程：

### 步骤 0：组装初始消息

```python
messages = self._build_initial_messages(user_input, context_messages)
```

这一步委托给 `self.brain.build_context()`，按固定顺序组装出 LLM 能理解的 messages 列表：

```
[系统] system_prompt（角色设定："你是数学专家..."）
[系统] 长期记忆（如有）："以下是你的长期记忆..."
[系统] 向量召回（如有）："以下是相关的历史经验..."
[历史] 此前的对话轮次（多轮上下文）
[额外] context_messages（外部传入的上下文）
[用户] user_input（当前问题）         ← 永远在最后
```

> 这个组装顺序很关键：角色设定在最前定基调，记忆和上下文在中间提供参考，用户当前输入在最后让模型直接回应。

### 步骤 1：进入循环（最多 max_steps 轮）

```python
for step in range(self.max_steps):   # 默认最多 5 轮，防止死循环
    result = self.llm.chat(messages, tools=..., tool_choice="auto")
```

每次循环调用一次 LLM。LLM 的返回有两种可能：

**情况 A：模型给出最终答复**（`result.tool_calls` 为空）

```python
if not result.tool_calls:
    return result.content   # 直接返回，循环结束
```

**情况 B：模型请求调用工具**（`result.tool_calls` 非空）

```python
# 1. 把模型的工具调用请求加入对话历史（推理模型还会带上 reasoning_content）
messages.append({"role": "assistant", "tool_calls": [...], ...})

# 2. 执行工具（多个工具并发执行，结果按声明顺序回填对话）
tool_results = self._execute_tool_calls(result.tool_calls, ...)
for tc, tool_result in zip(result.tool_calls, tool_results):
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)})

# 3. 继续下一轮循环——带着工具结果再问 LLM
```

> 同一步返回的多个工具调用会在线程池里**并发执行**（上限 4），但结果严格按
> `tool_calls` 的声明顺序回填——对话协议的确定性不受并发影响。

### 完整时序

```
think("计算 (15+27)*3")
  │
  ├─ 组装 messages（system_prompt + 问题）
  │
  └─ 第1轮循环
       ├─ LLM.chat(messages, tools=[calculator])
       ├─ LLM 返回：tool_calls=[{name:"calculator", args:"(15+27)*3"}]
       ├─ 执行 calculator → 得到 126
       ├─ 把 {role:"tool", content:"126"} 加入 messages
       │
       └─ 第2轮循环
            ├─ LLM.chat(messages)   ← 这次 messages 里已有工具结果
            ├─ LLM 返回："(15+27)*3 的结果是 126"（无 tool_calls）
            └─ 返回 "结果是 126"    ← 循环结束
```

### 步骤 2：兜底（超过 max_steps 仍未结束）

如果循环跑完 max_steps 轮模型还在调工具，做一次不带工具的收尾调用，强制得到一个文本答复。

## 2.4 关键设计：为什么是"循环"而不是"一次调用"

如果只调用一次 LLM，模型只能基于自己训练时的知识回答——它**算不准算术、查不到实时信息、读不了文件**。

循环让模型可以：**先说要调什么工具 → 真的去调 → 拿到真实结果 → 基于结果继续推理**。这就是 Agent 区别于聊天机器人的本质。

## 2.5 工具调用的安全边界（参数校验 + 异常边界）

工具执行有两层防护。第一层在 `ToolRegistry.execute`：**按工具声明的 JSON Schema 校验参数**——必填检查 + 轻量类型矫正（`"3"`→3、`3`→`"3"`、`"true"`→True，LLM 给错类型是常事）。校验失败抛出**带 schema 说明**的错误，让模型拿到自纠所需的信息。

第二层是 engine 的异常边界 `_execute_tool_safe`：

```python
def _execute_tool_safe(self, tool_name, arguments):
    try:
        return self.tool_registry.execute(tool_name, **arguments)
    except Exception as e:
        return {"error": f"工具 '{tool_name}' 执行失败: {str(e)}"}
```

校验错误和执行错误都不会中断 think() 循环——错误作为 tool 结果回传给 LLM，模型下一步就能自己纠正参数或换个做法。

## 2.6 流式输出、事件流与混合推理模型

### 真流式（stream_callback）

只要传入 `stream_callback`，ReAct 主循环的**每一步 LLM 调用都走 `chat_stream` 真流式**——首字延迟 ≈ 模型出第一个字的时间，而不是整个答复生成完：

```python
def on_chunk(text):
    print(text, end="", flush=True)   # 实时打印每个片段

agent.think("介绍量子纠缠", stream_callback=on_chunk)
```

注意一个流式语义细节：模型在发起工具调用前常会先输出一段叙述性文字（"我先搜索一下…"），这段文字也会实时流到回调里——这和主流 Agent 产品的行为一致。

流式中的工具调用不会丢：`chat_stream` 把工具调用拆成碎片（`delta_tool_calls`）逐块送达，engine 按 index 累积拼装成完整的 `ToolCall`。流在还没产出任何内容时失败，会安全回退一次非流式调用。

### 结构化事件流（events_callback）

文本回调之外，还可以拿到**执行过程的全部事件**——前端据此渲染"正在思考/正在调用工具"：

```python
from core import AgentStreamEvent

def on_event(ev: AgentStreamEvent):
    if ev.type == "reasoning":   # 思考过程增量（推理模型）
        print(f"\033[2m{ev.content}\033[0m", end="")
    elif ev.type == "text":      # 文本增量
        print(ev.content, end="")
    elif ev.type == "tool_call": # {"name": ..., "arguments": ...}
        print(f"\n· 调用工具 {ev.content['name']}")

agent.think("搜索量子纠缠进展", events_callback=on_event)
```

事件类型：`reasoning` / `text` / `tool_call` / `tool_result`，都带 `step` 编号。CLI 的 `chat` 命令就是这么实现的。

### 混合推理模型（thinking / reasoning_content）

GLM-4.5+ 是混合推理模型，开启深度思考后响应里带 `reasoning_content`（思考过程）。框架全程透传：`ChatResult.reasoning_content` 字段、流式的 `delta_reasoning` 增量、`reasoning` 事件。带工具调用的 assistant 消息会**回传 reasoning_content**，多轮推理保持连贯（Preserved Thinking）。

```python
llm = GLMClient(model="glm-4.7-flash", thinking=True)   # 开思考，输出含推理过程
llm = GLMClient(model="glm-4.7-flash", thinking=False)  # 关思考，省 token 降延迟
llm = GLMClient(model="glm-4-flash")                    # 不传，由模型默认决定
```

### 相关延伸

- **结构化输出**：`think(..., response_format={"type": "json_object"})` 约束模型输出合法 JSON（第 6 章 Leader 规划就在用）
- **异步执行**：`await agent.think_async(...)`，asyncio 程序与多 Agent 并发的入口（第 7 章）

## 2.7 Token 统计

每次 LLM 调用的 token 用量会自动累计：

```python
agent.think("复杂问题")
print(agent.get_token_usage())
# TokenUsage(prompt_tokens=850, completion_tokens=320, total_tokens=1170)
```

跨多次 think 调用持续累加，方便核算成本。

## 2.8 能力声明（capabilities）

```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__("id", "名字", capabilities=["search", "calculate"])
```

`capabilities` 是给**团队协作系统**看的标签，声明这个 Agent 能做什么。协作策略据此匹配 Agent，而不是靠名字硬编码（详见第 6 章）。

## 2.9 自己实现一个 Agent

最简模板：

```python
from core import Agent, GLMClient, Message

llm = GLMClient()

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id="my",
            name="我的助手",
            system_prompt="你是一个友好的助手。",
            llm_client=llm,
        )

    def receive(self, message: Message):
        """收到其他 Agent 发来的消息时调用"""
        reply = self.think(message.content)
        print(reply)

agent = MyAgent()
print(agent.think("你好"))   # 直接调用 think
```

进阶：挂工具、注入记忆、声明能力，都在 `__init__` 里完成。

## 2.10 ReactEngine——推理引擎（内部架构）

`think()` 的 ReAct 循环逻辑实际由 **ReactEngine**（`core/react_engine.py`）承载。Agent 通过委托调用：

```python
# Agent 内部
self._engine = ReactEngine(self)

def think(self, ...):
    return self._engine.think(...)   # 一行委托
```

这样设计的好处：Agent 基类聚焦"身份 + 通信 + 工具"，推理细节内聚在 engine。后续的错误恢复、token 统计、钩子触发都在 engine 一处维护。

## 2.11 错误恢复——优雅降级

当 LLM 调用失败时（GLMClient 内部重试已耗尽），`think()` 不会直接崩溃：

- **第一步就失败**（无任何上下文）：抛出带说明的 `RuntimeError`
- **循环中途失败**（已有工具结果）：尝试一次无工具的降级调用，基于已有信息给出"尽力而为"的答复
- **降级也失败**：抛错

这让 Agent 对网络抖动等瞬时故障更有韧性。

## 2.12 生命周期钩子

通过 `AgentHooks` 在 ReAct 过程的关键节点注入逻辑，无需子类化：

```python
from core import Agent, AgentHooks

hooks = AgentHooks(
    on_think_start=lambda agent, inp: print(f"开始思考：{inp}"),
    on_tool_call=lambda agent, name, args, res: print(f"调用了 {name}"),
    on_think_end=lambda agent, resp: print(f"完成：{resp[:20]}"),
    on_error=lambda agent, err, ctx: print(f"出错：{err}"),
)
agent = MyAgent(hooks=hooks)
```

`on_think_start` 返回 `False` 可**中止** think（用于 human-in-the-loop 审批）。钩子异常不影响主流程。

## 2.13 记忆持久化

默认 Agent 的记忆是纯内存的（重启丢失）。需要持久化时：

```python
agent.enable_persistence()              # 默认 memory_{agent_id}.json
agent.enable_persistence("my_mem.json") # 自定义路径
```

启用后长期记忆写入文件，重启恢复。多个 Agent 各自的文件互不干扰。

## 2.14 与其他模块的关系

```
Agent.think()
   ├── 调用 → LLMClient.chat() / chat_stream()    [第 7 章]
   ├── 调用 → ToolRegistry.execute()              [第 3 章]
   ├── 委托 → MemoryManager.build_context()       [第 4 章]
   └── 记录 → Tracer (可选)                        [第 7 章]
```

Agent 是中心枢纽，把 LLM、工具、记忆、追踪串在一起。

---

← [上一章：快速开始](01_快速开始.md) | 下一章：[工具系统 →](03_工具系统.md)
