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
# 1. 把模型的工具调用请求加入对话历史
messages.append({"role": "assistant", "tool_calls": [...]})

# 2. 逐个执行工具
for tc in result.tool_calls:
    tool_result = self._execute_tool_safe(tc.name, tc.arguments)
    messages.append({"role": "tool", "content": json.dumps(tool_result)})

# 3. 继续下一轮循环——带着工具结果再问 LLM
```

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

## 2.5 工具调用的安全边界

`_execute_tool_safe` 包裹了工具执行：

```python
def _execute_tool_safe(self, tool_name, arguments):
    try:
        return self.tool_registry.execute(tool_name, **arguments)
    except Exception as e:
        return {"error": f"工具 '{tool_name}' 执行失败: {str(e)}"}
```

工具失败不会中断 think() 循环——错误作为 tool 结果回传给 LLM，让模型自己决定怎么办（换个工具、放弃、或基于已知信息回答）。

## 2.6 流式输出

默认 `think()` 等模型完整生成才返回。传入 `stream_callback` 后，最终答复会**逐块实时回调**：

```python
def on_chunk(text):
    print(text, end="", flush=True)   # 实时打印每个片段

agent.think("介绍量子纠缠", stream_callback=on_chunk)
```

机制：当模型给出最终答复时（情况 A），用 `chat_stream()` 重新流式请求，每收到一段文本就回调一次。工具调用步骤仍用非流式（流式只用于最终输出）。

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

## 2.10 与其他模块的关系

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
