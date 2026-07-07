#!/usr/bin/env python3
"""
PyAgentKit P2 能力演示

演示三项"让它好用"的能力：
1. 能力驱动的协作：Agent 声明能力，协作策略按能力匹配（不依赖名字）
2. 向量记忆：存入经验后按语义召回相关内容
3. 多轮对话：Session 维护上下文，支持追问/指代

运行前需配置 ZHIPUAI_API_KEY（同 main.py）
"""

import os

# 自动加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core import (
    Router, GLMClient,
    DynamicCollaborationManager, CapabilityCollaborationStrategy,
    VectorMemory, GLMEmbedder, LocalTfidfEmbedder,
    ConversationSession,
)
from core.agent import Agent
from core.message import Message
from core.logging_config import setup_logging
from agents import ResearchAgent, WriterAgent, AnalyzerAgent


def demo1_capability_driven_collaboration():
    """演示1：基于能力声明的协作匹配（不依赖 Agent 名称）"""
    print("=" * 60)
    print("演示1：能力驱动的协作匹配")
    print("=" * 60)

    # 临时 Agent，故意起一个"看不出功能"的名字，验证靠能力而非名字匹配
    class CoolBot(Agent):
        """名字叫 CoolBot，但声明了 search 能力——协作策略应能识别它"""
        def __init__(self, llm):
            super().__init__(
                agent_id="cool_bot",
                name="CoolBot",
                system_prompt="你是研究助理。",
                llm_client=llm,
                capabilities=["search", "research"],
            )
        def receive(self, message):
            print(f"  [CoolBot] 收到任务: {message.content}")

    class Wordsmith(Agent):
        """名字叫 Wordsmith，声明 write 能力"""
        def __init__(self, llm):
            super().__init__(
                agent_id="wordsmith",
                name="Wordsmith",
                system_prompt="你是写作专家。",
                llm_client=llm,
                capabilities=["write"],
            )
        def receive(self, message):
            print(f"  [Wordsmith] 收到任务: {message.content}")

    # 注意：不创建 Researcher/Writer，验证匹配不靠名字
    llm = GLMClient(model="glm-4-flash")
    router = Router()
    router.register_agent(CoolBot(llm))
    router.register_agent(Wordsmith(llm))

    collab = DynamicCollaborationManager(router)
    # research 策略需要 search+write 能力，应匹配到 CoolBot(入口) 和 Wordsmith
    ok = collab.initiate_collaboration("研究量子计算的基础原理")
    print(f"  协作启动: {ok}（证明：靠能力匹配，而非名字）")
    print()


def demo2_vector_memory():
    """演示2：向量记忆的语义召回"""
    print("=" * 60)
    print("演示2：向量记忆（语义召回）")
    print("=" * 60)
    print("  注：默认使用本地 TF-IDF embedder（零成本离线）；")
    print("  充值 embedding-3 后可换 GLMEmbedder 获得更强语义效果\n")

    # 本地 embedder，零成本可用；recall 语义基于词项重叠
    embedder = LocalTfidfEmbedder()
    vm = VectorMemory(embedder=embedder)

    # 存入若干条"经验"
    experiences = [
        "Python 是一门解释型、动态类型的编程语言，强调代码可读性。",
        "用 pip 安装 Python 包：pip install 包名。",
        "Git 用于版本控制，常用命令有 clone、commit、push。",
        "今天午餐吃了红烧肉和米饭。",
        "HTTP 状态码 200 表示请求成功，404 表示资源未找到。",
    ]
    for exp in experiences:
        vm.add(exp)
    print(f"  已存入 {vm.count()} 条记忆")

    # 用不同的查询测试语义召回（注意查询词与原文不完全相同）
    queries = ["Python 语言的特点", "版本管理工具", "网络请求的状态"]
    for q in queries:
        print(f"\n  查询: 「{q}」")
        results = vm.search(q, top_k=2)
        for r in results:
            print(f"    [{r['similarity']:.3f}] {r['text']}")
    print()


def demo3_conversation_session():
    """演示3：多轮对话（上下文延续/追问）"""
    print("=" * 60)
    print("演示3：多轮对话（Session 上下文）")
    print("=" * 60)

    llm = GLMClient(model="glm-4-flash")
    agent = AnalyzerAgent(llm_client=llm)
    session = ConversationSession(agent)

    # 第一轮：给出信息
    print("\n  用户: 我有 3 个苹果，又买了 5 个")
    r1 = session.chat("我有 3 个苹果，又买了 5 个")
    print(f"  Agent: {r1}")

    # 第二轮：追问——需要记住上一轮的"8 个苹果"
    print("\n  用户: 那我再吃掉 2 个，还剩多少？")
    r2 = session.chat("那我再吃掉 2 个，还剩多少？")
    print(f"  Agent: {r2}")

    print(f"\n  会话摘要: {session.summary()}")
    print(f"  （若 Agent 能答出 6，说明成功利用了上文上下文）")
    print()


def main():
    setup_logging()

    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("未检测到 ZHIPUAI_API_KEY，请先配置（参考 README）。")
        return

    print("\nPyAgentKit P2 能力演示\n")
    demo1_capability_driven_collaboration()
    demo2_vector_memory()
    demo3_conversation_session()
    print("=" * 60)
    print("P2 演示完成")


if __name__ == "__main__":
    main()
