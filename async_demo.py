#!/usr/bin/env python3
"""
PyAgentKit 异步路由 + 流式输出 演示

演示两个新能力：
1. AsyncRouter：基于 asyncio.Queue 的异步路由，解决同步 Router 的栈溢出
2. 流式输出 + Token 统计：GLMClient.chat_stream 实时产出，Agent 累计 token

运行前需配置 ZHIPUAI_API_KEY
"""

import asyncio
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core import GLMClient, AsyncRouter
from core.agent import Agent
from core.message import Message
from core.logging_config import setup_logging


# --------------------------------------------------------------------
# 演示1：流式输出 + Token 统计
# --------------------------------------------------------------------

def demo_streaming_and_tokens(llm):
    print("=" * 60)
    print("演示1：流式输出 + Token 统计")
    print("=" * 60)

    class Assistant(Agent):
        def __init__(self):
            super().__init__(
                "assistant", "助手",
                system_prompt="你是一个简洁的中文助手，回答控制在 50 字内。",
                llm_client=llm,
            )
        def receive(self, m): pass

    agent = Assistant()

    print("\n问题：用三句话介绍量子纠缠（流式实时输出）\n")
    print("助手: ", end="", flush=True)

    # stream_callback：每收到一段文本就实时打印
    chunks = []
    def on_chunk(text):
        chunks.append(text)
        print(text, end="", flush=True)

    reply = agent.think("用三句话介绍量子纠缠", stream_callback=on_chunk)
    print("\n")
    print(f"（流式完成，共 {len(chunks)} 个片段）")

    # Token 统计
    usage = agent.get_token_usage()
    print(f"Token 用量: 输入={usage.prompt_tokens}, 输出={usage.completion_tokens}, 合计={usage.total_tokens}")
    print()


# --------------------------------------------------------------------
# 演示2：AsyncRouter 异步路由（不栈溢出）
# --------------------------------------------------------------------

async def demo_async_router(llm):
    print("=" * 60)
    print("演示2：AsyncRouter 异步路由（解决同步栈溢出）")
    print("=" * 60)
    print("构造 A↔B 互相转发的长链路，AsyncRouter 能正常完成（同步会栈溢出）\n")

    class PingPong(Agent):
        def __init__(self, aid, target, max_hops=10):
            super().__init__(aid, aid, llm_client=None)
            self.target = target
            self.max_hops = max_hops
            self.hops = 0

        def receive(self, message):
            self.hops += 1
            print(f"  [{self.agent_id}] 收到（第 {self.hops} 跳）")
            if self.hops < self.max_hops:
                self.send(self.target, message.content)

    router = AsyncRouter()
    a = PingPong("a", "b", max_hops=10)
    b = PingPong("b", "a", max_hops=10)
    router.register_agent(a)
    router.register_agent(b)

    await router.start()
    await router.route(Message("demo", "a", "ping"))
    await router.drain()   # 等待所有消息处理完
    await router.stop()

    print(f"\n完成：a 共 {a.hops} 跳，b 共 {b.hops} 跳，无栈溢出")
    print(f"路由统计: {router.get_router_stats()['stats']}")
    print()


async def main():
    setup_logging()

    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("未检测到 ZHIPUAI_API_KEY，请先配置。")
        return

    print("\nPyAgentKit 异步 + 流式 演示\n")
    llm = GLMClient(model="glm-4-flash")

    demo_streaming_and_tokens(llm)
    await demo_async_router(llm)

    print("=" * 60)
    print("演示完成")


if __name__ == "__main__":
    asyncio.run(main())
