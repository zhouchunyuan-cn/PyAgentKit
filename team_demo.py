#!/usr/bin/env python3
"""
PyAgentKit Team 模块演示

展示两种团队协作流程（用真实 GLM 跑通）：
1. Sequential：研究→写作→审核 的顺序流水线
2. Hierarchical：Leader 自动拆解任务并按能力分配成员

运行前需配置 ZHIPUAI_API_KEY（同 main.py）
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core import GLMClient
from core.agent import Agent
from core.message import Message
from core.team import Team, SequentialProcess, HierarchicalProcess
from core.logging_config import setup_logging


def make_simple_agent(agent_id, name, capabilities, llm, system_prompt):
    """构造一个通用 Team 成员（只走 think，不走 receive 路由）"""
    class _Member(Agent):
        def __init__(self):
            super().__init__(
                agent_id=agent_id, name=name,
                system_prompt=system_prompt,
                llm_client=llm, capabilities=capabilities,
            )

        def receive(self, message: Message):
            pass  # Team 流程不依赖消息路由，走 think

    return _Member()


def demo_sequential(llm):
    """演示1：Sequential 顺序流水线（研究→写作→审核）"""
    print("=" * 60)
    print("演示1：Sequential 顺序接力（研究 → 写作 → 审核）")
    print("=" * 60)
    print("流程特点：成员按声明顺序执行，前者产出作为后者上下文，确定性强。\n")

    members = [
        make_simple_agent(
            "researcher", "研究员", ["search"],
            llm, "你是研究员，负责收集和整理资料。输出简洁的调研要点。",
        ),
        make_simple_agent(
            "writer", "作家", ["write"],
            llm, "你是作家，基于研究员的资料撰写通顺的文章。",
        ),
        make_simple_agent(
            "reviewer", "审核员", ["review"],
            llm, "你是审核员，检查文章质量并给出终稿。直接输出修订后的最终版本。",
        ),
    ]

    team = Team(name="内容生产组", members=members, process=SequentialProcess())
    result = team.run("写一段关于人工智能在医疗领域应用的科普介绍")

    print("\n--- 最终结果（审核员产出）---")
    print(result)
    print("\n团队摘要:", team.summary()["process"], "成员数", team.summary()["member_count"])
    print()


def demo_hierarchical(llm):
    """演示2：Hierarchical Leader 自动编排"""
    print("=" * 60)
    print("演示2：Hierarchical Leader 自动编排")
    print("=" * 60)
    print("流程特点：Leader 用 LLM 分析任务，自动拆解为子任务并按能力分配成员。\n")

    leader = make_simple_agent(
        "leader", "项目经理", ["plan"],
        llm, "你是项目经理，擅长拆解任务、分配工作、汇总成果。",
    )
    members = [
        make_simple_agent(
            "researcher", "研究员", ["search"],
            llm, "你是研究员，擅长搜索整理资料。",
        ),
        make_simple_agent(
            "analyst", "分析师", ["analysis"],
            llm, "你是数据分析师，擅长分析和解读。",
        ),
        make_simple_agent(
            "writer", "作家", ["write"],
            llm, "你是作家，擅长把内容整理成报告。",
        ),
    ]

    team = Team(
        name="AI项目组", members=members,
        process=HierarchicalProcess(max_subtasks=4),
        leader=leader,
    )
    result = team.run("完成一份AI行业现状的简要分析")

    print("\n--- Leader 汇总的最终结果 ---")
    print(result)
    print()


def main():
    setup_logging()

    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("未检测到 ZHIPUAI_API_KEY，请先配置。")
        return

    print("\nPyAgentKit Team 模块演示\n")
    llm = GLMClient(model="glm-4-flash")

    demo_sequential(llm)
    demo_hierarchical(llm)

    print("=" * 60)
    print("Team 演示完成")
    print("对比：Sequential 是固定流水线；Hierarchical 让 Leader 智能分配。")


if __name__ == "__main__":
    main()
