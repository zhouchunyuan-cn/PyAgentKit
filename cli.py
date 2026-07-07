#!/usr/bin/env python3
"""
PyAgentKit 命令行入口

零额外依赖（argparse 标准库），提供四个子命令：

  pyagentkit chat   交互式多轮对话（默认通用助手）
  pyagentkit ask    单次提问，返回答复后退出
  pyagentkit run    加载 YAML 配置编排 Agent 后交互（含团队模式）
  pyagentkit tools  列出可用的内置工具

安装为命令：在项目根目录 `pip install -e .` 后可用 `pyagentkit`，
也可直接 `python cli.py ...`。
"""
import argparse
import os
import sys

# 自动加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.llm import GLMClient
from core.logging_config import setup_logging
from core.session import ConversationSession
from core.agent import Agent
from core.config import ConfigLoader


def _ensure_api_key():
    """检查 API Key，缺失时给出清晰提示并退出"""
    if not os.environ.get("ZHIPUAI_API_KEY"):
        print("错误：未检测到 ZHIPUAI_API_KEY 环境变量。")
        print("请复制 .env.example 为 .env 并填入 API Key，或运行：")
        print("  export ZHIPUAI_API_KEY=your_key  (Linux/Mac)")
        print("  set ZHIPUAI_API_KEY=your_key     (Windows)")
        sys.exit(1)


def _default_agent(model: str, tools=None) -> Agent:
    """构建一个默认的通用助手 Agent"""
    class _Assistant(Agent):
        def __init__(self, llm, tool_list):
            super().__init__(
                agent_id="assistant",
                name="助手",
                system_prompt="你是一个友好、专业的中文助手。根据需要可调用工具获取信息或计算。",
                llm_client=llm,
                capabilities=["chat"],
            )
            for t in (tool_list or []):
                self.tool_registry.register(t)

        def receive(self, message):
            pass  # CLI 场景由 session 直接拿 think 结果

    llm = GLMClient(model=model)
    return _Assistant(llm, tools)


def _interactive_loop(session: ConversationSession, prompt: str = "你") -> None:
    """交互式对话主循环，直到用户退出"""
    print("输入消息开始对话（输入 :q 退出，:clear 清空上下文）\n")
    while True:
        try:
            user_input = input(f"{prompt}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input in (":q", ":quit", "exit"):
            print("再见！")
            break
        if user_input == ":clear":
            session.clear_history()
            print("（已清空对话上下文）\n")
            continue

        try:
            reply = session.chat(user_input)
            print(f"\n助手: {reply}\n")
        except Exception as e:
            print(f"\n[错误] {e}\n")


# --------------------------------------------------------------------
# 子命令实现
# --------------------------------------------------------------------

def cmd_chat(args):
    """交互式对话（默认通用助手）"""
    _ensure_api_key()
    setup_logging()
    from core.tool_factory import default_factory

    tools = default_factory.create_many(args.tools) if args.tools else None
    agent = _default_agent(args.model, tools)
    session = ConversationSession(agent, max_history=args.max_history)
    _interactive_loop(session)


def cmd_ask(args):
    """单次提问"""
    _ensure_api_key()
    setup_logging()
    from core.tool_factory import default_factory

    tools = default_factory.create_many(args.tools) if args.tools else None
    agent = _default_agent(args.model, tools)
    session = ConversationSession(agent, max_history=2)

    try:
        reply = session.chat(args.question)
        print(reply)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args):
    """加载 YAML 配置并启动交互"""
    _ensure_api_key()
    setup_logging()

    loader = ConfigLoader()
    config = loader.load_file(args.config)
    print(f"已加载配置：模型={config.model}，agents={[a.id for a in config.agents]}")

    _, agents, session = loader.build(config)

    # 若配置含 team 块，进入团队协作模式
    team = loader.build_team(config, agents)
    if team is not None:
        print(f"团队模式：{team.name}（{team.process.name}流程），"
              f"成员={[m.agent_id for m in team.members]}")
        if team.leader:
            print(f"Leader：{team.leader.agent_id}")
        print()
        _team_interactive_loop(team)
    else:
        print(f"会话绑定 agent：{config.session_agent}\n")
        _interactive_loop(session, prompt=config.session_agent)


def _team_interactive_loop(team):
    """团队协作模式的交互循环：每个输入作为团队任务执行"""
    print("输入任务开始团队协作（输入 :q 退出，:summary 查看团队摘要）\n")
    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not task:
            continue
        if task in (":q", ":quit", "exit"):
            print("再见！")
            break
        if task == ":summary":
            print(team.summary())
            continue
        try:
            print(f"\n[团队执行中...]\n")
            result = team.run(task)
            print("--- 团队最终结果 ---")
            print(result)
            print()
        except Exception as e:
            print(f"\n[错误] {e}\n")


def cmd_list_tools(args):
    """列出所有可用的内置工具名"""
    from core.tool_factory import default_factory
    print("可用工具：")
    for name in default_factory.list_tools():
        print(f"  - {name}")


# --------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyagentkit",
        description="PyAgentKit 命令行 —— 基于 GLM 的多 Agent 框架",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # chat
    p_chat = sub.add_parser("chat", help="交互式多轮对话")
    p_chat.add_argument("--model", default="glm-4-flash", help="LLM 模型名（默认 glm-4-flash）")
    p_chat.add_argument("--tools", nargs="*", help="启用的工具，如 --tools web_search calculator")
    p_chat.add_argument("--max-history", type=int, default=10, help="保留的最大对话轮数")
    p_chat.set_defaults(func=cmd_chat)

    # ask
    p_ask = sub.add_parser("ask", help="单次提问")
    p_ask.add_argument("question", help="提问内容")
    p_ask.add_argument("--model", default="glm-4-flash", help="LLM 模型名")
    p_ask.add_argument("--tools", nargs="*", help="启用的工具")
    p_ask.set_defaults(func=cmd_ask)

    # run
    p_run = sub.add_parser("run", help="加载 YAML 配置编排 Agent")
    p_run.add_argument("config", help="YAML 配置文件路径")
    p_run.set_defaults(func=cmd_run)

    # tools
    p_tools = sub.add_parser("tools", help="列出可用工具")
    p_tools.set_defaults(func=cmd_list_tools)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
