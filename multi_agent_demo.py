#!/usr/bin/env python3
"""
PyAgentKit 多智能体对话系统演示
实现用户提问 -> 研究代理搜索 -> 作家代理整理报告 -> 返回用户的完整流程
"""

from core import Router, Message
from agents import UserAgent, ResearchAgent, WriterAgent


def main():
    """
    主函数 - 演示多智能体对话流程
    流程: UserAgent 发送问题 -> Router判断 -> 分发给 ResearchAgent
         ResearchAgent 调用 WebSearchTool -> 得到结果 -> 回复消息
         Router 将结果传给 WriterAgent -> 整理成报告 -> 输出给用户
    """
    print("PyAgentKit 多智能体对话系统演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    
    # 创建Agents
    user_agent = UserAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    
    # 注册Agents到路由器
    router.register_agent(user_agent)
    router.register_agent(research_agent)
    router.register_agent(writer_agent)
    
    
    # 模拟用户提问
    user_question = "人工智能"
    print(f"[System] 用户提问: {user_question}")
    
    # 创建初始消息
    initial_message = Message(
        sender="user",
        receiver="researcher",  # 初始发送给研究代理
        content=user_question,
        msg_type="research_request"
    )
    
    # 路由消息启动流程
    print("\n开始处理用户请求...")
    router.route_message(initial_message)
    
    print("\n演示完成")


if __name__ == "__main__":
    main()