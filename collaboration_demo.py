#!/usr/bin/env python3
"""
PyAgentKit 动态协作演示
展示Agent如何根据任务需求自动组合协作
"""

from core import Router, Message
from core.collaboration import DynamicCollaborationManager
from agents import UserAgent, ResearchAgent, WriterAgent, AnalyzerAgent


def main():
    """
    主函数 - 演示动态协作功能
    """
    print("PyAgentKit 动态协作演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    collaboration_manager = DynamicCollaborationManager(router)
    
    # 创建Agents
    user_agent = UserAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    analyzer_agent = AnalyzerAgent()
    
    # 注册Agents到路由器
    router.register_agent(user_agent)
    router.register_agent(research_agent)
    router.register_agent(writer_agent)
    router.register_agent(analyzer_agent)
    
    print("1. 演示研究任务的动态协作:")
    # 模拟研究任务
    research_task = "研究人工智能的发展历史"
    print(f"[System] 用户研究任务: {research_task}")
    
    # 启动协作
    success = collaboration_manager.initiate_collaboration(research_task)
    if success:
        print("研究任务协作已启动")
    else:
        print("研究任务协作启动失败")
    
    print("\n" + "="*50)
    print("2. 演示分析任务的动态协作:")
    # 模拟分析任务
    analysis_task = "计算2023年公司的收入增长率"
    print(f"[System] 用户分析任务: {analysis_task}")
    
    # 启动协作
    success = collaboration_manager.initiate_collaboration(analysis_task)
    if success:
        print("分析任务协作已启动")
    else:
        print("分析任务协作启动失败")
    
    print("\n" + "="*50)
    print("3. 协作统计信息:")
    # 显示协作统计信息
    stats = collaboration_manager.get_collaboration_stats()
    print(f"  协作策略数量: {stats['strategy_count']}")
    print(f"  任务历史数量: {stats['task_history_count']}")
    print("  最近任务:")
    for i, task in enumerate(stats['recent_tasks'], 1):
        print(f"    {i}. {task['description']} ({task['type']})")
    
    print("\n演示完成")


if __name__ == "__main__":
    main()