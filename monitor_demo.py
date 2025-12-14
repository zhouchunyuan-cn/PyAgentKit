#!/usr/bin/env python3
"""
PyAgentKit 监控系统演示
展示如何使用监控系统观察和调试多Agent交互
"""

from core import Router, Message
from core.monitor import AgentMonitor
from agents import UserAgent, ResearchAgent, WriterAgent
import time
import threading


def monitoring_worker(monitor: AgentMonitor):
    """
    监控工作线程
    """
    while monitor.is_monitoring:
        monitor.print_real_time_status()
        time.sleep(5)  # 每5秒打印一次状态


def main():
    """
    主函数 - 演示监控系统
    """
    print("PyAgentKit 监控系统演示")
    print("=" * 50)
    
    # 创建核心组件
    router = Router()
    monitor = AgentMonitor(router)
    
    # 创建Agents
    user_agent = UserAgent()
    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    
    # 注册Agents到路由器
    router.register_agent(user_agent)
    router.register_agent(research_agent)
    router.register_agent(writer_agent)
    
    # 启动监控
    monitor.start_monitoring()
    
    # 启动监控显示线程
    monitor_thread = threading.Thread(target=monitoring_worker, args=(monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 模拟用户提问
    user_question = "人工智能发展史"
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
    
    # 等待一段时间以便观察监控效果
    time.sleep(10)
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 导出监控报告
    monitor.export_monitoring_report("monitoring_report.json")
    router.export_message_log("message_log.json")
    
    print("\n监控报告已导出到 monitoring_report.json")
    print("消息日志已导出到 message_log.json")
    print("\n演示完成")


if __name__ == "__main__":
    main()