from core import Agent, Message, ToolRegistry, WebSearchTool, CalculatorTool, FileReadTool
from typing import Any


class UserAgent(Agent):
    """
    用户代理，代表与系统交互的用户
    """

    def __init__(self):
        super().__init__("user", "User")

    def receive(self, message: Message) -> None:
        """
        用户代理接收消息（通常是最终结果）
        """
        print(f"[{self.name}] 收到最终结果: {message.content}")
        print("=" * 50)


class ResearchAgent(Agent):
    """
    研究代理，负责搜索和收集信息
    """

    def __init__(self):
        super().__init__("researcher", "Researcher", 
                         system_prompt="你是一个研究助理，专门负责搜索和收集信息。")
        # 注册工具
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(CalculatorTool())

    def receive(self, message: Message) -> None:
        """
        研究代理接收研究请求并执行搜索
        """
        print(f"[{self.name}] 收到研究请求: {message.content}")
        
        # 执行搜索
        search_query = message.content
        try:
            search_result = self.tool_registry.execute("web_search", query=search_query)
            print(f"[{self.name}] 搜索完成，结果: {search_result['result']}")
            
            # 将结果发送给下一个Agent（WriterAgent）
            self.send(
                "writer", 
                {
                    "original_query": search_query,
                    "search_result": search_result
                },
                "research_result"
            )
        except Exception as e:
            self.send("writer", f"搜索出错: {str(e)}", "error")


class WriterAgent(Agent):
    """
    作家代理，负责整理和撰写报告
    """

    def __init__(self):
        super().__init__("writer", "Writer",
                         system_prompt="你是一个专业的内容整理者和报告撰写者。")

    def receive(self, message: Message) -> None:
        """
        作家代理接收研究结果并整理成报告
        """
        print(f"[{self.name}] 收到研究结果，开始撰写报告...")
        
        if message.type == "research_result":
            data = message.content
            original_query = data["original_query"]
            search_result = data["search_result"]
            
            # 整理成报告
            report = f"""
研究报告: {original_query}
================================

{search_result["result"]}

来源: {search_result["source"]}
            """.strip()
            
            # 将最终报告发送给用户
            self.send("user", report, "final_report")
        else:
            # 处理其他类型的消息
            error_report = f"无法处理的消息类型: {message.type}\n内容: {message.content}"
            self.send("user", error_report, "error_report")


class AnalyzerAgent(Agent):
    """
    分析代理，负责数据分析和处理
    """

    def __init__(self):
        super().__init__("analyzer", "Analyzer",
                         system_prompt="你是一个数据分析专家，专门负责处理和分析数据。")
        # 注册工具
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(FileReadTool())

    def receive(self, message: Message) -> None:
        """
        分析代理接收分析请求并执行分析
        """
        print(f"[{self.name}] 收到分析请求: {message.content}")
        
        # 根据消息类型执行不同的分析任务
        if message.type == "calculate":
            try:
                expression = message.content
                result = self.tool_registry.execute("calculator", expression=expression)
                print(f"[{self.name}] 计算完成: {result}")
                
                # 将结果发送给请求者
                self.send(
                    message.sender,
                    result,
                    "calculation_result"
                )
            except Exception as e:
                self.send(message.sender, f"计算出错: {str(e)}", "error")
        elif message.type == "mcp_command":
            # 处理MCP工具命令
            try:
                command_data = message.content
                tool_name = command_data.get("tool_name", "")
                command = command_data.get("command", "")
                params = command_data.get("params", {})
                
                result = self.tool_registry.execute_mcp_tool(tool_name, command=command, params=params)
                print(f"[{self.name}] MCP工具执行完成: {result}")
                
                # 将结果发送给请求者
                self.send(
                    message.sender,
                    result,
                    "mcp_result"
                )
            except Exception as e:
                self.send(message.sender, f"MCP工具执行出错: {str(e)}", "error")
        else:
            self.send(message.sender, "未知的分析任务", "error")