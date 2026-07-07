"""
PyAgentKit 内置 Agent 实现

所有 Agent 均基于 LLM 驱动：
- 通过 think() 进入 ReAct 循环，由 GLM 自主决定是否调用工具
- system_prompt 真正生效，定义各 Agent 的角色
- 交互过程存入 memory，保持上下文
"""

import json
from typing import Optional

from core import Agent, Message, ToolRegistry, WebSearchTool, CalculatorTool, FileReadTool
from core.llm import LLMClient


class UserAgent(Agent):
    """
    用户代理，代表与系统交互的终端用户。

    本身不接 LLM（用户即真人），仅负责接收并展示最终结果。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(
            agent_id="user",
            name="User",
            system_prompt=None,
            llm_client=None,  # 用户代理不接 LLM
        )

    def receive(self, message: Message) -> None:
        """
        用户代理接收消息（通常是最终结果）
        """
        print(f"[{self.name}] 收到最终结果:")
        print("-" * 50)
        print(message.content)
        print("-" * 50)


class ResearchAgent(Agent):
    """
    研究代理，负责搜索和收集信息。

    由 GLM 自主决定是否调用 web_search / calculator 工具，
    并将整理后的自然语言摘要发送给下游 Writer。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(
            agent_id="researcher",
            name="Researcher",
            system_prompt=(
                "你是一个研究助理，擅长搜索和整理信息。"
                "收到研究请求后，使用 web_search 工具获取资料，"
                "然后用简洁的中文总结要点。如果涉及数值计算，可使用 calculator 工具。"
                "只返回整理后的研究摘要，不要添加额外格式说明。"
            ),
            llm_client=llm_client,
            capabilities=["search", "research"],  # 能力声明，供协作策略匹配
        )
        # 注册工具
        self.tool_registry.register(WebSearchTool())
        self.tool_registry.register(CalculatorTool())

    def receive(self, message: Message) -> None:
        """
        研究代理接收研究请求，经 LLM 推理后将结果转发给 Writer
        """
        print(f"[{self.name}] 收到研究请求: {message.content}")

        try:
            # 进入 ReAct 循环：LLM 自主决定是否调工具并产出摘要
            summary = self.think(message.content)
            print(f"[{self.name}] 研究完成，生成摘要")

            # 把结构化结果发送给 Writer
            self.send(
                "writer",
                {
                    "original_query": message.content,
                    "research_summary": summary,
                },
                "research_result",
            )
        except Exception as e:
            print(f"[{self.name}] 研究过程出错: {e}")
            self.send("writer", {"original_query": message.content, "error": str(e)}, "error")


class WriterAgent(Agent):
    """
    作家代理，负责将研究结果整理成结构化报告。

    由 GLM 真正生成报告内容（而非字符串拼接）。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(
            agent_id="writer",
            name="Writer",
            system_prompt=(
                "你是一个专业的内容整理者和报告撰写者。"
                "你会收到一份研究摘要，请将其整理成结构清晰、语言流畅的中文报告，"
                "包含标题、要点和简短结论。直接输出报告正文。"
            ),
            llm_client=llm_client,
            capabilities=["write"],  # 能力声明
        )

    def receive(self, message: Message) -> None:
        """
        作家代理接收研究结果并生成报告
        """
        print(f"[{self.name}] 收到研究结果，开始撰写报告...")

        if message.type == "research_result":
            data = message.content
            original_query = data.get("original_query", "")
            summary = data.get("research_summary", "")

            # 用 LLM 把摘要整理成正式报告
            prompt = f"研究主题：{original_query}\n\n研究摘要：\n{summary}"
            try:
                report = self.think(prompt, use_tools=False)
                self.send("user", report, "final_report")
            except Exception as e:
                self.send("user", f"报告生成失败: {str(e)}", "error_report")
        elif message.type == "error":
            self.send(
                "user",
                f"研究过程出错: {message.content}",
                "error_report",
            )
        else:
            self.send("user", f"无法处理的消息类型: {message.type}", "error_report")


class AnalyzerAgent(Agent):
    """
    分析代理，负责数据分析和数值计算。

    由 GLM 自主决定使用 calculator / file_read 工具，并返回分析结论。
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        super().__init__(
            agent_id="analyzer",
            name="Analyzer",
            system_prompt=(
                "你是一个数据分析专家，擅长数值计算和数据处理。"
                "收到分析请求后，使用 calculator 工具进行数学计算，"
                "或使用 file_read 工具读取文件内容，然后给出清晰的分析结论。"
                "直接输出分析结果，不要添加额外格式说明。"
            ),
            llm_client=llm_client,
            capabilities=["analysis", "calculate"],  # 能力声明
        )
        # 注册工具
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(FileReadTool())

    def receive(self, message: Message) -> None:
        """
        分析代理接收分析请求，经 LLM 推理后返回结论
        """
        print(f"[{self.name}] 收到分析请求: {message.content}")

        try:
            # ReAct 循环：LLM 自主选择工具并给出结论
            conclusion = self.think(message.content)
            print(f"[{self.name}] 分析完成")
            self.send(message.sender, conclusion, "calculation_result")
        except Exception as e:
            self.send(message.sender, f"分析出错: {str(e)}", "error")
