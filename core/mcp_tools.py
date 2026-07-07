"""
MCP（Model Context Protocol）工具集成

提供接入符合 MCP 协议的外部工具的能力，作为 Agent 工具系统的扩展入口。
"""

from typing import Dict, Any, Optional, List, Union
import json
import uuid
import logging
import requests

logger = logging.getLogger(__name__)

class MCPTool:
    """
    MCP (Model Context Protocol) 工具基类
    用于接入符合MCP协议的外部工具
    """

    def __init__(self, name: str, description: str, mcp_endpoint: str):
        """
        初始化MCP工具
        
        Args:
            name: 工具名称
            description: 工具描述
            mcp_endpoint: MCP服务端点URL
        """
        self.name = name
        self.description = description
        self.mcp_endpoint = mcp_endpoint
        self.capabilities = []
        self.initialized = False

    def initialize(self) -> bool:
        """
        初始化MCP工具连接
        
        Returns:
            是否初始化成功
        """
        try:
            # 尝试连接MCP服务并获取功能列表
            response = requests.get(f"{self.mcp_endpoint}/capabilities", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.capabilities = data.get("capabilities", [])
                self.initialized = True
                return True
            else:
                self.initialized = False
                return False
        except Exception as e:
            logger.warning("初始化MCP工具失败: %s", e)
            self.initialized = False
            return False

    def get_capabilities(self) -> List[str]:
        """
        获取工具支持的功能列表
        
        Returns:
            功能列表
        """
        if not self.initialized:
            self.initialize()
        return self.capabilities

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行MCP工具命令
        
        Args:
            **kwargs: 命令参数
            
        Returns:
            执行结果
        """
        if not self.initialized:
            if not self.initialize():
                return {
                    "success": False,
                    "error": "Failed to initialize MCP tool connection",
                    "tool": self.name
                }
            
        command = kwargs.get("command", "")
        params = kwargs.get("params", {})
        
        try:
            # 构造请求数据
            request_data = {
                "command": command,
                "params": params
            }
            
            # 发送POST请求到MCP服务
            response = requests.post(
                f"{self.mcp_endpoint}/execute",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result,
                    "tool": self.name
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "tool": self.name
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }


class MCPToolRegistry:
    """
    MCP工具注册表
    管理所有MCP工具的注册和执行
    """

    def __init__(self):
        """
        初始化MCP工具注册表
        """
        self.mcp_tools: Dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool) -> bool:
        """
        注册MCP工具
        
        Args:
            tool: MCP工具实例
            
        Returns:
            是否注册成功
        """
        if not isinstance(tool, MCPTool):
            return False
            
        self.mcp_tools[tool.name] = tool
        return True

    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销MCP工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否注销成功
        """
        if tool_name in self.mcp_tools:
            del self.mcp_tools[tool_name]
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """
        获取MCP工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            MCP工具实例或None
        """
        return self.mcp_tools.get(tool_name)

    def list_tools(self) -> Dict[str, str]:
        """
        列出所有注册的MCP工具
        
        Returns:
            工具名称和描述的字典
        """
        return {name: tool.description for name, tool in self.mcp_tools.items()}

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行MCP工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 执行参数
            
        Returns:
            执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"MCP tool '{tool_name}' not found",
                "tool": tool_name
            }
            
        return tool.run(**kwargs)


class MCPIntegrationTool:
    """
    MCP集成工具基类
    作为Agent访问MCP工具的统一入口
    """

    def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        # 入参 schema，供 function-calling 使用
        self.parameters = parameters or {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "要调用的MCP工具名称"},
                "command": {"type": "string", "description": "要执行的命令"},
                "params": {"type": "object", "description": "命令参数"},
            },
            "required": ["tool_name"],
        }

    def run(self, **kwargs) -> Any:
        """
        执行工具
        """
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 GLM/OpenAI function-calling 要求的工具 schema 格式
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ConcreteMCPIntegrationTool(MCPIntegrationTool):
    """
    具体的MCP集成工具实现
    """

    def __init__(self, mcp_registry: MCPToolRegistry):
        """
        初始化MCP集成工具
        
        Args:
            mcp_registry: MCP工具注册表
        """
        super().__init__("mcp_integration", "集成的MCP工具访问接口")
        self.mcp_registry = mcp_registry

    def run(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行指定的MCP工具
        
        Args:
            tool_name: MCP工具名称
            **kwargs: 工具执行参数
            
        Returns:
            执行结果
        """
        return self.mcp_registry.execute_tool(tool_name, **kwargs)