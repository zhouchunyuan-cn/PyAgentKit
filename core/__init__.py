# PyAgentKit Core Package
#
# 版本号：与 pyproject.toml 的 [project].version 保持同步。
# 安装后可通过 importlib.metadata 读取（兼容 pip install -e 场景）。
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pyagentkit")
except PackageNotFoundError:  # 未安装为包（如直接从源码运行）
    __version__ = "0.0.0+local"

from .agent import Agent
from .async_router import AsyncRouter
from .audit import AuditLogger, redact
from .checkpoint import CheckpointBusyError, FileCheckpointStore, RunOutcome
from .collaboration import (
    CapabilityCollaborationStrategy,
    CollaborationStrategy,
    DynamicCollaborationManager,
)
from .evals import EvalCase, EvalReport, make_llm_judge, run_eval
from .hooks import AgentHooks
from .llm import ChatResult, GLMClient, LLMClient, StreamChunk, TokenUsage, ToolCall
from .mcp_client import (
    MCPClientError,
    MCPServerConfig,
    MCPToolProvider,
    RemoteMCPTool,
    validate_mcp_url,
)
from .mcp_tools import MCPIntegrationTool, MCPTool, MCPToolRegistry
from .memory import Memory
from .memory_manager import MemoryManager
from .message import Message
from .openai_client import OpenAICompatClient
from .react_engine import AgentStreamEvent, ReactEngine
from .router import Router
from .session import ConversationSession, SessionManager
from .team import (
    HierarchicalProcess,
    Process,
    SequentialProcess,
    SharedContext,
    Task,
    Team,
)
from .tools import CalculatorTool, DatabaseTool, FileReadTool, Tool, ToolRegistry, WebSearchTool
from .trace import JsonlTraceExporter, Trace, Tracer, TraceSpan
from .vector_memory import Embedder, GLMEmbedder, LocalTfidfEmbedder, VectorMemory

__all__ = [
    "Agent",
    "AuditLogger",
    "redact",
    "CheckpointBusyError",
    "EvalCase",
    "EvalReport",
    "run_eval",
    "make_llm_judge",
    "JsonlTraceExporter",
    "AgentHooks",
    "AgentStreamEvent",
    "FileCheckpointStore",
    "RunOutcome",
    "Message",
    "Router",
    "Memory",
    "MemoryManager",
    "Tool",
    "WebSearchTool",
    "CalculatorTool",
    "FileReadTool",
    "DatabaseTool",
    "ToolRegistry",
    "MCPTool",
    "MCPToolRegistry",
    "MCPIntegrationTool",
    "MCPClientError",
    "MCPServerConfig",
    "MCPToolProvider",
    "RemoteMCPTool",
    "validate_mcp_url",
    "LLMClient",
    "GLMClient",
    "OpenAICompatClient",
    "ChatResult",
    "ToolCall",
    "TokenUsage",
    "StreamChunk",
    "CollaborationStrategy",
    "CapabilityCollaborationStrategy",
    "DynamicCollaborationManager",
    "VectorMemory",
    "GLMEmbedder",
    "LocalTfidfEmbedder",
    "Embedder",
    "ConversationSession",
    "SessionManager",
    "Team",
    "Process",
    "SequentialProcess",
    "HierarchicalProcess",
    "SharedContext",
    "Task",
    "AsyncRouter",
    "ReactEngine",
    "Tracer",
    "Trace",
    "TraceSpan",
]
