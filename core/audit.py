"""
审计日志（Audit）

企业合规的硬需求：谁（actor）、何时、让哪个 Agent 做了什么——
包括每次工具调用的名称、参数与结果摘要。以结构化 JSONL 落盘，
便于后续检索、对账与合规审查。

设计：复用 AgentHooks 生命周期钩子（零侵入——不改 Agent/Engine 一行代码），
AuditLogger.make_hooks() 产出一组钩子挂到 Agent 上即可开始记录。

敏感信息脱敏：参数与结果中命中 redact_keys（password/token/secret/api_key 等，
大小写不敏感）的键值会被掩码为 ***，防止密钥/口令进入审计文件。

用法：
    from core import AuditLogger

    audit = AuditLogger("audit.jsonl", actor="user-123")
    agent = MyAgent(hooks=audit.make_hooks())      # 新建 Agent 时
    audit.install(agent)                            # 或对已有 Agent 安装（替换现有钩子）

    agent.think("...")   # 每次推理/工具调用/错误都追加一行 JSON
"""

import json
import logging
import threading
from datetime import datetime
from typing import Any

from .hooks import AgentHooks

logger = logging.getLogger(__name__)

# 默认脱敏的参数键（大小写不敏感、子串匹配）
DEFAULT_REDACT_KEYS = (
    "password",
    "passwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
)


def redact(value: Any, redact_keys: tuple[str, ...] = DEFAULT_REDACT_KEYS, depth: int = 6) -> Any:
    """递归脱敏：命中敏感键的值替换为 ***（防止密钥/口令进入审计与日志）"""
    if depth <= 0:
        return "..."
    if isinstance(value, dict):
        return {
            k: (
                "***"
                if any(r in str(k).lower() for r in redact_keys)
                else redact(v, redact_keys, depth - 1)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v, redact_keys, depth - 1) for v in value]
    return value


def _preview(value: Any, limit: int = 500) -> Any:
    """长文本截断，控制审计行体积"""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...(截断，共{len(value)}字符)"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > limit:
            return text[:limit] + "...(截断)"
    return value


class AuditLogger:
    """
    结构化审计日志器（JSONL 追加写，线程安全）

    每行一个 JSON 对象：{"ts", "actor", "agent", "event", ...fields}
    """

    def __init__(
        self,
        path: str = "audit.jsonl",
        actor: str = "unknown",
        redact_keys: tuple[str, ...] = DEFAULT_REDACT_KEYS,
    ):
        """
        Args:
            path: 审计文件路径（JSONL，逐行追加）
            actor: 操作者标识（用户 ID / 系统名）——合规回答"谁"
            redact_keys: 需要脱敏的参数键（子串匹配，大小写不敏感）
        """
        self.path = path
        self.actor = actor
        self.redact_keys = tuple(k.lower() for k in redact_keys)
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        """追加一条审计记录（写失败仅告警，不影响主流程）"""
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "actor": self.actor,
            "event": event,
        }
        record.update(redact(fields, self.redact_keys))
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # 审计失败不阻断业务，但必须可见
            logger.error("审计日志写入失败（%s）: %s", self.path, e)

    # ------------------------------------------------------------------
    # 与 Agent 集成（复用 AgentHooks，零侵入）
    # ------------------------------------------------------------------
    def make_hooks(self) -> AgentHooks:
        """构造挂接到 Agent 生命周期的审计钩子"""
        return AgentHooks(
            on_think_start=self._on_think_start,
            on_tool_call=self._on_tool_call,
            on_think_end=self._on_think_end,
            on_error=self._on_error,
        )

    def install(self, *agents: Any) -> None:
        """对已有 Agent 安装审计钩子（注意：会替换该 Agent 现有钩子）"""
        hooks = self.make_hooks()
        for agent in agents:
            agent.hooks = hooks

    # 钩子实现 ------------------------------------------------------------
    def _on_think_start(self, agent, user_input):
        self.log(
            "think_start",
            agent=agent.agent_id,
            input_preview=_preview(str(user_input)[:200]),
        )

    def _on_tool_call(self, agent, tool_name, arguments, result):
        self.log(
            "tool_call",
            agent=agent.agent_id,
            tool=tool_name,
            arguments=_preview(arguments),
            result_preview=_preview(result),
        )

    def _on_think_end(self, agent, response):
        self.log("think_end", agent=agent.agent_id, output_preview=_preview(str(response)[:300]))

    def _on_error(self, agent, error, context):
        self.log("error", agent=agent.agent_id, error=str(error)[:300], context=str(context)[:200])
