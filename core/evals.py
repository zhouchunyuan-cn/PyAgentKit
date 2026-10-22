"""
评估（Eval）——Agent 行为的回归基准

单元测试守护"代码没坏"，eval 守护"Agent 表现没变差"。改一版 prompt、
换一个模型，跑同一组用例对比通过率——这是企业 agent 工程的标配闭环。

两种判定方式（可组合）：
- 关键词断言：回答必须包含指定的关键词（大小写不敏感），零成本、确定性
- 自定义判定 / LLM-as-judge：judge(question, reply) -> bool，
  用另一个 LLM 按评分标准判断回答质量（make_llm_judge 一行构造）

用法：
    from core import EvalCase, run_eval, make_llm_judge

    cases = [
        EvalCase(input="计算 (15+27)*3", expect_keywords=["126"]),
        EvalCase(input="介绍量子计算", judge=make_llm_judge(judge_llm, "准确且通俗易懂")),
    ]
    report = run_eval(agent, cases, name="冒烟集")
    print(report.format_text())
    # 通过率、逐用例结果、token 成本一目了然
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """
    单个评估用例

    Attributes:
        input: 喂给 agent.think 的输入
        expect_keywords: 回答应包含的关键词（全部命中才通过，大小写不敏感）
        judge: 自定义判定 (question, reply) -> bool；优先于关键词
        id: 用例标识（默认取输入前 24 字符）
    """

    input: str
    expect_keywords: list[str] = field(default_factory=list)
    judge: Callable[[str, str], bool] | None = None
    id: str = ""


@dataclass
class CaseResult:
    """单个用例的评估结果"""

    case_id: str
    input: str
    reply: str
    passed: bool
    reason: str
    tokens: int = 0


@dataclass
class EvalReport:
    """一次评估的汇总报告"""

    name: str = "eval"
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / len(self.results) if self.results else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def format_text(self) -> str:
        lines = [
            f"=== Eval: {self.name} ===",
            f"通过: {self.passed_count}/{len(self.results)}"
            f"（{self.pass_rate:.0%}）  token 合计: {self.total_tokens}",
            "",
        ]
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            lines.append(f"  {mark} [{r.case_id}] {r.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed_count,
            "total": len(self.results),
            "pass_rate": self.pass_rate,
            "total_tokens": self.total_tokens,
            "results": [
                {
                    "case_id": r.case_id,
                    "input": r.input,
                    "reply": r.reply,
                    "passed": r.passed,
                    "reason": r.reason,
                    "tokens": r.tokens,
                }
                for r in self.results
            ],
        }

    def save(self, path: str) -> None:
        """报告落盘 JSON（版本对比/CI 存档用）"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def run_eval(
    agent: Any,
    cases: list[EvalCase],
    name: str = "eval",
    use_tools: bool = True,
    isolate_cases: bool = True,
) -> EvalReport:
    """
    对 Agent 跑一组评估用例

    Args:
        agent: 被评估的 Agent（需支持 think 与 get_token_usage）
        cases: 用例列表
        name: 报告名（如 "冒烟集"/"v3-prompt-变更前"）
        use_tools: 是否允许调用工具
        isolate_cases: 用例间隔离（清空会话与短期记忆，避免互相污染；
                       评估的是单轮能力。多轮场景请自行组织用例并关闭）

    Returns:
        EvalReport（通过率、逐用例结果、token 成本）
    """
    report = EvalReport(name=name)
    for case in cases:
        if isolate_cases:
            agent.brain.clear_conversation()
            agent.memory.clear(memory_type="short")
        tokens_before = agent.get_token_usage().total_tokens
        try:
            reply = agent.think(case.input, use_tools=use_tools)
        except Exception as e:
            reply = ""
            logger.exception("用例 %s 执行异常", case.id or case.input[:24])
            report.results.append(
                CaseResult(
                    case_id=case.id or case.input[:24],
                    input=case.input,
                    reply="",
                    passed=False,
                    reason=f"执行异常: {e}",
                    tokens=agent.get_token_usage().total_tokens - tokens_before,
                )
            )
            continue
        tokens_used = agent.get_token_usage().total_tokens - tokens_before

        if case.judge is not None:
            try:
                passed = bool(case.judge(case.input, reply))
                reason = "judge 判定通过" if passed else "judge 判定未通过"
            except Exception as e:
                passed, reason = False, f"judge 异常: {e}"
        elif case.expect_keywords:
            missing = [k for k in case.expect_keywords if k.lower() not in reply.lower()]
            passed = not missing
            reason = "关键词全部命中" if passed else f"缺少关键词: {missing}"
        else:
            passed, reason = True, "无断言（仅记录）"

        report.results.append(
            CaseResult(
                case_id=case.id or case.input[:24],
                input=case.input,
                reply=reply,
                passed=passed,
                reason=reason,
                tokens=tokens_used,
            )
        )
    return report


def make_llm_judge(
    judge_llm: LLMClient, rubric: str = "回答准确、完整、忠实于事实"
) -> Callable[[str, str], bool]:
    """
    构造 LLM-as-judge 判定函数

    Args:
        judge_llm: 评审用 LLM 客户端（建议与被评 Agent 不同实例，避免自评偏置）
        rubric: 评分标准（自然语言描述）

    Returns:
        judge(question, reply) -> bool
    """

    def judge(question: str, reply: str) -> bool:
        prompt = (
            f"评分标准：{rubric}\n\n问题：{question}\n\n待评回答：\n{reply}\n\n"
            "该回答是否达到标准？只输出 PASS 或 FAIL（首行）。"
        )
        result = judge_llm.chat(
            [{"role": "user", "content": prompt}], tools=None, tool_choice="none"
        )
        verdict = (result.content or "").strip().upper()
        return verdict.startswith("PASS")

    return judge
