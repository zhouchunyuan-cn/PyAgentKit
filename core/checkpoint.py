"""
执行检查点（Checkpoint）

为 Team 任务提供断点续跑能力：
- 每完成一个成员/子任务，把进度（含共享上下文）原子落盘
- 进程崩溃或主动暂停后，同一 run_id 再次执行会从断点继续，
  已完成的步骤不再重跑（LLM 不重复付费）
- 已完成的 run 再次执行直接返回缓存结果（幂等）

用法：
    from core import FileCheckpointStore

    team = Team(name="内容组", members=[...], process=SequentialProcess(),
                checkpoint_store=FileCheckpointStore(".checkpoints"))

    # 方式一：普通执行 + 崩溃自动恢复（同 run_id 重跑从断点继续）
    result = team.run("写报告", run_id="report-101")

    # 方式二：分步执行（人工审批/低预算分段）
    outcome = team.run_resumable("写报告", run_id="report-101", step_limit=1)
    if not outcome.completed:
        ...  # 人工检查/择时续跑
        outcome = team.run_resumable("写报告", run_id="report-101")
    print(outcome.result)
"""

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# run_id 中允许直接出现在文件名里的字符；其余替换为短哈希，防路径注入
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class RunOutcome:
    """
    一次可续跑执行的结局

    Attributes:
        result: 最终结果文本；未完成时为 None
        completed: 任务是否已完整执行（含 Leader 汇总）
        steps_done: 本次与历史累计完成的成员产出数
        run_id: 检查点标识（未启用检查点时为 None）
    """

    result: str | None
    completed: bool
    steps_done: int
    run_id: str | None = None


class FileCheckpointStore:
    """
    基于文件的检查点存储（原子写入，损坏文件自动忽略）

    每个检查点是一个 JSON 文件：{task, process, process_state, steps, context, ...}。
    序列化在截断之前完成，写入失败不会破坏既有检查点。
    """

    def __init__(self, dir_path: str = ".checkpoints"):
        self.dir_path = dir_path
        os.makedirs(dir_path, exist_ok=True)

    def _path(self, run_id: str) -> str:
        if not _SAFE_RUN_ID.match(run_id or ""):
            # 含路径分隔符等字符的 run_id：用哈希替代，保文件名安全且互不冲突
            digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
            run_id = f"run-{digest}"
        return os.path.join(self.dir_path, f"{run_id}.json")

    def save(self, run_id: str, state: dict) -> None:
        """保存检查点（原子写）；state 必须可 JSON 序列化，否则抛出且不动旧文件"""
        path = self._path(run_id)
        payload = json.dumps(state, ensure_ascii=False, indent=2)
        fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", suffix=".tmp", dir=self.dir_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        logger.debug("检查点已保存: %s", path)

    def load(self, run_id: str) -> dict | None:
        """读取检查点；不存在或损坏（无法解析/非字典）返回 None"""
        path = self._path(run_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("检查点格式非法（应为 JSON 对象）")
            return data
        except Exception as e:
            logger.warning("检查点 %s 无法读取，忽略: %s", path, e)
            return None

    def delete(self, run_id: str) -> bool:
        """删除检查点（重新开始任务时使用）"""
        path = self._path(run_id)
        if os.path.exists(path):
            os.unlink(path)
            return True
        return False
