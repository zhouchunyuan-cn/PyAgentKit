"""
统一日志配置

集中管理 PyAgentKit 的日志输出，替代散落在各模块的 print。
要点：
- 应用日志默认 INFO 级别，可由环境变量 LOG_LEVEL 覆盖
- 日志走 stderr：stdout 留给程序输出（如 `pyagentkit ask q > out.txt`
  只会捕获答复，不会混入日志行）
- 第三方库噪音（httpx 请求日志、zhipuai）统一压到 WARNING，避免淹没业务输出
- 在程序入口调用 setup_logging() 一次即可
"""

import logging
import os
import sys

# 默认格式：时间 [级别] 模块名: 消息
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

# 这些第三方库的 INFO 日志很吵（每次 HTTP 请求都打一行），统一降到 WARNING
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "zhipuai")


def setup_logging(level: str | None = None) -> None:
    """
    初始化全局日志配置

    Args:
        level: 日志级别字符串（"DEBUG"/"INFO"/"WARNING"/"ERROR"），
               默认从环境变量 LOG_LEVEL 读取，再缺省则 INFO

    Raises:
        ValueError: 日志级别字符串非法（附可用级别提示）
    """
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if resolved not in valid_levels:
        raise ValueError(
            f"无效的日志级别 {resolved!r}，可用: {sorted(valid_levels)}"
            "（也可通过环境变量 LOG_LEVEL 配置）"
        )

    # 配置根 logger（输出到 stderr，与程序 stdout 输出分离）
    logging.basicConfig(
        level=resolved,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        stream=sys.stderr,
        force=True,  # 覆盖此前可能存在的配置（如 basicConfig 重复调用）
    )

    # 压制第三方库噪音
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
