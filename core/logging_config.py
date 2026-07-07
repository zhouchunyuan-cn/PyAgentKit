"""
统一日志配置

集中管理 PyAgentKit 的日志输出，替代散落在各模块的 print。
要点：
- 应用日志默认 INFO 级别，可由环境变量 LOG_LEVEL 覆盖
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


def setup_logging(level: str = None) -> None:
    """
    初始化全局日志配置

    Args:
        level: 日志级别字符串（"DEBUG"/"INFO"/"WARNING"/"ERROR"），
               默认从环境变量 LOG_LEVEL 读取，再缺省则 INFO
    """
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()

    # 配置根 logger
    logging.basicConfig(
        level=resolved,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        stream=sys.stdout,
        force=True,  # 覆盖此前可能存在的配置（如 basicConfig 重复调用）
    )

    # 压制第三方库噪音
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
