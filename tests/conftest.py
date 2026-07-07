"""
pytest 公共配置

确保项目根目录在 sys.path 中，使测试可以用 `from core import ...`
"""
import os
import sys

# 将项目根目录（tests 的上级）加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
