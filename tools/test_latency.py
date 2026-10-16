"""延迟测试 CLI

实现已移到运行时模块 src/latency_tester.py
（界面里的「延迟测试」对话框复用其中的 LatencyTester），
这里只保留命令行入口。

用法：
    python tools/test_latency.py --api --local
    python tools/test_latency.py --e2e -n 10
    python tools/test_latency.py --help
"""

import os
import sys

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from latency_tester import _cli  # noqa: E402


if __name__ == '__main__':
    _cli()
