"""LST - 程序入口
只负责：初始化日志 → 启动 UI
"""

import logging
import multiprocessing
import os
import sys


# ==================== 无控制台模式兜底 ====================
# PyInstaller console=False 时 sys.stdout / sys.stderr 为 None。
# 部分三方库（如 huggingface_hub 的 tqdm）会直接写入，导致下载等功能异常。
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w', encoding='utf-8')
# ==========================================================


# ==================== 初始化日志（最先） ====================
from logger import setup as setup_logger, get_logger

try:
    setup_logger('app', to_console=True)
except Exception:
    # 日志初始化不能阻断程序启动；极端情况下退化为无文件日志。
    logging.basicConfig(level=logging.INFO)
log = get_logger('main')
# ==========================================================


def main():
    """主入口"""
    from version import APP_NAME, __version__
    log.info(f"===== {APP_NAME} v{__version__} 启动 =====")

    from ui import TranslatorApp

    app = TranslatorApp()
    app.mainloop()

    log.info(f"===== {APP_NAME} v{__version__} 退出 =====")


if __name__ == '__main__':
    # ⚠️ PyInstaller 打包 + multiprocessing 必须
    multiprocessing.freeze_support()

    try:
        main()
    except Exception:
        # console=False 时看不到标准错误，把启动异常写入日志便于排查。
        log.exception("启动失败")
        raise
