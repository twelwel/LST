"""LST - 程序入口
只负责：初始化日志 → 启动 UI
"""

import sys


# ==================== 初始化日志（最先） ====================
from logger import setup as setup_logger, get_logger

setup_logger('app', to_console=True)
log = get_logger('main')
# ==========================================================


def main():
    """主入口"""
    from version import APP_NAME, __version__
    log.info(f"===== {APP_NAME} v{__version__} 启动 =====")

    # 不再强制检查 API 配置，直接启动主窗口
    # 用户可在主界面点"API 设置"按钮自行配置
    from ui import TranslatorApp

    app = TranslatorApp()
    app.mainloop()

    log.info(f"===== {APP_NAME} v{__version__} 退出 =====")


if __name__ == '__main__':
    main()