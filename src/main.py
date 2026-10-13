"""LST - 程序入口
只负责：初始化日志 → 检查 API 配置 → 启动 UI
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

    # 1) 检查 API 配置
    from api_config import is_configured

    if not is_configured():
        log.info("首次启动，弹出 API 配置对话框")
        from api_setup_dialog import APISetupDialog

        dlg = APISetupDialog()
        dlg.mainloop()

        if not dlg.result:
            log.info("[启动] 用户未完成 API 配置，退出")
            sys.exit(0)

    # 2) 启动主窗口
    from ui import TranslatorApp

    app = TranslatorApp()
    app.mainloop()

    log.info(f"===== {APP_NAME} v{__version__} 退出 =====")


if __name__ == '__main__':
    main()