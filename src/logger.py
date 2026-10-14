"""
日志模块
- 输出到文件 + 控制台
- 按大小自动轮转
- 支持日志级别
- 线程安全
"""

import os
import sys
import logging
import threading
from logging.handlers import RotatingFileHandler


# ==================== 配置 ====================
LOG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator', 'logs')
LOG_FILENAME = 'app.log'
LOG_MAX_BYTES = 5 * 1024 * 1024      # 单文件最大 5MB
LOG_BACKUP_COUNT = 5                  # 保留 5 个备份
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# ==============================================


def _build_file_handler():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, LOG_FILENAME)
    handler = RotatingFileHandler(
        path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    handler.setLevel(LOG_LEVEL)
    return handler


def _build_console_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    handler.setLevel(LOG_LEVEL)
    return handler


# ==================== 全局初始化 ====================

_initialized = False
_init_lock = threading.Lock()


def setup(name='app', level=logging.INFO, to_console=True):
    """初始化日志系统（幂等）"""
    global _initialized
    with _init_lock:
        if _initialized:
            return logging.getLogger(name)

        root = logging.getLogger()
        root.setLevel(level)

        for h in list(root.handlers):
            root.removeHandler(h)

        root.addHandler(_build_file_handler())
        if to_console:
            root.addHandler(_build_console_handler())

        for noisy in ('mss', 'PIL', 'urllib3', 'requests', 'matplotlib'):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _initialized = True
        return logging.getLogger(name)


def get_logger(name):
    if not _initialized:
        setup()
    return logging.getLogger(name)


# ==================== 便捷函数 ====================

def log_info(msg, name='app'):
    get_logger(name).info(msg)


def log_warning(msg, name='app'):
    get_logger(name).warning(msg)


def log_error(msg, name='app'):
    get_logger(name).error(msg)


def log_debug(msg, name='app'):
    get_logger(name).debug(msg)


def log_exception(msg, name='app'):
    get_logger(name).exception(msg)


# ==================== 日志文件操作 ====================

def get_log_path():
    return os.path.join(LOG_DIR, LOG_FILENAME)


def get_log_dir():
    return LOG_DIR


def clear_logs():
    """清空所有日志文件（需要先关闭文件 handler）"""
    try:
        if not os.path.isdir(LOG_DIR):
            print("[日志] 日志目录不存在")
            return False

        # 关闭所有 handler 释放文件锁
        root = logging.getLogger()
        handlers_backup = list(root.handlers)
        for h in handlers_backup:
            try:
                h.close()
                root.removeHandler(h)
            except Exception:
                pass

        # 删除文件
        removed = 0
        for fname in os.listdir(LOG_DIR):
            if fname.startswith('app.log'):
                try:
                    os.remove(os.path.join(LOG_DIR, fname))
                    removed += 1
                except Exception:
                    pass

        # 重新挂上 handler
        global _initialized
        _initialized = False
        setup()

        print(f"[日志] 已清空 {removed} 个文件")
        return True
    except Exception as e:
        print(f"[日志] 清空失败：{e}")
        return False


def open_log_dir():
    """用资源管理器打开日志目录"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(LOG_DIR)
        elif sys.platform == 'darwin':
            os.system(f'open "{LOG_DIR}"')
        else:
            os.system(f'xdg-open "{LOG_DIR}"')
        return True
    except Exception as e:
        print(f"[日志] 打开目录失败：{e}")
        return False