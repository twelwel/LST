"""日志模块
- 每次启动生成一份独立日志文件
- 文件名带时间戳：app_YYYYMMDD_HHMMSS.log
- 退出时记录使用时间段
- 输出到文件 + 控制台
- 线程安全
"""

import os
import sys
import atexit
import logging
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler


# ==================== 配置 ====================
LOG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator', 'logs')
LOG_PREFIX = 'app_'
LOG_EXT = '.log'

# 单文件最大 5MB，保留 5 个轮转备份
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# 历史日志最大保留份数（不同启动会话）
MAX_LOG_FILES = 30

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s | %(levelname)-5s | %(name)-12s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# ==============================================


# ==================== 全局状态 ====================
_initialized = False
_init_lock = threading.Lock()
_start_time = None
_current_log_path = None
_session_id = None
_exit_registered = False
# =================================================


def _build_log_filename():
    """生成带时间戳的日志文件名"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{LOG_PREFIX}{ts}{LOG_EXT}"


def _clean_old_logs():
    """清理旧日志，只保留最近 MAX_LOG_FILES 份"""
    try:
        if not os.path.isdir(LOG_DIR):
            return
        files = []
        for fn in os.listdir(LOG_DIR):
            if fn.startswith(LOG_PREFIX) and LOG_EXT in fn:
                full = os.path.join(LOG_DIR, fn)
                if os.path.isfile(full):
                    files.append((os.path.getmtime(full), full))
        # 按修改时间倒序（新的在前）
        files.sort(reverse=True)
        # 删除多余的
        for _, path in files[MAX_LOG_FILES:]:
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        pass


def _build_file_handler():
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        _current_log_path,
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


def _format_duration(seconds):
    """格式化时长：Xh Ym Zs"""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}小时{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def _on_exit():
    """程序退出时记录使用时长"""
    global _start_time
    if _start_time is None:
        return
    try:
        now = datetime.now()
        elapsed = (now - _start_time).total_seconds()
        duration = _format_duration(elapsed)

        logger = logging.getLogger('app')
        logger.info("=" * 60)
        logger.info(f"会话结束 | 使用时长：{duration}")
        logger.info(f"启动时间：{_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"结束时间：{now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 确保全部写出
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass


def setup(name='app', level=logging.INFO, to_console=True):
    """初始化日志系统（幂等）"""
    global _initialized, _start_time, _current_log_path, _session_id
    global _exit_registered

    with _init_lock:
        if _initialized:
            return logging.getLogger(name)

        # 只记录一次启动时间
        if _start_time is None:
            _start_time = datetime.now()
            _session_id = _start_time.strftime('%Y%m%d_%H%M%S')
            _current_log_path = os.path.join(LOG_DIR, _build_log_filename())

        os.makedirs(LOG_DIR, exist_ok=True)
        _clean_old_logs()

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

        logger = logging.getLogger(name)
        logger.info("=" * 60)
        logger.info(f"会话开始 | Session ID: {_session_id}")
        logger.info(f"启动时间：{_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"日志文件：{_current_log_path}")
        logger.info("=" * 60)

        # 只注册一次退出回调
        if not _exit_registered:
            atexit.register(_on_exit)
            _exit_registered = True

        return logger


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
    """当前会话的日志文件路径"""
    return _current_log_path or os.path.join(LOG_DIR, 'app.log')


def get_log_dir():
    return LOG_DIR


def get_session_id():
    """当前会话 ID（YYYYMMDD_HHMMSS）"""
    return _session_id


def get_start_time():
    """会话启动时间"""
    return _start_time


def clear_logs():
    """清空所有日志文件"""
    global _initialized
    try:
        if not os.path.isdir(LOG_DIR):
            return False

        # 关闭 handler 释放文件锁
        root = logging.getLogger()
        handlers_backup = list(root.handlers)
        for h in handlers_backup:
            try:
                h.close()
                root.removeHandler(h)
            except Exception:
                pass

        # 删除所有 app_*.log 及其轮转备份
        removed = 0
        for fname in os.listdir(LOG_DIR):
            if fname.startswith(LOG_PREFIX) and LOG_EXT in fname:
                try:
                    os.remove(os.path.join(LOG_DIR, fname))
                    removed += 1
                except Exception:
                    pass

        # 重新挂 handler（不重置 _start_time）
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