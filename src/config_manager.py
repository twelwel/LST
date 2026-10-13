"""本地配置保存/加载
保存位置：~/.screen_translator/user_config.json
"""

import os
import json
import threading

from logger import get_logger


log = get_logger('config')


# ==================== 路径 ====================
CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'user_config.json')
# =================================================


# ==================== 默认配置 ====================
DEFAULT_USER_CONFIG = {
    "version": 1,

    # 区域
    "region": {
        "top": 378,
        "left": 982,
        "width": 847,
        "height": 432,
    },

    # 性能模式（'fast' / 'medium' / 'slow'）
    "performance_mode": "medium",

    # 加速方式
    "use_gpu": False,
    "use_multithread_ocr": False,
    "multithread_pool_size": 2,

    # 优化开关
    "enable_frame_diff": True,
    "enable_block_ocr": True,
    "enable_filter": True,
    "enable_async_translate": True,
    "enable_window_pool": True,

    # 显示设置
    "scan_interval": 0.30,
    "font_size": 14,
    "use_inpainting": False,
    "freeze_mode": "follow",          # 'follow' 或 'pin'

    # 缓存
    "auto_clean": True,
    "cache_max_size": 500,

    # 热键
    "hotkeys": {
        "toggle": "ctrl+alt+t",
        "config": "f2",
        "border": "f3",
        "freeze": "ctrl+alt+l",
        "quit":   "ctrl+esc",
    },

    # 窗口
    "window_geometry": "740x820+50+50",
}
# =====================================================


class ConfigManager:
    """线程安全的配置管理器"""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self.load()

    # ============ 加载 ============

    def load(self):
        """从磁盘加载配置"""
        with self._lock:
            self._data = dict(DEFAULT_USER_CONFIG)

            if not os.path.exists(CONFIG_FILE):
                log.info("[配置] 首次运行，使用默认配置")
                return

            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self._merge(loaded)
                log.info(f"[配置] 已加载：{CONFIG_FILE}")
            except Exception as e:
                log.error(f"[配置] 加载失败，使用默认：{e}")

    def _merge(self, loaded):
        """深度合并（用户配置覆盖默认配置）"""
        if not isinstance(loaded, dict):
            return
        for key, val in loaded.items():
            if key == 'hotkeys' and isinstance(val, dict):
                self._data['hotkeys'].update(val)
            elif key == 'region' and isinstance(val, dict):
                self._data['region'].update(val)
            else:
                self._data[key] = val

    # ============ 保存 ============

    def save(self):
        """保存到磁盘"""
        with self._lock:
            try:
                os.makedirs(CONFIG_DIR, exist_ok=True)
                tmp = CONFIG_FILE + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                # 原子替换
                os.replace(tmp, CONFIG_FILE)
                log.info(f"[配置] 已保存：{CONFIG_FILE}")
                return True
            except Exception as e:
                log.error(f"[配置] 保存失败：{e}")
                return False

    # ============ 读写接口 ============

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get_region(self):
        with self._lock:
            return dict(self._data.get('region',
                                       DEFAULT_USER_CONFIG['region']))

    def set_region(self, region):
        with self._lock:
            self._data['region'] = dict(region)

    def get_hotkeys(self):
        with self._lock:
            return dict(self._data.get('hotkeys',
                                       DEFAULT_USER_CONFIG['hotkeys']))

    def set_hotkeys(self, hotkeys):
        with self._lock:
            self._data['hotkeys'] = dict(hotkeys)

    def update(self, **kwargs):
        """批量更新"""
        with self._lock:
            for k, v in kwargs.items():
                self._data[k] = v

    def reset_to_default(self):
        """恢复默认"""
        with self._lock:
            self._data = dict(DEFAULT_USER_CONFIG)
        self.save()


# ==================== 全局单例 ====================
_instance = None
_instance_lock = threading.Lock()


def get_config():
    """获取全局配置管理器实例"""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ConfigManager()
        return _instance


def get_config_path():
    """返回配置文件路径"""
    return CONFIG_FILE


def open_config_dir():
    """用资源管理器打开配置目录"""
    import sys
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(CONFIG_DIR)
        elif sys.platform == 'darwin':
            os.system(f'open "{CONFIG_DIR}"')
        else:
            os.system(f'xdg-open "{CONFIG_DIR}"')
        return True
    except Exception as e:
        log.error(f"[配置] 打开目录失败：{e}")
        return False