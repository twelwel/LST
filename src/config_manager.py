"""本地配置保存/加载
保存位置：~/.screen_translator/user_config.json
"""

import os
import json
import threading

from logger import get_logger


log = get_logger('config')


CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'user_config.json')


DEFAULT_USER_CONFIG = {
    "version": 1,

    # 区域
    "region": {
        "top": 378,
        "left": 982,
        "width": 847,
        "height": 432,
    },

    # 性能模式
    "performance_mode": "medium",

    # 加速模式
    "accel_mode": "",

    # CPU 引擎数（可调）
    "cpu_engine_count": 2,
    # OCR 模型大小（'server' / 'mobile'）
    "ocr_model_size": "server",

    # 高精度 OCR（多引擎投票）
    "enable_high_accuracy_ocr": False,

    # DirectML 多进程 worker 数
    "dml_worker_count": 2,

    # 译文模式
    "follow_mode": "follow",

    # 显示模式
    "display_mode": "overlay",

    # 译文颜色
    "fg_color": "auto",

    # 优化开关
    "enable_frame_diff": True,
    "enable_block_ocr": True,
    "enable_filter": True,
    "enable_async_translate": True,
    "enable_window_pool": True,

    # 快速响应层
    "enable_fast_lookup": True,

    # 显示设置
    "scan_interval": 0.30,
    "font_size": 14,
    "use_inpainting": False,

    # 缓存
    "auto_clean": True,
    "cache_max_size": 500,

    "phrase_table_path": "",

    # 热键
    "hotkeys": {
        "toggle": "ctrl+alt+t",
        "config": "f2",
        "border": "f3",
        "freeze": "ctrl+alt+l",
        "quit":   "ctrl+esc",
    },

    # 窗口
    "window_geometry": "760x800+80+40",
    "enable_fine_text": False,
    "enable_async_overlay": False,
    "enable_general_filter": True,
    "local_dict_path": "",

    # 翻译策略
    "translation_strategy": "single_translate",
}


class ConfigManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None
        self.load()

    def load(self):
        with self._lock:
            self._data = {
                **DEFAULT_USER_CONFIG,
                'region': dict(DEFAULT_USER_CONFIG['region']),
                'hotkeys': dict(DEFAULT_USER_CONFIG['hotkeys']),
            }

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
        if not isinstance(loaded, dict):
            return

        for key, val in loaded.items():
            if key == 'hotkeys' and isinstance(val, dict):
                self._data['hotkeys'].update(val)
            elif key == 'region' and isinstance(val, dict):
                self._data['region'].update(val)
            else:
                self._data[key] = val

        # 迁移：旧 translation_mode → 新 translation_strategy
        if 'translation_strategy' not in loaded:
            old = loaded.get('translation_mode')
            if old in ('local', 'hybrid', 'online'):
                migrate = {
                    'local':  'local_only',
                    'hybrid': 'hybrid',
                    'online': 'single_translate',
                }
                self._data['translation_strategy'] = migrate[old]

        # 迁移：旧 enable_phrase_cache → 新 enable_fast_lookup
        if 'enable_fast_lookup' not in loaded:
            old_flag = loaded.get('enable_phrase_cache')
            if old_flag is not None:
                self._data['enable_fast_lookup'] = bool(old_flag)

        # 迁移：旧 bilingual_fg → 新 fg_color
        if 'fg_color' not in loaded:
            old_fg = loaded.get('bilingual_fg')
            if old_fg:
                self._data['fg_color'] = old_fg

    def save(self):
        with self._lock:
            try:
                os.makedirs(CONFIG_DIR, exist_ok=True)
                tmp = CONFIG_FILE + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, CONFIG_FILE)
                return True
            except Exception as e:
                log.error(f"[配置] 保存失败：{e}")
                return False

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
        with self._lock:
            for k, v in kwargs.items():
                self._data[k] = v

    def reset_to_default(self):
        with self._lock:
            self._data = {
                **DEFAULT_USER_CONFIG,
                'region': dict(DEFAULT_USER_CONFIG['region']),
                'hotkeys': dict(DEFAULT_USER_CONFIG['hotkeys']),
            }
        self.save()


_instance = None
_instance_lock = threading.Lock()


def get_config():
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ConfigManager()
        return _instance


def get_config_path():
    return CONFIG_FILE


def open_config_dir():
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