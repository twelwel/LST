"""全局配置"""

import ctypes


def _get_screen_size():
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _make_default_region():
    sw, sh = _get_screen_size()
    w = int(sw * 0.70)
    h = int(sh * 0.30)
    x = (sw - w) // 2
    y = int(sh * 0.35)
    return {"top": y, "left": x, "width": w, "height": h}


INITIAL_REGION = _make_default_region()


# ==================== 性能模式 ====================
PERFORMANCE_MODES = {
    'fast': {
        'name': '快',
        'scan_interval': 0.15,
        'change_ratio_threshold': 0.02,
        'full_frame_ratio': 0.15,
        'bbox_padding': 80,
        'max_ocr_width': 1600,
        'max_ocr_height': 700,
        'fast_mode': True,          # 分段翻译 + 事件驱动
    },
    'medium': {
        'name': '中',
        'scan_interval': 0.30,
        'change_ratio_threshold': 0.03,
        'full_frame_ratio': 0.25,
        'bbox_padding': 60,
        'max_ocr_width': 1400,
        'max_ocr_height': 600,
        'fast_mode': False,
    },
    'slow': {
        'name': '慢',
        'scan_interval': 0.80,
        'change_ratio_threshold': 0.05,
        'full_frame_ratio': 0.35,
        'bbox_padding': 40,
        'max_ocr_width': 1000,
        'max_ocr_height': 500,
        'fast_mode': False,
    },
}

DEFAULT_MODE = 'medium'

# ==================== 优化功能开关 ====================
DEFAULT_ENABLE_FRAME_DIFF = True
DEFAULT_ENABLE_BLOCK_OCR = True
DEFAULT_ENABLE_FILTER = True
DEFAULT_ENABLE_ASYNC_TRANSLATE = True
DEFAULT_ENABLE_WINDOW_POOL = True
DEFAULT_ENABLE_MULTITHREAD_OCR = False

# 多线程 OCR
MULTITHREAD_OCR_POOL_SIZE = 2

# 精细文字替换
DEFAULT_ENABLE_FINE_TEXT = False

# ==================== GPU 加速 ====================
DEFAULT_USE_GPU = False
GPU_USE_CUDA = True
GPU_CUDNN_SEARCH = "DEFAULT"
GPU_CLS_ON_CPU = True

# ==================== UI 默认值 ====================
INITIAL_SCAN_INTERVAL = PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval']
INITIAL_FONT_SIZE = 15

SCAN_MIN, SCAN_MAX = 0.10, 3.0
FONT_MIN, FONT_MAX = 8, 28

IDLE_THRESHOLD_FOR_SLOWDOWN = 3
IDLE_SLOWDOWN_FACTOR = 3.0

FONT_NAME = 'Microsoft YaHei'
REUSE_MIN_OVERLAP = 0.5

INITIAL_HOTKEYS = {
    'toggle': 'ctrl+alt+t',
    'config': 'f2',
    'border': 'f3',
    'freeze': 'ctrl+alt+l',
    'quit':   'ctrl+esc',
}

HOTKEY_LABELS = {
    'toggle': '开始 / 暂停',
    'config': '修改区域',
    'border': '隐藏/显示边框',
    'freeze': '切换译文模式',
    'quit':   '退出程序',
}

FRAME_SIMILARITY_THRESHOLD = 0.9
POSITION_QUANTIZE = 10

DEFAULT_USE_INPAINTING = False

DEFAULT_ENABLE_ASYNC_OVERLAY = False   # 异步覆盖（逐字显示）

DEFAULT_ENABLE_GENERAL_FILTER = True   # 通用过滤（艺术字/数字/单位等）

DEFAULT_ENABLE_PHRASE_CACHE = True    # 短语缓存池

DEFAULT_PHRASE_TABLE_PATH = ""       # 外部短语表路径（用户指定）

DEFAULT_TRANSLATION_MODE = "hybrid"     # 'local' / 'hybrid' / 'online'
DEFAULT_LOCAL_DICT_PATH = ""            # 本地词典路径