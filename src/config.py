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
        'fast_mode': True,
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

# CPU 多线程引擎数范围
CPU_ENGINE_COUNT_MIN = 1
CPU_ENGINE_COUNT_MAX = 8
CPU_ENGINE_COUNT_DEFAULT = 2

# 兼容旧常量
MULTITHREAD_OCR_POOL_SIZE = CPU_ENGINE_COUNT_DEFAULT

# DirectML 单进程内的引擎数（固定 1）
DIRECTML_ENGINE_COUNT = 1

# DirectML 多进程 worker 数
DML_WORKER_COUNT_MIN = 1
DML_WORKER_COUNT_MAX = 4
DML_WORKER_COUNT_DEFAULT = 2

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

DEFAULT_ENABLE_ASYNC_OVERLAY = False

DEFAULT_ENABLE_GENERAL_FILTER = True

DEFAULT_PHRASE_TABLE_PATH = ""

DEFAULT_LOCAL_DICT_PATH = ""

# ==================== 翻译策略 ====================
TRANSLATION_STRATEGIES = {
    'local_only':          '离线',
    'single_translate':    '单翻译 API',
    'multi_translate_rr':  '多翻译 API 轮询',
    'hybrid':              '翻译 API + AI 混合',
    'single_ai':           '单 AI 翻译',
    'multi_ai_vote':       '多 AI 高精度投票',
}

TRANSLATION_STRATEGY_DESC = {
    'local_only':          '纯离线：本地词典/短语未命中则跳过，不联网',
    'single_translate':    '主翻译 API，失败自动尝试其他翻译 API',
    'multi_translate_rr':  '多个翻译 API 轮流使用，均衡负载',
    'hybrid':              '翻译 API 与 AI 混合（顺序可在 API 设置里配置）',
    'single_ai':           '主 AI 大语言模型翻译',
    'multi_ai_vote':       '多个 AI 并发翻译，相似度/长度投票',
}

DEFAULT_TRANSLATION_STRATEGY = 'single_translate'

# ==================== 快速响应层 ====================
DEFAULT_ENABLE_FAST_LOOKUP = True
LOCAL_LOOKUP_MAX_WORDS = 8
LOCAL_LOOKUP_MAX_CHARS = 50

# ==================== 显示模式 ====================
DISPLAY_MODES = {
    'overlay':   '覆盖',
    'bilingual': '双语',
    'window':    '小窗口',
}

DISPLAY_MODE_DESC = {
    'overlay':   '译文覆盖原文位置',
    'bilingual': '原文保留，下方显示小字译文',
    'window':    '译文从独立浮窗输出，可拖动、可选中复制',
}

DEFAULT_DISPLAY_MODE = 'overlay'

BILINGUAL_FONT_SCALE = 0.75
BILINGUAL_PADDING = 4

# ==================== 小窗口（字幕）模式 ====================
CAPTION_DEFAULT_W = 460
CAPTION_DEFAULT_H = 360
CAPTION_OPACITY = 0.94
CAPTION_BG = '#1e1e1e'
CAPTION_FG = '#eeeeee'
CAPTION_BAR_BG = '#2b2b2b'
CAPTION_BAR_FG = '#eeeeee'
CAPTION_SELECT_BG = '#444488'
CAPTION_FONT_SIZE = 12

# ==================== 译文颜色（全模式通用） ====================
FG_COLOR_PRESETS = {
    'auto':    '自动（按背景反色）',
    '#000000': '黑色',
    '#FFFFFF': '白色',
    '#FFEB3B': '黄色',
    '#00E676': '绿色',
    '#40C4FF': '蓝色',
    '#FF5252': '红色',
    '#FF9800': '橙色',
}

DEFAULT_FG_COLOR = 'auto'

# OCR 模型大小（PP-OCRv6）
# - 'server'：PP-OCRv6 small/medium，精度高，慢
# - 'tiny'：PP-OCRv6 tiny，速度极快，参数约 1.1M
OCR_MODEL_SIZE_SERVER = 'server'
OCR_MODEL_SIZE_TINY = 'tiny'
OCR_MODEL_SIZE_DEFAULT = OCR_MODEL_SIZE_SERVER

# 高精度 OCR（多引擎投票）
DEFAULT_ENABLE_HIGH_ACCURACY_OCR = False
# 融合时位置对齐的 IoU 阈值
HIGH_ACCURACY_IOU_THRESHOLD = 0.5

# OCR 加速优化（关方向分类 + det 限长 + cv2 缩放，一个开关统一控制）
DEFAULT_ENABLE_OCR_SPEED_OPT = True

# 英文模式（只识别包含英文字母的文本）
DEFAULT_ENABLE_ENGLISH_ONLY = False

# 标题/Logo 过滤（全大写 + 超大字号）
DEFAULT_ENABLE_TITLE_FILTER = False

# 二次 OCR 验证（视频场景防一闪而过）
DEFAULT_ENABLE_OCR_VERIFY = False

# ==================== 多语言 ====================

SOURCE_LANGS = {
    'auto': '自适应',
    'en':   '英语',
    'zh':   '中文',
    'ja':   '日语',
    'ko':   '韩语',
    'fr':   '法语',
    'de':   '德语',
    'es':   '西班牙语',
    'ru':   '俄语',
}

TARGET_LANGS = {
    'zh':   '中文',
    'en':   '英语',
    'ja':   '日语',
    'ko':   '韩语',
    'fr':   '法语',
    'de':   '德语',
    'es':   '西班牙语',
    'ru':   '俄语',
}

DEFAULT_SOURCE_LANG = 'auto'
DEFAULT_TARGET_LANG = 'zh'

# 百度翻译语言代码
BAIDU_LANG_MAP = {
    'auto': 'auto',
    'en':   'en',
    'zh':   'zh',
    'ja':   'jp',
    'ko':   'kor',
    'fr':   'fra',
    'de':   'de',
    'es':   'spa',
    'ru':   'ru',
}

# AI 翻译提示词的语言名
TARGET_LANG_NAME = {
    'zh': '简体中文',
    'en': 'English',
    'ja': '日本語',
    'ko': '한국어',
    'fr': 'Français',
    'de': 'Deutsch',
    'es': 'Español',
    'ru': 'Русский',
}

# 源语言 → OCR lang_type 映射（rapidocr 枚举名）
SOURCE_LANG_TO_OCR = {
    'auto': 'CH',
    'en':   'EN',
    'zh':   'CH',
    'ja':   'JAPAN',
    'ko':   'KOREAN',
    'fr':   'FRENCH',
    'de':   'GERMAN',
    'es':   'SPANISH',
    'ru':   'RUSSIAN',
}

# ==================== 请求节流 ====================

DEFAULT_ENABLE_THROTTLE = False
DEFAULT_THROTTLE_QPS = 1.0       # 每个 API 每秒请求数
THROTTLE_QPS_MIN = 0.2
THROTTLE_QPS_MAX = 20.0