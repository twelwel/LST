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
        'change_ratio_threshold': 0.00015,
        'full_frame_ratio': 0.10,
        'bbox_padding': 80,
        'max_ocr_width': 1600,
        'max_ocr_height': 700,
        'fast_mode': True,
    },
    'medium': {
        'name': '中',
        'scan_interval': 0.30,
        'change_ratio_threshold': 0.0002,
        'full_frame_ratio': 0.12,
        'bbox_padding': 60,
        'max_ocr_width': 1400,
        'max_ocr_height': 600,
        'fast_mode': False,
    },
    'slow': {
        'name': '慢',
        'scan_interval': 0.80,
        'change_ratio_threshold': 0.0003,
        'full_frame_ratio': 0.15,
        'bbox_padding': 40,
        'max_ocr_width': 1000,
        'max_ocr_height': 500,
        'fast_mode': False,
    },
}

DEFAULT_MODE = 'medium'

# ==================== 优化功能开关 ====================
# 帧差检测与分块 OCR 合并为“智能 OCR”：
# 开启 → 只在画面变化时 OCR，并只识别变化的区块；
# 关闭 → 每帧整屏 OCR。
DEFAULT_ENABLE_SMART_OCR = True
DEFAULT_ENABLE_FRAME_DIFF = DEFAULT_ENABLE_SMART_OCR
DEFAULT_ENABLE_BLOCK_OCR = DEFAULT_ENABLE_SMART_OCR
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
    'local_api':           '离线 + API',
    'single_translate':    '单翻译 API',
    'multi_translate_rr':  '多翻译 API 轮询',
    'hybrid':              '翻译 API + AI 混合',
    'single_ai':           '单 AI 翻译',
    'multi_ai_vote':       '多 AI 高精度投票',
}

TRANSLATION_STRATEGY_DESC = {
    'local_only':          '只用本地模型 + 短语缓存，完全离线',
    'local_api':           '优先本地模型，未命中走翻译 API / AI',
    'single_translate':    '主翻译 API，失败尝试其他 API（本地请用「本地流式」）',
    'multi_translate_rr':  '多个翻译 API 轮流使用（本地请用「本地流式」）',
    'hybrid':              '翻译 API 与 AI 混合，顺序可配置（本地请用「本地流式」）',
    'single_ai':           '主 AI 大语言模型翻译（本地请用「本地流式」）',
    'multi_ai_vote':       '多个 AI 并发翻译投票（本地请用「本地流式」）',
}

# 默认使用远程翻译；本地翻译通过主界面「⚡ 本地流式」按钮显式启动。
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

# ==================== 覆盖层外观 ====================
# 浅黑面板 + 可调不透明度。
# 0.70 = 30% 透明（默认）：能隐约看到底图，白字仍清晰。
# 注意：这是整窗 alpha，文字会一起半透明；深底白字下观感正常。
OVERLAY_BG_COLOR = '#0e0e0e'
DEFAULT_OVERLAY_BG_OPACITY = 0.70
OVERLAY_BG_OPACITY_MIN = 0.10
OVERLAY_BG_OPACITY_MAX = 1.00

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

# ==================== 本地模型设备 ====================

LOCAL_MODEL_DEVICES = {
    'auto':     '自动（推荐：CPU）',
    'cuda':     'NVIDIA CUDA',
    'directml': 'DirectML（慢，勿选）',
    'cpu':      'CPU（不占显卡）',
}

DEFAULT_LOCAL_MODEL_DEVICE = 'auto'

# ==================== 本地模型批量推理 ====================
# 一次 generate 处理多条，摊薄逐条推理的固定开销。
# 实测（8 条短句）：CPU 56.1ms/条 → 18.4ms/条（3.1×）
LOCAL_BATCH_MAX = 8               # 单批最多几条
# 批内最长/最短超过此倍数才另起一批。实测放到 4.0 最优：
# padding 浪费远小于"多切一批"的代价（3 批 30.7ms/条 → 1 批 18.8ms/条）。
# 仍然保留倍率是为了防"8 字符 vs 500 字符"混批时 padding 爆炸。
LOCAL_BATCH_PADDING_RATIO = 4.0

# ==================== 段落级译文复用（抗结构抖动） ====================
# 背景变动会让段落「合并 / 拆分」（多一行、少一行），p_hash 随之改变，
# 翻译缓存必然失效 → 重新翻译 + 重绘，表现为抖动。
# 这里按「同位置 + 行集合相似度」复用旧译文，减少抖动与重复请求。
PARAGRAPH_REUSE_JACCARD = 0.66     # 行集合 Jaccard 下限（约 2/3）
PARAGRAPH_REUSE_LEN_RATIO = 0.5    # 原文长度比下限，避免短标签套用长译文
PARAGRAPH_MEMORY_MAX = 300         # 记忆表上限（条）

# ==================== 文字合并强度 ====================
# 游戏 UI 里「短标签 + 短数值」常被拆成碎片；而左右分栏/独立菜单项
# 又容易被错误拼成一句。这里给四档，用户可按场景自己调。
MERGE_LEVELS = {
    'off':    '关闭（每行独立）',
    'weak':   '弱（只并长行）',
    'mid':    '中（默认）',
    'strong': '强（相邻短词也并）',
}
DEFAULT_MERGE_LEVEL = 'mid'

# ==================== 缓存范围 ====================
CACHE_MIN = 100
CACHE_MAX = 5000
CACHE_DEFAULT = 1000