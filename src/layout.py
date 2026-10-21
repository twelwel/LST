"""
自动排版模块
- OCR 异常框过滤
- 通用过滤（只过滤数字、单位、图像）
- 英语特化过滤（只保留字母主导的段落）
- 标题/Logo 过滤（全大写 + 超大字号）
- 目标语言过滤（按 target_lang 动态判断）
- 段落合并（三向对齐 + 宽度相似）
- 日语优化（归一化 hash + 行长度判断）
- 自适应字号（像素级估算，中英文区分）
"""

import hashlib
import re


# ==================== 段落合并参数 ====================
LINE_GAP_RATIO = 0.6
X_TOLERANCE = 80

MIN_MERGE_LINE_LENGTH = 15
MIN_MERGE_LINE_WIDTH = 150
LENGTH_SIMILAR_RATIO = 0.35

WIDTH_SIMILAR_RATIO = 0.3
ALIGN_TOLERANCE_RATIO = 2.0

DEBUG_MERGE = False


# ==================== OCR 异常框过滤参数 ====================
MAX_WIDTH_RATIO = 0.98
MAX_HEIGHT_RATIO = 0.50
MAX_AREA_RATIO = 0.30
ABS_MAX_WIDTH = 3000
ABS_MAX_HEIGHT = 400
MIN_CONFIDENCE_LENGTH = 2

# ==================== 通用过滤参数 ====================
DIGIT_RATIO_THRESHOLD = 0.6

UNIT_PATTERN = re.compile(
    r'^\s*[\d\.,]+\s*'
    r'(?:[a-zA-Z°%‰㎡m²m³sSminh℃℉pxPXpt°]+|'
    r'[米厘毫]|[个只条张页块]|[元角分]|[年月日时分秒]|'
    r'[kKMGT]?[mMgGlL]?[bB]?)\s*$'
)

DECORATIVE_CHARS = set(
    '★☆●○◎◆◇■□▲△▼▽◀▶◁▷♠♥♦♣♤♡♢♧'
    '→←↑↓↔↕⇒⇐⇑⇓'
    '※§¶†‡•·※'
    '☎☏✉✂✎✏✐✓✔✗✘'
    '①②③④⑤⑥⑦⑧⑨⑩'
    'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ'
)

# ==================== 英语特化参数 ====================
ENGLISH_LETTER_RATIO = 0.5
ENGLISH_MAX_CHINESE_RATIO = 0.2

# ==================== 标题/Logo 过滤参数 ====================
TITLE_HEIGHT_RATIO = 1.8

# ==================== 字号参数 ====================
MIN_FONT_SIZE = 9
MAX_FONT_SIZE = 28
LINE_HEIGHT_RATIO = 1.25

WIDTH_CJK = 1.0
WIDTH_ASCII_WIDE = 0.65
WIDTH_ASCII = 0.55
WIDTH_ASCII_NARROW = 0.35

# ==================== 位置量化 ====================
POSITION_KEY_QUANTIZE = 40
POSITION_KEY_QUANTIZE_Y = 20
POSITION_KEY_QUANTIZE_H = 12


# ==================== 哈希工具 ====================

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _has_kana(text):
    """检查文本是否含日文假名"""
    if not text:
        return False
    return any('\u3040' <= ch <= '\u30ff' for ch in text)


def _normalize_jp(text):
    """
    日语归一化：
    - 去掉首尾标点
    - 去掉末尾可能的 OCR 抖动（末尾重复字符）
    """
    if not text:
        return ''
    t = text.strip()
    t = t.strip('。、！？「」『』（）()[]【】.,!? \t')
    if len(t) >= 4:
        last = t[-1]
        if t[-2] == last and t[-3] == last:
            t = t.rstrip(last)
    return t


def _normalize_for_hash(text):
    """归一化用于缓存哈希的文本（所有语言）。

    背景变动时 OCR 常多出/少掉空格或首尾标点。旧实现只对日语归一化，
    中英文直接用原文哈希 → 同一句话算出不同 key → 翻译缓存全部失效
    → 每帧重新翻译 + 重绘，表现就是「译文抖动」。

    这里只做不改变语义的归一化：
    - 删除所有空白（空格抖动最常见）
    - 去掉首尾标点
    - 保留大小写与内部标点（避免把 "1.5" 与 "15" 混为一句）
    """
    if not text:
        return ''
    t = ''.join(text.split())
    return t.strip('.,;:!?"\'`´“”‘’、。，！？；：…·~～')


def hash_paragraph(paragraph):
    """段落内容 hash（不含位置，用于翻译缓存）

    ⭐ 日语优化：日语文本用归一化 hash，减少 OCR 抖动导致的缓存失效
    """
    line_texts = []
    for line in paragraph['lines']:
        raw = line['text'].strip()
        if _has_kana(raw):
            line_texts.append(_normalize_jp(raw))
        else:
            line_texts.append(_normalize_for_hash(raw))

    line_hashes = sorted(hash_text(t) for t in line_texts)
    return hashlib.md5('|'.join(line_hashes).encode('utf-8')).hexdigest()


def get_line_hashes(paragraph):
    """返回段落每行的 hash 集合（用于缓存复用）"""
    result = set()
    for line in paragraph['lines']:
        raw = line['text'].strip()
        if _has_kana(raw):
            result.add(hash_text(_normalize_jp(raw)))
        else:
            result.add(hash_text(_normalize_for_hash(raw)))
    return frozenset(result)


def paragraph_display_key(paragraph):
    """
    段落显示 key：
    - 用「中心位置 + 尺寸」粗量化做 key
    - 不依赖 OCR 文本，避免逐字抖动
    """
    box = paragraph.get('box', (0, 0, 0, 0))
    try:
        x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                          int(box[2]), int(box[3]))
    except Exception:
        x1 = y1 = x2 = y2 = 0

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h = max(1, y2 - y1)

    qx = POSITION_KEY_QUANTIZE
    qy = POSITION_KEY_QUANTIZE_Y
    qh = POSITION_KEY_QUANTIZE_H

    key = f"{cx // qx}|{cy // qy}|{h // qh}"
    return hash_text(key)


# ==================== OCR 异常框过滤 ====================

def filter_invalid_items(items, region_w=None, region_h=None):
    """过滤 OCR 误识别的异常文本框"""
    if not items:
        return []

    if region_w and region_h:
        max_w = min(ABS_MAX_WIDTH, int(region_w * MAX_WIDTH_RATIO))
        max_h = min(ABS_MAX_HEIGHT, int(region_h * MAX_HEIGHT_RATIO))
        max_area = int(region_w * region_h * MAX_AREA_RATIO)
    else:
        max_w = ABS_MAX_WIDTH
        max_h = ABS_MAX_HEIGHT
        max_area = 999999

    valid = []
    for it in items:
        text = it['text'].strip()
        x1, y1, x2, y2 = it['box']
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if w > max_w or h > max_h:
            continue
        if area > max_area:
            continue
        if len(text) < MIN_CONFIDENCE_LENGTH:
            continue

        if not any(ch.isalnum() or '\u4e00' <= ch <= '\u9fff'
                   or '\u3040' <= ch <= '\u30ff'
                   or '\uac00' <= ch <= '\ud7af'
                   for ch in text):
            continue

        if h > 0 and (w / h) > 50:
            continue
        if w > 0 and (h / w) > 15:
            continue

        valid.append(it)
    return valid


# ==================== 通用过滤（只过滤数字/单位/图像） ====================

def _digit_ratio(text):
    if not text:
        return 0
    digits = sum(1 for ch in text if ch.isdigit() or ch in '.,')
    return digits / len(text)


def _is_decorative(text):
    """纯装饰字符（视为图像）"""
    if not text:
        return True
    deco = sum(1 for ch in text if ch in DECORATIVE_CHARS)
    return deco / len(text) > 0.5


def _is_number_or_unit(text):
    """纯数字、纯单位、数字+单位"""
    t = text.strip()
    if not t:
        return True

    if re.fullmatch(r'[\d\.,]+', t):
        return True

    if UNIT_PATTERN.match(t):
        return True

    if _digit_ratio(t) >= DIGIT_RATIO_THRESHOLD:
        if not any('\u4e00' <= ch <= '\u9fff' for ch in t):
            return True

    return False


def _is_symbol_only(text):
    """纯符号（无字母、无任何语言字符、无数字）"""
    if not text:
        return True
    for ch in text:
        if ch.isalnum():
            return False
        if '\u4e00' <= ch <= '\u9fff':
            return False
        if '\u3040' <= ch <= '\u30ff':
            return False
        if '\uac00' <= ch <= '\ud7af':
            return False
        if '\u0400' <= ch <= '\u04ff':
            return False
    return True


def is_general_content_filtered(text, debug=False):
    """通用过滤：只过滤数字/单位/装饰字符/纯符号"""
    if not text or not text.strip():
        return True

    t = text.strip()

    if _is_number_or_unit(t):
        if debug:
            print(f"[通用过滤] 数字/单位: '{t}'")
        return True

    if _is_decorative(t):
        if debug:
            print(f"[通用过滤] 装饰字符: '{t}'")
        return True

    if _is_symbol_only(t):
        if debug:
            print(f"[通用过滤] 纯符号: '{t}'")
        return True

    return False


def filter_general_content(items, debug=False):
    if not items:
        return []
    valid = []
    for it in items:
        text = it['text'].strip()
        if not text:
            continue
        if is_general_content_filtered(text, debug=debug):
            continue
        valid.append(it)
    return valid


# ==================== 英语特化过滤 ====================

def is_english_filtered(text, debug=False):
    """
    英语特化过滤：
    - 必须含英文字母
    - 英文字母占非空白字符 >= 50%
    - 中文占非空白字符 <= 20%
    """
    if not text or not text.strip():
        return True

    t = text.strip()

    letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    if letters == 0:
        if debug:
            print(f"[英语特化] 无英文字母: '{t}'")
        return True

    non_space = sum(1 for ch in t if not ch.isspace())
    if non_space == 0:
        return True

    letter_ratio = letters / non_space
    if letter_ratio < ENGLISH_LETTER_RATIO:
        if debug:
            print(f"[英语特化] 字母占比 {letter_ratio:.2f} < "
                  f"{ENGLISH_LETTER_RATIO}: '{t}'")
        return True

    cn = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff')
    cn_ratio = cn / non_space
    if cn_ratio > ENGLISH_MAX_CHINESE_RATIO:
        if debug:
            print(f"[英语特化] 中文占比 {cn_ratio:.2f} > "
                  f"{ENGLISH_MAX_CHINESE_RATIO}: '{t}'")
        return True

    return False


def filter_english_only(items, debug=False):
    """英语特化过滤（供 pipeline 调用）"""
    if not items:
        return []
    valid = []
    for it in items:
        text = it['text'].strip()
        if not text:
            continue
        if is_english_filtered(text, debug=debug):
            continue
        valid.append(it)
    return valid


# ==================== 标题 / Logo 过滤 ====================

def is_title_logo_filtered(text, box, med_height=None, debug=False):
    """
    过滤标题 / Logo / 品牌名：
    1. 全大写字母（≥ 2 个字母且全大写，无小写）
    2. 字号过大（框高 > 中位高度 × TITLE_HEIGHT_RATIO）
    """
    t = text.strip()
    if not t:
        return True

    letters = [ch for ch in t if ch.isascii() and ch.isalpha()]
    if len(letters) >= 2:
        has_lower = any(ch.islower() for ch in letters)
        if not has_lower:
            if debug:
                print(f"[标题过滤] 全大写: '{t}'")
            return True

    if med_height is not None and med_height > 0:
        h = box[3] - box[1]
        if h > med_height * TITLE_HEIGHT_RATIO:
            if debug:
                print(f"[标题过滤] 字号过大 "
                      f"h={h} > {med_height * TITLE_HEIGHT_RATIO:.0f}: "
                      f"'{t}'")
            return True

    return False


def filter_title_logo(items, debug=False):
    """过滤标题/Logo（需先算框中位高度）"""
    if not items:
        return []

    heights = sorted((it['box'][3] - it['box'][1]) for it in items)
    med_height = heights[len(heights) // 2] if heights else 0

    valid = []
    for it in items:
        text = it['text'].strip()
        if not text:
            continue
        if is_title_logo_filtered(text, it['box'], med_height,
                                   debug=debug):
            continue
        valid.append(it)
    return valid


# ==================== 目标语言过滤 ====================

def is_target_language_text(text, target_lang='zh', threshold=0.5):
    """
    判断文本是否已经是目标语言（是则跳过翻译）

    ⭐ 日语优化：含假名的文本即使汉字多，也不算中文
    """
    if not text:
        return False
    t = text.strip()
    total = len(t)
    if total == 0:
        return False

    has_kana = any('\u3040' <= ch <= '\u30ff' for ch in t)
    has_hangul = any('\uac00' <= ch <= '\ud7af' for ch in t)

    if target_lang == 'zh':
        # 含假名/韩文 → 不算中文
        if has_kana or has_hangul:
            return False
        cn = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff')
        return cn / total >= threshold

    if target_lang == 'ja':
        jp = sum(1 for ch in t if '\u3040' <= ch <= '\u30ff')
        cn = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff')
        if jp == 0:
            return False
        return (jp + cn) / total >= threshold

    if target_lang == 'ko':
        kr = sum(1 for ch in t if '\uac00' <= ch <= '\ud7af')
        return kr / total >= threshold

    if target_lang == 'ru':
        ru = sum(1 for ch in t if '\u0400' <= ch <= '\u04ff')
        return ru / total >= threshold

    if target_lang == 'en':
        letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())
        return letters / total >= threshold

    # fr/de/es 都用拉丁字母
    letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())
    return letters / total >= threshold


# 兼容旧接口
def is_chinese_text(text, threshold=0.5):
    return is_target_language_text(text, 'zh', threshold)


# ==================== 文本相似度 ====================

def text_similarity(a, b):
    """
    归一化 + 双向包含 + SequenceMatcher
    """
    if not a or not b:
        return 0.0

    from difflib import SequenceMatcher as _SM

    a_n = re.sub(r'\s+', '', a).lower()
    b_n = re.sub(r'\s+', '', b).lower()

    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    if a_n in b_n or b_n in a_n:
        return 1.0

    return _SM(None, a_n, b_n).ratio()


# ==================== 合并强度预设 ====================
# 游戏 UI 里「短标签 + 短数值」很常见，默认阈值会把它们拆成碎片；
# 而左右分栏 / 独立菜单项又容易被错误拼成一句。这里给四档交给用户按场景调。
# 注意：'mid' 的数值与历史行为完全一致，改主流程默认行为时请新增档位。
MERGE_PRESETS = {
    'off': {
        'enabled': False,
    },
    'weak': {
        'enabled': True,
        'require_long': 'both',          # 必须两条都算长行
        'min_len': 18, 'min_width': 180,
        'jp_min_len': 4, 'jp_min_width': 72,
        'align_ratio': 1.5, 'width_sim': 0.40,
        'gap_ratio': 0.45, 'length_sim': 0.50,
        'h_gap_ratio': None,
    },
    'mid': {
        'enabled': True,
        'require_long': 'one',           # 至少一条长行（历史行为）
        'min_len': MIN_MERGE_LINE_LENGTH, 'min_width': MIN_MERGE_LINE_WIDTH,
        'jp_min_len': 3, 'jp_min_width': 60,
        'align_ratio': ALIGN_TOLERANCE_RATIO,
        'width_sim': WIDTH_SIMILAR_RATIO,
        'gap_ratio': LINE_GAP_RATIO,
        'length_sim': LENGTH_SIMILAR_RATIO,
        'h_gap_ratio': None,
    },
    'strong': {
        'enabled': True,
        'require_long': 'none',          # 不再要求长行：同排短词也合并
        'min_len': 8, 'min_width': 60,
        'jp_min_len': 2, 'jp_min_width': 24,
        'align_ratio': 2.5, 'width_sim': 0.20,
        'gap_ratio': 0.75, 'length_sim': 0.25,
        # 同行水平间距上限（× 字高）：防止把同一行里相隔很远的元素粘成一句
        'h_gap_ratio': 1.5,
    },
}

DEFAULT_MERGE_LEVEL = 'mid'


def get_merge_preset(level=None):
    """按名称取合并预设；未知名称回退到 mid。"""
    key = str(level or DEFAULT_MERGE_LEVEL).strip().lower()
    return MERGE_PRESETS.get(key, MERGE_PRESETS['mid'])


# ==================== 行长度判断 ====================

def _is_long_line(item, preset=None):
    preset = preset or MERGE_PRESETS['mid']
    text = item['text'].strip()
    x1, y1, x2, y2 = item['box']
    width = x2 - x1

    # ⭐ 日语行：字符数阈值降低
    if _has_kana(text):
        return (len(text) >= preset['jp_min_len']
                and width >= preset['jp_min_width'])

    return (len(text) >= preset['min_len']
            and width >= preset['min_width'])


def _length_similar(text_a, text_b, ratio=None):
    a = len(text_a.strip())
    b = len(text_b.strip())
    if a == 0 or b == 0:
        return False
    return min(a, b) / max(a, b) >= (ratio or LENGTH_SIMILAR_RATIO)


# ==================== 对齐检测 ====================

def _is_aligned(prev, cur, avg_h, ratio=None):
    px1, py1, px2, py2 = prev['box']
    x1, y1, x2, y2 = cur['box']

    left_d   = abs(x1 - px1)
    right_d  = abs(x2 - px2)
    center_d = abs((x1 + x2) / 2 - (px1 + px2) / 2)
    min_align = min(left_d, right_d, center_d)

    return min_align <= avg_h * (ratio or ALIGN_TOLERANCE_RATIO)


def _width_similar(prev, cur, ratio=None):
    pw = prev['box'][2] - prev['box'][0]
    cw = cur['box'][2] - cur['box'][0]
    if pw <= 0 or cw <= 0:
        return False
    return min(pw, cw) / max(pw, cw) >= (ratio or WIDTH_SIMILAR_RATIO)


# ==================== 段落合并 ====================

def merge_lines_to_paragraphs(items, level=None):
    """把 OCR 行合并成段落。

    level: 'off' / 'weak' / 'mid' / 'strong'（None → mid，与历史行为一致）
    """
    if not items:
        return []

    preset = get_merge_preset(level)
    if not preset.get('enabled', True):
        # 关闭合并：每个 OCR 行独立成段
        return [_build([it]) for it in items]

    align_ratio = preset['align_ratio']
    width_sim = preset['width_sim']
    gap_ratio = preset['gap_ratio']
    length_sim = preset['length_sim']
    h_gap_ratio = preset.get('h_gap_ratio')
    require_long = preset.get('require_long', 'one')

    sorted_items = sorted(items, key=lambda it: (it['box'][1], it['box'][0]))
    paragraphs = []
    current = [sorted_items[0]]

    for it in sorted_items[1:]:
        prev = current[-1]
        px1, py1, px2, py2 = prev['box']
        x1, y1, x2, y2 = it['box']

        prev_h = max(1, py2 - py1)
        cur_h = max(1, y2 - y1)
        avg_h = (prev_h + cur_h) / 2

        gap = y1 - py2
        if gap < -avg_h * 0.5:
            same_line = True
        else:
            same_line = gap < avg_h * gap_ratio

        aligned = _is_aligned(prev, it, avg_h, align_ratio)
        width_ok = _width_similar(prev, it, width_sim)

        prev_long = _is_long_line(prev, preset)
        cur_long = _is_long_line(it, preset)
        if require_long == 'none':
            long_ok = True
        elif require_long == 'both':
            long_ok = prev_long and cur_long
        else:
            long_ok = prev_long or cur_long

        if prev_long and cur_long:
            length_ok = _length_similar(prev['text'], it['text'], length_sim)
        else:
            length_ok = True

        # 同行水平间距上限（仅 strong 启用）
        h_ok = True
        if h_gap_ratio is not None:
            h_gap = x1 - px2
            if h_gap > avg_h * h_gap_ratio:
                h_ok = False

        can_merge = (
            same_line and aligned and width_ok
            and long_ok and length_ok and h_ok
        )

        if can_merge:
            current.append(it)
        else:
            paragraphs.append(_build(current))
            current = [it]

    if current:
        paragraphs.append(_build(current))

    if DEBUG_MERGE:
        for i, p in enumerate(paragraphs):
            print(f"[合并] P{i} box={p['box']} "
                  f"lines={len(p['lines'])} text={p['text'][:40]!r}")

    return paragraphs


def _build(lines):
    parts = []
    for i, it in enumerate(lines):
        t = it['text'].strip()
        if not t:
            continue
        # 日语不加空格
        if parts and t[0] not in ',.;:!?，。；：！？)】」':
            if not _has_kana(t) and not _has_kana(parts[-1]):
                parts.append(' ')
        parts.append(t)
    text = ''.join(parts)

    x1 = min(it['box'][0] for it in lines)
    y1 = min(it['box'][1] for it in lines)
    x2 = max(it['box'][2] for it in lines)
    y2 = max(it['box'][3] for it in lines)

    return {'text': text, 'box': (x1, y1, x2, y2), 'lines': lines}


# ==================== 字号（像素级估算） ====================

_NARROW_CHARS = set("iIl1|!.,;:'\"`^ \t\r\n")
_WIDE_ASCII = set("MW@%&#")


def _char_width(ch, size):
    code = ord(ch)
    # CJK 汉字 / 假名 / 全角
    if (0x4E00 <= code <= 0x9FFF or
            0x3000 <= code <= 0x303F or
            0xFF00 <= code <= 0xFFEF or
            0x3040 <= code <= 0x30FF or
            0xAC00 <= code <= 0xD7AF):
        return size
    if ch in _NARROW_CHARS:
        return max(1, int(size * WIDTH_ASCII_NARROW))
    if ch in _WIDE_ASCII:
        return max(1, int(size * WIDTH_ASCII_WIDE))
    return max(1, int(size * WIDTH_ASCII))


def estimate_text_width(text, size):
    if not text:
        return 0
    return sum(_char_width(ch, size) for ch in text)


def count_lines(text, size, max_width):
    if not text or max_width <= 0:
        return 1
    lines = 1
    current = 0
    for ch in text:
        cw = _char_width(ch, size)
        if current + cw > max_width and current > 0:
            lines += 1
            current = cw
        else:
            current += cw
    return lines


def calc_font_size(text, w, h, base_size):
    if not text or w <= 0 or h <= 0:
        return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size))

    size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size))
    while size > MIN_FONT_SIZE:
        single_w = estimate_text_width(text, size)
        if single_w <= w:
            line_count = 1
        else:
            line_count = count_lines(text, size, w)
        total_h = line_count * size * LINE_HEIGHT_RATIO
        if total_h <= h:
            return size
        size -= 1
    return MIN_FONT_SIZE

# ==================== 字幕判定 ====================

SUBTITLE_MAX_HEIGHT = 120
SUBTITLE_MIN_ASPECT = 2.5
SUBTITLE_MIN_CHARS = 3
SUBTITLE_MAX_CHARS = 200


def is_subtitle_text(text, box):
    """
    判断段落是否像视频字幕：
    - 文本长度 3-200 字符
    - 框高 < 120px（不超过 3 行）
    - 宽高比 >= 2.5（宽扁形状）
    """
    if not text or not text.strip():
        return False

    t = text.strip()
    if not (SUBTITLE_MIN_CHARS <= len(t) <= SUBTITLE_MAX_CHARS):
        return False

    try:
        x1, y1, x2, y2 = box
    except Exception:
        return False

    w = x2 - x1
    h = max(1, y2 - y1)

    if h > SUBTITLE_MAX_HEIGHT:
        return False

    aspect = w / h
    if aspect < SUBTITLE_MIN_ASPECT:
        return False

    return True