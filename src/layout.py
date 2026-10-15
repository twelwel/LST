"""
自动排版模块
- OCR 异常框过滤
- 通用过滤（只过滤数字、单位、图像）
- 英文模式过滤（只保留包含英文字母的文本）
- 段落合并（三向对齐 + 宽度相似）
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

# ==================== 英文模式参数 ====================
ENGLISH_LETTER_RATIO = 0.3

# ==================== 字号参数 ====================
MIN_FONT_SIZE = 9
MAX_FONT_SIZE = 28
LINE_HEIGHT_RATIO = 1.25

WIDTH_CJK = 1.0
WIDTH_ASCII_WIDE = 0.65
WIDTH_ASCII = 0.55
WIDTH_ASCII_NARROW = 0.35

# ==================== 位置量化（静止模式） ====================
POSITION_KEY_QUANTIZE = 40
POSITION_KEY_QUANTIZE_Y = 20
POSITION_KEY_QUANTIZE_H = 12


# ==================== 哈希工具 ====================

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def hash_paragraph(paragraph):
    """段落内容 hash（不含位置，用于翻译缓存）"""
    line_hashes = sorted(
        hash_text(line['text'].strip()) for line in paragraph['lines']
    )
    return hashlib.md5('|'.join(line_hashes).encode('utf-8')).hexdigest()


def get_line_hashes(paragraph):
    return frozenset(
        hash_text(line['text'].strip()) for line in paragraph['lines']
    )


def paragraph_display_key(paragraph):
    """段落显示 key（位置量化，稳定）"""
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

        if not any(ch.isalnum() or '\u4e00' <= ch <= '\u9fff' for ch in text):
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
    if not text:
        return True
    deco = sum(1 for ch in text if ch in DECORATIVE_CHARS)
    return deco / len(text) > 0.5


def _is_number_or_unit(text):
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
    if not text:
        return True
    for ch in text:
        if ch.isalnum():
            return False
        if '\u4e00' <= ch <= '\u9fff':
            return False
    return True


def is_general_content_filtered(text, debug=False):
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

# 英文字母占非空白字符的最低比例
ENGLISH_LETTER_RATIO = 0.5
# 中文占非空白字符的最高比例
ENGLISH_MAX_CHINESE_RATIO = 0.2


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


# ==================== 行长度判断 ====================

def _is_long_line(item):
    text = item['text'].strip()
    x1, y1, x2, y2 = item['box']
    width = x2 - x1
    return (len(text) >= MIN_MERGE_LINE_LENGTH
            and width >= MIN_MERGE_LINE_WIDTH)


def _length_similar(text_a, text_b):
    a = len(text_a.strip())
    b = len(text_b.strip())
    if a == 0 or b == 0:
        return False
    return min(a, b) / max(a, b) >= LENGTH_SIMILAR_RATIO


# ==================== 对齐检测 ====================

def _is_aligned(prev, cur, avg_h):
    px1, py1, px2, py2 = prev['box']
    x1, y1, x2, y2 = cur['box']

    left_d   = abs(x1 - px1)
    right_d  = abs(x2 - px2)
    center_d = abs((x1 + x2) / 2 - (px1 + px2) / 2)
    min_align = min(left_d, right_d, center_d)

    return min_align <= avg_h * ALIGN_TOLERANCE_RATIO


def _width_similar(prev, cur):
    pw = prev['box'][2] - prev['box'][0]
    cw = cur['box'][2] - cur['box'][0]
    if pw <= 0 or cw <= 0:
        return False
    return min(pw, cw) / max(pw, cw) >= WIDTH_SIMILAR_RATIO


# ==================== 段落合并 ====================

def merge_lines_to_paragraphs(items):
    if not items:
        return []

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
            same_line = gap < avg_h * LINE_GAP_RATIO

        aligned = _is_aligned(prev, it, avg_h)
        width_ok = _width_similar(prev, it)

        prev_long = _is_long_line(prev)
        cur_long = _is_long_line(it)
        at_least_one_long = prev_long or cur_long

        if prev_long and cur_long:
            length_ok = _length_similar(prev['text'], it['text'])
        else:
            length_ok = True

        can_merge = (
            same_line and aligned and width_ok
            and at_least_one_long and length_ok
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
        if parts and t[0] not in ',.;:!?，。；：！？)】」':
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
    if (0x4E00 <= code <= 0x9FFF or
            0x3000 <= code <= 0x303F or
            0xFF00 <= code <= 0xFFEF or
            0x3040 <= code <= 0x30FF):
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


def is_chinese_text(text, threshold=0.5):
    if not text:
        return False
    total = len(text.strip())
    if total == 0:
        return False
    cn = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cn / total >= threshold

# ==================== 标题 / Logo 过滤 ====================

# 框高 > 本帧中位高度 × 该值 → 视为标题/Logo
TITLE_HEIGHT_RATIO = 1.8


def is_title_logo_filtered(text, box, med_height=None, debug=False):
    """
    过滤标题 / Logo / 品牌名：
    1. 全大写字母（≥ 2 个字母且全大写，无小写）
    2. 字号过大（框高 > 中位高度 × TITLE_HEIGHT_RATIO）
    """
    t = text.strip()
    if not t:
        return True

    # 规则 1：全大写
    letters = [ch for ch in t if ch.isascii() and ch.isalpha()]
    if len(letters) >= 2:
        has_lower = any(ch.islower() for ch in letters)
        if not has_lower:
            if debug:
                print(f"[标题过滤] 全大写: '{t}'")
            return True

    # 规则 2：字号过大
    if med_height is not None and med_height > 0:
        h = box[3] - box[1]
        if h > med_height * TITLE_HEIGHT_RATIO:
            if debug:
                print(f"[标题过滤] 字号过大 "
                      f"h={h} > {med_height * TITLE_HEIGHT_RATIO:.0f}: '{t}'")
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
        if is_title_logo_filtered(text, it['box'], med_height, debug=debug):
            continue
        valid.append(it)
    return valid