"""
自动排版模块
- OCR 异常框过滤
- 通用内容过滤（艺术字/招牌/数字/单位等）
- 段落合并
- 自适应字号
"""

import hashlib
import re


# ==================== 段落合并参数 ====================
LINE_GAP_RATIO = 0.6
X_TOLERANCE = 80

MIN_MERGE_LINE_LENGTH = 15
MIN_MERGE_LINE_WIDTH = 150
LENGTH_SIMILAR_RATIO = 0.35

# ==================== OCR 异常框过滤参数 ====================
MAX_WIDTH_RATIO = 0.98
MAX_HEIGHT_RATIO = 0.50
MAX_AREA_RATIO = 0.30
ABS_MAX_WIDTH = 3000
ABS_MAX_HEIGHT = 400

MIN_CONFIDENCE_LENGTH = 2

# ==================== 通用过滤参数 ====================
# 数字占比超过此值 → 过滤（如 "123.45"、"2024"）
DIGIT_RATIO_THRESHOLD = 0.6
# 短文本长度（含）→ 过滤（如 "OK"、"Hi"、"A"）
SHORT_TEXT_LENGTH = 3
# 符号占比超过此值 → 过滤（如 "★→◆"）
SYMBOL_RATIO_THRESHOLD = 0.5

# 常见单位正则（数字+单位 → 过滤）
UNIT_PATTERN = re.compile(
    r'^\s*[\d\.,]+\s*'
    r'(?:[a-zA-Z°%‰㎡m²m³sSminh℃℉pxPXpt°]+|'
    r'[米厘毫]|[个只条张页块]|[元角分]|[年月日时分秒]|'
    r'[kKMGT]?[mMgGlL]?[bB]?)\s*$'
)

# 艺术字/装饰性字符
DECORATIVE_CHARS = set(
    '★☆●○◎◆◇■□▲△▼▽◀▶◁▷♠♥♦♣♤♡♢♧'
    '→←↑↓↔↕⇒⇐⇑⇓'
    '※§¶†‡•·※'
    '☎☏✉✂✎✏✐✓✔✗✘'
    '①②③④⑤⑥⑦⑧⑨⑩'
    'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ'
)

# ==================== 字号参数 ====================
MIN_FONT_SIZE = 9
MAX_FONT_SIZE = 28
LINE_HEIGHT_RATIO = 1.25
# ======================================================


# ==================== 哈希工具 ====================

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def hash_paragraph(paragraph):
    line_hashes = sorted(
        hash_text(line['text'].strip()) for line in paragraph['lines']
    )
    return hashlib.md5('|'.join(line_hashes).encode('utf-8')).hexdigest()


def get_line_hashes(paragraph):
    return frozenset(
        hash_text(line['text'].strip()) for line in paragraph['lines']
    )


def paragraph_display_key(paragraph):
    if not paragraph.get('lines'):
        return hash_text(paragraph['text'])
    return hash_text(paragraph['lines'][0]['text'].strip())


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

        has_meaningful = any(
            ch.isalnum() or '\u4e00' <= ch <= '\u9fff'
            for ch in text
        )
        if not has_meaningful:
            continue

        if h > 0 and (w / h) > 50:
            continue
        if w > 0 and (h / w) > 15:
            continue

        valid.append(it)
    return valid


# ==================== 通用内容过滤 ====================

def _digit_ratio(text):
    """数字字符占比"""
    if not text:
        return 0
    digits = sum(1 for ch in text if ch.isdigit() or ch in '.,')
    return digits / len(text)


def _symbol_ratio(text):
    """符号字符占比（非字母/数字/汉字）"""
    if not text:
        return 0
    symbols = sum(
        1 for ch in text
        if not ch.isalnum() and not ('\u4e00' <= ch <= '\u9fff')
        and not ch.isspace()
    )
    return symbols / len(text)


def _is_decorative(text):
    """是否主要为装饰性字符（艺术字、招牌）"""
    if not text:
        return True
    deco = sum(1 for ch in text if ch in DECORATIVE_CHARS)
    return deco / len(text) > 0.5


def _is_number_or_unit(text):
    """是否纯数字 / 数字+单位"""
    t = text.strip()
    if not t:
        return True

    # 纯数字（含小数、千分位）
    if re.fullmatch(r'[\d\.,]+', t):
        return True

    # 数字 + 单位
    if UNIT_PATTERN.match(t):
        return True

    # 数字占比过高（如 "3.14"、"2024 05 12"）
    if _digit_ratio(t) >= DIGIT_RATIO_THRESHOLD:
        # 但如果含中文，可能是正文
        if not any('\u4e00' <= ch <= '\u9fff' for ch in t):
            return True

    return False


def _is_too_short(text):
    """是否过短（如 "OK"、"A"、"Hi"）"""
    t = text.strip()
    if len(t) <= SHORT_TEXT_LENGTH:
        # 但如果是中文，可能是有意义的词
        if not any('\u4e00' <= ch <= '\u9fff' for ch in t):
            return True
    return False


def _is_symbol_heavy(text):
    """是否符号过多"""
    return _symbol_ratio(text) >= SYMBOL_RATIO_THRESHOLD


def is_general_content_filtered(text, debug=False):
    """
    通用过滤：判断是否应该过滤掉
    返回 True 表示应该过滤

    过滤类型：
    - 艺术字 / 装饰符号
    - 招牌（纯数字/单位）
    - 数字 / 单位
    - 过短文本
    - 符号过多
    """
    if not text or not text.strip():
        return True

    t = text.strip()

    # 1) 装饰性字符
    if _is_decorative(t):
        if debug:
            print(f"[通用过滤] 装饰字符: '{t}'")
        return True

    # 2) 数字 / 单位
    if _is_number_or_unit(t):
        if debug:
            print(f"[通用过滤] 数字/单位: '{t}'")
        return True

    # 3) 过短
    if _is_too_short(t):
        if debug:
            print(f"[通用过滤] 过短: '{t}'")
        return True

    # 4) 符号过多
    if _is_symbol_heavy(t):
        if debug:
            print(f"[通用过滤] 符号过多: '{t}'")
        return True

    return False


def filter_general_content(items, debug=False):
    """过滤 OCR 结果中的非正文内容"""
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
        gap = y1 - py2

        same_position = (
            gap < prev_h * LINE_GAP_RATIO
            and abs(x1 - px1) < X_TOLERANCE
        )

        prev_long = _is_long_line(prev)
        cur_long = _is_long_line(it)
        at_least_one_long = prev_long or cur_long

        if prev_long and cur_long:
            length_ok = _length_similar(prev['text'], it['text'])
        else:
            length_ok = True

        can_merge = same_position and at_least_one_long and length_ok

        if can_merge:
            current.append(it)
        else:
            paragraphs.append(_build(current))
            current = [it]

    if current:
        paragraphs.append(_build(current))

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

    return {
        'text': text,
        'box': (x1, y1, x2, y2),
        'lines': lines,
    }


# ==================== 字号 ====================

def calc_font_size(text, w, h, base_size):
    size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, base_size))
    while size > MIN_FONT_SIZE:
        chars_per_line = max(1, int(w / size))
        lines_needed = (len(text) + chars_per_line - 1) // chars_per_line
        total_h = lines_needed * size * LINE_HEIGHT_RATIO
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