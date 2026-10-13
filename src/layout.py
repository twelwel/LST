"""
自动排版模块
- OCR 异常框过滤（按截图区域自适应）
- 段落合并（长文本才合并，单个词/词组不合并）
- 段落哈希（对边界变化免疫）
- 段落显示 key（用于 overlay 平移）
- 自适应字号
"""

import hashlib

# ==================== 段落合并参数 ====================
LINE_GAP_RATIO = 0.6
X_TOLERANCE = 80

MIN_MERGE_LINE_LENGTH = 15
MIN_MERGE_LINE_WIDTH = 150
LENGTH_SIMILAR_RATIO = 0.35

# ==================== OCR 异常框过滤参数 ====================
# 按截图区域的相对比例来过滤（关键修复）
MAX_WIDTH_RATIO = 0.98          # 单个框宽度不能超过截图宽度的 98%
MAX_HEIGHT_RATIO = 0.50         # 单个框高度不能超过截图高度的 50%
MAX_AREA_RATIO = 0.30           # 单个框面积不能超过截图面积的 30%

# 绝对上限（兜底，防止区域太大时失效）
ABS_MAX_WIDTH = 3000
ABS_MAX_HEIGHT = 400

MIN_CONFIDENCE_LENGTH = 2

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
    """
    过滤 OCR 误识别的异常文本框
    关键：使用相对截图区域的**比例**判断，而不是绝对像素
    """
    if not items:
        return []

    # 计算动态上限
    if region_w and region_h:
        max_w = min(ABS_MAX_WIDTH,  int(region_w * MAX_WIDTH_RATIO))
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

        # 1) 尺寸过大（按比例）
        if w > max_w or h > max_h:
            # 只在明显异常时才打印，避免刷屏
            if w > max_w * 1.1 or h > max_h * 1.1:
                print(f"[过滤] 尺寸过大: '{text[:20]}' "
                      f"w={w}(max{max_w}) h={h}(max{max_h})")
            continue

        # 2) 面积过大
        if area > max_area:
            print(f"[过滤] 面积过大: '{text[:20]}' "
                  f"area={area}(max{max_area})")
            continue

        # 3) 文本过短
        if len(text) < MIN_CONFIDENCE_LENGTH:
            continue

        # 4) 必须包含字母、数字或汉字
        has_meaningful = any(
            ch.isalnum() or '\u4e00' <= ch <= '\u9fff'
            for ch in text
        )
        if not has_meaningful:
            continue

        # 5) 宽高比极端异常
        if h > 0 and (w / h) > 50:
            print(f"[过滤] 宽高比异常: '{text[:20]}' w/h={w/h:.1f}")
            continue
        if w > 0 and (h / w) > 15:
            print(f"[过滤] 宽高比异常: '{text[:20]}' h/w={h/w:.1f}")
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