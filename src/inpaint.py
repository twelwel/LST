"""图像修复工具
- inpaint_region: 整框修复（基础模式）
- inpaint_text_only: 笔画级修复（精细模式，加强版）
- sample_text_color: 文字颜色采样（加强版）
"""

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ==================== 整框修复参数 ====================
INPAINT_PADDING = 15
INPAINT_RADIUS = 3

# ==================== 笔画级修复参数（加强） ====================
TEXT_PADDING = 8
TEXT_THRESHOLD_BLOCK = 15
TEXT_THRESHOLD_C = 10
TEXT_DILATE_SIZE = 3
TEXT_DILATE_ITER = 2      # 加强：多膨胀一次，覆盖抗锯齿边缘
TEXT_INPAINT_RADIUS = 4   # 加强：稍大半径修复更彻底

# ==================== 颜色采样参数 ====================
COLOR_DARK_PERCENTILE = 25
COLOR_BG_PERCENTILE = 75


# ==================== 整框修复 ====================

def inpaint_region(image_np, x1, y1, x2, y2, padding=INPAINT_PADDING):
    """整框修复：把整个框视为待修复区域"""
    if not CV2_AVAILABLE:
        return None

    H, W = image_np.shape[:2]
    ex1 = max(0, x1 - padding)
    ey1 = max(0, y1 - padding)
    ex2 = min(W, x2 + padding)
    ey2 = min(H, y2 + padding)

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = image_np[ey1:ey2, ex1:ex2].copy()

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    mx1, my1 = x1 - ex1, y1 - ey1
    mx2, my2 = x2 - ex1, y2 - ey1
    mask[my1:my2, mx1:mx2] = 255

    try:
        inpainted = cv2.inpaint(roi, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    except Exception:
        return None

    result = inpainted[my1:my2, mx1:mx2]
    return Image.fromarray(result)


# ==================== 笔画级修复（加强版） ====================

def inpaint_text_only(image_np, x1, y1, x2, y2, padding=TEXT_PADDING):
    """
    笔画级修复：只擦除文字笔画，保留背景
    返回：PIL.Image（尺寸 = 原框大小）或 None
    """
    if not CV2_AVAILABLE:
        return None

    H, W = image_np.shape[:2]

    ex1 = max(0, x1 - padding)
    ey1 = max(0, y1 - padding)
    ex2 = min(W, x2 + padding)
    ey2 = min(H, y2 + padding)

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = image_np[ey1:ey2, ex1:ex2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 自适应阈值：文字比周围暗
    try:
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            TEXT_THRESHOLD_BLOCK,
            TEXT_THRESHOLD_C,
        )
    except Exception:
        return None

    # 加强：膨胀 2 次，覆盖抗锯齿边缘
    kernel = np.ones((TEXT_DILATE_SIZE, TEXT_DILATE_SIZE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=TEXT_DILATE_ITER)

    # 只在原框区域内修复
    inner = np.zeros_like(mask)
    mx1, my1 = x1 - ex1, y1 - ey1
    mx2, my2 = x2 - ex1, y2 - ey1
    inner[my1:my2, mx1:mx2] = 255
    mask = cv2.bitwise_and(mask, inner)

    if mask.sum() == 0:
        return None

    try:
        inpainted = cv2.inpaint(
            roi, mask, TEXT_INPAINT_RADIUS, cv2.INPAINT_TELEA
        )
    except Exception:
        return None

    result = inpainted[my1:my2, mx1:mx2]
    return Image.fromarray(result)


# ==================== 颜色采样（加强版） ====================

def sample_text_color(image_np, x1, y1, x2, y2):
    """
    从文字笔画采样前景色
    返回：#RRGGBB 字符串
    """
    if not CV2_AVAILABLE:
        return '#000000'

    H, W = image_np.shape[:2]
    cx1 = max(0, min(W - 1, x1))
    cy1 = max(0, min(H - 1, y1))
    cx2 = max(cx1 + 1, min(W, x2))
    cy2 = max(cy1 + 1, min(H, y2))

    roi = image_np[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return '#000000'

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 文字：暗色像素；背景：亮色像素
    text_thresh = np.percentile(gray, COLOR_DARK_PERCENTILE)
    bg_thresh = np.percentile(gray, COLOR_BG_PERCENTILE)

    text_mask = gray <= text_thresh
    bg_mask = gray >= bg_thresh

    if text_mask.sum() < 3:
        return '#000000'

    # 文字平均色
    text_pixels = roi[text_mask]
    b_t, g_t, r_t = text_pixels.mean(axis=0).astype(int)
    text_lum = r_t * 0.299 + g_t * 0.587 + b_t * 0.114

    # 背景平均色
    if bg_mask.sum() > 0:
        bg_pixels = roi[bg_mask]
        b_b, g_b, r_b = bg_pixels.mean(axis=0).astype(int)
        bg_lum = r_b * 0.299 + g_b * 0.587 + b_b * 0.114

        contrast = abs(text_lum - bg_lum)

        # 深色文字 + 亮背景 → 用纯黑
        if text_lum < 100 and bg_lum > 150 and contrast > 60:
            return '#000000'
        # 亮色文字 + 深背景 → 用纯白
        if text_lum > 155 and bg_lum < 100 and contrast > 60:
            return '#FFFFFF'
        # 对比度太低 → 用纯黑/纯白强制高对比
        if contrast < 40:
            return '#FFFFFF' if bg_lum < 128 else '#000000'

    return f'#{r_t:02x}{g_t:02x}{b_t:02x}'


# ==================== 场景适配判断（新增） ====================

def is_fine_text_suitable(image_np, x1, y1, x2, y2):
    """
    判断当前段落是否适合用精细模式
    条件（全部满足才返回 True）：
    - 框高度 >= 24 像素（小字看不清笔画）
    - 背景亮度 >= 70（深色 UI 不适合）
    - 文字/背景对比度 >= 50
    - 文字笔画占框面积 3% ~ 35%（太少无字，太多是色块）
    """
    if not CV2_AVAILABLE:
        return False

    h = y2 - y1
    w = x2 - x1

    # 条件 1：字号足够大
    if h < 24 or w < 30:
        return False

    H, W = image_np.shape[:2]
    cx1 = max(0, min(W - 1, x1))
    cy1 = max(0, min(H - 1, y1))
    cx2 = max(cx1 + 1, min(W, x2))
    cy2 = max(cy1 + 1, min(H, y2))

    roi = image_np[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return False

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False

    # 条件 2：背景亮度
    bg_lum = float(np.percentile(gray, 75))
    if bg_lum < 70:
        return False

    # 条件 3：对比度
    text_lum = float(np.percentile(gray, 15))
    contrast = abs(bg_lum - text_lum)
    if contrast < 50:
        return False

    # 条件 4：文字占比
    try:
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            TEXT_THRESHOLD_BLOCK,
            TEXT_THRESHOLD_C,
        )
    except Exception:
        return False

    text_ratio = mask.sum() / (mask.size * 255)
    if text_ratio < 0.03 or text_ratio > 0.35:
        return False

    return True