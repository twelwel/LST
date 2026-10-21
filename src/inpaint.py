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

# ==================== 算法增强参数 ====================
INPAINT_RADIUS_MIN = 3
INPAINT_RADIUS_MAX = 12
UNIFORM_STD_THRESHOLD = 12.0      # 背景判为纯色的标准差上限
TEXTURED_STD_THRESHOLD = 45.0     # 超过此值视为纹理背景 → TELEA
COLOR_DISTANCE_THRESHOLD = 60     # BGR 三通道绝对差之和（找文字像素）
TEXT_DILATE_MIN = 3
TEXT_DILATE_MAX = 11
TEXT_MIN_RATIO = 0.004            # 掩膜占比过低 = 没找到文字
TEXT_MAX_RATIO = 0.55             # 掩膜占比过高 = 判断错误，放弃
TEXT_INPAINT_RADIUS_MIN = 3
TEXT_INPAINT_RADIUS_MAX = 8
FEATHER_WIDTH = 3                 # 边缘羽化宽度（像素）


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _to_pil_rgb(bgr):
    """BGR(OpenCV) → RGB(PIL)。

    旧实现直接 Image.fromarray(BGR) 会被当成 RGB，导致红蓝互换；
    这里统一换序。
    """
    return Image.fromarray(bgr[:, :, ::-1])


def _ring_pixels(roi, inner):
    """取 inner 掩膜之外的像素（作为背景参考），返回 (N,3)；无则 None。"""
    keep = (inner == 0)
    if not keep.any():
        return None
    return roi[keep]


def _uniform_background_color(ring, std_threshold=UNIFORM_STD_THRESHOLD):
    """背景近似纯色时返回中位色（BGR 数组），否则 None。"""
    if ring is None or len(ring) < 8:
        return None
    if float(ring.std(axis=0).mean()) > std_threshold:
        return None
    return np.median(ring, axis=0).astype(np.uint8)


def _feather_edges(patched, original, width=FEATHER_WIDTH):
    """把修复块四边与原始像素渐变融合，消除方形接缝。"""
    if patched is None or original is None:
        return patched
    if patched.shape != original.shape:
        return patched
    h, w = patched.shape[:2]
    if h < 2 * width + 2 or w < 2 * width + 2:
        return patched
    out = patched.astype(np.float32)
    org = original.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, width + 1, dtype=np.float32)[1:]
    for i, a in enumerate(ramp):
        out[i, :, :] = org[i, :, :] * (1 - a) + out[i, :, :] * a
        out[h - 1 - i, :, :] = (org[h - 1 - i, :, :] * (1 - a) +
                                out[h - 1 - i, :, :] * a)
        out[:, i, :] = org[:, i, :] * (1 - a) + out[:, i, :] * a
        out[:, w - 1 - i, :] = (org[:, w - 1 - i, :] * (1 - a) +
                                out[:, w - 1 - i, :] * a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==================== 整框修复 ====================

def inpaint_region(image_np, x1, y1, x2, y2, padding=INPAINT_PADDING):
    """整框修复：把整个框视为待修复区域。

    相比旧实现的改进：
    1. 背景近似纯色 → 直接用中位色填充（比盲修复干净得多，也更快）；
    2. 背景有纹理 → 自适应半径 + 按纹理强度选 TELEA / NS；
    3. 掩膜外扩 1px，覆盖文字抗锯齿边缘；
    4. 四边羽化，消除方形接缝；
    5. BGR → RGB 正确换序（旧实现红蓝互换）。
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

    mx1 = _clamp(x1 - ex1, 0, roi.shape[1])
    my1 = _clamp(y1 - ey1, 0, roi.shape[0])
    mx2 = _clamp(x2 - ex1, 0, roi.shape[1])
    my2 = _clamp(y2 - ey1, 0, roi.shape[0])
    if mx2 - mx1 < 2 or my2 - my1 < 2:
        return None

    inner = np.zeros(roi.shape[:2], dtype=np.uint8)
    inner[my1:my2, mx1:mx2] = 255

    original = roi[my1:my2, mx1:mx2].copy()
    ring = _ring_pixels(roi, inner)
    flat = _uniform_background_color(ring)

    try:
        if flat is not None:
            # 纯色背景：直接填充，避免 TELEA 把周边杂色抹进来
            result = np.empty_like(original)
            result[:, :] = flat
        else:
            mask = cv2.dilate(inner, np.ones((3, 3), np.uint8), iterations=1)
            radius = _clamp(int(min(mx2 - mx1, my2 - my1) // 3),
                            INPAINT_RADIUS_MIN, INPAINT_RADIUS_MAX)
            ring_std = float(ring.std(axis=0).mean()) if ring is not None else 0.0
            flag = (cv2.INPAINT_NS if ring_std < TEXTURED_STD_THRESHOLD
                    else cv2.INPAINT_TELEA)
            inpainted = cv2.inpaint(roi, mask, radius, flag)
            result = inpainted[my1:my2, mx1:mx2]
    except Exception:
        return None

    result = _feather_edges(result, original)
    return _to_pil_rgb(result)


# ==================== 笔画级修复（加强版） ====================

def inpaint_text_only(image_np, x1, y1, x2, y2, padding=TEXT_PADDING):
    """
    笔画级修复：只擦除文字笔画，保留背景
    返回：PIL.Image（尺寸 = 原框大小）或 None

    相比旧实现的改进：
    1. 自动判断文字极性（深字浅底 / 浅字深底），不再固定 BINARY_INV；
    2. 掩膜 = 自适应阈值 ∪「与背景中位色的距离」，彩色文字也能覆盖；
    3. 膨胀尺寸随框高自适应，大字不会留下抗锯齿光晕；
    4. 掩膜占比过高/过低直接放弃（返回 None，交回调用方兜底）；
    5. 半径自适应 + 四边羽化 + BGR→RGB 正确换序。
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

    mx1 = _clamp(x1 - ex1, 0, roi.shape[1])
    my1 = _clamp(y1 - ey1, 0, roi.shape[0])
    mx2 = _clamp(x2 - ex1, 0, roi.shape[1])
    my2 = _clamp(y2 - ey1, 0, roi.shape[0])
    if mx2 - mx1 < 4 or my2 - my1 < 4:
        return None

    inner = np.zeros(roi.shape[:2], dtype=np.uint8)
    inner[my1:my2, mx1:mx2] = 255

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inner_gray = gray[my1:my2, mx1:mx2]

        # 极性无关的文字提取：先 Otsu 二分，再取「离背景中位值更远」的一类当文字。
        # 旧实现固定 THRESH_BINARY_INV → 深底浅字完全失效；
        # 而改成固定 BINARY 又会把暗背景整片判成文字（实测占比 0.68）。
        bg_med = float(np.median(inner_gray))
        _th, otsu_inner = cv2.threshold(
            inner_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = otsu_inner > 0
        if bright.any() and (~bright).any():
            mean_bright = float(inner_gray[bright].mean())
            mean_dark = float(inner_gray[~bright].mean())
            text_is_bright = (abs(mean_bright - bg_med) >
                              abs(mean_dark - bg_med))
        else:
            text_is_bright = bg_med < 128
        text_inner = (otsu_inner if text_is_bright
                      else cv2.bitwise_not(otsu_inner))
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        mask[my1:my2, mx1:mx2] = text_inner

        # 颜色距离掩膜：与框内背景中位色差异大的像素（覆盖彩色文字）
        bg_color = np.median(
            roi[my1:my2, mx1:mx2].reshape(-1, 3), axis=0).astype(np.int16)
        diff = np.abs(roi.astype(np.int16) - bg_color).sum(axis=2)
        color_mask = (diff > COLOR_DISTANCE_THRESHOLD).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, color_mask)

        # 膨胀尺寸随字号自适应（大字抗锯齿更宽）
        k = int(_clamp(int(round((my2 - my1) / 12)) | 1,
                       TEXT_DILATE_MIN, TEXT_DILATE_MAX))
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, inner)

        ratio = float(mask.sum()) / float(mask.size * 255)
        if ratio < TEXT_MIN_RATIO or ratio > TEXT_MAX_RATIO:
            return None

        radius = _clamp(int(max(mx2 - mx1, my2 - my1) // 40),
                        TEXT_INPAINT_RADIUS_MIN, TEXT_INPAINT_RADIUS_MAX)
        inpainted = cv2.inpaint(roi, mask, radius, cv2.INPAINT_TELEA)
    except Exception:
        return None

    original = roi[my1:my2, mx1:mx2]
    result = _feather_edges(inpainted[my1:my2, mx1:mx2], original)
    return _to_pil_rgb(result)


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

    # 条件 2：背景亮度（取较亮分位，避免被文字拉低）
    bg_lum = float(np.percentile(gray, 80))
    if bg_lum < 70:
        return False

    # 条件 3：对比度
    # 文字通常只占框内 5%~15%，用 15 分位会把背景像素当成文字，
    # 测出的对比度偏低（实测 16），导致精细模式在多数真实场景不生效。
    text_lum = float(np.percentile(gray, 5))
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