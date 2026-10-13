"""
图像修复工具
把原文区域用周围像素填充，生成"干净"背景
"""

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 修复参数
INPAINT_PADDING = 15      # 向外扩展的上下文像素
INPAINT_RADIUS = 3        # 修复半径


def inpaint_region(image_np, x1, y1, x2, y2, padding=INPAINT_PADDING):
    """
    对指定区域做图像修复
    返回 PIL.Image（尺寸 = 原框大小）或 None
    """
    if not CV2_AVAILABLE:
        return None

    H, W = image_np.shape[:2]

    # 向外扩展，获取周围上下文
    ex1 = max(0, x1 - padding)
    ey1 = max(0, y1 - padding)
    ex2 = min(W, x2 + padding)
    ey2 = min(H, y2 + padding)

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = image_np[ey1:ey2, ex1:ex2].copy()

    # Mask：只把原框区域标记为"待修复"
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    mx1, my1 = x1 - ex1, y1 - ey1
    mx2, my2 = x2 - ex1, y2 - ey1
    mask[my1:my2, mx1:mx2] = 255

    try:
        inpainted = cv2.inpaint(roi, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    except Exception:
        return None

    # 裁回原框区域
    result = inpainted[my1:my2, mx1:mx2]
    return Image.fromarray(result)