"""滚动检测独立线程
- 独立循环，不受 OCR / 翻译阻塞
- 高频截图检测垂直滚动偏移
- 通过回调通知 UI
"""

import threading
import time

import numpy as np
import mss
from PIL import Image as PILImage

from logger import get_logger


log = get_logger('scroll')


# ==================== 参数 ====================
THUMB_W = 160              # 缩略图宽
THUMB_H = 48               # 缩略图高
MAX_OFFSET_PX = 60         # 最大搜索位移（像素）
MIN_MATCH = 0.6            # 最小匹配分数
MIN_DY = 3                 # 小于此值不算滚动
DETECT_INTERVAL = 0.01     # 检测间隔（秒），约 16 fps
# =================================================


class ScrollDetector:
    """
    滚动检测器
    - 独立线程运行，与 pipeline 解耦
    - 通过 on_scroll(dy) 回调上报
    - 提供 scroll_boost 状态给 UI 用（决定截图频率）
    """

    def __init__(self, get_region_fn, on_scroll,
                 is_active_fn=None,
                 interval=DETECT_INTERVAL):
        """
        get_region_fn : 无参函数，返回当前截图区域 dict
        on_scroll     : 回调 (dy: int) -> None，dy 为内容位移（正=下移）
        is_active_fn  : 无参函数，返回是否应该运行检测（暂停/钉住时返回 False）
        interval      : 检测间隔（秒）
        """
        self.get_region_fn = get_region_fn
        self.on_scroll = on_scroll
        self.is_active_fn = is_active_fn or (lambda: True)
        self.interval = interval

        self._thread = None
        self._stop_event = threading.Event()
        self._prev_gray = None

        # 状态：滚动中标志（供 UI 决定截图频率）
        self.scroll_boost = False
        self._streak = 0
        self._lock = threading.Lock()

        # mss 实例（线程独占）
        self._sct = None

    # ================= 生命周期 =================

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='ScrollDetector'
        )
        self._thread.start()
        log.info("[滚动] 检测线程已启动")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._close_sct()
        log.info("[滚动] 检测线程已停止")

    def reset(self):
        """清空状态（切换区域或模式时调用）"""
        with self._lock:
            self._prev_gray = None
            self._streak = 0
            self.scroll_boost = False

    def is_scrolling(self):
        with self._lock:
            return self.scroll_boost

    # ================= 主循环 =================

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                if self.is_active_fn():
                    self._detect_once()
                else:
                    # 暂停时清缓存
                    with self._lock:
                        self._prev_gray = None
                        self._streak = 0
                        self.scroll_boost = False
            except Exception as e:
                log.exception(f"[滚动] 检测异常：{e}")

            # 用 Event.wait 实现可中断 sleep
            self._stop_event.wait(self.interval)

        self._close_sct()

    def _detect_once(self):
        """执行一次检测"""
        try:
            region = self.get_region_fn()
        except Exception:
            return

        # 截取区域
        img = self._grab(region)
        if img is None:
            return

        # 转灰度缩略图
        try:
            pil = PILImage.fromarray(img).convert('L').resize(
                (THUMB_W, THUMB_H), PILImage.Resampling.BILINEAR,
            )
            cur = np.asarray(pil, dtype=np.int16)
        except Exception:
            return

        with self._lock:
            prev = self._prev_gray
            self._prev_gray = cur

        if prev is None:
            return

        # 计算最佳垂直偏移
        h = img.shape[0]
        scale_y = h / THUMB_H
        max_off = int(MAX_OFFSET_PX / scale_y) + 1

        ref_y1 = THUMB_H // 4
        ref_y2 = 3 * THUMB_H // 4
        template = cur[ref_y1:ref_y2, :]

        best_off = 0
        best_score = float('inf')
        for off in range(-max_off, max_off + 1):
            y1 = ref_y1 + off
            y2 = ref_y2 + off
            if y1 < 0 or y2 > THUMB_H:
                continue
            diff = prev[y1:y2, :] - template
            score = float(np.mean(np.abs(diff)))
            if score < best_score:
                best_score = score
                best_off = off

        similarity = max(0.0, 1.0 - best_score / 40.0)
        dy = -int(best_off * scale_y)

        is_scrolling = (abs(dy) >= MIN_DY and similarity >= MIN_MATCH)

        with self._lock:
            if is_scrolling:
                self._streak += 1
                self.scroll_boost = True
            else:
                self._streak = 0
                self.scroll_boost = False

        if is_scrolling:
            try:
                self.on_scroll(dy)
            except Exception as e:
                log.exception(f"[滚动] 回调异常：{e}")

    # ================= mss 管理 =================

    def _grab(self, region):
        try:
            if self._sct is None:
                self._sct = mss.mss()
            shot = self._sct.grab(region)
            return np.array(shot)[:, :, :3]
        except Exception:
            # mss 实例异常，重置
            self._close_sct()
            try:
                self._sct = mss.mss()
                shot = self._sct.grab(region)
                return np.array(shot)[:, :, :3]
            except Exception as e:
                log.error(f"[滚动] 截图失败：{e}")
                return None

    def _close_sct(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None