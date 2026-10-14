"""
覆盖显示模块
- 多小窗口 + 位置量化防抖
- 支持窗口池
- 支持图像修复背景 + 精细文字替换
- 支持异步覆盖（打字机动画）
"""

import tkinter as tk
import ctypes
from PIL import ImageTk
from layout import calc_font_size
from logger import get_logger


log = get_logger('overlay')

DEFAULT_FONT = 'Microsoft YaHei'

WDA_EXCLUDEFROMCAPTURE = 0x00000011

POSITION_QUANTIZE = 6


def _get_toplevel_hwnd(hwnd):
    try:
        root = ctypes.windll.user32.GetAncestor(hwnd, 2)
        return root if root else hwnd
    except Exception:
        return hwnd


def exclude_from_capture(hwnd):
    try:
        top = _get_toplevel_hwnd(hwnd)
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(
            top, WDA_EXCLUDEFROMCAPTURE
        ))
    except Exception:
        return False


class TranslationBox(tk.Toplevel):
    """
    单个译文小窗口
    - 支持异步覆盖（逐字显示）
    - 位置量化防抖
    """

    def __init__(self, master, font_name=DEFAULT_FONT):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill='both', expand=True)

        self._font_name = font_name
        self._box_w = 0
        self._box_h = 0
        self._full_text = ''         # 完整译文
        self._visible_len = 0        # 当前可见字符数
        self._animating = False      # 是否正在动画
        self._base_font_size = 15
        self._pos = None
        self._bg_hex = '#FFFFFF'
        self._fg = '#000000'
        self._bg_image = None
        self._photo = None
        self._initialized = False

    def reset(self, x, y, w, h, text, bg_hex, fg, bg_image,
              base_font_size, async_enabled=False):
        """首次初始化（不复用）"""
        self._box_w = w
        self._box_h = h
        self._full_text = text
        self._base_font_size = base_font_size
        self._pos = (x, y)
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image

        if async_enabled and text:
            self._visible_len = 0
            self._animating = True
        else:
            self._visible_len = len(text)
            self._animating = False

        self.geometry(f"{w}x{h}+{x}+{y}")
        self._redraw()

        if not self._initialized:
            self.after(30, self._apply_capture_exclusion)
            self._initialized = True

    def _apply_capture_exclusion(self):
        try:
            self.update_idletasks()
            exclude_from_capture(self.winfo_id())
        except Exception:
            pass

    def _redraw(self):
        self.canvas.delete('all')
        self._photo = None

        if self._bg_image is not None:
            img = self._bg_image
            if img.size != (self._box_w, self._box_h):
                img = img.resize((self._box_w, self._box_h))
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=self._photo, anchor='nw')
        else:
            self.canvas.configure(bg=self._bg_hex)

        # 👇 字号按完整文本算，保证动画中字号不变
        font_size = calc_font_size(
            self._full_text, self._box_w, self._box_h, self._base_font_size
        )

        visible_text = self._full_text[:self._visible_len]

        self.canvas.create_text(
            2, self._box_h // 2,
            text=visible_text, fill=self._fg,
            font=(self._font_name, font_size),
            anchor='w', width=self._box_w - 4,
            justify='left',
        )

    def update_content(self, text, base_font_size, bg_hex, fg, bg_image,
                       async_enabled=False):
        """更新内容（位置不变）"""
        text_changed = (self._full_text != text)
        style_changed = (
            self._base_font_size != base_font_size or
            self._bg_hex != bg_hex or
            self._fg != fg or
            self._bg_image is not bg_image
        )

        if not text_changed and not style_changed:
            return

        self._full_text = text
        self._base_font_size = base_font_size
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image

        if text_changed:
            if async_enabled:
                # 新文本 → 从头动画
                self._visible_len = 0
                self._animating = True
            else:
                self._visible_len = len(text)
                self._animating = False

        self._redraw()

    def advance_animation(self, chars_per_tick=1):
        """推进动画，返回是否仍在动画中"""
        if not self._animating:
            return False

        self._visible_len = min(
            self._visible_len + chars_per_tick,
            len(self._full_text)
        )

        if self._visible_len >= len(self._full_text):
            self._animating = False

        self._redraw()
        return self._animating

    def skip_animation(self):
        """立即显示全部"""
        if self._animating:
            self._visible_len = len(self._full_text)
            self._animating = False
            self._redraw()

    def set_base_font_size(self, base_font_size):
        if self._base_font_size != base_font_size:
            self._base_font_size = base_font_size
            self._redraw()

    def move_to(self, x, y, quantize=True):
        if self._pos is None:
            self._pos = (x, y)
            self.geometry(f"+{x}+{y}")
            return

        old_x, old_y = self._pos

        if quantize:
            if abs(x - old_x) < POSITION_QUANTIZE and abs(y - old_y) < POSITION_QUANTIZE:
                return

        if (x, y) != self._pos:
            self._pos = (x, y)
            self.geometry(f"+{x}+{y}")

    def force_move_to(self, x, y):
        self._pos = (x, y)
        self.geometry(f"+{x}+{y}")

    # 兼容旧接口
    def hide(self):
        pass

    def show(self):
        pass


class OverlayWindow:
    """多小窗口覆盖，支持窗口池 + 异步覆盖"""

    POOL_MAX = 50

    def __init__(self, master, font_name=DEFAULT_FONT, alpha=1.0):
        self.master = master
        self.font_name = font_name
        self.boxes = {}
        self.base_font_size = 15

        self.use_pool = True
        self._pool = []

        # 异步覆盖
        self.async_enabled = False
        self.base_chars_per_tick = 2   # 基础速度

    def set_async_enabled(self, enabled):
        """开关异步覆盖"""
        self.async_enabled = enabled
        if not enabled:
            # 关闭时立即显示所有正在动画的
            for box in self.boxes.values():
                try:
                    box.skip_animation()
                except Exception:
                    pass

    def tick_animation(self):
        """
        推进所有正在动画的窗口
        按文本长度自适应速度：短文本慢，长文本快
        """
        if not self.async_enabled:
            return
        for box in self.boxes.values():
            try:
                if not box._animating:
                    continue
                text_len = len(box._full_text)
                if text_len < 20:
                    cpt = 1
                elif text_len < 60:
                    cpt = 2
                else:
                    cpt = 3
                box.advance_animation(cpt)
            except Exception:
                pass

    def set_pool_enabled(self, enabled):
        self.use_pool = enabled
        if not enabled:
            for box in self._pool:
                try:
                    box.destroy()
                except Exception:
                    pass
            self._pool.clear()

    def _acquire(self):
        if self.use_pool and self._pool:
            box = self._pool.pop()
            try:
                box.deiconify()
            except Exception:
                pass
            return box
        return TranslationBox(self.master, self.font_name)

    def _release(self, box):
        if self.use_pool and len(self._pool) < self.POOL_MAX:
            try:
                box.withdraw()
            except Exception:
                pass
            self._pool.append(box)
        else:
            try:
                box.destroy()
            except Exception:
                pass

    # 兼容旧接口
    def hide_all(self):
        pass

    def show_all(self):
        pass

    def set_base_font_size(self, size):
        self.base_font_size = size
        for box in self.boxes.values():
            try:
                box.set_base_font_size(size)
            except Exception:
                pass

    def update(self, items, base_font_size=15):
        """更新所有译文窗口"""
        self.base_font_size = base_font_size
        if not items:
            return

        new_map = {it['key']: it for it in items}

        for key, item in new_map.items():
            text = item.get('text', '')
            if not text or not text.strip():
                continue

            x1 = item['x1']
            y1 = item['y1']
            x2 = item['x2']
            y2 = item['y2']
            bg_hex = item.get('bg_hex', '#FFFFFF')
            fg = item.get('fg', '#000000')
            bg_image = item.get('bg_image')
            item_font = item.get('force_font_size', base_font_size)

            w = max(10, x2 - x1)
            h = max(10, y2 - y1)

            old = self.boxes.get(key)
            if old is None:
                box = self._acquire()
                box.reset(x1, y1, w, h, text, bg_hex, fg, bg_image,
                          item_font, async_enabled=self.async_enabled)
                self.boxes[key] = box
            else:
                old.move_to(x1, y1, quantize=True)
                old.update_content(text, item_font, bg_hex, fg, bg_image,
                                   async_enabled=self.async_enabled)

        for key in list(self.boxes.keys()):
            if key not in new_map:
                self._release(self.boxes[key])
                del self.boxes[key]

    # ========== 边界裁剪 ==========

    def prune_out_of_bounds(self, region, padding=0):
        if not self.boxes:
            return 0

        try:
            l = region['left'] - padding
            t = region['top'] - padding
            r = region['left'] + region['width'] + padding
            b = region['top'] + region['height'] + padding
        except Exception as e:
            log.error(f"[裁剪] region 参数异常：{e}")
            return 0

        removed = 0
        for key in list(self.boxes.keys()):
            try:
                box = self.boxes.get(key)
                if box is None:
                    continue
                pos = getattr(box, '_pos', None)
                if pos is None:
                    continue
                x1, y1 = pos

                bw = getattr(box, '_box_w', 0) or 0
                bh = getattr(box, '_box_h', 0) or 0
                cx = x1 + bw // 2
                cy = y1 + bh // 2

                if cx < l or cx > r or cy < t or cy > b:
                    try:
                        self._release(box)
                    except Exception:
                        pass
                    del self.boxes[key]
                    removed += 1
            except Exception as e:
                log.warning(f"[裁剪] 跳过异常 box {key}: {e}")
                try:
                    del self.boxes[key]
                except Exception:
                    pass

        return removed

    # ========== 滚动平移 ==========

    def scroll_by(self, dy):
        if dy == 0 or not self.boxes:
            return
        for box in self.boxes.values():
            try:
                x, y = box._pos if box._pos else (0, 0)
                box.force_move_to(x, y + dy)
            except Exception:
                pass

    def clear(self):
        for box in self.boxes.values():
            try:
                self._release(box)
            except Exception:
                pass
        self.boxes.clear()