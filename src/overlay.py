"""
覆盖显示模块
- 多小窗口
- 支持窗口池（复用空闲窗口）
- 支持图像修复背景
- 用户可见，截图不可见
- 支持边界裁剪（原文滚出区域后译文自动消失）
"""

import tkinter as tk
import ctypes
from PIL import ImageTk
from layout import calc_font_size
from logger import get_logger


log = get_logger('overlay')

DEFAULT_FONT = 'Microsoft YaHei'

WDA_EXCLUDEFROMCAPTURE = 0x00000011


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
    - 支持复用（从池中取出后调用 reset 重设内容和位置）
    - 首次创建时应用截图排除
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
        self._text = ''
        self._base_font_size = 15
        self._pos = (0, 0)
        self._bg_hex = '#FFFFFF'
        self._fg = '#000000'
        self._bg_image = None
        self._photo = None
        self._initialized = False

    def reset(self, x, y, w, h, text, bg_hex, fg, bg_image, base_font_size):
        """首次初始化或从池中取出复用时调用"""
        self._box_w = w
        self._box_h = h
        self._text = text
        self._base_font_size = base_font_size
        self._pos = (x, y)
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image

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

        font_size = calc_font_size(
            self._text, self._box_w, self._box_h, self._base_font_size
        )
        self.canvas.create_text(
            2, self._box_h // 2,
            text=self._text, fill=self._fg,
            font=(self._font_name, font_size),
            anchor='w', width=self._box_w - 4,
            justify='left',
        )

    def update_content(self, text, base_font_size, bg_hex, fg, bg_image):
        """更新内容（位置不变）"""
        if (self._text == text and
                self._base_font_size == base_font_size and
                self._bg_hex == bg_hex and
                self._fg == fg and
                self._bg_image is bg_image):
            return
        self._text = text
        self._base_font_size = base_font_size
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image
        self._redraw()

    def set_base_font_size(self, base_font_size):
        if self._base_font_size != base_font_size:
            self._base_font_size = base_font_size
            self._redraw()

    def move_to(self, x, y):
        if self._pos != (x, y):
            self._pos = (x, y)
            self.geometry(f"+{x}+{y}")

    # 兼容旧接口
    def hide(self):
        pass

    def show(self):
        pass


class OverlayWindow:
    """
    多小窗口覆盖，支持窗口池
    - use_pool=True：不销毁窗口，回池复用
    - use_pool=False：每个段落独立创建/销毁
    """

    POOL_MAX = 50

    def __init__(self, master, font_name=DEFAULT_FONT, alpha=1.0):
        self.master = master
        self.font_name = font_name
        self.boxes = {}
        self.base_font_size = 15

        self.use_pool = True
        self._pool = []

    def set_pool_enabled(self, enabled):
        """开关窗口池；关闭时清空池"""
        self.use_pool = enabled
        if not enabled:
            for box in self._pool:
                try:
                    box.destroy()
                except Exception:
                    pass
            self._pool.clear()

    def _acquire(self):
        """获取一个窗口（优先从池里取）"""
        if self.use_pool and self._pool:
            box = self._pool.pop()
            try:
                box.deiconify()
            except Exception:
                pass
            return box
        return TranslationBox(self.master, self.font_name)

    def _release(self, box):
        """释放窗口（回池或销毁）"""
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

            w = max(10, x2 - x1)
            h = max(10, y2 - y1)

            old = self.boxes.get(key)
            if old is None:
                box = self._acquire()
                box.reset(x1, y1, w, h, text, bg_hex, fg, bg_image,
                          base_font_size)
                self.boxes[key] = box
            else:
                old.move_to(x1, y1)
                old.update_content(text, base_font_size, bg_hex, fg, bg_image)

        # 释放不再显示的段落
        for key in list(self.boxes.keys()):
            if key not in new_map:
                self._release(self.boxes[key])
                del self.boxes[key]

    # ========== 边界裁剪（新增） ==========

    def prune_out_of_bounds(self, region, padding=0):
        """
        移除超出截图区域的译文（原文已滚出边界）
        - 带异常保护，避免单个坏 box 导致整个循环挂掉
        返回：移除的数量
        """
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

                # 安全访问属性（防止旧版本创建的 box 缺属性）
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
                # 单个 box 出错不影响其它
                log.warning(f"[裁剪] 跳过异常 box {key}: {e}")
                try:
                    del self.boxes[key]
                except Exception:
                    pass

        return removed

    # ======================================

    def clear(self):
        for box in self.boxes.values():
            try:
                self._release(box)
            except Exception:
                pass
        self.boxes.clear()
        
    def scroll_by(self, dy):
        """
        按 dy 平移所有译文窗口
        dy: 内容位置变化（正数=内容下移，译文下移；负数=内容上移，译文上移）
        """
        if dy == 0 or not self.boxes:
            return
        for box in self.boxes.values():
            try:
                x, y = box._pos
                box.move_to(x, y + dy)
            except Exception:
                pass