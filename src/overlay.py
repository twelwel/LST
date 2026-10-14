"""
覆盖显示模块
- 覆盖模式：多小窗口
- 双语模式：原文下方小字译文
- 小窗口模式：独立浮窗累积译文（段落级去重）

- 位置量化防抖
- 窗口池
- 图像修复背景 + 精细文字替换
- 异步覆盖（打字机动画）
"""

import re
import tkinter as tk
import ctypes
from PIL import ImageTk
from layout import calc_font_size
from logger import get_logger


log = get_logger('overlay')

DEFAULT_FONT = 'Microsoft YaHei'
WDA_EXCLUDEFROMCAPTURE = 0x00000011
POSITION_QUANTIZE = 6

# 双语模式
BILINGUAL_FONT_SCALE = 0.75
BILINGUAL_PADDING = 2
BILINGUAL_GAP = 1
BILINGUAL_MIN_H = 16
BILINGUAL_MIN_W = 40
BILINGUAL_ALPHA = 0.5
BILINGUAL_MAX_LINES = 2

# 小窗口模式
CAPTION_DEFAULT_W = 460
CAPTION_DEFAULT_H = 360
CAPTION_OPACITY = 0.94
CAPTION_BG = '#1e1e1e'
CAPTION_FG = '#eeeeee'
CAPTION_BAR_BG = '#2b2b2b'
CAPTION_BAR_FG = '#eeeeee'
CAPTION_SELECT_BG = '#444488'
CAPTION_FONT_SIZE = 12
CAPTION_MIN_TEXT_LEN = 2      # 少于该长度的片段丢弃


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


def _bilingual_chars_per_line(width, font_size):
    return max(1, int((width - 6) / (font_size * 0.95)))


def _calc_bilingual_height(text, width, base_font_size):
    size = max(9, int(base_font_size * BILINGUAL_FONT_SCALE))
    cpl = _bilingual_chars_per_line(width, size)
    total_lines = max(1, (len(text) + cpl - 1) // cpl)
    lines = min(total_lines, BILINGUAL_MAX_LINES)
    line_h = int(size * 1.15)
    return max(BILINGUAL_MIN_H, lines * line_h + 2 * BILINGUAL_PADDING)


def _truncate_bilingual(text, width, base_font_size):
    if not text:
        return text
    size = max(9, int(base_font_size * BILINGUAL_FONT_SCALE))
    cpl = _bilingual_chars_per_line(width, size)
    max_chars = cpl * BILINGUAL_MAX_LINES
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:1]
    return text[:max_chars - 1] + '…'


def _estimate_text_width(text, font_size):
    if not text:
        return 0
    w = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
            w += font_size
        else:
            w += int(font_size * 0.55)
    return w


def _calc_bilingual_width(text, base_font_size, max_w):
    size = max(9, int(base_font_size * BILINGUAL_FONT_SCALE))
    est = _estimate_text_width(text, size) + 10
    return max(BILINGUAL_MIN_W, min(int(est), int(max_w)))


# ==================== 覆盖/双语 单窗口 ====================

class TranslationBox(tk.Toplevel):
    """单个译文窗口（覆盖 / 双语）"""

    def __init__(self, master, font_name=DEFAULT_FONT):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill='both', expand=True)

        self._font_name = font_name
        self._box_w = 0
        self._box_h = 0
        self._full_text = ''
        self._visible_len = 0
        self._animating = False
        self._base_font_size = 15
        self._pos = None
        self._bg_hex = '#FFFFFF'
        self._fg = '#000000'
        self._bg_image = None
        self._photo = None
        self._initialized = False
        self._bilingual = False

    def _apply_alpha(self):
        try:
            if self._bilingual:
                self.attributes('-alpha', BILINGUAL_ALPHA)
            else:
                self.attributes('-alpha', 1.0)
        except Exception:
            pass

    def reset(self, x, y, w, h, text, bg_hex, fg, bg_image,
              base_font_size, async_enabled=False, bilingual=False):
        self._bilingual = bool(bilingual)
        self._box_w = w
        self._box_h = h
        self._full_text = text
        self._base_font_size = base_font_size
        self._pos = (x, y)
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image

        self._apply_alpha()

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

        if self._bg_image is not None and not self._bilingual:
            img = self._bg_image
            if img.size != (self._box_w, self._box_h):
                img = img.resize((self._box_w, self._box_h))
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=self._photo, anchor='nw')
        else:
            self.canvas.configure(bg=self._bg_hex)

        if self._bilingual:
            base_size = max(9, int(self._base_font_size *
                                   BILINGUAL_FONT_SCALE))
            font_size = calc_font_size(
                self._full_text, self._box_w, self._box_h, base_size)
            visible = self._full_text[:self._visible_len]
            visible = _truncate_bilingual(
                visible, self._box_w, self._base_font_size)
            self.canvas.create_text(
                3, self._box_h // 2,
                text=visible, fill=self._fg,
                font=(self._font_name, font_size),
                anchor='w', width=self._box_w - 6,
                justify='left',
            )
            return

        font_size = calc_font_size(
            self._full_text, self._box_w, self._box_h, self._base_font_size)
        visible = self._full_text[:self._visible_len]
        self.canvas.create_text(
            4, self._box_h // 2,
            text=visible, fill=self._fg,
            font=(self._font_name, font_size),
            anchor='w', width=self._box_w - 8,
            justify='left',
        )

    def update_content(self, text, base_font_size, bg_hex, fg, bg_image,
                       async_enabled=False, bilingual=False):
        text_changed = (self._full_text != text)
        style_changed = (
            self._base_font_size != base_font_size or
            self._bg_hex != bg_hex or
            self._fg != fg or
            self._bg_image is not bg_image or
            self._bilingual != bool(bilingual)
        )

        if not text_changed and not style_changed:
            return

        self._full_text = text
        self._base_font_size = base_font_size
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image
        self._bilingual = bool(bilingual)

        self._apply_alpha()

        if text_changed:
            if async_enabled:
                self._visible_len = 0
                self._animating = True
            else:
                self._visible_len = len(text)
                self._animating = False

        self._redraw()

    def set_size(self, w, h, x=None, y=None):
        w = max(10, int(w))
        h = max(10, int(h))
        if w == self._box_w and h == self._box_h:
            if x is not None and y is not None and self._pos != (x, y):
                self._pos = (x, y)
                self.geometry(f"{w}x{h}+{x}+{y}")
            return
        self._box_w = w
        self._box_h = h
        if x is not None and y is not None:
            self._pos = (x, y)
            self.geometry(f"{w}x{h}+{x}+{y}")
        elif self._pos:
            self.geometry(f"{w}x{h}+{self._pos[0]}+{self._pos[1]}")
        self._redraw()

    def advance_animation(self, chars_per_tick=1):
        if not self._animating:
            return False
        self._visible_len = min(
            self._visible_len + chars_per_tick, len(self._full_text))
        if self._visible_len >= len(self._full_text):
            self._animating = False
        self._redraw()
        return self._animating

    def skip_animation(self):
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

    def hide(self):
        pass

    def show(self):
        pass


# ==================== 小窗口（字幕）模式 ====================

class CaptionWindow:
    """
    独立浮窗，累积显示译文
    - 段落级去重：用「归一化文本」做 key（去空格、标点、转小写）
    - 过滤短片段（< CAPTION_MIN_TEXT_LEN 字符）
    - 顶部标题栏可拖动，正文可选中复制
    """

    def __init__(self, master, font_name=DEFAULT_FONT,
                 width=CAPTION_DEFAULT_W, height=CAPTION_DEFAULT_H,
                 on_close=None):
        self.master = master
        self.font_name = font_name
        self.on_close = on_close
        self._seen_norm = set()
        self._visible = False
        self._drag_start = None

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.attributes('-topmost', True)
        try:
            self.win.attributes('-alpha', CAPTION_OPACITY)
        except Exception:
            pass

        # ---- 标题栏 ----
        self.bar = tk.Frame(self.win, bg=CAPTION_BAR_BG, height=26)
        self.bar.pack(fill='x', side='top')
        self.bar.pack_propagate(False)

        tk.Label(
            self.bar, text='📋 译文',
            fg=CAPTION_BAR_FG, bg=CAPTION_BAR_BG,
            font=(font_name, 10, 'bold'),
        ).pack(side='left', padx=8)

        close_btn = tk.Label(
            self.bar, text='✕',
            fg='#cccccc', bg=CAPTION_BAR_BG,
            font=(font_name, 11, 'bold'),
            cursor='hand2',
        )
        close_btn.pack(side='right', padx=8)
        close_btn.bind('<Button-1>', lambda e: self.hide())

        clear_btn = tk.Label(
            self.bar, text='🗑',
            fg='#cccccc', bg=CAPTION_BAR_BG,
            font=(font_name, 11),
            cursor='hand2',
        )
        clear_btn.pack(side='right', padx=4)
        clear_btn.bind('<Button-1>', lambda e: self.clear())

        tk.Label(
            self.bar, text='(可拖动)',
            fg='#888888', bg=CAPTION_BAR_BG,
            font=(font_name, 8),
        ).pack(side='right', padx=4)

        # ---- 文本区 ----
        text_frame = tk.Frame(self.win, bg=CAPTION_BG)
        text_frame.pack(fill='both', expand=True)

        self.text = tk.Text(
            text_frame,
            bg=CAPTION_BG, fg=CAPTION_FG,
            font=(font_name, CAPTION_FONT_SIZE),
            wrap='word', bd=0, highlightthickness=0,
            insertbackground=CAPTION_FG,
            selectbackground=CAPTION_SELECT_BG,
            selectforeground='#ffffff',
            padx=8, pady=6,
        )
        self.text.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(
            text_frame, command=self.text.yview,
            bg=CAPTION_BAR_BG, troughcolor=CAPTION_BG,
            activebackground='#444444', bd=0,
            width=10,
        )
        scrollbar.pack(side='right', fill='y')
        self.text.configure(yscrollcommand=scrollbar.set)

        # ---- 拖动绑定 ----
        self.bar.bind('<ButtonPress-1>', self._on_drag_start)
        self.bar.bind('<B1-Motion>', self._on_drag)

        # ---- 初始位置：屏幕右下 ----
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = sw - width - 30
        y = sh - height - 80
        self.win.geometry(f"{width}x{height}+{x}+{y}")
        self.win.withdraw()

    # ---------- 拖动 ----------

    def _on_drag_start(self, event):
        self._drag_start = (
            event.x_root, event.y_root,
            self.win.winfo_x(), self.win.winfo_y(),
        )

    def _on_drag(self, event):
        if not self._drag_start:
            return
        sx, sy, wx, wy = self._drag_start
        dx = event.x_root - sx
        dy = event.y_root - sy
        self.win.geometry(f"+{wx + dx}+{wy + dy}")

    # ---------- 显隐 ----------

    def show(self):
        if not self._visible:
            try:
                self.win.deiconify()
            except Exception:
                pass
            self._visible = True

    def hide(self):
        if self._visible:
            try:
                self.win.withdraw()
            except Exception:
                pass
            self._visible = False
        if self.on_close:
            try:
                self.on_close()
            except Exception:
                pass

    def is_visible(self):
        return self._visible

    # ---------- 内容整理 ----------

    @staticmethod
    def _normalize(text):
        """
        归一化文本用于去重：
        - 去掉所有空白
        - 去掉常见中英文标点
        - 转小写
        这样 "Hello," 和 "hello" 会被视为同一段
        """
        if not text:
            return ''
        t = re.sub(r'\s+', '', text)
        t = re.sub(
            r"""[.,;:!?"'`~@#$%^&*()\[\]{}<>|\\/《》【】。，；：！？、""''…—\-_]+""",
            '', t,
        )
        return t.lower()

    def add_items(self, items):
        """
        追加译文（段落级去重）
        - 过滤短片段
        - 归一化后去重
        - 返回实际新增数量
        """
        added = 0
        for it in items:
            text = (it.get('text') or '').strip()
            if len(text) < CAPTION_MIN_TEXT_LEN:
                continue

            norm = self._normalize(text)
            if not norm or norm in self._seen_norm:
                continue

            self._seen_norm.add(norm)
            try:
                self.text.insert('end', text + '\n')
                added += 1
            except Exception:
                pass

        if added:
            try:
                self.text.see('end')
            except Exception:
                pass
            self.show()
            log.info(f"[小窗口] 新增 {added} 段译文")
        return added

    def clear(self):
        self._seen_norm.clear()
        try:
            self.text.delete('1.0', 'end')
        except Exception:
            pass

    def get_geometry(self):
        try:
            self.win.update_idletasks()
            return self.win.geometry()
        except Exception:
            return ''

    def destroy(self):
        try:
            self.win.destroy()
        except Exception:
            pass


# ==================== 覆盖管理器 ====================

class OverlayWindow:
    """覆盖 / 双语 多窗口管理器"""

    POOL_MAX = 50

    def __init__(self, master, font_name=DEFAULT_FONT, alpha=1.0):
        self.master = master
        self.font_name = font_name
        self.boxes = {}
        self.base_font_size = 15

        self.use_pool = True
        self._pool = []

        self.async_enabled = False

    def set_async_enabled(self, enabled):
        self.async_enabled = enabled
        if not enabled:
            for box in self.boxes.values():
                try:
                    box.skip_animation()
                except Exception:
                    pass

    def tick_animation(self):
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
            item_font = item.get('force_font_size', base_font_size)

            bilingual = item.get('bilingual', False)

            if bilingual:
                orig_y2 = item.get('orig_y2', y2)
                new_x1 = x1
                new_y1 = orig_y2 + BILINGUAL_GAP
                max_w = max(BILINGUAL_MIN_W, int(x2 - x1))
                new_w = _calc_bilingual_width(text, item_font, max_w)
                new_h = _calc_bilingual_height(text, new_w, item_font)
                bg_image = None
            else:
                new_x1 = x1
                new_y1 = y1
                new_w = max(10, x2 - x1)
                new_h = max(10, y2 - y1)

            old = self.boxes.get(key)
            if old is None:
                box = self._acquire()
                box.reset(
                    new_x1, new_y1, new_w, new_h,
                    text, bg_hex, fg, bg_image,
                    item_font,
                    async_enabled=self.async_enabled,
                    bilingual=bilingual,
                )
                self.boxes[key] = box
            else:
                old.move_to(new_x1, new_y1, quantize=True)
                old.set_size(new_w, new_h, new_x1, new_y1)
                old.update_content(
                    text, item_font, bg_hex, fg, bg_image,
                    async_enabled=self.async_enabled,
                    bilingual=bilingual,
                )

        for key in list(self.boxes.keys()):
            if key not in new_map:
                self._release(self.boxes[key])
                del self.boxes[key]

    # ========== 边界裁剪 ==========

    def prune_out_of_bounds(self, region, padding=0):
        if not self.boxes:
            return 0

        try:
            rl = region['left'] - padding
            rt = region['top'] - padding
            rr = region['left'] + region['width'] + padding
            rb = region['top'] + region['height'] + padding
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

                bx1, by1 = pos
                bw = getattr(box, '_box_w', 0) or 0
                bh = getattr(box, '_box_h', 0) or 0
                bx2 = bx1 + bw
                by2 = by1 + bh

                if bx2 < rl or bx1 > rr or by2 < rt or by1 > rb:
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