"""覆盖显示模块
- 覆盖模式：多小窗口
- 双语模式：原文下方小字译文（视频字幕场景优化）
  ⭐ 字幕自动单行：宽扁段落译文强制单行显示，字号自适应
- 小窗口模式：独立浮窗累积译文（段落级去重）

- 位置量化防抖
- 窗口池
- 图像修复背景 + 精细文字替换
- 异步覆盖（打字机动画）
"""

import re
import tkinter as tk
import ctypes
from difflib import SequenceMatcher
from PIL import ImageTk
from layout import calc_font_size
from logger import get_logger


log = get_logger('overlay')

DEFAULT_FONT = 'Microsoft YaHei'
WDA_EXCLUDEFROMCAPTURE = 0x00000011
POSITION_QUANTIZE = 6

# ==================== 窗口扩展样式（鼠标穿透） ====================
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000        # Tk 设置 -alpha 时必需
WS_EX_TRANSPARENT = 0x00000020    # 鼠标穿透（不接收点击）
WS_EX_NOACTIVATE = 0x08000000     # 不抢焦点
LWA_ALPHA = 0x00000002            # SetLayeredWindowAttributes 的「整体透明度」模式

# ==================== 双语模式参数 ====================
BILINGUAL_FONT_SCALE = 0.9
BILINGUAL_PADDING = 3
BILINGUAL_GAP = 2
BILINGUAL_MIN_H = 20
BILINGUAL_MIN_W = 300
BILINGUAL_MAX_W_RATIO = 0.85
BILINGUAL_MAX_W = 1600
BILINGUAL_MAX_LINES = 4
BILINGUAL_ALPHA = 0.85

# ==================== 小窗口模式参数 ====================
CAPTION_DEFAULT_W = 460
CAPTION_DEFAULT_H = 360
CAPTION_OPACITY = 0.94
CAPTION_BG = '#1e1e1e'
CAPTION_FG = '#eeeeee'
CAPTION_BAR_BG = '#2b2b2b'
CAPTION_BAR_FG = '#eeeeee'
CAPTION_SELECT_BG = '#444488'
CAPTION_FONT_SIZE = 12
CAPTION_MIN_TEXT_LEN = 2


def _is_minor_text_change(old, new, min_len_ratio=0.7, min_sim=0.7):
    """判断译文改动是否「小到不该重启动画」。

    背景变动会让 OCR 轻微抖动，进而让译文差一两个标点/字。
    这种改动若每次都重打字机动画，观感就是抖动；
    此处判定为 minor 时直接整体替换、不重启动画。
    """
    if not old or not new:
        return False
    la, lb = len(old), len(new)
    longest = max(la, lb)
    if not longest:
        return False
    if (min(la, lb) / longest) < min_len_ratio:
        return False
    try:
        return SequenceMatcher(None, old, new).ratio() >= min_sim
    except Exception:
        return False


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


_click_through_logged = False


def make_click_through(hwnd, alpha=1.0):
    """让窗口不接收鼠标事件（点击穿透到下层窗口）。

    覆盖层是「贴在屏幕上的译文」，任何点击都应该落到底下的游戏/程序。
    需要 WS_EX_TRANSPARENT，并加 WS_EX_NOACTIVATE 避免抢焦点。

    ⚠️ 两个必须注意的坑：
    1. Tk 的 -alpha 是「直接赋值」GWL_EXSTYLE，会抹掉 TRANSPARENT /
       NOACTIVATE，所以必须在设置 -alpha 之后调用本函数。
    2. 若本次把窗口变成「分层窗口」（补了 WS_EX_LAYERED），必须同时调用
       SetLayeredWindowAttributes 设置分层属性 —— 否则会得到一个
       「没有任何分层属性的分层窗口」，在 Windows 上**完全不可见**
       （表现为译文整片消失）。

    幂等：样式已正确时不会重复调用 SetWindowLong。
    """
    global _click_through_logged
    try:
        top = _get_toplevel_hwnd(hwnd)
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_long]
        ex = user32.GetWindowLongW(top, GWL_EXSTYLE)
        want = ex | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE
        if want != ex:
            user32.SetWindowLongW(top, GWL_EXSTYLE, want)
            if not (ex & WS_EX_LAYERED):
                # 刚刚变成分层窗口 → 立刻补分层属性，避免窗口变隐形
                try:
                    a = max(0, min(255, int(round(float(alpha) * 255))))
                    user32.SetLayeredWindowAttributes(
                        top, 0, a, LWA_ALPHA)
                except Exception as e:
                    log.warning(f"[覆盖层] 设置分层属性失败（可能不可见）：{e}")
        if not _click_through_logged:
            _click_through_logged = True
            log.info("[覆盖层] 鼠标穿透已启用（WS_EX_TRANSPARENT）")
        return True
    except Exception as e:
        if not _click_through_logged:
            _click_through_logged = True
            log.warning(f"[覆盖层] 鼠标穿透启用失败：{e}")
        return False


def is_click_through(hwnd):
    """诊断用：读取当前窗口是否已带鼠标穿透样式。"""
    try:
        top = _get_toplevel_hwnd(hwnd)
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ex = user32.GetWindowLongW(top, GWL_EXSTYLE)
        return bool(ex & WS_EX_TRANSPARENT)
    except Exception:
        return False


# ==================== 双语工具函数 ====================

def _estimate_text_width(text, font_size):
    """估算文本像素宽度（中文全宽，其他半宽）"""
    if not text:
        return 0
    w = 0
    for ch in text:
        if ('\u4e00' <= ch <= '\u9fff' or
                '\u3000' <= ch <= '\u303f' or
                '\u3040' <= ch <= '\u30ff' or
                '\uac00' <= ch <= '\ud7af'):
            w += font_size
        else:
            w += int(font_size * 0.55)
    return w


def _count_lines(text, font_size, width):
    """按宽度模拟换行，返回行数"""
    if not text or width <= 0:
        return 1
    lines = 1
    current = 0
    for ch in text:
        if ('\u4e00' <= ch <= '\u9fff' or
                '\u3000' <= ch <= '\u303f' or
                '\u3040' <= ch <= '\u30ff' or
                '\uac00' <= ch <= '\ud7af'):
            cw = font_size
        else:
            cw = int(font_size * 0.55)
        if current + cw > width and current > 0:
            lines += 1
            current = cw
        else:
            current += cw
    return lines


def _calc_single_line_font(text, max_width, base_size, min_size=8):
    """单行字号：从 base_size 递减到能塞进 max_width"""
    if not text or max_width <= 0:
        return base_size
    size = base_size
    while size > min_size:
        w = _estimate_text_width(text, size)
        if w <= max_width:
            return size
        size -= 1
    return min_size


def _truncate_single_line(text, max_width, min_size=8):
    """单行截断（即使最小字号也塞不下时，末尾加省略号）"""
    if not text or max_width <= 0:
        return text

    if _estimate_text_width(text, min_size) <= max_width:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        w = _estimate_text_width(text[:mid] + '…', min_size)
        if w <= max_width:
            lo = mid
        else:
            hi = mid - 1

    if lo == 0:
        return '…'
    return text[:lo] + '…'


def _calc_bilingual_width(text, base_font_size, orig_w, subtitle=False):
    """
    双语窗口宽度：
    - 至少 = 原文宽度
    - 字幕模式：自适应文本宽度
    - 上限 = BILINGUAL_MAX_W
    """
    size = max(10, int(base_font_size * BILINGUAL_FONT_SCALE))
    est_w = _estimate_text_width(text, size) + 12

    w = max(int(orig_w or 0), est_w)

    if w < BILINGUAL_MIN_W:
        w = BILINGUAL_MIN_W
    if w > BILINGUAL_MAX_W:
        w = BILINGUAL_MAX_W

    return int(w)


def _calc_bilingual_height(text, width, base_font_size, subtitle=False):
    """
    双语窗口高度：
    - subtitle=True：强制单行
    - subtitle=False：最多 BILINGUAL_MAX_LINES 行
    """
    size = max(10, int(base_font_size * BILINGUAL_FONT_SCALE))

    if subtitle:
        line_h = int(size * 1.35)
        h = line_h + 2 * BILINGUAL_PADDING
        return max(BILINGUAL_MIN_H, h)

    inner_w = max(10, width - 8)
    total_lines = _count_lines(text, size, inner_w)
    lines = min(total_lines, BILINGUAL_MAX_LINES)
    line_h = int(size * 1.35)
    h = lines * line_h + 2 * BILINGUAL_PADDING
    return max(BILINGUAL_MIN_H, h)


def _truncate_bilingual(text, width, base_font_size):
    """超出最大行数时截断，末尾加省略号（多行）"""
    if not text:
        return text
    size = max(10, int(base_font_size * BILINGUAL_FONT_SCALE))
    inner_w = max(10, width - 8)
    max_chars = 0
    lines = 1
    current = 0
    for i, ch in enumerate(text):
        if ('\u4e00' <= ch <= '\u9fff' or
                '\u3000' <= ch <= '\u303f' or
                '\u3040' <= ch <= '\u30ff' or
                '\uac00' <= ch <= '\ud7af'):
            cw = size
        else:
            cw = int(size * 0.55)
        if current + cw > inner_w and current > 0:
            lines += 1
            if lines > BILINGUAL_MAX_LINES:
                max_chars = i
                break
            current = cw
        else:
            current += cw
    if max_chars == 0:
        return text
    return text[:max(1, max_chars - 1)] + '…'


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
        self._subtitle = False
        self._bg_opacity = 1.0
        self._font_cache_key = None
        self._font_cache_text = ''
        self._font_cache_size = 15

    def _apply_alpha(self):
        try:
            alpha = (BILINGUAL_ALPHA if self._bilingual
                     else self._bg_opacity)
            self.attributes('-alpha', alpha)
            # 设置 -alpha 会让窗口变为分层窗口，而 Tk 是「赋值」GWL_EXSTYLE，
            # 会抹掉 TRANSPARENT / NOACTIVATE；这里在之后补一次
            # （幂等、开销极小），并把 alpha 传进去避免分层属性缺失导致隐形。
            make_click_through(self.winfo_id(), alpha)
        except Exception:
            pass

    def set_bg_opacity(self, value):
        """设置覆盖面板不透明度（1.0 = 完全不透明）。"""
        try:
            v = max(0.05, min(1.0, float(value)))
        except Exception:
            return
        if abs(v - self._bg_opacity) < 0.001:
            return
        self._bg_opacity = v
        self._apply_alpha()

    def reset(self, x, y, w, h, text, bg_hex, fg, bg_image,
              base_font_size, async_enabled=False, bilingual=False,
              subtitle=False, bg_opacity=None):
        self._bilingual = bool(bilingual)
        self._subtitle = bool(subtitle)
        if bg_opacity is not None:
            try:
                self._bg_opacity = max(0.05, min(1.0, float(bg_opacity)))
            except Exception:
                pass
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
        """应用窗口样式：排除截屏 + 鼠标穿透（窗口首次显示后执行）。"""
        try:
            self.update_idletasks()
            hwnd = self.winfo_id()
            alpha = (BILINGUAL_ALPHA if self._bilingual
                     else self._bg_opacity)
            exclude_from_capture(hwnd)
            make_click_through(hwnd, alpha)
        except Exception:
            pass

    def _redraw(self):
        self.canvas.delete('all')
        self._photo = None

        # ============ 双语模式 ============
        if self._bilingual:
            self.canvas.configure(bg=self._bg_hex)

            base_size = max(10, int(self._base_font_size *
                                   BILINGUAL_FONT_SCALE))

            # ⭐ 字幕：单行显示，字号自适应
            if self._subtitle:
                visible = self._full_text[:self._visible_len]
                inner_w = max(10, self._box_w - 8)

                # 先截断
                visible = _truncate_single_line(visible, inner_w)
                # 字号自适应
                font_size = _calc_single_line_font(
                    visible, inner_w, base_size)

                self.canvas.create_text(
                    4, BILINGUAL_PADDING,
                    text=visible, fill=self._fg,
                    font=(self._font_name, font_size),
                    anchor='nw',
                    justify='left',
                )
                return

            # 多行模式
            font_size = calc_font_size(
                self._full_text, self._box_w - 8,
                self._box_h - 2 * BILINGUAL_PADDING, base_size)

            visible = self._full_text[:self._visible_len]
            visible = _truncate_bilingual(
                visible, self._box_w, self._base_font_size)

            self.canvas.create_text(
                4, BILINGUAL_PADDING,
                text=visible, fill=self._fg,
                font=(self._font_name, font_size),
                anchor='nw', width=self._box_w - 8,
                justify='left',
            )
            return

        # ============ 覆盖模式 ============
        if self._bg_image is not None:
            img = self._bg_image
            if img.size != (self._box_w, self._box_h):
                img = img.resize((self._box_w, self._box_h))
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, image=self._photo, anchor='nw')
        else:
            self.canvas.configure(bg=self._bg_hex)

        # 字号滞后：盒子尺寸变化 1~2px 时不重算字号。
        # 否则框高抖动 1px 就可能让字号跳一级，字形重新排版 → 视觉抖动。
        _fkey = (self._box_w // 2, self._box_h // 2)
        if (self._font_cache_key == _fkey and
                self._font_cache_text == self._full_text):
            font_size = self._font_cache_size
        else:
            font_size = calc_font_size(
                self._full_text, self._box_w, self._box_h,
                self._base_font_size)
            self._font_cache_key = _fkey
            self._font_cache_text = self._full_text
            self._font_cache_size = font_size
        visible = self._full_text[:self._visible_len]

        self.canvas.create_text(
            4, self._box_h // 2,
            text=visible, fill=self._fg,
            font=(self._font_name, font_size),
            anchor='w', width=self._box_w - 8,
            justify='left',
        )

    def update_content(self, text, base_font_size, bg_hex, fg, bg_image,
                       async_enabled=False, bilingual=False,
                       subtitle=False, bg_opacity=None):
        new_opacity = self._bg_opacity
        if bg_opacity is not None:
            try:
                new_opacity = max(0.05, min(1.0, float(bg_opacity)))
            except Exception:
                new_opacity = self._bg_opacity
        old_text = self._full_text
        text_changed = (old_text != text)
        style_changed = (
            self._base_font_size != base_font_size or
            self._bg_hex != bg_hex or
            self._fg != fg or
            self._bg_image is not bg_image or
            self._bilingual != bool(bilingual) or
            self._subtitle != bool(subtitle) or
            abs(new_opacity - self._bg_opacity) > 0.001
        )
        self._bg_opacity = new_opacity

        if not text_changed and not style_changed:
            return

        self._full_text = text
        self._base_font_size = base_font_size
        self._bg_hex = bg_hex
        self._fg = fg
        self._bg_image = bg_image
        self._bilingual = bool(bilingual)
        self._subtitle = bool(subtitle)

        self._apply_alpha()

        if text_changed:
            # 小幅改动（OCR 抖动、标点差异）不重启动画：
            # 否则每帧都从第一个字重新打字，观感就是「译文抖一下」。
            minor = _is_minor_text_change(old_text, text)
            if async_enabled and not minor:
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
            if (abs(x - old_x) < POSITION_QUANTIZE and
                    abs(y - old_y) < POSITION_QUANTIZE):
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
    - 段落级去重：用「归一化文本」做 key
    - 过滤短片段
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

        self.bar.bind('<ButtonPress-1>', self._on_drag_start)
        self.bar.bind('<B1-Motion>', self._on_drag)

        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = sw - width - 30
        y = sh - height - 80
        self.win.geometry(f"{width}x{height}+{x}+{y}")
        self.win.withdraw()

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

    @staticmethod
    def _normalize(text):
        if not text:
            return ''
        t = re.sub(r'\s+', '', text)
        t = re.sub(
            r"""[.,;:!?"'`~@#$%^&*()\[\]{}<>|\\/《》【】。，；：！？、""''…—\-_]+""",
            '', t,
        )
        return t.lower()

    def add_items(self, items):
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

        self.bilingual_gap = BILINGUAL_GAP

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

    def set_bilingual_gap(self, gap):
        """设置双语模式下译文相对原文的距离（像素）"""
        try:
            self.bilingual_gap = int(gap)
        except Exception:
            return

    def update(self, items, base_font_size=15):
        self.base_font_size = base_font_size
        if not items:
            # 管线在整屏扫描后没有译文时会发送空列表：
            # 清掉旧屏幕残留的覆盖窗口。
            self.clear()
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
            bg_opacity = item.get('bg_opacity')
            item_font = item.get('force_font_size', base_font_size)

            bilingual = item.get('bilingual', False)
            subtitle = item.get('subtitle', False)
            orig_w = item.get('orig_w', x2 - x1)
            orig_h = item.get('orig_h', y2 - y1)

            if bilingual:
                orig_y2 = item.get('orig_y2', y2)
                new_x1 = x1
                new_y1 = orig_y2 + self.bilingual_gap
                new_w = _calc_bilingual_width(
                    text, item_font, orig_w, subtitle=subtitle)
                new_h = _calc_bilingual_height(
                    text, new_w, item_font, subtitle=subtitle)
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
                    subtitle=subtitle,
                    bg_opacity=bg_opacity,
                )
                self.boxes[key] = box
            else:
                old.move_to(new_x1, new_y1, quantize=True)
                old.set_size(new_w, new_h, new_x1, new_y1)
                old.update_content(
                    text, item_font, bg_hex, fg, bg_image,
                    async_enabled=self.async_enabled,
                    bilingual=bilingual,
                    subtitle=subtitle,
                    bg_opacity=bg_opacity,
                )

        for key in list(self.boxes.keys()):
            if key not in new_map:
                self._release(self.boxes[key])
                del self.boxes[key]

    def set_bg_opacity(self, value):
        """立即更新所有已存在窗口的不透明度（滑杆实时生效）。"""
        for box in list(self.boxes.values()):
            try:
                box.set_bg_opacity(value)
            except Exception:
                pass

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