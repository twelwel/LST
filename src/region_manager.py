"""
区域范围管理模块（自适应屏幕 + 鼠标拖动 resize）
"""

import tkinter as tk
from tkinter import ttk, messagebox

DEFAULT_BORDER_COLOR = '#00FF00'
DEFAULT_BORDER_WIDTH = 3
DEFAULT_TRANSPARENT = 'magenta'


def get_preset_regions(screen_w, screen_h):
    """常用区域预设"""
    half_w = screen_w // 2
    half_h = screen_h // 2
    band_h = int(screen_h * 0.4)
    return {
        "全屏":   {"left": 0,      "top": 0,      "width": screen_w, "height": screen_h},
        "上半区": {"left": 0,      "top": 0,      "width": screen_w, "height": band_h},
        "下半区": {"left": 0,      "top": screen_h - band_h, "width": screen_w, "height": band_h},
        "左半区": {"left": 0,      "top": 0,      "width": half_w,   "height": screen_h},
        "右半区": {"left": half_w, "top": 0,      "width": half_w,   "height": screen_h},
        "左上":   {"left": 0,      "top": 0,      "width": half_w,   "height": half_h},
        "右上":   {"left": half_w, "top": 0,      "width": half_w,   "height": half_h},
        "左下":   {"left": 0,      "top": half_h, "width": half_w,   "height": half_h},
        "右下":   {"left": half_w, "top": half_h, "width": half_w,   "height": half_h},
    }


class RegionOverlay(tk.Toplevel):
    """
    绿色边框，可鼠标拖动缩放（类似 Windows 窗口 resize）
    - 靠近边框 → 光标变化
    - 拖边框/角 → 缩放
    - 无方块手柄，只有纯绿框
    """

    SENSITIVE = 10       # 鼠标感应范围（像素）
    MIN_W = 50           # 最小宽度
    MIN_H = 50           # 最小高度

    def __init__(self, master, region, on_change=None,
                 color=DEFAULT_BORDER_COLOR,
                 border_width=DEFAULT_BORDER_WIDTH,
                 transparent_color=DEFAULT_TRANSPARENT):
        super().__init__(master)
        self.color = color
        self.border_width = border_width
        self.transparent_color = transparent_color
        self.on_change = on_change

        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.attributes('-transparentcolor', transparent_color)
        self.config(bg=transparent_color)

        self.sw = self.winfo_screenwidth()
        self.sh = self.winfo_screenheight()

        self._region = dict(region)
        self._drag_mode = None
        self._drag_start_root = None
        self._drag_start_region = None
        self._last_cursor = 'arrow'

        # Canvas 占满整个 Toplevel
        self.canvas = tk.Canvas(
            self, highlightthickness=0, bg=transparent_color,
        )
        self.canvas.pack(fill='both', expand=True)

        # 一次性创建 4 条矩形（后续只改坐标，不重建）
        bw = border_width
        w0 = region['width'] + 2 * bw
        h0 = region['height'] + 2 * bw
        c = color
        self.rect_top = self.canvas.create_rectangle(
            0, 0, w0, bw, fill=c, outline='')
        self.rect_bottom = self.canvas.create_rectangle(
            0, h0 - bw, w0, h0, fill=c, outline='')
        self.rect_left = self.canvas.create_rectangle(
            0, bw, bw, h0 - bw, fill=c, outline='')
        self.rect_right = self.canvas.create_rectangle(
            w0 - bw, bw, w0, h0 - bw, fill=c, outline='')

        # 事件绑定
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Leave>', self._on_leave)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        # 应用初始位置
        self._apply_geometry()
        self._apply_rect_coords()

    # ================= 位置更新 =================

    def _apply_geometry(self):
        """更新 Toplevel 窗口的尺寸和位置"""
        offset = self.border_width
        x = self._region['left'] - offset
        y = self._region['top'] - offset
        w = self._region['width'] + 2 * offset
        h = self._region['height'] + 2 * offset
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _apply_rect_coords(self):
        """只更新 4 条矩形的坐标（拖动时用，比 delete 快很多）"""
        bw = self.border_width
        w = self._region['width'] + 2 * bw
        h = self._region['height'] + 2 * bw
        self.canvas.coords(self.rect_top,    0,      0,      w,     bw)
        self.canvas.coords(self.rect_bottom, 0,      h - bw, w,     h)
        self.canvas.coords(self.rect_left,   0,      bw,     bw,    h - bw)
        self.canvas.coords(self.rect_right,  w - bw, bw,     w,     h - bw)

    def set_region(self, region):
        self._region = dict(region)
        self._apply_geometry()
        self._apply_rect_coords()

    def set_border_color(self, color):
        self.color = color
        for rid in (self.rect_top, self.rect_bottom,
                    self.rect_left, self.rect_right):
            self.canvas.itemconfigure(rid, fill=color)

    # ================= 命中测试 =================

    def _hit_test(self, x, y):
        """
        判断鼠标位置对应的操作：
        nw / ne / sw / se / n / s / w / e / None
        """
        bw = self.border_width
        w = self._region['width'] + 2 * bw
        h = self._region['height'] + 2 * bw
        s = self.SENSITIVE

        near_left = x < bw + s
        near_right = x > w - bw - s
        near_top = y < bw + s
        near_bottom = y > h - bw - s

        # 四角优先
        if near_left and near_top:
            return 'nw'
        if near_right and near_top:
            return 'ne'
        if near_left and near_bottom:
            return 'sw'
        if near_right and near_bottom:
            return 'se'

        # 四边
        if near_top:
            return 'n'
        if near_bottom:
            return 's'
        if near_left:
            return 'w'
        if near_right:
            return 'e'

        return None

    def _cursor_for(self, mode):
        return {
            'nw': 'size_nw_se', 'se': 'size_nw_se',
            'ne': 'size_ne_sw', 'sw': 'size_ne_sw',
            'n':  'size_ns',    's':  'size_ns',
            'w':  'size_we',    'e':  'size_we',
        }.get(mode, 'arrow')

    # ================= 鼠标事件 =================

    def _on_hover(self, event):
        if self._drag_mode:
            return
        mode = self._hit_test(event.x, event.y)
        cur = self._cursor_for(mode)
        if cur != self._last_cursor:
            try:
                self.canvas.configure(cursor=cur)
                self._last_cursor = cur
            except Exception:
                pass

    def _on_leave(self, event):
        if not self._drag_mode:
            try:
                self.canvas.configure(cursor='arrow')
                self._last_cursor = 'arrow'
            except Exception:
                pass

    def _on_press(self, event):
        mode = self._hit_test(event.x, event.y)
        if not mode:
            return
        self._drag_mode = mode
        self._drag_start_root = (event.x_root, event.y_root)
        self._drag_start_region = dict(self._region)

    def _on_drag(self, event):
        if not self._drag_mode:
            return

        dx = event.x_root - self._drag_start_root[0]
        dy = event.y_root - self._drag_start_root[1]
        sr = self._drag_start_region
        mode = self._drag_mode

        l = sr['left']
        t = sr['top']
        r = l + sr['width']
        b = t + sr['height']

        if 'n' in mode:
            t += dy
        if 's' in mode:
            b += dy
        if 'w' in mode:
            l += dx
        if 'e' in mode:
            r += dx

        # 边界限制
        l = max(0, min(l, self.sw - self.MIN_W))
        t = max(0, min(t, self.sh - self.MIN_H))
        r = max(l + self.MIN_W, min(r, self.sw))
        b = max(t + self.MIN_H, min(b, self.sh))

        self._region = {
            'left': int(l),
            'top': int(t),
            'width': int(r - l),
            'height': int(b - t),
        }

        # 只更新位置 + 坐标，不重建
        self._apply_geometry()
        self._apply_rect_coords()

    def _on_release(self, event):
        if not self._drag_mode:
            return
        self._drag_mode = None
        if self.on_change:
            self.on_change(dict(self._region))


class RegionConfigDialog(tk.Toplevel):
    """区域配置对话框"""

    def __init__(self, master, current_region, on_apply):
        super().__init__(master)
        self.title("修改截图区域")
        self.attributes('-topmost', True)
        self.resizable(False, False)
        self.on_apply = on_apply

        self.screen_w = self.winfo_screenwidth()
        self.screen_h = self.winfo_screenheight()

        dlg_w = min(620, int(self.screen_w * 0.8))
        dlg_w = max(480, dlg_w)
        dlg_h = min(660, int(self.screen_h * 0.75))
        dlg_h = max(420, dlg_h)

        pos_x = (self.screen_w - dlg_w) // 2
        pos_y = (self.screen_h - dlg_h) // 2
        self.geometry(f"{dlg_w}x{dlg_h}+{pos_x}+{pos_y}")

        self._dlg_w = dlg_w

        self.ranges = {
            'top':    (0, self.screen_h),
            'left':   (0, self.screen_w),
            'width':  (50, self.screen_w),
            'height': (50, self.screen_h),
        }
        self.field_labels = {
            'top':    '顶部 Y 坐标',
            'left':   '左侧 X 坐标',
            'width':  '宽度',
            'height': '高度',
        }
        self.entry_vars = {}
        self.slider_vars = {}
        self.sliders = {}

        self._build_ui(current_region)

    def _build_ui(self, current_region):
        main = ttk.Frame(self, padding=10)
        main.pack(fill='both', expand=True)

        info = tk.Label(
            main,
            text=f"屏幕分辨率：{self.screen_w} × {self.screen_h}",
            font=('Microsoft YaHei', 9),
            fg='#666666',
        )
        info.pack(anchor='w', pady=(0, 6))

        quick_frame = ttk.LabelFrame(main, text="快捷区域", padding=6)
        quick_frame.pack(fill='x', pady=(0, 8))

        presets = get_preset_regions(self.screen_w, self.screen_h)
        names = list(presets.keys())

        for i, name in enumerate(names):
            row = i // 3
            col = i % 3
            btn = ttk.Button(
                quick_frame, text=name, width=9,
                command=lambda n=name: self._apply_preset(presets[n]),
            )
            btn.grid(row=row, column=col, padx=3, pady=2, sticky='ew')

        for c in range(3):
            quick_frame.grid_columnconfigure(c, weight=1)

        params_frame = ttk.LabelFrame(main, text="精确设置", padding=8)
        params_frame.pack(fill='both', expand=True)

        slider_len = max(180, self._dlg_w - 260)

        for i, key in enumerate(['top', 'left', 'width', 'height']):
            self._build_slider_row(
                params_frame, key, i,
                current_region.get(key, 0),
                slider_len,
            )

        self.preview_label = tk.Label(
            main, text="", font=('Microsoft YaHei', 9), fg='#444444',
        )
        self.preview_label.pack(pady=(6, 3))
        self._update_preview()

        btn_frame = ttk.Frame(main)
        btn_frame.pack(pady=(6, 0))

        ttk.Button(btn_frame, text="应用", width=10,
                   command=self._apply).pack(side='left', padx=6)
        ttk.Button(btn_frame, text="取消", width=10,
                   command=self.destroy).pack(side='left', padx=6)

        self.bind('<Return>', lambda e: self._apply())
        self.bind('<Escape>', lambda e: self.destroy())

    def _build_slider_row(self, parent, key, row, value, slider_len):
        ttk.Label(
            parent, text=self.field_labels[key] + "：",
            font=('Microsoft YaHei', 10),
        ).grid(row=row, column=0, sticky='w', pady=6, padx=(0, 4))

        lo, hi = self.ranges[key]
        slider_var = tk.IntVar(value=int(value))
        slider = ttk.Scale(
            parent, from_=lo, to=hi, orient='horizontal',
            variable=slider_var, length=slider_len,
            command=lambda v, k=key: self._on_slider_move(k, v),
        )
        slider.grid(row=row, column=1, sticky='ew', padx=4)
        self.slider_vars[key] = slider_var
        self.sliders[key] = slider

        entry_var = tk.StringVar(value=str(int(value)))
        entry = ttk.Entry(
            parent, textvariable=entry_var, width=7, justify='right',
        )
        entry.grid(row=row, column=2, sticky='w', padx=(4, 0))
        entry.bind('<Return>', lambda e, k=key: self._on_entry_commit(k))
        entry.bind('<FocusOut>', lambda e, k=key: self._on_entry_commit(k))
        self.entry_vars[key] = entry_var

        ttk.Label(parent, text="px",
                  font=('Microsoft YaHei', 9),
                  foreground='gray').grid(row=row, column=3, padx=(3, 0))

        parent.grid_columnconfigure(1, weight=1)

    def _on_slider_move(self, key, value):
        v = int(float(value))
        self.entry_vars[key].set(str(v))
        self._update_preview()

    def _on_entry_commit(self, key):
        try:
            v = int(self.entry_vars[key].get())
        except ValueError:
            self.entry_vars[key].set(str(self.slider_vars[key].get()))
            return

        lo, hi = self.ranges[key]
        v = max(lo, min(hi, v))
        self.slider_vars[key].set(v)
        self.entry_vars[key].set(str(v))
        self._update_preview()

    def _update_preview(self):
        try:
            r = self._read_current()
            self.preview_label.configure(
                text=f"预览：left={r['left']}  top={r['top']}  "
                     f"width={r['width']}  height={r['height']}"
            )
        except Exception:
            pass

    def _apply_preset(self, preset):
        for key in ['top', 'left', 'width', 'height']:
            v = preset[key]
            self.slider_vars[key].set(v)
            self.entry_vars[key].set(str(v))
        self._update_preview()

    def _read_current(self):
        return {
            'top':    int(self.entry_vars['top'].get()),
            'left':   int(self.entry_vars['left'].get()),
            'width':  int(self.entry_vars['width'].get()),
            'height': int(self.entry_vars['height'].get()),
        }

    def _apply(self):
        try:
            new_region = self._read_current()

            if new_region['width'] <= 0 or new_region['height'] <= 0:
                raise ValueError("宽高必须大于 0")
            if new_region['top'] < 0 or new_region['left'] < 0:
                raise ValueError("坐标不能为负数")
            if new_region['left'] + new_region['width'] > self.screen_w:
                raise ValueError(
                    f"区域超出屏幕宽度\n"
                    f"left({new_region['left']}) + "
                    f"width({new_region['width']}) > 屏宽({self.screen_w})"
                )
            if new_region['top'] + new_region['height'] > self.screen_h:
                raise ValueError(
                    f"区域超出屏幕高度\n"
                    f"top({new_region['top']}) + "
                    f"height({new_region['height']}) > 屏高({self.screen_h})"
                )

            self.on_apply(new_region)
            self.destroy()
        except ValueError as e:
            messagebox.showerror("输入错误", str(e), parent=self)


class RegionManager:
    """区域管理器对外接口"""

    def __init__(self, master, region,
                 border_color=DEFAULT_BORDER_COLOR,
                 border_width=DEFAULT_BORDER_WIDTH,
                 on_change=None):
        self.master = master
        self.region = dict(region)
        self.on_change = on_change

        self.overlay = RegionOverlay(
            master, self.region,
            color=border_color,
            border_width=border_width,
            on_change=self._on_drag_end,
        )

    def _on_drag_end(self, new_region):
        self.region = dict(new_region)
        if self.on_change:
            self.on_change(dict(self.region))

    def get_region(self):
        return dict(self.region)

    def update_region(self, new_region):
        for k in ('top', 'left', 'width', 'height'):
            if k in new_region:
                self.region[k] = int(new_region[k])
        self.overlay.set_region(self.region)
        if self.on_change:
            self.on_change(self.get_region())

    def set_border_visible(self, visible):
        if visible:
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def open_config(self):
        RegionConfigDialog(self.master, self.region, self.update_region)

    def destroy(self):
        try:
            self.overlay.destroy()
        except Exception:
            pass