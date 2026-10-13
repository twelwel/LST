"""实时屏幕翻译 - UI 模块
包含主窗口和所有界面组件（带版本号 + 配置持久化）
"""

import queue
import time
import threading

import customtkinter as ctk

from config import (
    INITIAL_REGION, INITIAL_FONT_SIZE,
    FONT_NAME, REUSE_MIN_OVERLAP,
    INITIAL_HOTKEYS, HOTKEY_LABELS,
    SCAN_MIN, SCAN_MAX, FONT_MIN, FONT_MAX,
    FRAME_SIMILARITY_THRESHOLD, POSITION_QUANTIZE,
    DEFAULT_USE_INPAINTING,
    PERFORMANCE_MODES, DEFAULT_MODE,
    IDLE_THRESHOLD_FOR_SLOWDOWN, IDLE_SLOWDOWN_FACTOR,
    DEFAULT_ENABLE_FRAME_DIFF, DEFAULT_ENABLE_FILTER,
    DEFAULT_ENABLE_BLOCK_OCR,
    DEFAULT_ENABLE_ASYNC_TRANSLATE, DEFAULT_ENABLE_WINDOW_POOL,
    DEFAULT_ENABLE_MULTITHREAD_OCR, MULTITHREAD_OCR_POOL_SIZE,
    DEFAULT_USE_GPU,
)
from ocr_engine import OCREngine
from cache import ParagraphCache
from pipeline import Pipeline, AppState
from overlay import OverlayWindow
from region_manager import RegionManager
from translator import Translator
from hotkeys import HotkeyManager
from inpaint import CV2_AVAILABLE
from logger import get_logger, open_log_dir, clear_logs
from scroll_detector import ScrollDetector
from config_manager import get_config, open_config_dir, get_config_path
from version import get_window_title, __version__, APP_NAME


log = get_logger('ui')


# ==================== 常量 ====================
CACHE_MIN = 100
CACHE_MAX = 2000
CACHE_DEFAULT = 500
TICK_MS = 16
BOUNDARY_PADDING = 10


# ==================== 可折叠面板 ====================

class CollapsibleFrame(ctk.CTkFrame):
    def __init__(self, master, title="标题", expanded=True, **kwargs):
        super().__init__(master, **kwargs)
        self._title = title
        self._expanded = expanded

        self.title_btn = ctk.CTkButton(
            self,
            text=self._make_text(),
            anchor='w',
            height=32,
            fg_color=('gray85', 'gray22'),
            hover_color=('gray75', 'gray32'),
            text_color=('black', 'white'),
            command=self.toggle,
            font=(FONT_NAME, 12, 'bold'),
        )
        self.title_btn.pack(fill='x', padx=2, pady=2)

        self.content = ctk.CTkFrame(self, fg_color='transparent')
        if expanded:
            self.content.pack(fill='x', padx=2, pady=(0, 5))

    def _make_text(self):
        arrow = "▼" if self._expanded else "▶"
        return f"{arrow}   {self._title}"

    def toggle(self):
        self._expanded = not self._expanded
        self.title_btn.configure(text=self._make_text())
        if self._expanded:
            self.content.pack(fill='x', padx=2, pady=(0, 5))
        else:
            self.content.pack_forget()


# ==================== 主窗口 ====================

class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 窗口标题带版本
        self.title(get_window_title())

        # ============ 加载用户配置 ============
        self.user_cfg = get_config()
        log.info(f"配置文件：{get_config_path()}")

        geo = self.user_cfg.get('window_geometry', "740x820+50+50")
        self.geometry(geo)
        self.minsize(660, 600)
        self.attributes('-topmost', True)

        # 读取配置
        initial_region = self.user_cfg.get_region()
        initial_font = self.user_cfg.get('font_size', INITIAL_FONT_SIZE)
        saved_mode = self.user_cfg.get('performance_mode', DEFAULT_MODE)
        if saved_mode not in PERFORMANCE_MODES:
            saved_mode = DEFAULT_MODE
        saved_hotkeys = self.user_cfg.get_hotkeys()

        saved_use_gpu = self.user_cfg.get('use_gpu', DEFAULT_USE_GPU)
        saved_mt = self.user_cfg.get('use_multithread_ocr',
                                     DEFAULT_ENABLE_MULTITHREAD_OCR)
        saved_pool_size = self.user_cfg.get('multithread_pool_size',
                                            MULTITHREAD_OCR_POOL_SIZE)

        saved_frame_diff = self.user_cfg.get('enable_frame_diff',
                                             DEFAULT_ENABLE_FRAME_DIFF)
        saved_block = self.user_cfg.get('enable_block_ocr',
                                        DEFAULT_ENABLE_BLOCK_OCR)
        saved_filter = self.user_cfg.get('enable_filter',
                                         DEFAULT_ENABLE_FILTER)
        saved_async = self.user_cfg.get('enable_async_translate',
                                        DEFAULT_ENABLE_ASYNC_TRANSLATE)
        saved_wpool = self.user_cfg.get('enable_window_pool',
                                        DEFAULT_ENABLE_WINDOW_POOL)

        saved_scan = self.user_cfg.get(
            'scan_interval',
            PERFORMANCE_MODES[saved_mode]['scan_interval'],
        )
        saved_inpaint = self.user_cfg.get('use_inpainting',
                                          DEFAULT_USE_INPAINTING)
        saved_cache_max = self.user_cfg.get('cache_max_size', CACHE_DEFAULT)
        saved_auto_clean = self.user_cfg.get('auto_clean', True)
        saved_freeze_mode = self.user_cfg.get('freeze_mode', 'follow')

        log.info(f"========== 启动 v{__version__} ==========")

        # 互斥：GPU 优先
        if saved_use_gpu and saved_mt:
            saved_mt = False

        # ============ 装配核心组件 ============
        self.translator = Translator()
        self.ocr_engine = OCREngine()

        pool_size = saved_pool_size if saved_mt else 1
        self._actual_ocr_mode = self.ocr_engine.init(
            use_gpu=saved_use_gpu, pool_size=pool_size,
        )
        log.info(f"OCR 引擎初始化完成：{self._actual_ocr_mode} × {pool_size}")

        self.cache = ParagraphCache(
            min_overlap=REUSE_MIN_OVERLAP,
            max_size=saved_cache_max,
        )
        self.cache.set_auto_clean(saved_auto_clean)

        default_cfg = PERFORMANCE_MODES[saved_mode]
        self.app_state = AppState(
            initial_region,
            frame_similarity_threshold=FRAME_SIMILARITY_THRESHOLD,
            position_quantize=POSITION_QUANTIZE,
            use_inpainting=saved_inpaint,
            mode_cfg=default_cfg,
            enable_frame_diff=saved_frame_diff,
            enable_filter=saved_filter,
            enable_block_ocr=saved_block,
            enable_async_translate=saved_async,
            enable_window_pool=saved_wpool,
            enable_multithread_ocr=saved_mt,
        )
        self.app_state.performance_mode = saved_mode
        self.app_state.base_font_size = initial_font
        self.app_state.freeze_overlay = (saved_freeze_mode == 'pin')
        self.app_state.scan_interval = saved_scan

        self.pipeline = Pipeline(
            self.ocr_engine, self.cache, self.translator, self.app_state,
        )
        self.pipeline.start()

        self.overlay = OverlayWindow(self, font_name=FONT_NAME)
        self.overlay.set_pool_enabled(saved_wpool)

        self.region_manager = RegionManager(
            self, initial_region,
            border_color='#00FF00', border_width=3,
            on_change=self.on_region_changed,
        )

        # 滚动检测独立线程
        self.scroll_detector = ScrollDetector(
            get_region_fn=self.app_state.snapshot_region,
            on_scroll=self._on_scroll_detected,
            is_active_fn=self._is_scroll_active,
        )
        self.scroll_detector.start()

        # 热键
        self.hotkeys = dict(saved_hotkeys)
        self.hk_mgr = HotkeyManager()
        self._register_all_hotkeys()

        self._last_capture = 0
        self._last_cache_ui_update = 0
        self._border_visible = True
        self._current_mode = saved_mode
        self._freeze_mode = saved_freeze_mode

        # UI
        self._build_ui()

        self.after(100, self._sync_gpu_ui)

        # 恢复 UI 状态
        self._restore_ui_from_config(
            saved_mode, saved_scan, initial_font, saved_inpaint,
            saved_frame_diff, saved_block, saved_filter, saved_async,
            saved_wpool, saved_cache_max, saved_auto_clean,
            saved_freeze_mode, saved_mt,
        )

        self.after(TICK_MS, self._tick)

    # ================= 恢复 UI 状态 =================

    def _restore_ui_from_config(self, mode, scan, font, inpaint,
                                frame_diff, block, filt, async_t,
                                wpool, cache_max, auto_clean,
                                freeze_mode, mt_ocr):
        try:
            self.mode_selector.set(PERFORMANCE_MODES[mode]['name'])
            self._update_mode_desc(mode)

            self.scan_slider.set(scan)
            self.scan_value_label.configure(text=f"{scan:.2f} s")
            self.font_slider.set(font)
            self.font_value_label.configure(text=str(font))

            if inpaint and CV2_AVAILABLE:
                self.inpaint_switch.select()

            if frame_diff:
                self.frame_diff_switch.select()
            else:
                self.frame_diff_switch.deselect()

            if block:
                self.block_ocr_switch.select()
            else:
                self.block_ocr_switch.deselect()

            if filt:
                self.filter_switch.select()
            else:
                self.filter_switch.deselect()

            if async_t:
                self.async_translate_switch.select()
            else:
                self.async_translate_switch.deselect()

            if wpool:
                self.window_pool_switch.select()
            else:
                self.window_pool_switch.deselect()

            self.cache_limit_slider.set(cache_max)
            self.cache_limit_label.configure(text=str(cache_max))
            self.cache.max_size = cache_max

            if auto_clean:
                self.auto_clean_switch.select()
            else:
                self.auto_clean_switch.deselect()

            if freeze_mode == 'pin':
                self.btn_freeze.configure(
                    text="🔄 恢复跟随",
                    fg_color='#ff9900', hover_color='#e68a00',
                )
                self.bottom_label.configure(
                    text="📌 已钉住（配置恢复）：译文固定")

            if mt_ocr:
                self.multithread_ocr_switch.select()

        except Exception as e:
            log.exception(f"[UI] 恢复配置失败：{e}")

    # ================= 滚动回调 =================

    def _is_scroll_active(self):
        return (self.app_state.running
                and not self.app_state.freeze_overlay)

    def _on_scroll_detected(self, dy):
        self.app_state.scroll_boost = True
        try:
            self.after(0, lambda: self._apply_scroll(dy))
        except Exception:
            pass

    def _apply_scroll(self, dy):
        if self.app_state.freeze_overlay:
            return
        try:
            self.overlay.scroll_by(dy)
        except Exception as e:
            log.exception(f"[UI] 滚动平移失败：{e}")

    # ================= 热键 =================

    def _get_callback(self, name):
        return {
            'toggle': self.toggle_running,
            'config': self.open_config,
            'border': self.toggle_border,
            'freeze': self.toggle_freeze,
            'quit':   self.quit_app,
        }[name]

    def _register_all_hotkeys(self):
        for name, hk in self.hotkeys.items():
            self.hk_mgr.register(name, hk, self._get_callback(name))

    # ================= UI 构建 =================

    def _build_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(fill='x', padx=10, pady=(10, 5))

        self.status_label = ctk.CTkLabel(
            top, text="状态：已暂停",
            font=(FONT_NAME, 13, 'bold'), anchor='w',
        )
        self.status_label.pack(side='left', padx=10)

        # 版本号（右上角）
        self.version_label = ctk.CTkLabel(
            top, text=f"{APP_NAME} v{__version__}",
            font=(FONT_NAME, 10), text_color='#3a86ff',
        )
        self.version_label.pack(side='right', padx=10)

        self.hotkey_hint = ctk.CTkLabel(
            top, text='', font=(FONT_NAME, 10), text_color='gray',
        )
        self.hotkey_hint.pack(side='right', padx=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill='x', padx=10, pady=5)

        self.btn_toggle = ctk.CTkButton(
            btn_frame, text="▶ 开始", width=90,
            command=self.toggle_running,
        )
        self.btn_toggle.pack(side='left', padx=3, pady=10)

        ctk.CTkButton(
            btn_frame, text="清空译文", width=80,
            command=self.clear_overlay,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame, text="修改区域", width=80,
            command=self.open_config,
        ).pack(side='left', padx=3)

        self.btn_border = ctk.CTkButton(
            btn_frame, text="隐藏边框", width=80,
            command=self.toggle_border,
        )
        self.btn_border.pack(side='left', padx=3)

        self.btn_freeze = ctk.CTkButton(
            btn_frame, text="📌 钉住译文", width=100,
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=self.toggle_freeze,
        )
        self.btn_freeze.pack(side='left', padx=3)

        btn_frame2 = ctk.CTkFrame(self)
        btn_frame2.pack(fill='x', padx=10, pady=5)

        ctk.CTkButton(
            btn_frame2, text="API 设置", width=85,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.open_api_config,
        ).pack(side='left', padx=3, pady=6)

        ctk.CTkButton(
            btn_frame2, text="打开日志", width=85,
            fg_color='#17a2b8', hover_color='#138496',
            command=self.open_logs,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="清空日志", width=85,
            fg_color='#dc3545', hover_color='#c82333',
            command=self.clear_logs_ui,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="打开配置", width=85,
            fg_color='#6f42c1', hover_color='#5a35a0',
            command=self.open_config_dir_ui,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="关于", width=70,
            fg_color='#495057', hover_color='#343a40',
            command=self.open_about,
        ).pack(side='left', padx=3)

        self.region_label = ctk.CTkLabel(
            self, text='', font=(FONT_NAME, 10), text_color='gray',
        )
        self.region_label.pack(pady=(0, 5))
        self._update_region_label()

        self.scroll_frame = ctk.CTkScrollableFrame(
            self, fg_color='transparent',
        )
        self.scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.display_panel = CollapsibleFrame(
            self.scroll_frame, title="显示设置", expanded=True,
        )
        self.display_panel.pack(fill='x', padx=5, pady=(5, 3))
        self._build_display_tab(self.display_panel.content)

        self.hotkey_panel = CollapsibleFrame(
            self.scroll_frame, title="热键设置", expanded=False,
        )
        self.hotkey_panel.pack(fill='x', padx=5, pady=3)
        self._build_hotkey_tab(self.hotkey_panel.content)

        self.bottom_label = ctk.CTkLabel(
            self, text="准备就绪", font=(FONT_NAME, 10), text_color='gray',
        )
        self.bottom_label.pack(side='bottom', pady=5)

        self._refresh_hotkey_hint()

    def _build_display_tab(self, parent):
        ctk.CTkLabel(
            parent, text="性能模式：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=0, column=0, sticky='w', padx=(15, 5), pady=(15, 8))

        self.mode_selector = ctk.CTkSegmentedButton(
            parent,
            values=["快", "中", "慢"],
            command=self._on_mode_change,
            font=(FONT_NAME, 11),
        )
        self.mode_selector.set(PERFORMANCE_MODES[DEFAULT_MODE]['name'])
        self.mode_selector.grid(
            row=0, column=1, columnspan=2,
            sticky='ew', padx=10, pady=(15, 8),
        )

        self.mode_desc_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
        )
        self.mode_desc_label.grid(
            row=1, column=0, columnspan=3,
            sticky='w', padx=(15, 15), pady=(0, 10),
        )
        self._update_mode_desc(DEFAULT_MODE)

        ctk.CTkLabel(
            parent, text="加速方式（二选一）：",
            font=(FONT_NAME, 11, 'bold'),
        ).grid(row=2, column=0, columnspan=3,
               sticky='w', padx=(15, 5), pady=(8, 4))

        self.gpu_switch = ctk.CTkSwitch(
            parent, text="GPU 加速（NVIDIA 显卡）",
            command=self._on_gpu_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_USE_GPU:
            self.gpu_switch.select()
        self.gpu_switch.grid(
            row=3, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.multithread_ocr_switch = ctk.CTkSwitch(
            parent,
            text=f"多线程 OCR（{MULTITHREAD_OCR_POOL_SIZE} 引擎并行，CPU）",
            command=self._on_multithread_ocr_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_MULTITHREAD_OCR:
            self.multithread_ocr_switch.select()
        self.multithread_ocr_switch.grid(
            row=4, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.gpu_status_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
        )
        self.gpu_status_label.grid(
            row=5, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 10),
        )

        ctk.CTkLabel(
            parent, text="优化功能：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=6, column=0, sticky='w', padx=(15, 5), pady=(8, 4))

        self.frame_diff_switch = ctk.CTkSwitch(
            parent, text="帧差检测（静止时跳过）",
            command=self._on_frame_diff_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FRAME_DIFF:
            self.frame_diff_switch.select()
        self.frame_diff_switch.grid(
            row=7, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.block_ocr_switch = ctk.CTkSwitch(
            parent, text="分块 OCR（只识别变化的区域）",
            command=self._on_block_ocr_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_BLOCK_OCR:
            self.block_ocr_switch.select()
        self.block_ocr_switch.grid(
            row=8, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.filter_switch = ctk.CTkSwitch(
            parent, text="文字过滤（过滤异常框）",
            command=self._on_filter_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FILTER:
            self.filter_switch.select()
        self.filter_switch.grid(
            row=9, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.async_translate_switch = ctk.CTkSwitch(
            parent, text="异步翻译（不阻塞 OCR）",
            command=self._on_async_translate_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_ASYNC_TRANSLATE:
            self.async_translate_switch.select()
        self.async_translate_switch.grid(
            row=10, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4),
        )

        self.window_pool_switch = ctk.CTkSwitch(
            parent, text="窗口池（复用译文窗口）",
            command=self._on_window_pool_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_WINDOW_POOL:
            self.window_pool_switch.select()
        self.window_pool_switch.grid(
            row=11, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 10),
        )

        ctk.CTkLabel(
            parent, text="刷新间隔 (秒)：", font=(FONT_NAME, 11),
        ).grid(row=12, column=0, sticky='w', padx=(15, 5), pady=8)

        self.scan_value_label = ctk.CTkLabel(
            parent,
            text=f"{PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval']:.2f} s",
            width=60, font=(FONT_NAME, 11),
        )
        self.scan_value_label.grid(row=12, column=2, sticky='e', padx=(5, 15))

        n_steps = int((SCAN_MAX - SCAN_MIN) / 0.05)
        self.scan_slider = ctk.CTkSlider(
            parent, from_=SCAN_MIN, to=SCAN_MAX,
            number_of_steps=n_steps,
            command=self._on_scan_change,
        )
        self.scan_slider.set(PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval'])
        self.scan_slider.grid(row=12, column=1, sticky='ew', padx=10)

        ctk.CTkLabel(
            parent, text="文字大小：", font=(FONT_NAME, 11),
        ).grid(row=13, column=0, sticky='w', padx=(15, 5), pady=8)

        self.font_value_label = ctk.CTkLabel(
            parent, text=str(INITIAL_FONT_SIZE), width=60,
            font=(FONT_NAME, 11),
        )
        self.font_value_label.grid(row=13, column=2, sticky='e', padx=(5, 15))

        self.font_slider = ctk.CTkSlider(
            parent, from_=FONT_MIN, to=FONT_MAX,
            number_of_steps=FONT_MAX - FONT_MIN,
            command=self._on_font_change,
        )
        self.font_slider.set(INITIAL_FONT_SIZE)
        self.font_slider.grid(row=13, column=1, sticky='ew', padx=10)

        ctk.CTkLabel(
            parent, text="图像修复（类原生文字）：", font=(FONT_NAME, 11),
        ).grid(row=14, column=0, sticky='w', padx=(15, 5), pady=(15, 8))

        self.inpaint_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_inpaint_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_USE_INPAINTING and CV2_AVAILABLE:
            self.inpaint_switch.select()
        self.inpaint_switch.grid(row=14, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.inpaint_switch.configure(state='disabled')

        ctk.CTkLabel(
            parent, text="自动清理缓存：", font=(FONT_NAME, 11),
        ).grid(row=15, column=0, sticky='w', padx=(15, 5), pady=(15, 8))

        self.auto_clean_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_auto_clean_toggle,
            onvalue=True, offvalue=False,
        )
        self.auto_clean_switch.select()
        self.auto_clean_switch.grid(row=15, column=1, sticky='w', padx=10)

        ctk.CTkLabel(
            parent, text="缓存上限：", font=(FONT_NAME, 11),
        ).grid(row=16, column=0, sticky='w', padx=(15, 5), pady=8)

        self.cache_limit_label = ctk.CTkLabel(
            parent, text=str(CACHE_DEFAULT), width=60,
            font=(FONT_NAME, 11),
        )
        self.cache_limit_label.grid(row=16, column=2, sticky='e', padx=(5, 15))

        self.cache_limit_slider = ctk.CTkSlider(
            parent, from_=CACHE_MIN, to=CACHE_MAX,
            number_of_steps=(CACHE_MAX - CACHE_MIN) // 50,
            command=self._on_cache_limit_change,
        )
        self.cache_limit_slider.set(CACHE_DEFAULT)
        self.cache_limit_slider.grid(row=16, column=1, sticky='ew', padx=10)

        cache_row = ctk.CTkFrame(parent, fg_color='transparent')
        cache_row.grid(row=17, column=0, columnspan=3,
                       sticky='ew', padx=(15, 15), pady=(5, 15))

        self.cache_size_label = ctk.CTkLabel(
            cache_row, text="缓存条目：0",
            font=(FONT_NAME, 10), text_color='gray',
        )
        self.cache_size_label.pack(side='left')

        ctk.CTkButton(
            cache_row, text="立即清理", width=80,
            command=self._manual_clear_cache,
        ).pack(side='right')

        parent.grid_columnconfigure(1, weight=1)

    def _build_hotkey_tab(self, parent):
        self.hotkey_entries = {}
        for i, (key, label) in enumerate(HOTKEY_LABELS.items()):
            ctk.CTkLabel(
                parent, text=f"{label}：", font=(FONT_NAME, 11),
            ).grid(row=i, column=0, sticky='w', padx=(15, 5), pady=8)

            entry = ctk.CTkEntry(parent, width=180)
            entry.insert(0, self.hotkeys[key])
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=8)
            self.hotkey_entries[key] = entry

            ctk.CTkButton(
                parent, text="捕获", width=70,
                command=lambda k=key: self._capture_hotkey(k),
            ).grid(row=i, column=2, sticky='w', padx=5, pady=8)

        ctk.CTkButton(
            parent, text="应用所有热键", width=140,
            command=self._apply_hotkeys,
        ).grid(row=len(HOTKEY_LABELS), column=0, columnspan=3,
               pady=(15, 10))

        parent.grid_columnconfigure(1, weight=1)

    # ================= 日志/配置/关于 =================

    def open_logs(self):
        open_log_dir()
        self.bottom_label.configure(text="已打开日志目录")

    def clear_logs_ui(self):
        clear_logs()
        self.bottom_label.configure(text="日志已清空")
        log.info("用户清空日志")

    def open_config_dir_ui(self):
        open_config_dir()
        self.bottom_label.configure(text="已打开配置目录")

    def open_about(self):
        """打开关于窗口"""
        try:
            from about_dialog import AboutDialog
            dlg = AboutDialog(self)
            dlg.grab_set()
        except Exception as e:
            log.exception(f"[关于] 打开失败：{e}")

    # ================= 跟随 / 钉住 =================

    def toggle_freeze(self):
        if self._freeze_mode == 'follow':
            self._freeze_mode = 'pin'
            self.app_state.freeze_overlay = True
            self.app_state.scroll_boost = False
            self.scroll_detector.reset()
            self.btn_freeze.configure(
                text="🔄 恢复跟随",
                fg_color='#ff9900', hover_color='#e68a00',
            )
            self.bottom_label.configure(
                text="📌 已钉住：译文固定，滚页不动")
            log.info("[模式] 切换到钉住")
        else:
            self._freeze_mode = 'follow'
            self.app_state.freeze_overlay = False

            try:
                while True:
                    self.pipeline.result_queue.get_nowait()
            except queue.Empty:
                pass
            self.overlay.clear()
            self.app_state.reset_signature()
            self.scroll_detector.reset()

            self.btn_freeze.configure(
                text="📌 钉住译文",
                fg_color='#3a86ff', hover_color='#2a6fd8',
            )
            self.bottom_label.configure(
                text="🔄 已恢复跟随：译文随原文滚动")
            log.info("[模式] 切换到跟随")

        self.user_cfg.set('freeze_mode', self._freeze_mode)
        self.user_cfg.save()

    # ================= GPU / 多线程 =================

    def _sync_gpu_ui(self):
        gpu_available = self.ocr_engine.check_gpu_available()
        mode = self._actual_ocr_mode

        if mode == 'gpu':
            self.gpu_status_label.configure(
                text="✅ 已启用 GPU 加速（多线程 OCR 已禁用）",
                text_color='#00a000')
            self.multithread_ocr_switch.configure(state='disabled')
            if self.multithread_ocr_switch.get():
                self.multithread_ocr_switch.deselect()
                self.app_state.enable_multithread_ocr = False

        elif not gpu_available:
            self.gpu_switch.configure(state='disabled')
            if self.gpu_switch.get():
                self.gpu_switch.deselect()
            self.gpu_status_label.configure(
                text="❌ 未检测到 CUDA 环境，可使用多线程 OCR（CPU）",
                text_color='#cc0000')
            self.multithread_ocr_switch.configure(state='normal')

        else:
            self.gpu_status_label.configure(
                text="ℹ️ 选择加速方式：GPU 或多线程 OCR（二选一）",
                text_color='#666666')
            self.multithread_ocr_switch.configure(state='normal')

    def _on_gpu_toggle(self):
        enabled = bool(self.gpu_switch.get())

        if not enabled:
            self.bottom_label.configure(text="正在切换到 CPU...")
            self.update_idletasks()
            self._restart_ocr_engine(use_gpu=False)
            self.multithread_ocr_switch.configure(state='normal')
            return

        if not self.ocr_engine.check_gpu_available():
            self.gpu_switch.deselect()
            self.gpu_status_label.configure(
                text="❌ 未检测到 CUDA 环境", text_color='#cc0000')
            self.bottom_label.configure(text="GPU 不可用")
            return

        if self.multithread_ocr_switch.get():
            self.multithread_ocr_switch.deselect()
            self.app_state.enable_multithread_ocr = False
            log.info("[互斥] GPU 开启，自动关闭多线程 OCR")

        self.multithread_ocr_switch.configure(state='disabled')

        self.bottom_label.configure(text="正在初始化 GPU 引擎...")
        self.gpu_status_label.configure(
            text="⏳ 初始化中...", text_color='#666666')
        self.update_idletasks()
        self._restart_ocr_engine(use_gpu=True)

    def _on_multithread_ocr_toggle(self):
        enabled = bool(self.multithread_ocr_switch.get())

        if enabled and self.gpu_switch.get():
            self.gpu_switch.deselect()
            log.info("[互斥] 多线程 OCR 开启，自动关闭 GPU")
            self.gpu_switch.configure(state='disabled')

            self.bottom_label.configure(text="正在切换到 CPU 多线程引擎...")
            self.update_idletasks()

            pool_size = MULTITHREAD_OCR_POOL_SIZE
            self._reinit_ocr_pool(use_gpu=False, pool_size=pool_size)
            self.app_state.enable_multithread_ocr = True

            self.gpu_status_label.configure(
                text=f"✅ 已启用 CPU 多线程 OCR（{pool_size} 引擎）",
                text_color='#00a000')
            self.bottom_label.configure(
                text=f"多线程 OCR：已开启（{pool_size} 引擎）")
            log.info(f"[开关] 多线程 OCR：已开启（{pool_size} 引擎）")

            self.user_cfg.set('use_multithread_ocr', True)
            self.user_cfg.set('use_gpu', False)
            self.user_cfg.save()
            return

        if enabled:
            if self.ocr_engine.check_gpu_available():
                self.gpu_switch.configure(state='normal')
            self.app_state.enable_multithread_ocr = True
            pool_size = MULTITHREAD_OCR_POOL_SIZE
            self._reinit_ocr_pool(use_gpu=False, pool_size=pool_size)
            self.gpu_status_label.configure(
                text=f"✅ 已启用 CPU 多线程 OCR（{pool_size} 引擎）",
                text_color='#00a000')
            self.bottom_label.configure(
                text=f"多线程 OCR：已开启（{pool_size} 引擎）")
            log.info(f"[开关] 多线程 OCR：已开启（{pool_size} 引擎）")
        else:
            self.app_state.enable_multithread_ocr = False
            self._reinit_ocr_pool(use_gpu=False, pool_size=1)
            if self.ocr_engine.check_gpu_available():
                self.gpu_switch.configure(state='normal')
            self.gpu_status_label.configure(
                text="ℹ️ 选择加速方式：GPU 或多线程 OCR（二选一）",
                text_color='#666666')
            self.bottom_label.configure(text="多线程 OCR：已关闭（单引擎）")
            log.info("[开关] 多线程 OCR：已关闭")

        self.user_cfg.set('use_multithread_ocr',
                          self.app_state.enable_multithread_ocr)
        self.user_cfg.save()

    def _restart_ocr_engine(self, use_gpu):
        old_running = self.app_state.running
        self.app_state.running = False
        pool_size = 1
        try:
            mode = self.ocr_engine.init(use_gpu=use_gpu, pool_size=pool_size)
            self._actual_ocr_mode = mode
            self.cache.clear()
            if hasattr(self.pipeline, 'inpaint_cache'):
                self.pipeline.inpaint_cache.clear()
            self.app_state.reset_signature()
            self.overlay.clear()
            if mode == 'gpu':
                self.gpu_switch.select()
                self.gpu_status_label.configure(
                    text="✅ 已启用 GPU 加速（多线程 OCR 已禁用）",
                    text_color='#00a000')
                self.bottom_label.configure(text="✅ GPU 加速已启用")
            else:
                self.gpu_switch.deselect()
                err = self.ocr_engine.last_error or "未知原因"
                self.gpu_status_label.configure(
                    text=f"⚠️ GPU 失败，已回退 CPU：{err[:40]}",
                    text_color='#cc6600')
                self.bottom_label.configure(text="GPU 失败，已回退 CPU")
            log.info(f"[GPU] OCR 引擎切换到 {mode}")

            self.user_cfg.set('use_gpu', (mode == 'gpu'))
            self.user_cfg.save()
        except Exception as e:
            log.exception(f"[GPU] 切换失败：{e}")
            self.bottom_label.configure(text=f"切换失败：{e}")
        finally:
            self.app_state.running = old_running

    def _reinit_ocr_pool(self, use_gpu, pool_size):
        old_running = self.app_state.running
        self.app_state.running = False
        try:
            mode = self.ocr_engine.init(use_gpu=use_gpu, pool_size=pool_size)
            self._actual_ocr_mode = mode
            self.cache.clear()
            if hasattr(self.pipeline, 'inpaint_cache'):
                self.pipeline.inpaint_cache.clear()
            self.app_state.reset_signature()
            self.overlay.clear()
            log.info(f"[OCR] 引擎池重建：{mode} × {len(self.ocr_engine.engines)}")
        except Exception as e:
            log.exception(f"[OCR] 重建失败：{e}")
        finally:
            self.app_state.running = old_running

    # ================= 性能模式 =================

    def _on_mode_change(self, mode_name):
        mode_key = None
        for k, v in PERFORMANCE_MODES.items():
            if v['name'] == mode_name:
                mode_key = k
                break
        if not mode_key:
            return
        cfg = PERFORMANCE_MODES[mode_key]
        self._current_mode = mode_key
        self.app_state.performance_mode = mode_key
        self.app_state.apply_mode(cfg)
        self.scan_slider.set(cfg['scan_interval'])
        self.scan_value_label.configure(text=f"{cfg['scan_interval']:.2f} s")
        self._update_mode_desc(mode_key)
        self.app_state.reset_signature()
        self.bottom_label.configure(text=f"性能模式：{cfg['name']}")
        log.info(f"[模式] 切换为 {cfg['name']}")

        self.user_cfg.set('performance_mode', mode_key)
        self.user_cfg.set('scan_interval', cfg['scan_interval'])
        self.user_cfg.save()

    def _update_mode_desc(self, mode_key):
        cfg = PERFORMANCE_MODES[mode_key]
        desc = (
            f"扫描 {cfg['scan_interval']:.2f}s | "
            f"灵敏度 {cfg['change_ratio_threshold']:.2f} | "
            f"外扩 {cfg['bbox_padding']}px | "
            f"OCR {cfg['max_ocr_width']}×{cfg['max_ocr_height']}"
        )
        self.mode_desc_label.configure(text=desc)

    # ================= 开关 =================

    def _on_frame_diff_toggle(self):
        enabled = bool(self.frame_diff_switch.get())
        self.app_state.enable_frame_diff = enabled
        self.app_state.reset_signature()
        msg = "帧差检测：已开启" if enabled else "帧差检测：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_frame_diff', enabled)
        self.user_cfg.save()

    def _on_block_ocr_toggle(self):
        enabled = bool(self.block_ocr_switch.get())
        self.app_state.enable_block_ocr = enabled
        self.app_state.reset_signature()
        msg = "分块 OCR：已开启" if enabled else "分块 OCR：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_block_ocr', enabled)
        self.user_cfg.save()

    def _on_filter_toggle(self):
        enabled = bool(self.filter_switch.get())
        self.app_state.enable_filter = enabled
        self.app_state.reset_signature()
        msg = "文字过滤：已开启" if enabled else "文字过滤：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_filter', enabled)
        self.user_cfg.save()

    def _on_async_translate_toggle(self):
        enabled = bool(self.async_translate_switch.get())
        self.app_state.enable_async_translate = enabled
        msg = "异步翻译：已开启" if enabled else "异步翻译：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_async_translate', enabled)
        self.user_cfg.save()

    def _on_window_pool_toggle(self):
        enabled = bool(self.window_pool_switch.get())
        self.app_state.enable_window_pool = enabled
        self.overlay.set_pool_enabled(enabled)
        msg = "窗口池：已开启" if enabled else "窗口池：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_window_pool', enabled)
        self.user_cfg.save()

    # ================= 其他设置 =================

    def _on_scan_change(self, value):
        value = round(value / 0.05) * 0.05
        value = max(SCAN_MIN, min(SCAN_MAX, value))
        self.app_state.scan_interval = round(value, 2)
        self.scan_value_label.configure(text=f"{value:.2f} s")
        self.user_cfg.set('scan_interval', round(value, 2))
        self.user_cfg.save()

    def _on_font_change(self, value):
        value = int(round(value))
        self.app_state.base_font_size = value
        self.font_value_label.configure(text=str(value))
        self.overlay.set_base_font_size(value)
        self.user_cfg.set('font_size', value)
        self.user_cfg.save()

    def _on_inpaint_toggle(self):
        enabled = bool(self.inpaint_switch.get())
        self.app_state.use_inpainting = enabled
        self.overlay.clear()
        self.app_state.reset_signature()
        msg = "图像修复：已启用" if enabled else "图像修复：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('use_inpainting', enabled)
        self.user_cfg.save()

    def _on_auto_clean_toggle(self):
        enabled = bool(self.auto_clean_switch.get())
        self.cache.set_auto_clean(enabled)
        msg = "自动清理缓存：已开启" if enabled else "自动清理缓存：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('auto_clean', enabled)
        self.user_cfg.save()

    def _on_cache_limit_change(self, value):
        value = int(round(value / 50.0) * 50)
        value = max(CACHE_MIN, min(CACHE_MAX, value))
        self.cache_limit_label.configure(text=str(value))
        self.cache.max_size = value
        self.user_cfg.set('cache_max_size', value)
        self.user_cfg.save()

    def _manual_clear_cache(self):
        self.cache.clear()
        if hasattr(self.pipeline, 'inpaint_cache'):
            self.pipeline.inpaint_cache.clear()
        self._update_cache_label()
        self.bottom_label.configure(text="缓存已清空")
        log.info("[缓存] 手动清空")

    def _update_cache_label(self):
        try:
            n = self.cache.size()
            cap = self.cache.max_size
            self.cache_size_label.configure(text=f"缓存条目：{n} / {cap}")
        except Exception:
            pass

    # ================= API =================

    def open_api_config(self):
        from api_setup_dialog import APISetupDialog
        dlg = APISetupDialog()
        dlg.mainloop()
        if dlg.result:
            self.translator.reload_config()
            self.overlay.clear()
            self.app_state.reset_signature()
            self.bottom_label.configure(text="✅ API 配置已更新")
            log.info("[API] 配置已更新")
        else:
            self.bottom_label.configure(text="API 配置未更改")

    # ================= 热键 =================

    def _capture_hotkey(self, name):
        self.bottom_label.configure(text="请按下你想要的组合键...")

        def worker():
            hk = self.hk_mgr.capture()
            self.after(0, lambda: self._set_captured(name, hk))

        threading.Thread(target=worker, daemon=True).start()

    def _set_captured(self, name, hk):
        if not hk:
            self.bottom_label.configure(text="捕获失败")
            return
        entry = self.hotkey_entries[name]
        entry.delete(0, 'end')
        entry.insert(0, hk)
        self.bottom_label.configure(
            text=f"已捕获：{hk}（点击「应用所有热键」生效）")

    def _apply_hotkeys(self):
        new_hotkeys = {}
        for name, entry in self.hotkey_entries.items():
            hk = entry.get().strip()
            if not hk:
                self.bottom_label.configure(text=f"{name} 热键为空")
                return
            new_hotkeys[name] = hk
        values = list(new_hotkeys.values())
        if len(values) != len(set(values)):
            self.bottom_label.configure(text="热键重复，请检查")
            return
        failed = []
        for name, hk in new_hotkeys.items():
            cb = self._get_callback(name)
            if self.hk_mgr.register(name, hk, cb):
                self.hotkeys[name] = hk
            else:
                failed.append(name)
        self._refresh_hotkey_hint()
        if failed:
            self.bottom_label.configure(
                text=f"部分热键注册失败：{', '.join(failed)}")
        else:
            self.bottom_label.configure(text="✅ 所有热键已更新")
            log.info("[热键] 全部更新成功")

        self.user_cfg.set_hotkeys(new_hotkeys)
        self.user_cfg.save()

    def _refresh_hotkey_hint(self):
        hk = self.hotkeys
        self.hotkey_hint.configure(
            text=f"{hk['toggle']} 开始/暂停 | "
                 f"{hk['freeze']} 钉住 | "
                 f"{hk['quit']} 退出")

    # ================= 交互 =================

    def toggle_running(self):
        self.app_state.running = not self.app_state.running
        if self.app_state.running:
            self.status_label.configure(text="状态：运行中", text_color='green')
            self.btn_toggle.configure(text="⏸ 暂停")
            self.bottom_label.configure(text="正在扫描屏幕...")
            log.info("状态：运行中")
        else:
            self.status_label.configure(text="状态：已暂停", text_color='gray')
            self.btn_toggle.configure(text="▶ 开始")
            self.bottom_label.configure(text="已暂停")
            self.scroll_detector.reset()
            self.app_state.scroll_boost = False
            log.info("状态：已暂停")

    def clear_overlay(self):
        self.overlay.clear()
        self.bottom_label.configure(text="已清空当前译文")
        log.info("[覆盖层] 手动清空")

    def open_config(self):
        self.region_manager.open_config()

    def toggle_border(self):
        self._border_visible = not self._border_visible
        self.region_manager.set_border_visible(self._border_visible)
        if self._border_visible:
            self.btn_border.configure(text="隐藏边框")
            self.bottom_label.configure(text="区域框：已显示")
        else:
            self.btn_border.configure(text="显示边框")
            self.bottom_label.configure(text="区域框：已隐藏")

    def on_region_changed(self, new_region):
        self.app_state.update_region(new_region)
        self.app_state.reset_signature()
        self.scroll_detector.reset()
        self.overlay.clear()
        self._update_region_label()
        self.bottom_label.configure(text="区域已更新")

        self.user_cfg.set_region(new_region)
        self.user_cfg.save()

        log.info(f"[区域] 更新为 {new_region}")

    def _update_region_label(self):
        r = self.app_state.snapshot_region()
        self.region_label.configure(
            text=f"区域: left={r['left']}, top={r['top']}, "
                 f"w={r['width']}, h={r['height']}")

    # ================= 主循环 =================

    def _tick(self):
        try:
            if not self.app_state.freeze_overlay:
                try:
                    while True:
                        msg = self.pipeline.result_queue.get_nowait()
                        kind = msg[0]
                        if kind == 'items':
                            try:
                                self.overlay.update(
                                    msg[1],
                                    base_font_size=self.app_state.base_font_size,
                                )
                            except Exception as e:
                                log.exception(f"[UI] overlay.update 失败：{e}")
                except queue.Empty:
                    pass

                try:
                    region = self.app_state.snapshot_region()
                    pruned = self.overlay.prune_out_of_bounds(
                        region, padding=BOUNDARY_PADDING)
                    if pruned:
                        log.info(f"[跟随] 移除 {pruned} 个超出边界的译文")
                except Exception as e:
                    log.exception(f"[UI] 边界裁剪失败：{e}")

            now = time.time()

            if now - self._last_cache_ui_update >= 5:
                try:
                    self._update_cache_label()
                except Exception:
                    pass
                self._last_cache_ui_update = now

            if (self.app_state.running
                    and not self.app_state.freeze_overlay):
                base = self.app_state.scan_interval
                idle = self.app_state.idle_count

                if self.scroll_detector.is_scrolling():
                    interval = base * 0.25
                    self.app_state.scroll_boost = True
                else:
                    interval = base
                    self.app_state.scroll_boost = False
                    if idle >= IDLE_THRESHOLD_FOR_SLOWDOWN:
                        interval = base * IDLE_SLOWDOWN_FACTOR
                    elif idle >= 1:
                        interval = base * 1.5

                if now - self._last_capture >= interval:
                    try:
                        self._do_capture()
                    except Exception as e:
                        log.exception(f"[UI] 截图失败：{e}")
                    self._last_capture = now

        except Exception as e:
            log.exception(f"[UI] _tick 异常：{e}")
        finally:
            self.after(TICK_MS, self._tick)

    def _do_capture(self):
        try:
            region = self.app_state.snapshot_region()
            img = self.ocr_engine.capture(region)
            self.pipeline.submit(img)
        except Exception as e:
            log.exception(f"[截图错误] {e}")

    def quit_app(self):
        log.info("========== 退出 ==========")

        try:
            self.user_cfg.set('window_geometry', self.geometry())
            self.user_cfg.save()
        except Exception as e:
            log.error(f"[退出] 保存配置失败：{e}")

        try:
            self.scroll_detector.stop()
        except Exception:
            pass
        self.hk_mgr.remove_all()
        self.region_manager.destroy()
        self.overlay.clear()
        self.destroy()