"""LST - UI 模块
三模式翻译：本地 / 混合 / 联网
短语缓存 + 本地词典（支持文件夹）+ API
"""

import os
import queue
import sys
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
    DEFAULT_ENABLE_FINE_TEXT,
    DEFAULT_ENABLE_ASYNC_OVERLAY,
    DEFAULT_ENABLE_GENERAL_FILTER,
    DEFAULT_ENABLE_PHRASE_CACHE,
    DEFAULT_PHRASE_TABLE_PATH,
    DEFAULT_TRANSLATION_MODE,
    DEFAULT_LOCAL_DICT_PATH,
    PERFORMANCE_MODES, DEFAULT_MODE,
    IDLE_THRESHOLD_FOR_SLOWDOWN, IDLE_SLOWDOWN_FACTOR,
    DEFAULT_ENABLE_FRAME_DIFF, DEFAULT_ENABLE_FILTER,
    DEFAULT_ENABLE_BLOCK_OCR,
    DEFAULT_ENABLE_ASYNC_TRANSLATE, DEFAULT_ENABLE_WINDOW_POOL,
    MULTITHREAD_OCR_POOL_SIZE,
    DIRECTML_ENGINE_COUNT_MIN,
    DIRECTML_ENGINE_COUNT_MAX,
    DIRECTML_ENGINE_COUNT_DEFAULT,
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
from cuda_setup import (
    check_onnxruntime_variant, check_cuda_ready, check_directml_ready,
    is_frozen, list_gpu_devices,
)


log = get_logger('ui')


CACHE_MIN = 100
CACHE_MAX = 2000
CACHE_DEFAULT = 500
TICK_MS = 10
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
        self.title(get_window_title())

        self._ui_ready = False

        # ============ 配置 ============
        self.user_cfg = get_config()
        log.info(f"配置文件：{get_config_path()}")

        geo = self.user_cfg.get('window_geometry', "740x1080+50+50")
        self.geometry(geo)
        self.minsize(660, 600)
        self.attributes('-topmost', True)

        initial_region = self.user_cfg.get_region()
        initial_font = self.user_cfg.get('font_size', INITIAL_FONT_SIZE)
        saved_mode = self.user_cfg.get('performance_mode', DEFAULT_MODE)
        if saved_mode not in PERFORMANCE_MODES:
            saved_mode = DEFAULT_MODE
        saved_hotkeys = self.user_cfg.get_hotkeys()

        saved_frame_diff = self.user_cfg.get('enable_frame_diff',
                                             DEFAULT_ENABLE_FRAME_DIFF)
        saved_block = self.user_cfg.get('enable_block_ocr',
                                        DEFAULT_ENABLE_BLOCK_OCR)
        saved_filter = self.user_cfg.get('enable_filter',
                                         DEFAULT_ENABLE_FILTER)
        saved_general_filter = self.user_cfg.get(
            'enable_general_filter', DEFAULT_ENABLE_GENERAL_FILTER)
        saved_phrase_cache = self.user_cfg.get(
            'enable_phrase_cache', DEFAULT_ENABLE_PHRASE_CACHE)
        saved_phrase_table_path = self.user_cfg.get(
            'phrase_table_path', DEFAULT_PHRASE_TABLE_PATH)
        saved_translation_mode = self.user_cfg.get(
            'translation_mode', DEFAULT_TRANSLATION_MODE)
        saved_local_dict_path = self.user_cfg.get(
            'local_dict_path', DEFAULT_LOCAL_DICT_PATH)
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
        saved_fine_text = self.user_cfg.get('enable_fine_text',
                                            DEFAULT_ENABLE_FINE_TEXT)
        saved_async_overlay = self.user_cfg.get(
            'enable_async_overlay', DEFAULT_ENABLE_ASYNC_OVERLAY)
        saved_cache_max = self.user_cfg.get('cache_max_size', CACHE_DEFAULT)
        saved_auto_clean = self.user_cfg.get('auto_clean', True)
        saved_follow_mode = self.user_cfg.get('follow_mode', 'follow')
        if saved_follow_mode not in ('default', 'follow', 'pin'):
            saved_follow_mode = 'follow'

        # ===== GPU 枚举 & 引擎分配 =====
        self.gpu_devices = list_gpu_devices()
        saved_counts = self.user_cfg.get('gpu_engine_counts', []) or []
        self.gpu_engine_counts = self._merge_gpu_counts(
            self.gpu_devices, saved_counts)

        log.info(f"========== 启动 v{__version__} ==========")
        if is_frozen():
            log.info("[启动] 打包模式（exe）")

        # ============ 核心组件 ============
        self.translator = Translator()
        self.ocr_engine = OCREngine()

        saved_accel_mode = self.user_cfg.get('accel_mode', '')
        current_variant = check_onnxruntime_variant() or 'cpu'

        if saved_accel_mode == 'directml' and current_variant == 'directml':
            init_mode = 'directml'
        elif saved_accel_mode == 'cuda' and current_variant == 'gpu':
            init_mode = 'cuda'
        elif saved_accel_mode == 'cpu_multi':
            init_mode = 'cpu_multi'
        else:
            if current_variant == 'directml':
                init_mode = 'directml'
            elif current_variant == 'gpu':
                init_mode = 'cuda'
            else:
                init_mode = 'cpu_multi'

        variant = self.ocr_engine.init(
            mode=init_mode,
            pool_size=MULTITHREAD_OCR_POOL_SIZE,
            gpu_engine_counts=self.gpu_engine_counts,
        )
        log.info(f"OCR 引擎初始化：{variant} × {len(self.ocr_engine.engines)}"
                 f" devices={self.ocr_engine.engine_device_ids}")

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
            enable_fine_text=saved_fine_text,
            enable_general_filter=saved_general_filter,
            enable_phrase_cache=saved_phrase_cache,
            translation_mode=saved_translation_mode,
            local_dict_path=saved_local_dict_path,
            mode_cfg=default_cfg,
            enable_frame_diff=saved_frame_diff,
            enable_filter=saved_filter,
            enable_block_ocr=saved_block,
            enable_async_translate=saved_async,
            enable_window_pool=saved_wpool,
            enable_multithread_ocr=True,
        )
        self.app_state.performance_mode = saved_mode
        self.app_state.base_font_size = initial_font
        self.app_state.scan_interval = saved_scan
        self.app_state.follow_mode = saved_follow_mode
        self.app_state.freeze_overlay = (saved_follow_mode == 'pin')

        self.pipeline = Pipeline(
            self.ocr_engine, self.cache, self.translator, self.app_state,
        )
        self.pipeline.start()

        try:
            self.pipeline.phrase_cache.set_external_path(
                saved_phrase_table_path)
            if saved_phrase_table_path:
                self.pipeline.phrase_cache.load_external()
        except Exception as e:
            log.exception(f"[短语表] 初始化失败：{e}")

        self.overlay = OverlayWindow(self, font_name=FONT_NAME)
        self.overlay.set_pool_enabled(saved_wpool)
        self.overlay.set_async_enabled(saved_async_overlay)

        self.region_manager = RegionManager(
            self, initial_region,
            border_color='#00FF00', border_width=3,
            on_change=self.on_region_changed,
        )

        self.scroll_detector = ScrollDetector(
            get_region_fn=self.app_state.snapshot_region,
            on_scroll=self._on_scroll_detected,
            is_active_fn=self._is_scroll_active,
        )
        self.scroll_detector.start()

        self.hotkeys = dict(saved_hotkeys)
        self.hk_mgr = HotkeyManager()
        self._register_all_hotkeys()

        self._last_capture = 0
        self._last_cache_ui_update = 0
        self._border_visible = True
        self._current_mode = saved_mode

        # ===== GPU 引擎 UI 引用 =====
        self.gpu_engine_sliders = {}
        self.gpu_engine_labels = {}

        # UI
        self._build_ui()

        self.after(100, self._sync_accel_ui)
        self.after(100, self._sync_follow_mode_ui)
        self._restore_ui_from_config(
            saved_mode, saved_scan, initial_font, saved_inpaint,
            saved_fine_text, saved_async_overlay,
            saved_general_filter, saved_phrase_cache,
            saved_phrase_table_path,
            saved_translation_mode, saved_local_dict_path,
            saved_frame_diff, saved_block, saved_filter, saved_async,
            saved_wpool, saved_cache_max, saved_auto_clean,
        )

        self._ui_ready = True
        self.after(TICK_MS, self._tick)

    # ================= GPU 引擎数合并 =================

    @staticmethod
    def _merge_gpu_counts(devices, saved):
        n = len(devices)
        if n <= 0:
            return [DIRECTML_ENGINE_COUNT_DEFAULT]

        result = [DIRECTML_ENGINE_COUNT_DEFAULT] * n

        if isinstance(saved, dict):
            for i in range(n):
                if i in saved:
                    try:
                        result[i] = int(saved[i])
                    except Exception:
                        pass
        elif isinstance(saved, (list, tuple)):
            for i in range(min(n, len(saved))):
                try:
                    result[i] = int(saved[i])
                except Exception:
                    pass

        result = [
            max(DIRECTML_ENGINE_COUNT_MIN,
                min(DIRECTML_ENGINE_COUNT_MAX, x))
            for x in result
        ]

        if sum(result) < 1 and n > 0:
            result[0] = DIRECTML_ENGINE_COUNT_DEFAULT

        return result

    # ================= 恢复 UI =================

    def _restore_ui_from_config(self, mode, scan, font, inpaint,
                                fine_text, async_overlay,
                                general_filter, phrase_cache,
                                phrase_table_path,
                                translation_mode, local_dict_path,
                                frame_diff, block, filt, async_t,
                                wpool, cache_max, auto_clean):
        try:
            self.mode_selector.set(PERFORMANCE_MODES[mode]['name'])
            self._update_mode_desc(mode)

            self.scan_slider.set(scan)
            self.scan_value_label.configure(text=f"{scan:.2f} s")
            self.font_slider.set(font)
            self.font_value_label.configure(text=str(font))

            if inpaint and CV2_AVAILABLE:
                self.inpaint_switch.select()
            else:
                self.inpaint_switch.deselect()

            if fine_text and CV2_AVAILABLE:
                self.fine_text_switch.select()
            else:
                self.fine_text_switch.deselect()

            if async_overlay:
                self.async_overlay_switch.select()
            else:
                self.async_overlay_switch.deselect()

            if general_filter:
                self.general_filter_switch.select()
            else:
                self.general_filter_switch.deselect()

            if phrase_cache:
                self.phrase_cache_switch.select()
            else:
                self.phrase_cache_switch.deselect()

            mode_map = {
                'local': '本地',
                'hybrid': '混合',
                'online': '联网',
            }
            self.trans_mode_selector.set(
                mode_map.get(translation_mode, '混合'))

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

            # GPU 引擎数滑块
            for device_id, slider in self.gpu_engine_sliders.items():
                if device_id < len(self.gpu_engine_counts):
                    v = self.gpu_engine_counts[device_id]
                    slider.set(v)
                    if device_id in self.gpu_engine_labels:
                        self.gpu_engine_labels[device_id].configure(
                            text=str(v))

            self._update_phrase_count()
            self._update_dict_count()
        except Exception as e:
            log.exception(f"[UI] 恢复配置失败：{e}")

    # ================= 滚动 =================

    def _is_scroll_active(self):
        if self.app_state.follow_mode != 'follow':
            return False
        return self.app_state.running

    def _on_scroll_detected(self, dy):
        if self.app_state.follow_mode != 'follow':
            return
        self.app_state.scroll_boost = True
        try:
            self.after(0, lambda: self._apply_scroll(dy))
        except Exception:
            pass

    def _apply_scroll(self, dy):
        if self.app_state.follow_mode != 'follow':
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
            'freeze': self.toggle_freeze_hotkey,
            'quit':   self.quit_app,
        }[name]

    def _register_all_hotkeys(self):
        for name, hk in self.hotkeys.items():
            self.hk_mgr.register(name, hk, self._get_callback(name))

    def toggle_freeze_hotkey(self):
        mode = self.app_state.follow_mode
        next_mode = {
            'default': 'follow',
            'follow': 'pin',
            'pin': 'default',
        }[mode]
        self.set_follow_mode(next_mode)

    # ================= UI 构建 =================

    def _build_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(fill='x', padx=10, pady=(10, 5))

        self.status_label = ctk.CTkLabel(
            top, text="状态：已暂停",
            font=(FONT_NAME, 13, 'bold'), anchor='w',
        )
        self.status_label.pack(side='left', padx=10)

        self.version_label = ctk.CTkLabel(
            top, text=f"{APP_NAME} v{__version__}",
            font=(FONT_NAME, 10), text_color='#3a86ff',
        )
        self.version_label.pack(side='right', padx=10)

        self.hotkey_hint = ctk.CTkLabel(
            top, text='', font=(FONT_NAME, 10), text_color='gray',
        )
        self.hotkey_hint.pack(side='right', padx=10)

        # 翻译模式
        trans_frame = ctk.CTkFrame(self)
        trans_frame.pack(fill='x', padx=10, pady=5)

        ctk.CTkLabel(
            trans_frame, text="翻译模式：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(5, 8))

        self.trans_mode_selector = ctk.CTkSegmentedButton(
            trans_frame,
            values=["本地", "混合", "联网"],
            command=self._on_trans_mode_change,
            font=(FONT_NAME, 11),
        )
        self.trans_mode_selector.set("混合")
        self.trans_mode_selector.pack(
            side='left', fill='x', expand=True, padx=5)

        self.trans_mode_desc = ctk.CTkLabel(
            trans_frame, text="", font=(FONT_NAME, 9),
            text_color='#666666', width=280, anchor='w',
        )
        self.trans_mode_desc.pack(side='left', padx=8)

        # 译文模式
        follow_frame = ctk.CTkFrame(self)
        follow_frame.pack(fill='x', padx=10, pady=5)

        ctk.CTkLabel(
            follow_frame, text="译文模式：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(5, 8))

        self.follow_selector = ctk.CTkSegmentedButton(
            follow_frame,
            values=["默认", "跟随", "钉住"],
            command=self._on_follow_mode_change,
            font=(FONT_NAME, 11),
        )
        self.follow_selector.pack(side='left', fill='x', expand=True, padx=5)

        self.follow_desc_label = ctk.CTkLabel(
            follow_frame, text="", font=(FONT_NAME, 9),
            text_color='#666666', width=260, anchor='w',
        )
        self.follow_desc_label.pack(side='left', padx=8)

        # 按钮行 1
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

        # 按钮行 2
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
        self._sync_trans_mode_ui()

    def _build_display_tab(self, parent):
        # ===== 性能模式 =====
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

        # ===== 加速方式 =====
        ctk.CTkLabel(
            parent, text="加速方式（运行时切换）：",
            font=(FONT_NAME, 11, 'bold'),
        ).grid(row=2, column=0, columnspan=3,
               sticky='w', padx=(15, 5), pady=(8, 4))

        self.ort_variant_label = ctk.CTkLabel(
            parent, text="当前引擎：检测中...",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w',
        )
        self.ort_variant_label.grid(
            row=3, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 6))

        accel_btns = ctk.CTkFrame(parent, fg_color='transparent')
        accel_btns.grid(
            row=4, column=0, columnspan=3,
            sticky='ew', padx=(15, 15), pady=(0, 6))

        self.btn_cpu_multi = ctk.CTkButton(
            accel_btns,
            text=f"多线程 CPU\n({MULTITHREAD_OCR_POOL_SIZE} 引擎)",
            width=140, height=50,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=lambda: self.switch_accel_mode('cpu_multi'),
        )
        self.btn_cpu_multi.pack(side='left', padx=3)

        self.btn_directml = ctk.CTkButton(
            accel_btns, text="DirectML GPU",
            width=140, height=50,
            font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=lambda: self.switch_accel_mode('directml'),
        )
        self.btn_directml.pack(side='left', padx=3)

        self.btn_cuda = ctk.CTkButton(
            accel_btns, text="NVIDIA CUDA\n(单引擎)",
            width=140, height=50,
            font=(FONT_NAME, 10),
            fg_color='#28a745', hover_color='#1e7e34',
            command=lambda: self.switch_accel_mode('cuda'),
        )
        self.btn_cuda.pack(side='left', padx=3)

        # ===== GPU 引擎分配（多行） =====
        gpu_box = ctk.CTkFrame(parent, fg_color='transparent')
        gpu_box.grid(
            row=5, column=0, columnspan=3,
            sticky='ew', padx=(25, 15), pady=(4, 6))

        ctk.CTkLabel(
            gpu_box, text="GPU 引擎分配（每块独立）：",
            font=(FONT_NAME, 9), text_color='#444444', anchor='w',
        ).pack(anchor='w')

        if not self.gpu_devices:
            ctk.CTkLabel(
                gpu_box, text="（未检测到 GPU 设备）",
                font=(FONT_NAME, 9), text_color='#999999',
            ).pack(anchor='w', pady=2)
        else:
            for device_id, name, vram in self.gpu_devices:
                row = ctk.CTkFrame(gpu_box, fg_color='transparent')
                row.pack(fill='x', pady=2)

                label_text = name
                if len(label_text) > 42:
                    label_text = label_text[:39] + "..."
                if vram > 0:
                    label_text += f"（{vram}MB）"

                ctk.CTkLabel(
                    row, text=f"[{device_id}] {label_text}",
                    font=(FONT_NAME, 9), text_color='#333333',
                    anchor='w', width=280,
                ).pack(side='left')

                val_lbl = ctk.CTkLabel(
                    row, text="0", width=20,
                    font=(FONT_NAME, 10, 'bold'),
                    text_color='#3a86ff',
                )
                val_lbl.pack(side='right')

                slider = ctk.CTkSlider(
                    row,
                    from_=DIRECTML_ENGINE_COUNT_MIN,
                    to=DIRECTML_ENGINE_COUNT_MAX,
                    number_of_steps=(
                        DIRECTML_ENGINE_COUNT_MAX -
                        DIRECTML_ENGINE_COUNT_MIN),
                    command=lambda v, d=device_id:
                        self._on_gpu_engine_change(d, v),
                    height=14, width=140,
                )
                if device_id < len(self.gpu_engine_counts):
                    slider.set(self.gpu_engine_counts[device_id])
                    val_lbl.configure(
                        text=str(self.gpu_engine_counts[device_id]))
                else:
                    slider.set(DIRECTML_ENGINE_COUNT_DEFAULT)
                    val_lbl.configure(
                        text=str(DIRECTML_ENGINE_COUNT_DEFAULT))
                slider.pack(side='right', padx=(4, 6))

                self.gpu_engine_sliders[device_id] = slider
                self.gpu_engine_labels[device_id] = val_lbl

        # ===== 加速状态 =====
        self.accel_status_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
            wraplength=600,
        )
        self.accel_status_label.grid(
            row=6, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 10))

        # ===== 优化功能 =====
        ctk.CTkLabel(
            parent, text="优化功能：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=7, column=0, sticky='w', padx=(15, 5), pady=(8, 4))

        # 1. 帧差
        self.frame_diff_switch = ctk.CTkSwitch(
            parent, text="帧差检测（静止时跳过）",
            command=self._on_frame_diff_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FRAME_DIFF:
            self.frame_diff_switch.select()
        self.frame_diff_switch.grid(
            row=8, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 2. 分块 OCR
        self.block_ocr_switch = ctk.CTkSwitch(
            parent, text="分块 OCR（只识别变化的区域）",
            command=self._on_block_ocr_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_BLOCK_OCR:
            self.block_ocr_switch.select()
        self.block_ocr_switch.grid(
            row=9, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 3. 文字过滤
        self.filter_switch = ctk.CTkSwitch(
            parent, text="文字过滤（过滤异常框）",
            command=self._on_filter_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FILTER:
            self.filter_switch.select()
        self.filter_switch.grid(
            row=10, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 4. 通用过滤
        self.general_filter_switch = ctk.CTkSwitch(
            parent, text="通用过滤（艺术字/数字/单位）",
            command=self._on_general_filter_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_GENERAL_FILTER:
            self.general_filter_switch.select()
        self.general_filter_switch.grid(
            row=11, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 5. 短语缓存
        self.phrase_cache_switch = ctk.CTkSwitch(
            parent, text="短语缓存（常见短语立即显示）",
            command=self._on_phrase_cache_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_PHRASE_CACHE:
            self.phrase_cache_switch.select()
        self.phrase_cache_switch.grid(
            row=12, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 短语表按钮
        phrase_btns = ctk.CTkFrame(parent, fg_color='transparent')
        phrase_btns.grid(
            row=13, column=0, columnspan=3,
            sticky='ew', padx=(45, 15), pady=(0, 8))

        ctk.CTkButton(
            phrase_btns, text="选择短语表", width=110, height=26,
            font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=self.select_phrase_file,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            phrase_btns, text="重载", width=70, height=26,
            font=(FONT_NAME, 10),
            fg_color='#6f42c1', hover_color='#5a35a0',
            command=self.reload_phrase_file,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            phrase_btns, text="清除路径", width=80, height=26,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.clear_phrase_file,
        ).pack(side='left', padx=3)

        self.phrase_count_label = ctk.CTkLabel(
            phrase_btns, text="", font=(FONT_NAME, 9),
            text_color='#666666',
        )
        self.phrase_count_label.pack(side='left', padx=8)

        # 本地词典
        ctk.CTkLabel(
            parent, text="本地词典：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=14, column=0, sticky='w', padx=(15, 5), pady=(8, 4))

        dict_btns = ctk.CTkFrame(parent, fg_color='transparent')
        dict_btns.grid(
            row=15, column=0, columnspan=3,
            sticky='ew', padx=(25, 15), pady=(0, 8))

        ctk.CTkButton(
            dict_btns, text="选择词典", width=100, height=26,
            font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=self.select_dict_file,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            dict_btns, text="重载", width=70, height=26,
            font=(FONT_NAME, 10),
            fg_color='#6f42c1', hover_color='#5a35a0',
            command=self.reload_dict_file,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            dict_btns, text="清除路径", width=80, height=26,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.clear_dict_file,
        ).pack(side='left', padx=3)

        self.dict_count_label = ctk.CTkLabel(
            dict_btns, text="", font=(FONT_NAME, 9),
            text_color='#666666',
        )
        self.dict_count_label.pack(side='left', padx=8)

        # 6. 异步翻译
        self.async_translate_switch = ctk.CTkSwitch(
            parent, text="异步翻译（不阻塞 OCR）",
            command=self._on_async_translate_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_ASYNC_TRANSLATE:
            self.async_translate_switch.select()
        self.async_translate_switch.grid(
            row=16, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        # 7. 窗口池
        self.window_pool_switch = ctk.CTkSwitch(
            parent, text="窗口池（复用译文窗口）",
            command=self._on_window_pool_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_WINDOW_POOL:
            self.window_pool_switch.select()
        self.window_pool_switch.grid(
            row=17, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 10))

        # 刷新间隔
        ctk.CTkLabel(
            parent, text="刷新间隔 (秒)：", font=(FONT_NAME, 11),
        ).grid(row=18, column=0, sticky='w', padx=(15, 5), pady=8)

        self.scan_value_label = ctk.CTkLabel(
            parent,
            text=f"{PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval']:.2f} s",
            width=60, font=(FONT_NAME, 11),
        )
        self.scan_value_label.grid(row=18, column=2, sticky='e', padx=(5, 15))

        n_steps = int((SCAN_MAX - SCAN_MIN) / 0.05)
        self.scan_slider = ctk.CTkSlider(
            parent, from_=SCAN_MIN, to=SCAN_MAX,
            number_of_steps=n_steps,
            command=self._on_scan_change,
        )
        self.scan_slider.set(PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval'])
        self.scan_slider.grid(row=18, column=1, sticky='ew', padx=10)

        ctk.CTkLabel(
            parent, text="文字大小：", font=(FONT_NAME, 11),
        ).grid(row=19, column=0, sticky='w', padx=(15, 5), pady=8)

        self.font_value_label = ctk.CTkLabel(
            parent, text=str(INITIAL_FONT_SIZE), width=60,
            font=(FONT_NAME, 11),
        )
        self.font_value_label.grid(row=19, column=2, sticky='e', padx=(5, 15))

        self.font_slider = ctk.CTkSlider(
            parent, from_=FONT_MIN, to=FONT_MAX,
            number_of_steps=FONT_MAX - FONT_MIN,
            command=self._on_font_change,
        )
        self.font_slider.set(INITIAL_FONT_SIZE)
        self.font_slider.grid(row=19, column=1, sticky='ew', padx=10)

        # 图像修复
        ctk.CTkLabel(
            parent, text="图像修复（整框）：", font=(FONT_NAME, 11),
        ).grid(row=20, column=0, sticky='w', padx=(15, 5), pady=(15, 8))

        self.inpaint_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_inpaint_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_USE_INPAINTING and CV2_AVAILABLE:
            self.inpaint_switch.select()
        self.inpaint_switch.grid(row=20, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.inpaint_switch.configure(state='disabled')

        # 精细文字替换
        ctk.CTkLabel(
            parent, text="精细文字替换（保留背景）：",
            font=(FONT_NAME, 11),
        ).grid(row=21, column=0, sticky='w', padx=(15, 5), pady=(10, 8))

        self.fine_text_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_fine_text_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_FINE_TEXT and CV2_AVAILABLE:
            self.fine_text_switch.select()
        self.fine_text_switch.grid(row=21, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.fine_text_switch.configure(state='disabled')

        # 异步覆盖
        ctk.CTkLabel(
            parent, text="异步覆盖（逐字显示）：",
            font=(FONT_NAME, 11),
        ).grid(row=22, column=0, sticky='w', padx=(15, 5), pady=(10, 8))

        self.async_overlay_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_async_overlay_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_ASYNC_OVERLAY:
            self.async_overlay_switch.select()
        self.async_overlay_switch.grid(row=22, column=1, sticky='w', padx=10)

        # 自动清理
        ctk.CTkLabel(
            parent, text="自动清理缓存：", font=(FONT_NAME, 11),
        ).grid(row=23, column=0, sticky='w', padx=(15, 5), pady=(15, 8))

        self.auto_clean_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_auto_clean_toggle,
            onvalue=True, offvalue=False,
        )
        self.auto_clean_switch.select()
        self.auto_clean_switch.grid(row=23, column=1, sticky='w', padx=10)

        # 缓存上限
        ctk.CTkLabel(
            parent, text="缓存上限：", font=(FONT_NAME, 11),
        ).grid(row=24, column=0, sticky='w', padx=(15, 5), pady=8)

        self.cache_limit_label = ctk.CTkLabel(
            parent, text=str(CACHE_DEFAULT), width=60,
            font=(FONT_NAME, 11),
        )
        self.cache_limit_label.grid(row=24, column=2, sticky='e', padx=(5, 15))

        self.cache_limit_slider = ctk.CTkSlider(
            parent, from_=CACHE_MIN, to=CACHE_MAX,
            number_of_steps=(CACHE_MAX - CACHE_MIN) // 50,
            command=self._on_cache_limit_change,
        )
        self.cache_limit_slider.set(CACHE_DEFAULT)
        self.cache_limit_slider.grid(row=24, column=1, sticky='ew', padx=10)

        cache_row = ctk.CTkFrame(parent, fg_color='transparent')
        cache_row.grid(row=25, column=0, columnspan=3,
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

        self._update_phrase_count()
        self._update_dict_count()

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

    # ================= 翻译模式 =================

    def _on_trans_mode_change(self, mode_name):
        mode_map = {
            '本地': 'local',
            '混合': 'hybrid',
            '联网': 'online',
        }
        mode = mode_map.get(mode_name, 'hybrid')
        self.app_state.translation_mode = mode
        self._sync_trans_mode_ui()

        self.user_cfg.set('translation_mode', mode)
        self.user_cfg.save()

        log.info(f"[翻译模式] 切换为 {mode}")
        self.bottom_label.configure(text=f"翻译模式：{mode_name}")

    def _sync_trans_mode_ui(self):
        mode = self.app_state.translation_mode

        name_map = {
            'local': '本地',
            'hybrid': '混合',
            'online': '联网',
        }
        try:
            self.trans_mode_selector.set(name_map.get(mode, '混合'))
        except Exception:
            pass

        desc_map = {
            'local': '只用本地词典+短语缓存，无网络',
            'hybrid': '本地优先，未命中走 API',
            'online': '短语缓存优先，其余走 API',
        }
        try:
            self.trans_mode_desc.configure(
                text=desc_map.get(mode, ''))
        except Exception:
            pass

    # ================= 译文模式 =================

    def _on_follow_mode_change(self, mode_name):
        mode_map = {
            '默认': 'default',
            '跟随': 'follow',
            '钉住': 'pin',
        }
        mode = mode_map.get(mode_name, 'follow')
        self.set_follow_mode(mode)

    def set_follow_mode(self, mode):
        if mode not in ('default', 'follow', 'pin'):
            return

        self.app_state.follow_mode = mode
        self.app_state.freeze_overlay = (mode == 'pin')

        self.scroll_detector.reset()
        self.app_state.scroll_boost = False

        if mode != 'pin':
            try:
                while True:
                    self.pipeline.result_queue.get_nowait()
            except queue.Empty:
                pass
            self.overlay.clear()
            self.app_state.reset_signature()

        self._sync_follow_mode_ui()

        self.user_cfg.set('follow_mode', mode)
        self.user_cfg.save()

        log.info(f"[译文模式] 切换为 {mode}")

    def _sync_follow_mode_ui(self):
        mode = self.app_state.follow_mode

        name_map = {
            'default': '默认',
            'follow': '跟随',
            'pin': '钉住',
        }
        try:
            self.follow_selector.set(name_map.get(mode, '跟随'))
        except Exception:
            pass

        desc_map = {
            'default': '滚动时重新识别（不平移）',
            'follow': '滚动时译文实时跟随',
            'pin': '译文固定，滚页不动',
        }
        try:
            self.follow_desc_label.configure(
                text=desc_map.get(mode, ''))
        except Exception:
            pass

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
        try:
            from about_dialog import AboutDialog
            dlg = AboutDialog(self)
            dlg.grab_set()
        except Exception as e:
            log.exception(f"[关于] 打开失败：{e}")

    # ================= 短语表 =================

    def select_phrase_file(self):
        from tkinter import filedialog

        try:
            current = self.pipeline.phrase_cache.get_external_path()
            init_dir = os.path.dirname(current) if current else ''

            path = filedialog.askopenfilename(
                title="选择短语表文件",
                initialdir=init_dir or None,
                filetypes=[
                    ("文本文件", "*.txt"),
                    ("CSV 文件", "*.csv"),
                    ("JSON 文件", "*.json"),
                    ("所有文件", "*.*"),
                ],
            )
            if not path:
                return

            self.pipeline.phrase_cache.set_external_path(path)
            count = self.pipeline.phrase_cache.load_external()

            self.user_cfg.set('phrase_table_path', path)
            self.user_cfg.save()

            self._update_phrase_count()
            self.bottom_label.configure(
                text=f"✅ 已选择短语表：{os.path.basename(path)}"
                     f"（加载 {count} 条）")
            log.info(f"[短语表] 用户选择：{path}")

        except Exception as e:
            log.exception(f"[短语表] 选择失败：{e}")
            self.bottom_label.configure(text="选择短语表失败")

    def reload_phrase_file(self):
        try:
            path = self.pipeline.phrase_cache.get_external_path()
            if not path:
                self.bottom_label.configure(
                    text="⚠️ 未指定短语表文件")
                return

            count = self.pipeline.phrase_cache.reload_external()
            self._update_phrase_count()
            self.bottom_label.configure(
                text=f"✅ 已重载：{os.path.basename(path)}"
                     f"（共 {count} 条）")
            log.info(f"[短语表] 重载：{path}")
        except Exception as e:
            log.exception(f"[短语表] 重载失败：{e}")
            self.bottom_label.configure(text="重载失败")

    def clear_phrase_file(self):
        try:
            self.pipeline.phrase_cache.set_external_path('')
            self.user_cfg.set('phrase_table_path', '')
            self.user_cfg.save()
            self._update_phrase_count()
            self.bottom_label.configure(text="已清除外部短语表路径")
            log.info("[短语表] 已清除路径")
        except Exception as e:
            log.exception(f"[短语表] 清除失败：{e}")

    def _update_phrase_count(self):
        try:
            stats = self.pipeline.phrase_cache.stats()
            path = self.pipeline.phrase_cache.get_external_path()

            if path:
                fname = os.path.basename(path)
                text = (f"共 {stats['total']} 条"
                        f"（内置 {stats['builtin']} + 其他 {stats['user']}）"
                        f" | 表：{fname}")
            else:
                text = (f"共 {stats['total']} 条"
                        f"（内置 {stats['builtin']} + 自定义 {stats['user']}）")

            self.phrase_count_label.configure(text=text)
        except Exception:
            pass

    # ================= 本地词典 =================

    def select_dict_file(self):
        from tkinter import messagebox, filedialog

        try:
            choice = messagebox.askyesnocancel(
                "选择方式",
                "点【是】选择文件夹（自动扫描所有词典文件）\n"
                "点【否】选择单个文件\n"
                "点【取消】放弃",
            )

            if choice is None:
                return

            if choice:
                current = self.pipeline.local_dict.get_path()
                init_dir = current if os.path.isdir(current) else ''

                path = filedialog.askdirectory(
                    title="选择词典文件夹（自动扫描）",
                    initialdir=init_dir or None,
                )
            else:
                current = self.pipeline.local_dict.get_path()
                init_dir = os.path.dirname(current) if current else ''

                path = filedialog.askopenfilename(
                    title="选择词典文件",
                    initialdir=init_dir or None,
                    filetypes=[
                        ("词典文件", "*.txt *.csv *.json *.tsv"),
                        ("文本文件", "*.txt"),
                        ("CSV 文件", "*.csv"),
                        ("JSON 文件", "*.json"),
                        ("所有文件", "*.*"),
                    ],
                )

            if not path:
                return

            self.pipeline.local_dict.set_path(path)
            count = self.pipeline.local_dict.load(path)

            self.user_cfg.set('local_dict_path', path)
            self.user_cfg.save()

            self._update_dict_count()

            name = os.path.basename(path)
            if os.path.isdir(path):
                self.bottom_label.configure(
                    text=f"✅ 已加载词典文件夹：{name}"
                         f"（{count} 条词条）")
            else:
                self.bottom_label.configure(
                    text=f"✅ 已加载词典文件：{name}"
                         f"（{count} 条词条）")

            log.info(f"[本地词典] 用户选择：{path}")

        except Exception as e:
            log.exception(f"[本地词典] 选择失败：{e}")
            self.bottom_label.configure(text="选择本地词典失败")

    def reload_dict_file(self):
        try:
            path = self.pipeline.local_dict.get_path()
            if not path:
                self.bottom_label.configure(
                    text="⚠️ 未指定本地词典")
                return

            count = self.pipeline.local_dict.reload()
            self._update_dict_count()

            name = os.path.basename(path)
            self.bottom_label.configure(
                text=f"✅ 已重载：{name}（共 {count} 条）")
            log.info(f"[本地词典] 重载：{path}")
        except Exception as e:
            log.exception(f"[本地词典] 重载失败：{e}")
            self.bottom_label.configure(text="重载失败")

    def clear_dict_file(self):
        try:
            self.pipeline.local_dict.set_path('')
            self.pipeline.local_dict.clear()
            self.user_cfg.set('local_dict_path', '')
            self.user_cfg.save()
            self._update_dict_count()
            self.bottom_label.configure(text="已清除本地词典")
            log.info("[本地词典] 已清除")
        except Exception as e:
            log.exception(f"[本地词典] 清除失败：{e}")

    def _update_dict_count(self):
        try:
            n = self.pipeline.local_dict.size()
            path = self.pipeline.local_dict.get_path()
            stats = self.pipeline.local_dict.stats()

            if path:
                name = os.path.basename(path)
                is_folder = stats.get('is_folder', False)
                total_files = stats.get('total_files', 1)
                enc = stats.get('encodings', '?')
                skipped = stats.get('skipped', 0)

                if is_folder:
                    text = (f"共 {n} 条 | 文件夹 {name}"
                            f"（{total_files} 个文件）"
                            f" | {enc} | 跳过 {skipped} 行")
                else:
                    text = (f"共 {n} 条 | {name}"
                            f" | {enc} | 跳过 {skipped} 行")
            else:
                text = "未指定词典文件/文件夹"

            self.dict_count_label.configure(text=text)
        except Exception:
            pass

    # ================= 加速方式 =================

    def switch_accel_mode(self, mode, force=False):
        current_variant = check_onnxruntime_variant() or 'cpu'

        if mode == 'directml' and current_variant != 'directml':
            self._show_need_install('DirectML', 'onnxruntime-directml', '50MB')
            return
        if mode == 'cuda' and current_variant != 'gpu':
            self._show_need_install('NVIDIA CUDA', 'onnxruntime-gpu', '1.6GB')
            return

        if self.ocr_engine.mode == mode and not force:
            self.bottom_label.configure(text="已经是该模式")
            return

        old_running = self.app_state.running
        self.app_state.running = False

        self.accel_status_label.configure(
            text=f"⏳ 正在切换到 {mode}...", text_color='#666666')
        self.bottom_label.configure(text=f"切换到 {mode}...")
        self.update_idletasks()
        log.info(f"[加速] 切换：{self.ocr_engine.mode} → {mode}"
                 f"（GPU 引擎分配：{self.gpu_engine_counts}）")

        try:
            actual = self.ocr_engine.init(
                mode=mode,
                pool_size=MULTITHREAD_OCR_POOL_SIZE,
                gpu_engine_counts=self.gpu_engine_counts,
            )
            log.info(f"[加速] 切换完成：{actual} "
                     f"× {len(self.ocr_engine.engines)}"
                     f" devices={self.ocr_engine.engine_device_ids}")

            self.cache.clear()
            if hasattr(self.pipeline, 'inpaint_cache'):
                self.pipeline.inpaint_cache.clear()
            if hasattr(self.pipeline, 'fine_cache'):
                self.pipeline.fine_cache.clear()
            if hasattr(self.pipeline, 'color_cache'):
                self.pipeline.color_cache.clear()
            self.app_state.reset_signature()
            self.overlay.clear()

            self.user_cfg.set('accel_mode', actual)
            self.user_cfg.save()

            self._sync_accel_ui()
            n = len(self.ocr_engine.engines)
            self.bottom_label.configure(
                text=f"✅ 已切换到 {actual}（{n} 引擎）")
        except Exception as e:
            log.exception(f"[加速] 切换失败")
            self.bottom_label.configure(text=f"切换失败：{e}")
        finally:
            self.app_state.running = old_running

    def _show_need_install(self, name, pkg, size):
        from tkinter import messagebox
        messagebox.showinfo(
            f"需要安装 {name}",
            f"当前 onnxruntime 版本不支持 {name}。\n\n"
            f"需要先安装：{pkg}（约 {size}）\n\n"
            f"请使用源码运行（python main.py）后安装，\n"
            f"或下载对应版本的 exe。"
        )

    def _sync_accel_ui(self):
        try:
            current_variant = check_onnxruntime_variant() or 'cpu'
            current_mode = self.ocr_engine.mode
            n_engines = len(self.ocr_engine.engines)

            mode_map = {
                'cpu_multi': f'多线程 CPU（{n_engines} 引擎）',
                'directml': f'DirectML GPU（{n_engines} 引擎）',
                'cuda': 'NVIDIA CUDA',
            }
            self.ort_variant_label.configure(
                text=f"当前引擎：{mode_map.get(current_mode, current_mode)}")

            self._highlight_buttons(current_mode, current_variant)

            if current_mode == 'cuda':
                ok, msg = check_cuda_ready()
                if ok:
                    self.accel_status_label.configure(
                        text="✅ CUDA 加速运行中", text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            elif current_mode == 'directml':
                ok, msg = check_directml_ready()
                if ok:
                    detail = self.ocr_engine.get_mode_text()
                    self.accel_status_label.configure(
                        text=f"✅ DirectML 运行中：{detail}",
                        text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            else:
                self.accel_status_label.configure(
                    text=f"✅ CPU 多线程运行中（{n_engines} 引擎）",
                    text_color='#00a000')
        except Exception as e:
            log.exception("[加速] 状态刷新失败")
            self.accel_status_label.configure(
                text=f"状态获取失败：{e}", text_color='#cc0000')

    def _highlight_buttons(self, current_mode, variant):
        try:
            self.btn_cpu_multi.configure(
                fg_color='#28a745' if current_mode == 'cpu_multi' else '#6c757d',
                state='normal',
            )
            if variant == 'directml':
                self.btn_directml.configure(
                    fg_color='#28a745' if current_mode == 'directml' else '#3a86ff',
                    state='normal')
            else:
                self.btn_directml.configure(
                    fg_color='#adb5bd', state='disabled')
            if variant == 'gpu':
                self.btn_cuda.configure(
                    fg_color='#28a745' if current_mode == 'cuda' else '#28a745',
                    state='normal')
            else:
                self.btn_cuda.configure(
                    fg_color='#adb5bd', state='disabled')
        except Exception:
            pass

    # ================= GPU 引擎数 =================

    def _on_gpu_engine_change(self, device_id, value):
        v = int(round(float(value)))
        v = max(DIRECTML_ENGINE_COUNT_MIN,
                min(DIRECTML_ENGINE_COUNT_MAX, v))

        if device_id in self.gpu_engine_labels:
            self.gpu_engine_labels[device_id].configure(text=str(v))

        if not getattr(self, '_ui_ready', False):
            return

        while len(self.gpu_engine_counts) <= device_id:
            self.gpu_engine_counts.append(0)
        if self.gpu_engine_counts[device_id] == v:
            return
        self.gpu_engine_counts[device_id] = v

        if sum(self.gpu_engine_counts) < 1:
            self.gpu_engine_counts[device_id] = DIRECTML_ENGINE_COUNT_DEFAULT
            if device_id in self.gpu_engine_sliders:
                self.gpu_engine_sliders[device_id].set(
                    DIRECTML_ENGINE_COUNT_DEFAULT)
            if device_id in self.gpu_engine_labels:
                self.gpu_engine_labels[device_id].configure(
                    text=str(DIRECTML_ENGINE_COUNT_DEFAULT))
            self.bottom_label.configure(
                text="⚠️ 至少需要 1 个引擎")
            return

        self.user_cfg.set('gpu_engine_counts', list(self.gpu_engine_counts))
        self.user_cfg.set('gpu_device_names',
                          [d[1] for d in self.gpu_devices])
        self.user_cfg.save()

        if self.ocr_engine.mode == 'directml':
            self.bottom_label.configure(
                text=f"GPU 引擎分配已更新，正在重新加载...")
            self.after(
                300,
                lambda: self.switch_accel_mode('directml', force=True),
            )
        else:
            total = sum(self.gpu_engine_counts)
            self.bottom_label.configure(
                text=f"GPU 引擎分配：{self.gpu_engine_counts}"
                     f"（共 {total} 引擎，切到 DirectML 时生效）")

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

        fast_tag = "（分段翻译）" if cfg.get('fast_mode') else "（批量翻译）"
        self.bottom_label.configure(text=f"性能模式：{cfg['name']}{fast_tag}")
        log.info(f"[模式] 切换为 {cfg['name']} {fast_tag}")

        self.user_cfg.set('performance_mode', mode_key)
        self.user_cfg.set('scan_interval', cfg['scan_interval'])
        self.user_cfg.save()

    def _update_mode_desc(self, mode_key):
        cfg = PERFORMANCE_MODES[mode_key]
        mode_tag = "分段翻译" if cfg.get('fast_mode') else "批量翻译"
        desc = (
            f"扫描 {cfg['scan_interval']:.2f}s | "
            f"灵敏度 {cfg['change_ratio_threshold']:.2f} | "
            f"外扩 {cfg['bbox_padding']}px | "
            f"OCR {cfg['max_ocr_width']}×{cfg['max_ocr_height']} | "
            f"{mode_tag}"
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

    def _on_general_filter_toggle(self):
        enabled = bool(self.general_filter_switch.get())
        self.app_state.enable_general_filter = enabled
        self.app_state.reset_signature()
        msg = "通用过滤：已开启" if enabled else "通用过滤：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_general_filter', enabled)
        self.user_cfg.save()

    def _on_phrase_cache_toggle(self):
        enabled = bool(self.phrase_cache_switch.get())
        self.app_state.enable_phrase_cache = enabled
        self.app_state.reset_signature()
        msg = "短语缓存：已开启" if enabled else "短语缓存：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_phrase_cache', enabled)
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

    def _on_fine_text_toggle(self):
        enabled = bool(self.fine_text_switch.get())
        self.app_state.enable_fine_text = enabled
        self.overlay.clear()
        self.app_state.reset_signature()
        msg = ("精细文字替换：已启用" if enabled
               else "精细文字替换：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('enable_fine_text', enabled)
        self.user_cfg.save()

    def _on_async_overlay_toggle(self):
        enabled = bool(self.async_overlay_switch.get())
        self.overlay.set_async_enabled(enabled)
        msg = ("异步覆盖：已启用（逐字显示）" if enabled
               else "异步覆盖：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('enable_async_overlay', enabled)
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
        msg = "图像修复（整框）：已启用" if enabled else "图像修复（整框）：已关闭"
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
        if hasattr(self.pipeline, 'fine_cache'):
            self.pipeline.fine_cache.clear()
        if hasattr(self.pipeline, 'color_cache'):
            self.pipeline.color_cache.clear()
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
                 f"{hk['freeze']} 切换模式 | "
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
            mode = self.app_state.follow_mode
            is_pin = (mode == 'pin')
            is_follow = (mode == 'follow')
            is_fast = self.app_state.fast_mode

            if not is_follow:
                self.app_state.scroll_boost = False

            if not is_pin:
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
                    self.overlay.tick_animation()
                except Exception as e:
                    log.exception(f"[UI] 动画推进失败：{e}")

                try:
                    region = self.app_state.snapshot_region()
                    pruned = self.overlay.prune_out_of_bounds(
                        region, padding=BOUNDARY_PADDING)
                    if pruned:
                        log.info(f"[{mode}] 移除 {pruned} 个超出边界的译文")
                except Exception as e:
                    log.exception(f"[UI] 边界裁剪失败：{e}")

            now = time.time()

            if now - self._last_cache_ui_update >= 5:
                try:
                    self._update_cache_label()
                    self._update_phrase_count()
                    self._update_dict_count()
                except Exception:
                    pass
                self._last_cache_ui_update = now

            if self.app_state.running and not is_pin:
                base = self.app_state.scan_interval
                idle = self.app_state.idle_count

                try:
                    qsize = self.pipeline.task_queue.qsize()
                except Exception:
                    qsize = 0

                if is_fast:
                    if is_follow and self.scroll_detector.is_scrolling():
                        interval = max(0.03, base * 0.2)
                        self.app_state.scroll_boost = True
                    else:
                        interval = base * 0.3
                        self.app_state.scroll_boost = False

                    if qsize >= 1:
                        interval *= 2.0
                else:
                    pending = getattr(self.app_state, 'pending_change', False)

                    if is_follow and self.scroll_detector.is_scrolling():
                        interval = base * 0.25
                        self.app_state.scroll_boost = True
                    elif pending:
                        interval = base * 0.3
                        self.app_state.scroll_boost = False
                    else:
                        interval = base
                        self.app_state.scroll_boost = False
                        if idle >= IDLE_THRESHOLD_FOR_SLOWDOWN:
                            interval = base * IDLE_SLOWDOWN_FACTOR
                        elif idle >= 1:
                            interval = base * 1.5

                        if qsize >= 1:
                            interval *= 2.0

                if now - self._last_capture >= interval:
                    try:
                        self._do_capture()
                        self.app_state.pending_change = True
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
        log.info(f"========== 退出 v{__version__} ==========")

        try:
            self.user_cfg.set('window_geometry', self.geometry())
            self.user_cfg.save()
        except Exception as e:
            log.error(f"[退出] 保存配置失败：{e}")

        try:
            self.pipeline.phrase_cache.save_user_phrases()
        except Exception:
            pass

        try:
            self.scroll_detector.stop()
        except Exception:
            pass
        self.hk_mgr.remove_all()
        self.region_manager.destroy()
        self.overlay.clear()
        self.destroy()