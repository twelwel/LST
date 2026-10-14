"""LST - UI 模块
翻译策略：离线 / 单翻译 API / 多翻译轮询 / 混合 / 单 AI / 多 AI 投票
快速响应层：本地词典 + 短语缓存（所有策略共用）
显示模式：覆盖 / 双语 / 小窗口
译文颜色：全模式通用
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
    DEFAULT_ENABLE_FAST_LOOKUP,
    DEFAULT_PHRASE_TABLE_PATH,
    DEFAULT_LOCAL_DICT_PATH,
    PERFORMANCE_MODES, DEFAULT_MODE,
    IDLE_THRESHOLD_FOR_SLOWDOWN, IDLE_SLOWDOWN_FACTOR,
    DEFAULT_ENABLE_FRAME_DIFF, DEFAULT_ENABLE_FILTER,
    DEFAULT_ENABLE_BLOCK_OCR,
    DEFAULT_ENABLE_ASYNC_TRANSLATE, DEFAULT_ENABLE_WINDOW_POOL,
    CPU_ENGINE_COUNT_MIN,
    CPU_ENGINE_COUNT_MAX,
    CPU_ENGINE_COUNT_DEFAULT,
    TRANSLATION_STRATEGIES,
    TRANSLATION_STRATEGY_DESC,
    DEFAULT_TRANSLATION_STRATEGY,
    DISPLAY_MODES,
    DISPLAY_MODE_DESC,
    DEFAULT_DISPLAY_MODE,
    FG_COLOR_PRESETS,
    DEFAULT_FG_COLOR,
)
from ocr_engine import OCREngine
from cache import ParagraphCache
from pipeline import Pipeline, AppState
from overlay import OverlayWindow, CaptionWindow
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
from translation.strategies import (
    HYBRID_ORDER_LABELS, VOTE_STRATEGY_LABELS,
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

        self.content = ctk.CTkFrame(self, fg_color='transparent', height=1)
        self.content.pack_propagate(True)
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

        geo = self.user_cfg.get('window_geometry', "760x800+80+40")
        self.geometry(geo)
        self.minsize(700, 560)
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
        saved_fast_lookup = self.user_cfg.get(
            'enable_fast_lookup', DEFAULT_ENABLE_FAST_LOOKUP)
        saved_phrase_table_path = self.user_cfg.get(
            'phrase_table_path', DEFAULT_PHRASE_TABLE_PATH)
        saved_local_dict_path = self.user_cfg.get(
            'local_dict_path', DEFAULT_LOCAL_DICT_PATH)
        saved_async = self.user_cfg.get('enable_async_translate',
                                        DEFAULT_ENABLE_ASYNC_TRANSLATE)
        saved_wpool = self.user_cfg.get('enable_window_pool',
                                        DEFAULT_ENABLE_WINDOW_POOL)

        # CPU 引擎数
        saved_cpu_engine = self.user_cfg.get(
            'cpu_engine_count', CPU_ENGINE_COUNT_DEFAULT)
        try:
            saved_cpu_engine = int(saved_cpu_engine)
        except Exception:
            saved_cpu_engine = CPU_ENGINE_COUNT_DEFAULT
        saved_cpu_engine = max(CPU_ENGINE_COUNT_MIN,
                               min(CPU_ENGINE_COUNT_MAX, saved_cpu_engine))
        self.cpu_engine_count = saved_cpu_engine

        # 翻译策略
        saved_translation_strategy = self.user_cfg.get(
            'translation_strategy', DEFAULT_TRANSLATION_STRATEGY)
        if saved_translation_strategy not in TRANSLATION_STRATEGIES:
            saved_translation_strategy = DEFAULT_TRANSLATION_STRATEGY

        try:
            from translation import load_strategy, save_strategy
            _st = load_strategy()
            if _st.get('mode') != saved_translation_strategy:
                _st['mode'] = saved_translation_strategy
                save_strategy(_st)
                log.info(f"[启动] 策略同步为 {saved_translation_strategy}")
        except Exception as e:
            log.exception(f"[启动] 策略同步失败：{e}")

        # 显示模式
        saved_display_mode = self.user_cfg.get(
            'display_mode', DEFAULT_DISPLAY_MODE)
        if saved_display_mode not in DISPLAY_MODES:
            saved_display_mode = DEFAULT_DISPLAY_MODE

        # 译文颜色
        saved_fg_color = self.user_cfg.get('fg_color', DEFAULT_FG_COLOR)
        if saved_fg_color not in FG_COLOR_PRESETS:
            saved_fg_color = DEFAULT_FG_COLOR

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

        # GPU 枚举（用于显示）
        self.gpu_devices = list_gpu_devices()

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
            pool_size=self.cpu_engine_count,
        )
        log.info(f"OCR 引擎初始化：{variant} × "
                 f"{len(self.ocr_engine.engines)}")

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
            enable_fast_lookup=saved_fast_lookup,
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
        self.app_state.translation_strategy = saved_translation_strategy
        self.app_state.display_mode = saved_display_mode
        self.app_state.fg_color = saved_fg_color

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

        self.caption_window = CaptionWindow(
            self, font_name=FONT_NAME,
            on_close=self._on_caption_closed,
        )

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

        # 翻译策略 UI 引用
        self._strategy_labels = list(TRANSLATION_STRATEGIES.values())
        self._strategy_label_to_key = {
            v: k for k, v in TRANSLATION_STRATEGIES.items()
        }
        self._hybrid_order_rev = {v: k for k, v in HYBRID_ORDER_LABELS.items()}
        self._vote_strategy_rev = {
            v: k for k, v in VOTE_STRATEGY_LABELS.items()
        }

        # 显示模式 UI 引用
        self._display_labels = list(DISPLAY_MODES.values())
        self._display_label_to_key = {
            v: k for k, v in DISPLAY_MODES.items()
        }

        # 译文颜色 UI 引用
        self._fg_label_to_key = {
            v: k for k, v in FG_COLOR_PRESETS.items()
        }

        # UI
        self._build_ui()

        self.after(100, self._sync_accel_ui)
        self.after(100, self._sync_follow_mode_ui)
        self._restore_ui_from_config(
            saved_mode, saved_scan, initial_font, saved_inpaint,
            saved_fine_text, saved_async_overlay,
            saved_general_filter, saved_fast_lookup,
            saved_phrase_table_path,
            saved_translation_strategy, saved_local_dict_path,
            saved_frame_diff, saved_block, saved_filter, saved_async,
            saved_wpool, saved_cache_max, saved_auto_clean,
        )

        self._ui_ready = True
        self.after(TICK_MS, self._tick)

    # ================= 恢复 UI =================

    def _restore_ui_from_config(self, mode, scan, font, inpaint,
                                fine_text, async_overlay,
                                general_filter, fast_lookup,
                                phrase_table_path,
                                translation_strategy, local_dict_path,
                                frame_diff, block, filt, async_t,
                                wpool, cache_max, auto_clean):
        try:
            self.mode_selector.set(PERFORMANCE_MODES[mode]['name'])
            self._update_mode_desc(mode)

            self.scan_slider.set(scan)
            self.scan_value_label.configure(text=f"{scan:.2f} s")
            self.font_slider.set(font)
            self.font_value_label.configure(text=str(font))

            # CPU 引擎数滑块
            self.cpu_count_slider.set(self.cpu_engine_count)
            self.cpu_count_label.configure(text=str(self.cpu_engine_count))

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

            if fast_lookup:
                self.fast_lookup_switch.select()
            else:
                self.fast_lookup_switch.deselect()

            self._sync_strategy_ui()
            self._sync_display_mode_ui()
            self._sync_fg_color_ui()

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

            self._update_phrase_count()
            self._update_dict_count()

            if getattr(self.app_state, 'display_mode',
                       'overlay') == 'window':
                self.caption_window.show()
            else:
                self.caption_window.hide()
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
        # 顶部状态栏
        top = ctk.CTkFrame(self, height=1)
        top.pack(fill='x', padx=10, pady=(8, 4))
        top.pack_propagate(True)

        self.status_label = ctk.CTkLabel(
            top, text="状态：已暂停",
            font=(FONT_NAME, 13, 'bold'), anchor='w',
        )
        self.status_label.pack(side='left', padx=8)

        self.version_label = ctk.CTkLabel(
            top, text=f"{APP_NAME} v{__version__}",
            font=(FONT_NAME, 10), text_color='#3a86ff',
        )
        self.version_label.pack(side='right', padx=8)

        self.hotkey_hint = ctk.CTkLabel(
            top, text='', font=(FONT_NAME, 9), text_color='gray',
        )
        self.hotkey_hint.pack(side='right', padx=8)

        # ===== 翻译策略 =====
        strategy_frame = ctk.CTkFrame(self, height=1)
        strategy_frame.pack(fill='x', padx=10, pady=(4, 2))
        strategy_frame.pack_propagate(True)

        row1 = ctk.CTkFrame(strategy_frame, fg_color='transparent', height=1)
        row1.pack(fill='x')
        row1.pack_propagate(True)

        ctk.CTkLabel(
            row1, text="翻译策略：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 6))

        self.strategy_selector = ctk.CTkOptionMenu(
            row1,
            values=self._strategy_labels,
            command=self._on_strategy_change,
            font=(FONT_NAME, 11),
            width=170,
        )
        self.strategy_selector.set(
            TRANSLATION_STRATEGIES[DEFAULT_TRANSLATION_STRATEGY])
        self.strategy_selector.pack(side='left', padx=4)

        self.strategy_desc = ctk.CTkLabel(
            row1, text="",
            font=(FONT_NAME, 9),
            text_color='#666666', anchor='w',
        )
        self.strategy_desc.pack(side='left', padx=6)

        # 子参数行
        self.sub_param_frame = ctk.CTkFrame(
            strategy_frame, fg_color='transparent', height=1)
        self.sub_param_frame.pack(fill='x', pady=(2, 0))
        self.sub_param_frame.pack_propagate(True)

        self.hybrid_order_label = ctk.CTkLabel(
            self.sub_param_frame, text="混合顺序：",
            font=(FONT_NAME, 10), text_color='#555555',
        )
        self.hybrid_order_selector = ctk.CTkOptionMenu(
            self.sub_param_frame,
            values=list(HYBRID_ORDER_LABELS.values()),
            command=self._on_hybrid_order_change,
            font=(FONT_NAME, 10), width=110,
        )
        self.hybrid_order_selector.set('翻译优先')

        self.vote_count_label = ctk.CTkLabel(
            self.sub_param_frame, text="投票数：",
            font=(FONT_NAME, 10), text_color='#555555',
        )
        self.vote_count_selector = ctk.CTkOptionMenu(
            self.sub_param_frame,
            values=['2', '3', '4', '5'],
            command=self._on_vote_count_change,
            font=(FONT_NAME, 10), width=55,
        )
        self.vote_count_selector.set('3')

        self.vote_strategy_label = ctk.CTkLabel(
            self.sub_param_frame, text="投票策略：",
            font=(FONT_NAME, 10), text_color='#555555',
        )
        self.vote_strategy_selector = ctk.CTkOptionMenu(
            self.sub_param_frame,
            values=list(VOTE_STRATEGY_LABELS.values()),
            command=self._on_vote_strategy_change,
            font=(FONT_NAME, 10), width=110,
        )
        self.vote_strategy_selector.set('相似度居中')

        # 提示行
        hint_frame = ctk.CTkFrame(self, fg_color='transparent', height=1)
        hint_frame.pack(fill='x', padx=10, pady=(0, 2))
        hint_frame.pack_propagate(True)
        ctk.CTkLabel(
            hint_frame,
            text="💡 所有策略都会先查快速响应层（本地词典 + 短语缓存），"
                 "仅处理单词和词组，未命中才按策略走网络",
            font=(FONT_NAME, 9),
            text_color='#888888', anchor='w',
        ).pack(side='left', padx=6)

        # ===== 译文模式 =====
        follow_frame = ctk.CTkFrame(self, height=1)
        follow_frame.pack(fill='x', padx=10, pady=3)
        follow_frame.pack_propagate(True)

        ctk.CTkLabel(
            follow_frame, text="译文模式：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 6))

        self.follow_selector = ctk.CTkSegmentedButton(
            follow_frame,
            values=["默认", "跟随", "钉住"],
            command=self._on_follow_mode_change,
            font=(FONT_NAME, 11),
        )
        self.follow_selector.pack(side='left', padx=4)

        self.follow_desc_label = ctk.CTkLabel(
            follow_frame, text="", font=(FONT_NAME, 9),
            text_color='#666666', anchor='w',
        )
        self.follow_desc_label.pack(side='left', padx=8)

        # ===== 显示模式 =====
        display_frame = ctk.CTkFrame(self, height=1)
        display_frame.pack(fill='x', padx=10, pady=3)
        display_frame.pack_propagate(True)

        ctk.CTkLabel(
            display_frame, text="显示模式：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 6))

        self.display_selector = ctk.CTkSegmentedButton(
            display_frame,
            values=self._display_labels,
            command=self._on_display_mode_change,
            font=(FONT_NAME, 11),
        )
        self.display_selector.set(DISPLAY_MODES[DEFAULT_DISPLAY_MODE])
        self.display_selector.pack(side='left', padx=4)

        self.display_desc_label = ctk.CTkLabel(
            display_frame, text="", font=(FONT_NAME, 9),
            text_color='#666666', anchor='w',
        )
        self.display_desc_label.pack(side='left', padx=8)

        # ===== 按钮行 1 =====
        btn_frame = ctk.CTkFrame(self, height=1)
        btn_frame.pack(fill='x', padx=10, pady=(4, 2))
        btn_frame.pack_propagate(True)

        self.btn_toggle = ctk.CTkButton(
            btn_frame, text="▶ 开始", width=90, height=30,
            command=self.toggle_running,
        )
        self.btn_toggle.pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame, text="清空译文", width=80, height=30,
            command=self.clear_overlay,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame, text="修改区域", width=80, height=30,
            command=self.open_config,
        ).pack(side='left', padx=3)

        self.btn_border = ctk.CTkButton(
            btn_frame, text="隐藏边框", width=80, height=30,
            command=self.toggle_border,
        )
        self.btn_border.pack(side='left', padx=3)

        # ===== 按钮行 2 =====
        btn_frame2 = ctk.CTkFrame(self, height=1)
        btn_frame2.pack(fill='x', padx=10, pady=2)
        btn_frame2.pack_propagate(True)

        ctk.CTkButton(
            btn_frame2, text="API 设置", width=80, height=28,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.open_api_config,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="打开日志", width=80, height=28,
            fg_color='#17a2b8', hover_color='#138496',
            command=self.open_logs,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="清空日志", width=80, height=28,
            fg_color='#dc3545', hover_color='#c82333',
            command=self.clear_logs_ui,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="打开配置", width=80, height=28,
            fg_color='#6f42c1', hover_color='#5a35a0',
            command=self.open_config_dir_ui,
        ).pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame2, text="关于", width=60, height=28,
            fg_color='#495057', hover_color='#343a40',
            command=self.open_about,
        ).pack(side='left', padx=3)

        # ===== 区域标签 =====
        self.region_label = ctk.CTkLabel(
            self, text='', font=(FONT_NAME, 9), text_color='gray',
        )
        self.region_label.pack(pady=(0, 3))
        self._update_region_label()

        # ===== 设置面板 =====
        self.scroll_frame = ctk.CTkScrollableFrame(
            self, fg_color='transparent',
        )
        self.scroll_frame.pack(fill='both', expand=True, padx=5, pady=3)

        self.display_panel = CollapsibleFrame(
            self.scroll_frame, title="显示设置", expanded=True,
        )
        self.display_panel.pack(fill='x', padx=5, pady=(3, 2))
        self._build_display_tab(self.display_panel.content)

        self.hotkey_panel = CollapsibleFrame(
            self.scroll_frame, title="热键设置", expanded=False,
        )
        self.hotkey_panel.pack(fill='x', padx=5, pady=2)
        self._build_hotkey_tab(self.hotkey_panel.content)

        # ===== 底部状态 =====
        self.bottom_label = ctk.CTkLabel(
            self, text="准备就绪", font=(FONT_NAME, 10), text_color='gray',
        )
        self.bottom_label.pack(side='bottom', pady=4)

        self._refresh_hotkey_hint()
        self._sync_strategy_ui()
        self._sync_display_mode_ui()
        self._sync_fg_color_ui()

    def _build_display_tab(self, parent):
        # ===== 性能模式 =====
        ctk.CTkLabel(
            parent, text="性能模式：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=0, column=0, sticky='w', padx=(15, 5), pady=(12, 6))

        self.mode_selector = ctk.CTkSegmentedButton(
            parent,
            values=["快", "中", "慢"],
            command=self._on_mode_change,
            font=(FONT_NAME, 11),
        )
        self.mode_selector.set(PERFORMANCE_MODES[DEFAULT_MODE]['name'])
        self.mode_selector.grid(
            row=0, column=1, columnspan=2,
            sticky='ew', padx=10, pady=(12, 6),
        )

        self.mode_desc_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
        )
        self.mode_desc_label.grid(
            row=1, column=0, columnspan=3,
            sticky='w', padx=(15, 15), pady=(0, 8),
        )
        self._update_mode_desc(DEFAULT_MODE)

        # ===== 加速方式 =====
        ctk.CTkLabel(
            parent, text="加速方式（运行时切换）：",
            font=(FONT_NAME, 11, 'bold'),
        ).grid(row=2, column=0, columnspan=3,
               sticky='w', padx=(15, 5), pady=(6, 4))

        self.ort_variant_label = ctk.CTkLabel(
            parent, text="当前引擎：检测中...",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w',
        )
        self.ort_variant_label.grid(
            row=3, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 4))

        accel_btns = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        accel_btns.grid(
            row=4, column=0, columnspan=3,
            sticky='ew', padx=(15, 15), pady=(0, 4))
        accel_btns.pack_propagate(True)

        self.btn_cpu_multi = ctk.CTkButton(
            accel_btns, text="多线程 CPU",
            width=140, height=48,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=lambda: self.switch_accel_mode('cpu_multi'),
        )
        self.btn_cpu_multi.pack(side='left', padx=3)

        self.btn_directml = ctk.CTkButton(
            accel_btns, text="DirectML GPU\n(单引擎)",
            width=140, height=48,
            font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=lambda: self.switch_accel_mode('directml'),
        )
        self.btn_directml.pack(side='left', padx=3)

        self.btn_cuda = ctk.CTkButton(
            accel_btns, text="NVIDIA CUDA\n(单引擎)",
            width=140, height=48,
            font=(FONT_NAME, 10),
            fg_color='#28a745', hover_color='#1e7e34',
            command=lambda: self.switch_accel_mode('cuda'),
        )
        self.btn_cuda.pack(side='left', padx=3)

        # ===== CPU 引擎数 =====
        cpu_row = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        cpu_row.grid(
            row=5, column=0, columnspan=3,
            sticky='ew', padx=(25, 15), pady=(6, 4))
        cpu_row.pack_propagate(True)

        ctk.CTkLabel(
            cpu_row, text="CPU 引擎数：",
            font=(FONT_NAME, 10), text_color='#333333', anchor='w',
        ).pack(side='left')

        self.cpu_count_label = ctk.CTkLabel(
            cpu_row, text=str(CPU_ENGINE_COUNT_DEFAULT), width=26,
            font=(FONT_NAME, 10, 'bold'), text_color='#3a86ff',
        )
        self.cpu_count_label.pack(side='right')

        self.cpu_count_slider = ctk.CTkSlider(
            cpu_row,
            from_=CPU_ENGINE_COUNT_MIN,
            to=CPU_ENGINE_COUNT_MAX,
            number_of_steps=CPU_ENGINE_COUNT_MAX - CPU_ENGINE_COUNT_MIN,
            command=self._on_cpu_count_change,
            height=14, width=160,
        )
        self.cpu_count_slider.set(CPU_ENGINE_COUNT_DEFAULT)
        self.cpu_count_slider.pack(side='right', padx=4)

        ctk.CTkLabel(
            parent,
            text="💡 CPU 引擎数仅在「多线程 CPU」模式生效；"
                 "DirectML / CUDA 固定单引擎",
            font=(FONT_NAME, 9), text_color='#888888', anchor='w',
        ).grid(row=6, column=0, columnspan=3,
               sticky='w', padx=(25, 15), pady=(0, 6))

        # ===== 加速状态 =====
        self.accel_status_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
            wraplength=600,
        )
        self.accel_status_label.grid(
            row=7, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 8))

        # ===== 优化功能 =====
        ctk.CTkLabel(
            parent, text="优化功能：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=8, column=0, sticky='w', padx=(15, 5), pady=(6, 4))

        self.frame_diff_switch = ctk.CTkSwitch(
            parent, text="帧差检测（静止时跳过）",
            command=self._on_frame_diff_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FRAME_DIFF:
            self.frame_diff_switch.select()
        self.frame_diff_switch.grid(
            row=9, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        self.block_ocr_switch = ctk.CTkSwitch(
            parent, text="分块 OCR（只识别变化的区域）",
            command=self._on_block_ocr_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_BLOCK_OCR:
            self.block_ocr_switch.select()
        self.block_ocr_switch.grid(
            row=10, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        self.filter_switch = ctk.CTkSwitch(
            parent, text="文字过滤（过滤异常框）",
            command=self._on_filter_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FILTER:
            self.filter_switch.select()
        self.filter_switch.grid(
            row=11, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        self.general_filter_switch = ctk.CTkSwitch(
            parent, text="通用过滤（艺术字/数字/单位）",
            command=self._on_general_filter_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_GENERAL_FILTER:
            self.general_filter_switch.select()
        self.general_filter_switch.grid(
            row=12, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        self.fast_lookup_switch = ctk.CTkSwitch(
            parent, text="快速响应层（本地词典 + 短语缓存，只处理单词和词组）",
            command=self._on_fast_lookup_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FAST_LOOKUP:
            self.fast_lookup_switch.select()
        self.fast_lookup_switch.grid(
            row=13, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        # 短语表按钮
        phrase_btns = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        phrase_btns.grid(
            row=14, column=0, columnspan=3,
            sticky='ew', padx=(45, 15), pady=(0, 6))
        phrase_btns.pack_propagate(True)

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
        ).grid(row=15, column=0, sticky='w', padx=(15, 5), pady=(6, 4))

        dict_btns = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        dict_btns.grid(
            row=16, column=0, columnspan=3,
            sticky='ew', padx=(25, 15), pady=(0, 6))
        dict_btns.pack_propagate(True)

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

        # 异步翻译
        self.async_translate_switch = ctk.CTkSwitch(
            parent, text="异步翻译（不阻塞 OCR）",
            command=self._on_async_translate_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_ASYNC_TRANSLATE:
            self.async_translate_switch.select()
        self.async_translate_switch.grid(
            row=17, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 3))

        # 窗口池
        self.window_pool_switch = ctk.CTkSwitch(
            parent, text="窗口池（复用译文窗口）",
            command=self._on_window_pool_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_WINDOW_POOL:
            self.window_pool_switch.select()
        self.window_pool_switch.grid(
            row=18, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 8))

        # 刷新间隔
        ctk.CTkLabel(
            parent, text="刷新间隔 (秒)：", font=(FONT_NAME, 11),
        ).grid(row=19, column=0, sticky='w', padx=(15, 5), pady=6)

        self.scan_value_label = ctk.CTkLabel(
            parent,
            text=f"{PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval']:.2f} s",
            width=60, font=(FONT_NAME, 11),
        )
        self.scan_value_label.grid(row=19, column=2, sticky='e', padx=(5, 15))

        n_steps = int((SCAN_MAX - SCAN_MIN) / 0.05)
        self.scan_slider = ctk.CTkSlider(
            parent, from_=SCAN_MIN, to=SCAN_MAX,
            number_of_steps=n_steps,
            command=self._on_scan_change,
        )
        self.scan_slider.set(PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval'])
        self.scan_slider.grid(row=19, column=1, sticky='ew', padx=10)

        # 文字大小
        ctk.CTkLabel(
            parent, text="文字大小：", font=(FONT_NAME, 11),
        ).grid(row=20, column=0, sticky='w', padx=(15, 5), pady=6)

        self.font_value_label = ctk.CTkLabel(
            parent, text=str(INITIAL_FONT_SIZE), width=60,
            font=(FONT_NAME, 11),
        )
        self.font_value_label.grid(row=20, column=2, sticky='e', padx=(5, 15))

        self.font_slider = ctk.CTkSlider(
            parent, from_=FONT_MIN, to=FONT_MAX,
            number_of_steps=FONT_MAX - FONT_MIN,
            command=self._on_font_change,
        )
        self.font_slider.set(INITIAL_FONT_SIZE)
        self.font_slider.grid(row=20, column=1, sticky='ew', padx=10)

        # 译文颜色
        ctk.CTkLabel(
            parent, text="译文颜色：", font=(FONT_NAME, 11),
        ).grid(row=21, column=0, sticky='w', padx=(15, 5), pady=6)

        self.fg_color_selector = ctk.CTkOptionMenu(
            parent,
            values=list(FG_COLOR_PRESETS.values()),
            command=self._on_fg_color_change,
            font=(FONT_NAME, 10),
            width=200,
        )
        self.fg_color_selector.set(FG_COLOR_PRESETS[DEFAULT_FG_COLOR])
        self.fg_color_selector.grid(
            row=21, column=1, columnspan=2,
            sticky='w', padx=10, pady=6)

        # 图像修复
        ctk.CTkLabel(
            parent, text="图像修复（整框）：", font=(FONT_NAME, 11),
        ).grid(row=22, column=0, sticky='w', padx=(15, 5), pady=(12, 6))

        self.inpaint_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_inpaint_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_USE_INPAINTING and CV2_AVAILABLE:
            self.inpaint_switch.select()
        self.inpaint_switch.grid(row=22, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.inpaint_switch.configure(state='disabled')

        # 精细文字替换
        ctk.CTkLabel(
            parent, text="精细文字替换（保留背景）：",
            font=(FONT_NAME, 11),
        ).grid(row=23, column=0, sticky='w', padx=(15, 5), pady=(8, 6))

        self.fine_text_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_fine_text_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_FINE_TEXT and CV2_AVAILABLE:
            self.fine_text_switch.select()
        self.fine_text_switch.grid(row=23, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.fine_text_switch.configure(state='disabled')

        # 异步覆盖
        ctk.CTkLabel(
            parent, text="异步覆盖（逐字显示）：",
            font=(FONT_NAME, 11),
        ).grid(row=24, column=0, sticky='w', padx=(15, 5), pady=(8, 6))

        self.async_overlay_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_async_overlay_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_ASYNC_OVERLAY:
            self.async_overlay_switch.select()
        self.async_overlay_switch.grid(row=24, column=1, sticky='w', padx=10)

        # 自动清理
        ctk.CTkLabel(
            parent, text="自动清理缓存：", font=(FONT_NAME, 11),
        ).grid(row=25, column=0, sticky='w', padx=(15, 5), pady=(12, 6))

        self.auto_clean_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_auto_clean_toggle,
            onvalue=True, offvalue=False,
        )
        self.auto_clean_switch.select()
        self.auto_clean_switch.grid(row=25, column=1, sticky='w', padx=10)

        # 缓存上限
        ctk.CTkLabel(
            parent, text="缓存上限：", font=(FONT_NAME, 11),
        ).grid(row=26, column=0, sticky='w', padx=(15, 5), pady=6)

        self.cache_limit_label = ctk.CTkLabel(
            parent, text=str(CACHE_DEFAULT), width=60,
            font=(FONT_NAME, 11),
        )
        self.cache_limit_label.grid(row=26, column=2, sticky='e', padx=(5, 15))

        self.cache_limit_slider = ctk.CTkSlider(
            parent, from_=CACHE_MIN, to=CACHE_MAX,
            number_of_steps=(CACHE_MAX - CACHE_MIN) // 50,
            command=self._on_cache_limit_change,
        )
        self.cache_limit_slider.set(CACHE_DEFAULT)
        self.cache_limit_slider.grid(row=26, column=1, sticky='ew', padx=10)

        cache_row = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        cache_row.grid(row=27, column=0, columnspan=3,
                       sticky='ew', padx=(15, 15), pady=(4, 12))
        cache_row.pack_propagate(True)

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
            ).grid(row=i, column=0, sticky='w', padx=(15, 5), pady=6)

            entry = ctk.CTkEntry(parent, width=180)
            entry.insert(0, self.hotkeys[key])
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=6)
            self.hotkey_entries[key] = entry

            ctk.CTkButton(
                parent, text="捕获", width=70,
                command=lambda k=key: self._capture_hotkey(k),
            ).grid(row=i, column=2, sticky='w', padx=5, pady=6)

        ctk.CTkButton(
            parent, text="应用所有热键", width=140,
            command=self._apply_hotkeys,
        ).grid(row=len(HOTKEY_LABELS), column=0, columnspan=3,
               pady=(12, 8))

        parent.grid_columnconfigure(1, weight=1)

    # ================= CPU 引擎数 =================

    def _on_cpu_count_change(self, value):
        v = int(round(float(value)))
        v = max(CPU_ENGINE_COUNT_MIN, min(CPU_ENGINE_COUNT_MAX, v))
        self.cpu_count_label.configure(text=str(v))

        if not getattr(self, '_ui_ready', False):
            return

        if v == self.cpu_engine_count:
            return

        self.cpu_engine_count = v
        self.user_cfg.set('cpu_engine_count', v)
        self.user_cfg.save()

        # 当前是 CPU 模式 → 立即重载
        if self.ocr_engine.mode == 'cpu_multi':
            self.bottom_label.configure(
                text=f"CPU 引擎数改为 {v}，正在重新加载...")
            self.after(
                200,
                lambda: self.switch_accel_mode('cpu_multi', force=True),
            )
        else:
            self.bottom_label.configure(
                text=f"CPU 引擎数设为 {v}（切到 CPU 模式时生效）")

    # ================= 翻译策略 =================

    def _on_strategy_change(self, label):
        key = self._strategy_label_to_key.get(
            label, DEFAULT_TRANSLATION_STRATEGY)
        self.app_state.translation_strategy = key
        self._sync_strategy_ui()
        self._update_sub_params()

        self.user_cfg.set('translation_strategy', key)
        self.user_cfg.save()

        try:
            from translation import load_strategy, save_strategy
            cfg = load_strategy()
            cfg['mode'] = key
            save_strategy(cfg)
            self.translator.reload_config()
            log.info(f"[翻译策略] 已同步到 manager：mode={key}")
        except Exception as e:
            log.exception(f"[翻译策略] 同步失败：{e}")

        self.overlay.clear()
        try:
            self.caption_window.clear()
        except Exception:
            pass
        self.app_state.reset_signature()

        try:
            status = self.translator._manager.get_status_text()
        except Exception:
            status = label
        log.info(f"[翻译策略] 切换为 {key} | {status}")
        self.bottom_label.configure(text=f"翻译策略：{label} | {status}")

    def _sync_strategy_ui(self):
        key = getattr(self.app_state, 'translation_strategy',
                      DEFAULT_TRANSLATION_STRATEGY)
        label = TRANSLATION_STRATEGIES.get(key, '单翻译 API')
        try:
            self.strategy_selector.set(label)
        except Exception:
            pass
        try:
            self.strategy_desc.configure(
                text=TRANSLATION_STRATEGY_DESC.get(key, ''))
        except Exception:
            pass

        try:
            self._sync_subparams_from_cfg()
        except Exception:
            pass
        try:
            self._update_sub_params()
        except Exception:
            pass

    def _update_sub_params(self):
        mode = getattr(self.app_state, 'translation_strategy',
                       DEFAULT_TRANSLATION_STRATEGY)
        for w in (self.hybrid_order_label, self.hybrid_order_selector,
                  self.vote_count_label, self.vote_count_selector,
                  self.vote_strategy_label, self.vote_strategy_selector):
            try:
                w.pack_forget()
            except Exception:
                pass

        if mode == 'hybrid':
            self.hybrid_order_label.pack(side='left', padx=(30, 4))
            self.hybrid_order_selector.pack(side='left', padx=2)
        elif mode == 'multi_ai_vote':
            self.vote_count_label.pack(side='left', padx=(30, 4))
            self.vote_count_selector.pack(side='left', padx=2)
            self.vote_strategy_label.pack(side='left', padx=(12, 4))
            self.vote_strategy_selector.pack(side='left', padx=2)

    def _sync_subparams_from_cfg(self):
        from translation import load_strategy
        cfg = load_strategy()
        order = cfg.get('hybrid_order', 'translate_first')
        try:
            self.hybrid_order_selector.set(
                HYBRID_ORDER_LABELS.get(order, '翻译优先'))
        except Exception:
            pass
        try:
            self.vote_count_selector.set(str(cfg.get('vote_count', 3)))
        except Exception:
            self.vote_count_selector.set('3')
        vs = cfg.get('vote_strategy', 'similarity')
        try:
            self.vote_strategy_selector.set(
                VOTE_STRATEGY_LABELS.get(vs, '相似度居中'))
        except Exception:
            pass

    def _save_strategy_subparams(self):
        try:
            from translation import load_strategy, save_strategy
            cfg = load_strategy()
            cfg['mode'] = getattr(self.app_state, 'translation_strategy',
                                  'single_translate')
            cfg['hybrid_order'] = self._hybrid_order_rev.get(
                self.hybrid_order_selector.get(), 'translate_first')
            try:
                cfg['vote_count'] = int(self.vote_count_selector.get())
            except Exception:
                cfg['vote_count'] = 3
            cfg['vote_strategy'] = self._vote_strategy_rev.get(
                self.vote_strategy_selector.get(), 'similarity')
            save_strategy(cfg)
            self.translator.reload_config()
            log.info(f"[策略子参数] 已保存：{cfg}")
        except Exception as e:
            log.exception(f"[策略子参数] 保存失败：{e}")

    def _on_hybrid_order_change(self, label):
        self._save_strategy_subparams()
        self.bottom_label.configure(text=f"混合顺序：{label}")

    def _on_vote_count_change(self, label):
        self._save_strategy_subparams()
        self.bottom_label.configure(text=f"投票数：{label}")

    def _on_vote_strategy_change(self, label):
        self._save_strategy_subparams()
        self.bottom_label.configure(text=f"投票策略：{label}")

    # ================= 显示模式 =================

    def _on_display_mode_change(self, label):
        key = self._display_label_to_key.get(
            label, DEFAULT_DISPLAY_MODE)
        self.app_state.display_mode = key
        self._sync_display_mode_ui()

        self.user_cfg.set('display_mode', key)
        self.user_cfg.save()

        self.overlay.clear()
        if key == 'window':
            self.caption_window.show()
        else:
            self.caption_window.hide()

        self.app_state.reset_signature()
        log.info(f"[显示模式] 切换为 {key}")
        self.bottom_label.configure(text=f"显示模式：{label}")

    def _sync_display_mode_ui(self):
        key = getattr(self.app_state, 'display_mode', DEFAULT_DISPLAY_MODE)
        label = DISPLAY_MODES.get(key, '覆盖')
        try:
            self.display_selector.set(label)
        except Exception:
            pass
        try:
            self.display_desc_label.configure(
                text=DISPLAY_MODE_DESC.get(key, ''))
        except Exception:
            pass

    def _on_caption_closed(self):
        if self.app_state.display_mode != 'window':
            return
        self.app_state.display_mode = 'overlay'
        self._sync_display_mode_ui()
        self.user_cfg.set('display_mode', 'overlay')
        self.user_cfg.save()
        self.bottom_label.configure(
            text="小窗口已关闭，显示模式切回覆盖")

    # ================= 译文颜色 =================

    def _on_fg_color_change(self, label):
        key = self._fg_label_to_key.get(label, DEFAULT_FG_COLOR)
        self.app_state.fg_color = key
        self.user_cfg.set('fg_color', key)
        self.user_cfg.save()

        self.overlay.clear()
        self.app_state.reset_signature()

        log.info(f"[译文颜色] 切换为 {key}")
        self.bottom_label.configure(text=f"译文颜色：{label}")

    def _sync_fg_color_ui(self):
        key = getattr(self.app_state, 'fg_color', DEFAULT_FG_COLOR)
        label = FG_COLOR_PRESETS.get(key, '自动（按背景反色）')
        try:
            self.fg_color_selector.set(label)
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
        log.info(f"[加速] 切换：{self.ocr_engine.mode} → {mode}")

        try:
            actual = self.ocr_engine.init(
                mode=mode,
                pool_size=self.cpu_engine_count,
            )
            log.info(f"[加速] 切换完成：{actual} "
                     f"× {len(self.ocr_engine.engines)}")

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
                'directml': 'DirectML GPU（单引擎）',
                'cuda': 'NVIDIA CUDA（单引擎）',
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
                    self.accel_status_label.configure(
                        text="✅ DirectML 运行中（单引擎）",
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

    def _on_fast_lookup_toggle(self):
        enabled = bool(self.fast_lookup_switch.get())
        self.app_state.enable_fast_lookup = enabled
        self.app_state.reset_signature()
        msg = "快速响应层：已开启" if enabled else "快速响应层：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_fast_lookup', enabled)
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
        try:
            from api_setup_dialog import APISetupDialog
            dlg = APISetupDialog()
            dlg.mainloop()
            result = dlg.result
        except Exception as e:
            log.exception(f"[API] 对话框异常：{e}")
            return

        if not result:
            try:
                self.bottom_label.configure(text="API 配置未更改")
            except Exception:
                pass
            return

        try:
            self.translator.reload_config()
        except Exception as e:
            log.exception(f"[API] translator 重载失败：{e}")

        try:
            self.overlay.clear()
        except Exception:
            pass
        try:
            self.caption_window.clear()
        except Exception:
            pass
        try:
            self.app_state.reset_signature()
        except Exception:
            pass

        try:
            self.bottom_label.configure(text="✅ API 配置已更新")
        except Exception:
            pass
        log.info("[API] 配置已更新")

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
        try:
            self.caption_window.clear()
        except Exception:
            pass
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
            is_window_mode = (self.app_state.display_mode == 'window')

            if not is_follow:
                self.app_state.scroll_boost = False

            try:
                while True:
                    msg = self.pipeline.result_queue.get_nowait()
                    kind = msg[0]
                    if kind == 'items':
                        try:
                            if is_window_mode:
                                self.caption_window.add_items(msg[1])
                            else:
                                self.overlay.update(
                                    msg[1],
                                    base_font_size=self.app_state.base_font_size,
                                )
                        except Exception as e:
                            log.exception(f"[UI] 分发译文失败：{e}")
            except queue.Empty:
                pass

            if not is_pin and not is_window_mode:
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

        try:
            self.caption_window.destroy()
        except Exception:
            pass

        self.hk_mgr.remove_all()
        self.region_manager.destroy()
        self.overlay.clear()
        self.destroy()