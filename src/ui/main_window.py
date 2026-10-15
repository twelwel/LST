"""LST - 主窗口

翻译策略：离线 / 单翻译 API / 多翻译轮询 / 混合 / 单 AI / 多 AI 投票
快速响应层：本地词典 + 短语缓存（所有策略共用）
显示模式：覆盖 / 双语 / 小窗口
译文颜色：全模式通用
OCR 模型：server / PP-OCRv6 tiny，可选高精度多引擎投票
通用过滤：只过滤数字、单位、图像
英文模式：只识别包含英文字母的文本
输出防抖：切换时先 reset_debouncer 再 clear overlay（防串扰）
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
    DEFAULT_ENABLE_ENGLISH_ONLY,
    DEFAULT_ENABLE_TITLE_FILTER,
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
    DML_WORKER_COUNT_MIN,
    DML_WORKER_COUNT_MAX,
    DML_WORKER_COUNT_DEFAULT,
    OCR_MODEL_SIZE_SERVER,
    OCR_MODEL_SIZE_TINY,
    OCR_MODEL_SIZE_DEFAULT,
    DEFAULT_ENABLE_HIGH_ACCURACY_OCR,
    DEFAULT_ENABLE_OCR_SPEED_OPT,
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

# ---- 本包内部依赖 ----
from .constants import (
    TICK_MS, BOUNDARY_PADDING, CACHE_DEFAULT,
    CACHE_MIN, CACHE_MAX
)
from .widgets import CollapsibleFrame
from .panels import DisplayPanelMixin, HotkeyPanelMixin


log = get_logger('ui')


class TranslatorApp(DisplayPanelMixin, HotkeyPanelMixin, ctk.CTk):
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
        saved_english_only = self.user_cfg.get(
            'enable_english_only', DEFAULT_ENABLE_ENGLISH_ONLY)
        saved_title_filter = self.user_cfg.get(
            'enable_title_filter', DEFAULT_ENABLE_TITLE_FILTER)
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

        # GPU 引擎数（CUDA）
        saved_dml_workers = self.user_cfg.get(
            'dml_worker_count', DML_WORKER_COUNT_DEFAULT)
        try:
            saved_dml_workers = int(saved_dml_workers)
        except Exception:
            saved_dml_workers = DML_WORKER_COUNT_DEFAULT
        saved_dml_workers = max(DML_WORKER_COUNT_MIN,
                                min(DML_WORKER_COUNT_MAX, saved_dml_workers))
        self.dml_worker_count = saved_dml_workers

        # OCR 模型大小（server / tiny）
        saved_ocr_model = self.user_cfg.get(
            'ocr_model_size', OCR_MODEL_SIZE_DEFAULT)
        if saved_ocr_model not in (OCR_MODEL_SIZE_SERVER,
                                    OCR_MODEL_SIZE_TINY):
            saved_ocr_model = OCR_MODEL_SIZE_DEFAULT
        self.ocr_model_size = saved_ocr_model

        # 高精度 OCR
        self.high_accuracy_ocr = self.user_cfg.get(
            'enable_high_accuracy_ocr', DEFAULT_ENABLE_HIGH_ACCURACY_OCR)

        # OCR 加速优化
        self.enable_ocr_speed_opt = self.user_cfg.get(
            'enable_ocr_speed_opt', DEFAULT_ENABLE_OCR_SPEED_OPT)

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
            dml_worker_count=self.dml_worker_count,
            model_size=self.ocr_model_size,
            high_accuracy=self.high_accuracy_ocr,
            speed_opt=self.enable_ocr_speed_opt,
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
            enable_english_only=saved_english_only,
            enable_title_filter=saved_title_filter,
            enable_fast_lookup=saved_fast_lookup,
            local_dict_path=saved_local_dict_path,
            mode_cfg=default_cfg,
            enable_frame_diff=saved_frame_diff,
            enable_filter=saved_filter,
            enable_block_ocr=saved_block,
            enable_async_translate=saved_async,
            enable_window_pool=saved_wpool,
            enable_multithread_ocr=True,
            enable_ocr_speed_opt=self.enable_ocr_speed_opt,
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

        self._strategy_labels = list(TRANSLATION_STRATEGIES.values())
        self._strategy_label_to_key = {
            v: k for k, v in TRANSLATION_STRATEGIES.items()
        }
        self._hybrid_order_rev = {v: k for k, v in HYBRID_ORDER_LABELS.items()}
        self._vote_strategy_rev = {
            v: k for k, v in VOTE_STRATEGY_LABELS.items()
        }

        self._display_labels = list(DISPLAY_MODES.values())
        self._display_label_to_key = {
            v: k for k, v in DISPLAY_MODES.items()
        }

        self._fg_label_to_key = {
            v: k for k, v in FG_COLOR_PRESETS.items()
        }

        self._build_ui()

        self.after(100, self._sync_accel_ui)
        self.after(100, self._sync_follow_mode_ui)
        self._restore_ui_from_config(
            saved_mode, saved_scan, initial_font, saved_inpaint,
            saved_fine_text, saved_async_overlay,
            saved_general_filter, saved_english_only,
            saved_fast_lookup,
            saved_phrase_table_path,
            saved_translation_strategy, saved_local_dict_path,
            saved_frame_diff, saved_block, saved_filter, saved_async,
            saved_wpool, saved_cache_max, saved_auto_clean,
            self.enable_ocr_speed_opt,
        )

        self._ui_ready = True
        self.after(TICK_MS, self._tick)

    # ================= 恢复 UI =================

    def _restore_ui_from_config(self, mode, scan, font, inpaint,
                                fine_text, async_overlay,
                                general_filter, english_only,
                                fast_lookup,
                                phrase_table_path,
                                translation_strategy, local_dict_path,
                                frame_diff, block, filt, async_t,
                                wpool, cache_max, auto_clean,
                                ocr_speed_opt):
        try:
            self.mode_selector.set(PERFORMANCE_MODES[mode]['name'])
            self._update_mode_desc(mode)

            self.scan_slider.set(scan)
            self.scan_value_label.configure(text=f"{scan:.2f} s")
            self.font_slider.set(font)
            self.font_value_label.configure(text=str(font))

            self.cpu_count_slider.set(self.cpu_engine_count)
            self.cpu_count_label.configure(text=str(self.cpu_engine_count))

            self.dml_count_slider.set(self.dml_worker_count)
            self.dml_count_label.configure(text=str(self.dml_worker_count))

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

            if english_only:
                self.english_only_switch.select()
            else:
                self.english_only_switch.deselect()

            try:
                if self.user_cfg.get('enable_title_filter',
                                     DEFAULT_ENABLE_TITLE_FILTER):
                    self.title_filter_switch.select()
                else:
                    self.title_filter_switch.deselect()
            except Exception:
                pass

            if fast_lookup:
                self.fast_lookup_switch.select()
            else:
                self.fast_lookup_switch.deselect()

            try:
                if self.ocr_model_size == OCR_MODEL_SIZE_TINY:
                    self.lightweight_ocr_switch.select()
                else:
                    self.lightweight_ocr_switch.deselect()
            except Exception:
                pass

            try:
                if self.high_accuracy_ocr:
                    self.high_accuracy_ocr_switch.select()
                else:
                    self.high_accuracy_ocr_switch.deselect()
            except Exception:
                pass

            try:
                if ocr_speed_opt:
                    self.ocr_speed_opt_switch.select()
                else:
                    self.ocr_speed_opt_switch.deselect()
            except Exception:
                pass

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

        # ⭐ 标记滚动状态（0.5s 有效），让 pipeline 用文本 hash key
        try:
            self.app_state.mark_scroll()
        except Exception:
            pass

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

    # ================= UI 构建（主框架） =================

    def _build_ui(self):
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

        self.region_label = ctk.CTkLabel(
            self, text='', font=(FONT_NAME, 9), text_color='gray',
        )
        self.region_label.pack(pady=(0, 3))
        self._update_region_label()

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

        self.bottom_label = ctk.CTkLabel(
            self, text="准备就绪", font=(FONT_NAME, 10), text_color='gray',
        )
        self.bottom_label.pack(side='bottom', pady=4)

        self._refresh_hotkey_hint()
        self._sync_strategy_ui()
        self._sync_display_mode_ui()
        self._sync_fg_color_ui()

    # ================= CPU / CUDA 引擎数 =================

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

        if self.ocr_engine.mode == 'cpu_multi':
            if self.app_state.running:
                self.bottom_label.configure(
                    text="⚠️ 请先暂停翻译再调整引擎数")
                return
            self.bottom_label.configure(
                text=f"CPU 引擎数改为 {v}，正在重新加载...")
            self.after(
                200,
                lambda: self.switch_accel_mode('cpu_multi', force=True),
            )
        else:
            self.bottom_label.configure(
                text=f"CPU 引擎数设为 {v}（切到 CPU 模式时生效）")

    def _on_dml_count_change(self, value):
        v = int(round(float(value)))
        v = max(DML_WORKER_COUNT_MIN, min(DML_WORKER_COUNT_MAX, v))
        self.dml_count_label.configure(text=str(v))

        if not getattr(self, '_ui_ready', False):
            return
        if v == self.dml_worker_count:
            return

        self.dml_worker_count = v
        self.user_cfg.set('dml_worker_count', v)
        self.user_cfg.save()

        if self.ocr_engine.mode == 'cuda':
            if self.app_state.running:
                self.bottom_label.configure(
                    text="⚠️ 请先暂停翻译再调整引擎数")
                return
            self.bottom_label.configure(
                text=f"CUDA 引擎数改为 {v}，正在重新加载...")
            self.after(
                200,
                lambda: self.switch_accel_mode('cuda', force=True),
            )
        else:
            self.bottom_label.configure(
                text=f"CUDA 引擎数设为 {v}（切到 CUDA 时生效）")

    # ================= OCR 开关统一重载 =================

    def _reload_ocr_safe(self, switch_widget):
        try:
            current = self.ocr_engine.mode
            self.switch_accel_mode(current, force=True)
            self.bottom_label.configure(text="✅ 已切换")
        except Exception as e:
            log.exception(f"[开关] 重载失败：{e}")
            self.bottom_label.configure(text=f"切换失败：{e}")
        finally:
            try:
                switch_widget.configure(state='normal')
            except Exception:
                pass

    def _check_running_and_revert(self, switch_widget, enabled):
        if not self.app_state.running:
            return False
        if enabled:
            switch_widget.deselect()
        else:
            switch_widget.select()
        self.bottom_label.configure(
            text="⚠️ 请先暂停翻译再切换此开关")
        return True

    def _on_lightweight_ocr_toggle(self):
        enabled = bool(self.lightweight_ocr_switch.get())
        new_size = (OCR_MODEL_SIZE_TINY if enabled
                    else OCR_MODEL_SIZE_SERVER)

        if new_size == self.ocr_model_size:
            return

        if self._check_running_and_revert(
                self.lightweight_ocr_switch, enabled):
            return

        self.ocr_model_size = new_size
        self.user_cfg.set('ocr_model_size', new_size)
        self.user_cfg.save()

        mode_tag = "轻量型（PP-OCRv6 tiny）" if enabled else "标准（server）"
        self.bottom_label.configure(
            text=f"OCR 模型：{mode_tag}，正在重载引擎...")
        log.info(f"[OCR 模型] 切换为 {new_size}")

        self.lightweight_ocr_switch.configure(state='disabled')
        self.after(
            200,
            lambda: self._reload_ocr_safe(self.lightweight_ocr_switch),
        )

    def _on_high_accuracy_toggle(self):
        enabled = bool(self.high_accuracy_ocr_switch.get())

        if enabled == self.high_accuracy_ocr:
            return

        if self._check_running_and_revert(
                self.high_accuracy_ocr_switch, enabled):
            return

        self.high_accuracy_ocr = enabled
        self.user_cfg.set('enable_high_accuracy_ocr', enabled)
        self.user_cfg.save()

        tag = "已启用（慢约 2 倍）" if enabled else "已关闭"
        self.bottom_label.configure(
            text=f"高精度 OCR {tag}，正在重载引擎...")
        log.info(f"[高精度 OCR] {'开' if enabled else '关'}")

        self.high_accuracy_ocr_switch.configure(state='disabled')
        self.after(
            200,
            lambda: self._reload_ocr_safe(self.high_accuracy_ocr_switch),
        )

    def _on_ocr_speed_opt_toggle(self):
        enabled = bool(self.ocr_speed_opt_switch.get())

        if enabled == self.enable_ocr_speed_opt:
            return

        if self._check_running_and_revert(
                self.ocr_speed_opt_switch, enabled):
            return

        self.enable_ocr_speed_opt = enabled
        self.app_state.enable_ocr_speed_opt = enabled
        self.user_cfg.set('enable_ocr_speed_opt', enabled)
        self.user_cfg.save()

        tag = "已开启" if enabled else "已关闭"
        self.bottom_label.configure(
            text=f"OCR 加速优化 {tag}，正在重载引擎...")
        log.info(f"[OCR 加速优化] {'开' if enabled else '关'}")

        self.ocr_speed_opt_switch.configure(state='disabled')
        self.after(
            200,
            lambda: self._reload_ocr_safe(self.ocr_speed_opt_switch),
        )

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

        # ⭐ 先 reset pipeline（清空 known + 队列 + generation+1）
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
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

        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
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

        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
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
            # ⭐ 先 reset pipeline（同时清空 result_queue）
            try:
                self.pipeline.reset_debouncer()
            except Exception:
                pass
            # 再清 overlay
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

        was_running = self.app_state.running
        if was_running:
            self.app_state.running = False
            try:
                self.btn_toggle.configure(text="▶ 开始")
                self.status_label.configure(
                    text="状态：已暂停", text_color='gray')
            except Exception:
                pass

        self.accel_status_label.configure(
            text=f"⏳ 正在切换到 {mode}...", text_color='#666666')
        self.bottom_label.configure(text=f"切换到 {mode}...")
        self.update_idletasks()
        log.info(f"[加速] 切换：{self.ocr_engine.mode} → {mode}"
                 f"（模型：{self.ocr_model_size}, "
                 f"高精度：{self.high_accuracy_ocr}, "
                 f"加速：{self.enable_ocr_speed_opt}）")

        try:
            actual = self.ocr_engine.init(
                mode=mode,
                pool_size=self.cpu_engine_count,
                dml_worker_count=self.dml_worker_count,
                model_size=self.ocr_model_size,
                high_accuracy=self.high_accuracy_ocr,
                speed_opt=self.enable_ocr_speed_opt,
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
            # ⭐ 先 reset pipeline
            try:
                self.pipeline.reset_debouncer()
            except Exception:
                pass
            # 再清 overlay
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
            if was_running:
                self.app_state.running = True
                try:
                    self.btn_toggle.configure(text="⏸ 暂停")
                    self.status_label.configure(
                        text="状态：运行中", text_color='green')
                except Exception:
                    pass

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

            model_tag = "tiny" if self.ocr_model_size == 'tiny' else "标准"
            ha_tag = " + 高精度" if self.high_accuracy_ocr else ""
            opt_tag = " + 加速" if self.enable_ocr_speed_opt else ""

            mode_map = {
                'cpu_multi': f'多线程 CPU（{n_engines} 引擎，{model_tag}{ha_tag}{opt_tag}）',
                'directml': f'DirectML（单引擎，{model_tag}{ha_tag}{opt_tag}）',
                'cuda': f'NVIDIA CUDA（{n_engines} 引擎，{model_tag}{ha_tag}{opt_tag}）',
            }
            self.ort_variant_label.configure(
                text=f"当前引擎：{mode_map.get(current_mode, current_mode)}")

            self._highlight_buttons(current_mode, current_variant)

            if current_mode == 'cuda':
                ok, msg = check_cuda_ready()
                if ok:
                    self.accel_status_label.configure(
                        text=f"✅ CUDA 多引擎运行中（{n_engines} 引擎，{model_tag}{ha_tag}{opt_tag}）",
                        text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            elif current_mode == 'directml':
                ok, msg = check_directml_ready()
                if ok:
                    self.accel_status_label.configure(
                        text=f"✅ DirectML 运行中（单引擎，{model_tag}{ha_tag}{opt_tag}）",
                        text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            else:
                self.accel_status_label.configure(
                    text=f"✅ CPU 多线程运行中（{n_engines} 引擎，{model_tag}{ha_tag}{opt_tag}）",
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
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        self.overlay.clear()

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
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        self.overlay.clear()
        msg = "通用过滤：已开启" if enabled else "通用过滤：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_general_filter', enabled)
        self.user_cfg.save()

    def _on_english_only_toggle(self):
        enabled = bool(self.english_only_switch.get())
        self.app_state.enable_english_only = enabled
        self.app_state.reset_signature()
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        self.overlay.clear()
        msg = "英语特化：已开启" if enabled else "英语特化：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_english_only', enabled)
        self.user_cfg.save()

    def _on_title_filter_toggle(self):
        enabled = bool(self.title_filter_switch.get())
        self.app_state.enable_title_filter = enabled
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()
        msg = "标题/Logo 过滤：已开启" if enabled else "标题/Logo 过滤：已关闭"
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_title_filter', enabled)
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
        self.app_state.reset_signature()
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        self.overlay.clear()
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
        self.app_state.reset_signature()
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        self.overlay.clear()
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
            self.app_state.reset_signature()
        except Exception:
            pass
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
        try:
            self.overlay.clear()
        except Exception:
            pass
        try:
            self.caption_window.clear()
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
        # ⭐ 先 reset pipeline（同时清空 result_queue + generation+1）
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
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
        # ⭐ 先 reset pipeline
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        # 再清 overlay
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

        try:
            self.ocr_engine.shutdown()
        except Exception:
            pass

        self.hk_mgr.remove_all()
        self.region_manager.destroy()
        self.overlay.clear()
        self.destroy()