"""LST - 主窗口"""

import os
import queue
import sys
import time
import threading
from pathlib import Path
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
    DEFAULT_ENABLE_OCR_VERIFY,
    DEFAULT_ENABLE_FAST_LOOKUP,
    DEFAULT_PHRASE_TABLE_PATH,
    PERFORMANCE_MODES, DEFAULT_MODE,
    IDLE_THRESHOLD_FOR_SLOWDOWN, IDLE_SLOWDOWN_FACTOR,
    DEFAULT_ENABLE_SMART_OCR, DEFAULT_ENABLE_FILTER,
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
    SOURCE_LANGS, TARGET_LANGS,
    DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG,
    DEFAULT_ENABLE_THROTTLE, DEFAULT_THROTTLE_QPS,
    THROTTLE_QPS_MIN, THROTTLE_QPS_MAX,
    LOCAL_MODEL_DEVICES, DEFAULT_LOCAL_MODEL_DEVICE,
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

from .constants import (
    TICK_MS, BOUNDARY_PADDING, CACHE_DEFAULT,
    CACHE_MIN, CACHE_MAX
)
from .download_task import DownloadTask
from .widgets import CollapsibleFrame
from .panels import DisplayPanelMixin, HotkeyPanelMixin


log = get_logger('ui')


class TranslatorApp(DisplayPanelMixin, HotkeyPanelMixin, ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(get_window_title())

        self._ui_ready = False
        self._downloading = False
        self._downloader = None
        self._download_task = None
        self._downloading_model_id = None
        self._downloading_variant = None
        self._download_ok = False
        self._download_status_text = ''
        self._current_model_label = None
        self._model_id_by_label = {}

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

        saved_frame_diff = self.user_cfg.get(
            'enable_frame_diff', DEFAULT_ENABLE_SMART_OCR)
        saved_block = self.user_cfg.get(
            'enable_block_ocr', DEFAULT_ENABLE_SMART_OCR)
        # 两个旧开关合并为一个“智能 OCR”开关；旧配置按“两者都开启”迁移。
        saved_smart = self.user_cfg.get(
            'enable_smart_ocr',
            bool(saved_frame_diff) and bool(saved_block))
        saved_filter = self.user_cfg.get(
            'enable_filter', DEFAULT_ENABLE_FILTER)
        saved_general_filter = self.user_cfg.get(
            'enable_general_filter', DEFAULT_ENABLE_GENERAL_FILTER)
        saved_english_only = self.user_cfg.get(
            'enable_english_only', DEFAULT_ENABLE_ENGLISH_ONLY)
        saved_title_filter = self.user_cfg.get(
            'enable_title_filter', DEFAULT_ENABLE_TITLE_FILTER)
        saved_ocr_verify = self.user_cfg.get(
            'enable_ocr_verify', DEFAULT_ENABLE_OCR_VERIFY)
        saved_fast_lookup = self.user_cfg.get(
            'enable_fast_lookup', DEFAULT_ENABLE_FAST_LOOKUP)
        saved_phrase_table_path = self.user_cfg.get(
            'phrase_table_path', DEFAULT_PHRASE_TABLE_PATH)
        saved_async = self.user_cfg.get(
            'enable_async_translate', DEFAULT_ENABLE_ASYNC_TRANSLATE)
        saved_wpool = self.user_cfg.get(
            'enable_window_pool', DEFAULT_ENABLE_WINDOW_POOL)

        saved_cpu_engine = self.user_cfg.get(
            'cpu_engine_count', CPU_ENGINE_COUNT_DEFAULT)
        try:
            saved_cpu_engine = int(saved_cpu_engine)
        except Exception:
            saved_cpu_engine = CPU_ENGINE_COUNT_DEFAULT
        saved_cpu_engine = max(CPU_ENGINE_COUNT_MIN,
                               min(CPU_ENGINE_COUNT_MAX, saved_cpu_engine))
        self.cpu_engine_count = saved_cpu_engine

        saved_dml_workers = self.user_cfg.get(
            'dml_worker_count', DML_WORKER_COUNT_DEFAULT)
        try:
            saved_dml_workers = int(saved_dml_workers)
        except Exception:
            saved_dml_workers = DML_WORKER_COUNT_DEFAULT
        saved_dml_workers = max(DML_WORKER_COUNT_MIN,
                                min(DML_WORKER_COUNT_MAX, saved_dml_workers))
        self.dml_worker_count = saved_dml_workers

        saved_ocr_model = self.user_cfg.get(
            'ocr_model_size', OCR_MODEL_SIZE_DEFAULT)
        if saved_ocr_model not in (OCR_MODEL_SIZE_SERVER,
                                    OCR_MODEL_SIZE_TINY):
            saved_ocr_model = OCR_MODEL_SIZE_DEFAULT
        self.ocr_model_size = saved_ocr_model

        from config import SOURCE_LANG_TO_OCR
        saved_source_lang_for_ocr = self.user_cfg.get(
            'source_lang', DEFAULT_SOURCE_LANG)
        saved_ocr_lang = self.user_cfg.get(
            'ocr_lang',
            SOURCE_LANG_TO_OCR.get(saved_source_lang_for_ocr, 'CH'))
        self._ocr_lang = saved_ocr_lang

        self.high_accuracy_ocr = self.user_cfg.get(
            'enable_high_accuracy_ocr', DEFAULT_ENABLE_HIGH_ACCURACY_OCR)
        self.enable_ocr_speed_opt = self.user_cfg.get(
            'enable_ocr_speed_opt', DEFAULT_ENABLE_OCR_SPEED_OPT)

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

        saved_display_mode = self.user_cfg.get(
            'display_mode', DEFAULT_DISPLAY_MODE)
        if saved_display_mode not in DISPLAY_MODES:
            saved_display_mode = DEFAULT_DISPLAY_MODE

        saved_fg_color = self.user_cfg.get('fg_color', DEFAULT_FG_COLOR)
        if saved_fg_color not in FG_COLOR_PRESETS:
            saved_fg_color = DEFAULT_FG_COLOR

        saved_source_lang = self.user_cfg.get(
            'source_lang', DEFAULT_SOURCE_LANG)
        if saved_source_lang not in SOURCE_LANGS:
            saved_source_lang = DEFAULT_SOURCE_LANG

        saved_target_lang = self.user_cfg.get(
            'target_lang', DEFAULT_TARGET_LANG)
        if saved_target_lang not in TARGET_LANGS:
            saved_target_lang = DEFAULT_TARGET_LANG

        saved_enable_throttle = self.user_cfg.get(
            'enable_throttle', DEFAULT_ENABLE_THROTTLE)
        saved_throttle_qps = self.user_cfg.get(
            'throttle_qps', DEFAULT_THROTTLE_QPS)

        saved_scan = self.user_cfg.get(
            'scan_interval',
            PERFORMANCE_MODES[saved_mode]['scan_interval'],
        )
        saved_inpaint = self.user_cfg.get(
            'use_inpainting', DEFAULT_USE_INPAINTING)
        saved_fine_text = self.user_cfg.get(
            'enable_fine_text', DEFAULT_ENABLE_FINE_TEXT)
        saved_async_overlay = self.user_cfg.get(
            'enable_async_overlay', DEFAULT_ENABLE_ASYNC_OVERLAY)
        saved_cache_max = self.user_cfg.get(
            'cache_max_size', CACHE_DEFAULT)
        saved_auto_clean = self.user_cfg.get('auto_clean', True)
        saved_follow_mode = self.user_cfg.get('follow_mode', 'follow')
        if saved_follow_mode not in ('default', 'follow', 'pin'):
            saved_follow_mode = 'follow'

        self.gpu_devices = list_gpu_devices()

        log.info(f"========== 启动 v{__version__} ==========")
        if is_frozen():
            log.info("[启动] 打包模式（exe）")

        # ============ 核心组件 ============
        self.translator = Translator(
            source=saved_source_lang, target=saved_target_lang)
        self.ocr_engine = OCREngine()

        # ⭐ 加速模式：首次运行默认 DirectML > CUDA > CPU
        saved_accel_mode = (self.user_cfg.get('accel_mode', '') or '').strip()
        current_variant = check_onnxruntime_variant() or 'cpu'

        if not saved_accel_mode:
            if current_variant == 'directml':
                saved_accel_mode = 'directml'
            elif current_variant == 'gpu':
                saved_accel_mode = 'cuda'
            else:
                saved_accel_mode = 'cpu_multi'
            self.user_cfg.set('accel_mode', saved_accel_mode)
            self.user_cfg.save()
            log.info(f"[启动] 首次运行，默认加速模式：{saved_accel_mode}")

        # 校验请求的 mode 与已安装的 variant 是否兼容
        if saved_accel_mode == 'directml' and current_variant != 'directml':
            log.warning("[启动] 请求 DirectML 但未安装，回退 CPU")
            saved_accel_mode = 'cpu_multi'
        elif saved_accel_mode == 'cuda' and current_variant != 'gpu':
            log.warning("[启动] 请求 CUDA 但未安装，回退 CPU")
            saved_accel_mode = 'cpu_multi'

        init_mode = saved_accel_mode

        # ⭐ 应用本地模型设备偏好（必须在引擎创建之前）
        saved_local_device = self.user_cfg.get(
            'local_model_device', DEFAULT_LOCAL_MODEL_DEVICE)
        if saved_local_device not in LOCAL_MODEL_DEVICES:
            saved_local_device = DEFAULT_LOCAL_MODEL_DEVICE
        try:
            from translation.local_models import set_preferred_device
            set_preferred_device(saved_local_device)
            log.info(f"[启动] 本地模型设备：{saved_local_device}")
        except Exception as e:
            log.exception(f"[启动] 设置本地模型设备失败：{e}")

        variant = self.ocr_engine.init(
            mode=init_mode,
            pool_size=self.cpu_engine_count,
            dml_worker_count=self.dml_worker_count,
            model_size=self.ocr_model_size,
            high_accuracy=self.high_accuracy_ocr,
            speed_opt=self.enable_ocr_speed_opt,
            lang_name=saved_ocr_lang,
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
            enable_ocr_verify=saved_ocr_verify,
            enable_fast_lookup=saved_fast_lookup,
            mode_cfg=default_cfg,
            enable_frame_diff=saved_smart,
            enable_filter=saved_filter,
            enable_block_ocr=saved_smart,
            enable_async_translate=saved_async,
            enable_window_pool=saved_wpool,
            enable_multithread_ocr=True,
            enable_ocr_speed_opt=self.enable_ocr_speed_opt,
            source_lang=saved_source_lang,
            target_lang=saved_target_lang,
        )
        self.app_state.performance_mode = saved_mode
        self.app_state.base_font_size = initial_font
        self.app_state.scan_interval = saved_scan
        self.app_state.follow_mode = saved_follow_mode
        self.app_state.freeze_overlay = (saved_follow_mode == 'pin')
        self.app_state.translation_strategy = saved_translation_strategy
        self.app_state.display_mode = saved_display_mode
        self.app_state.fg_color = saved_fg_color
        self.app_state.source_lang = saved_source_lang
        self.app_state.target_lang = saved_target_lang

        self.pipeline = Pipeline(
            self.ocr_engine, self.cache, self.translator, self.app_state,
        )
        self.pipeline.start()

        try:
            from translation.throttle import get_limiter
            limiter = get_limiter()
            limiter.set_enabled(saved_enable_throttle)
            limiter.set_qps(saved_throttle_qps)
            log.info(f"[节流] "
                     f"{'开启' if saved_enable_throttle else '关闭'} "
                     f"QPS={saved_throttle_qps}")
        except Exception as e:
            log.exception(f"[节流] 初始化失败：{e}")

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

        try:
            bg = int(self.user_cfg.get('bilingual_gap', 2))
            self.overlay.set_bilingual_gap(bg)
        except Exception:
            pass

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
        self._hybrid_order_rev = {
            v: k for k, v in HYBRID_ORDER_LABELS.items()
        }
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
            saved_title_filter, saved_ocr_verify,
            saved_fast_lookup,
            saved_phrase_table_path,
            saved_translation_strategy,
            saved_smart, saved_filter, saved_async,
            saved_wpool, saved_cache_max, saved_auto_clean,
            self.enable_ocr_speed_opt,
        )

        self._ui_ready = True
        self.after(TICK_MS, self._tick)

        # ⭐ 定时刷新设备提示
        self.after(2000, self._periodic_device_refresh)

    # ================= 恢复 UI =================

    def _restore_ui_from_config(self, mode, scan, font, inpaint,
                                fine_text, async_overlay,
                                general_filter, english_only,
                                title_filter, ocr_verify,
                                fast_lookup, phrase_table_path,
                                translation_strategy,
                                smart_ocr, filt, async_t,
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
            self.cpu_count_label.configure(
                text=str(self.cpu_engine_count))
            self.dml_count_slider.set(self.dml_worker_count)
            self.dml_count_label.configure(
                text=str(self.dml_worker_count))

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
                if title_filter:
                    self.title_filter_switch.select()
                else:
                    self.title_filter_switch.deselect()
            except Exception:
                pass
            try:
                if ocr_verify:
                    self.ocr_verify_switch.select()
                else:
                    self.ocr_verify_switch.deselect()
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
            self._sync_lang_ui()

            # ⭐ 恢复本地模型设备选择器
            try:
                saved_device = self.user_cfg.get(
                    'local_model_device', DEFAULT_LOCAL_MODEL_DEVICE)
                if saved_device not in LOCAL_MODEL_DEVICES:
                    saved_device = DEFAULT_LOCAL_MODEL_DEVICE
                self.local_model_device_selector.set(
                    LOCAL_MODEL_DEVICES[saved_device])
                self._update_local_model_device_hint()
            except Exception as e:
                log.exception(f"[UI] 恢复本地模型设备失败：{e}")

            if smart_ocr:
                self.smart_ocr_switch.select()
            else:
                self.smart_ocr_switch.deselect()
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

            try:
                bg = int(self.user_cfg.get('bilingual_gap', 2))
                self.bilingual_gap_slider.set(bg)
                self.bilingual_gap_value_label.configure(text=str(bg))
                self.overlay.set_bilingual_gap(bg)
            except Exception:
                pass

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

    # ================= UI 构建 =================

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

        row1 = ctk.CTkFrame(strategy_frame,
                             fg_color='transparent', height=1)
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

        # ===== 语言选择 =====
        lang_frame = ctk.CTkFrame(self, height=1)
        lang_frame.pack(fill='x', padx=10, pady=(2, 2))
        lang_frame.pack_propagate(True)

        ctk.CTkLabel(
            lang_frame, text="源语言：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 4))

        self.source_lang_selector = ctk.CTkOptionMenu(
            lang_frame,
            values=list(SOURCE_LANGS.values()),
            command=self._on_source_lang_change,
            font=(FONT_NAME, 11),
            width=110,
        )
        self.source_lang_selector.set(SOURCE_LANGS[DEFAULT_SOURCE_LANG])
        self.source_lang_selector.pack(side='left', padx=2)

        ctk.CTkLabel(
            lang_frame, text="目标语言：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(12, 4))

        self.target_lang_selector = ctk.CTkOptionMenu(
            lang_frame,
            values=list(TARGET_LANGS.values()),
            command=self._on_target_lang_change,
            font=(FONT_NAME, 11),
            width=110,
        )
        self.target_lang_selector.set(TARGET_LANGS[DEFAULT_TARGET_LANG])
        self.target_lang_selector.pack(side='left', padx=2)

        # ===== 本地模型 =====
        lm_frame = ctk.CTkFrame(self, height=1)
        lm_frame.pack(fill='x', padx=10, pady=(2, 2))
        lm_frame.pack_propagate(True)

        ctk.CTkLabel(
            lm_frame, text="本地模型：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 4))

        self.local_model_selector = ctk.CTkOptionMenu(
            lm_frame,
            values=['（加载中...）'],
            command=self._on_local_model_selected,
            font=(FONT_NAME, 10),
            width=200,
        )
        self.local_model_selector.set('（加载中...）')
        self.local_model_selector.pack(side='left', padx=2)

        self.local_model_status = ctk.CTkLabel(
            lm_frame, text="",
            font=(FONT_NAME, 10), text_color='#666666',
        )
        self.local_model_status.pack(side='left', padx=6)

        self.local_model_btn = ctk.CTkButton(
            lm_frame, text="下载", width=70, height=26,
            font=(FONT_NAME, 10),
            fg_color='#16a085', hover_color='#138d75',
            command=self._on_local_model_action,
        )
        self.local_model_btn.pack(side='left', padx=2)

        self.local_model_del_btn = ctk.CTkButton(
            lm_frame, text="删除", width=60, height=26,
            font=(FONT_NAME, 10),
            fg_color='#dc3545', hover_color='#c82333',
            command=self._on_local_model_delete,
            state='disabled',
        )
        self.local_model_del_btn.pack(side='left', padx=2)

        self.local_model_progress = ctk.CTkProgressBar(
            lm_frame, height=8, width=100,
        )
        self.local_model_progress.set(0)
        self.local_model_progress.pack(side='left', padx=4)

        ctk.CTkButton(
            lm_frame, text="管理", width=60, height=26,
            font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=self._open_model_manager,
        ).pack(side='left', padx=2)

        ctk.CTkButton(
            lm_frame, text="打开目录", width=75, height=26,
            font=(FONT_NAME, 10),
            fg_color='#17a2b8', hover_color='#138496',
            command=self._open_models_dir,
        ).pack(side='left', padx=2)

        ctk.CTkButton(
            lm_frame, text="刷新", width=55, height=26,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=self._refresh_local_models,
        ).pack(side='left', padx=2)

        # ⭐ 本地模型设备（本地模型行下方）
        device_frame = ctk.CTkFrame(self, height=1)
        device_frame.pack(fill='x', padx=10, pady=(0, 2))
        device_frame.pack_propagate(True)

        ctk.CTkLabel(
            device_frame, text="模型运算设备：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 4))

        self.local_model_device_selector = ctk.CTkOptionMenu(
            device_frame,
            values=list(LOCAL_MODEL_DEVICES.values()),
            command=self._on_local_model_device_change,
            font=(FONT_NAME, 10), width=160,
        )
        self.local_model_device_selector.set(
            LOCAL_MODEL_DEVICES[DEFAULT_LOCAL_MODEL_DEVICE])
        self.local_model_device_selector.pack(side='left', padx=2)

        self.local_model_device_hint = ctk.CTkLabel(
            device_frame, text="",
            font=(FONT_NAME, 10), text_color='#3a86ff',
            anchor='w',
        )
        self.local_model_device_hint.pack(side='left', padx=8)

        # ===== 请求节流 =====
        throttle_frame = ctk.CTkFrame(self, height=1)
        throttle_frame.pack(fill='x', padx=10, pady=(2, 2))
        throttle_frame.pack_propagate(True)

        self.throttle_switch = ctk.CTkSwitch(
            throttle_frame, text="请求节流",
            command=self._on_throttle_toggle,
            onvalue=True, offvalue=False,
            font=(FONT_NAME, 11, 'bold'),
        )
        if self.user_cfg.get('enable_throttle',
                              DEFAULT_ENABLE_THROTTLE):
            self.throttle_switch.select()
        self.throttle_switch.pack(side='left', padx=(6, 8))

        ctk.CTkLabel(
            throttle_frame, text="每个 API 每秒请求数：",
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=(6, 2))

        self.throttle_qps_value_label = ctk.CTkLabel(
            throttle_frame, text=f"{DEFAULT_THROTTLE_QPS:.1f}",
            width=40, font=(FONT_NAME, 10, 'bold'),
            text_color='#3a86ff',
        )
        self.throttle_qps_value_label.pack(side='right', padx=(2, 10))

        self.throttle_qps_slider = ctk.CTkSlider(
            throttle_frame,
            from_=THROTTLE_QPS_MIN, to=THROTTLE_QPS_MAX,
            number_of_steps=int((THROTTLE_QPS_MAX -
                                  THROTTLE_QPS_MIN) * 10),
            command=self._on_throttle_qps_change,
            height=14, width=160,
        )
        self.throttle_qps_slider.set(
            self.user_cfg.get('throttle_qps', DEFAULT_THROTTLE_QPS))
        self.throttle_qps_slider.pack(side='right', padx=4)

        # ===== 提示 =====
        hint_frame = ctk.CTkFrame(self, fg_color='transparent',
                                   height=1)
        hint_frame.pack(fill='x', padx=10, pady=(0, 2))
        hint_frame.pack_propagate(True)
        ctk.CTkLabel(
            hint_frame,
            text="💡 所有策略都会先查快速响应层（短语缓存），"
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
            btn_frame2, text="延迟测试", width=80, height=28,
            fg_color='#e67e22', hover_color='#d35400',
            command=self.open_latency_test,
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
        self.scroll_frame.pack(fill='both', expand=True,
                                padx=5, pady=3)

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
            self, text="准备就绪",
            font=(FONT_NAME, 10), text_color='gray',
        )
        self.bottom_label.pack(side='bottom', pady=4)

        self._refresh_hotkey_hint()
        self._sync_strategy_ui()
        self._sync_display_mode_ui()
        self._sync_fg_color_ui()
        self._sync_lang_ui()
        self._update_local_model_device_hint()

        self.after(300, self._refresh_local_models)

    # ================= 本地模型设备 =================

    def _get_current_model_actual_device(self):
        """读取当前已加载本地模型的实际设备名（若无则 None）"""
        try:
            from translation.local_models import TranslationModelFactory
            for mid, inst in (TranslationModelFactory._instances or {}).items():
                dev = getattr(inst, 'device', None)
                if dev:
                    return dev
        except Exception:
            pass
        return None

    def _set_device_hint(self, text, color='#3a86ff'):
        try:
            self.local_model_device_hint.configure(
                text=text, text_color=color)
        except Exception:
            pass

    def _set_device_selector_enabled(self, enabled):
        try:
            self.local_model_device_selector.configure(
                state='normal' if enabled else 'disabled')
        except Exception:
            pass

    def _set_device_strict(self, key):
        """强制设置设备偏好（不递归刷新 hint）"""
        if key not in LOCAL_MODEL_DEVICES:
            return
        self.user_cfg.set('local_model_device', key)
        self.user_cfg.save()
        try:
            from translation.local_models import (
                set_preferred_device, unload_all_models)
            set_preferred_device(key)
            # 默认 force=False：有已加载模型时不会在运行期销毁 DML 会话
            if not unload_all_models():
                log.info("[本地模型] 模型已加载，设备变更将在重启后生效")
        except Exception as e:
            log.exception(f"[本地模型] 设置设备失败：{e}")
        try:
            self.local_model_device_selector.set(
                LOCAL_MODEL_DEVICES[key])
        except Exception:
            pass

    def _apply_model_device_constraint(self) -> bool:
        """
        ⭐ 安全机制：绝对禁止 int8 模型运行在 GPU 上
        """
        label = getattr(self, '_current_model_label', None)
        if not label or label.startswith('（'):
            self._set_device_selector_enabled(True)
            return False

        mid = self._model_id_by_label.get(label)
        if not mid or mid.startswith('custom_'):
            self._set_device_selector_enabled(True)
            return False

        try:
            from translation.local_models.downloader import validate_model_files
            from translation.local_models import get_preferred_device
            has_cpu, _ = validate_model_files(mid, 'cpu')
            has_gpu, _ = validate_model_files(mid, 'gpu')
            pref = get_preferred_device()
        except Exception:
            return False

        # 如果用户选的是 GPU 相关的设备 (非纯 CPU)
        if pref != 'cpu':
            # 如果只有 int8，没有 FP32，直接拦截并强制切回 CPU
            if has_cpu and not has_gpu:
                self._set_device_selector_enabled(True)
                self._set_device_hint(
                    "⚠️ 当前仅下载了 int8 模型，已强行锁定为 CPU。如需 GPU，请去下载 FP32 版本",
                    '#cc0000')
                self._set_device_strict('cpu')
                return True
            
            # 如果连 GPU 版都没有
            if not has_gpu:
                self._set_device_selector_enabled(True)
                self._set_device_hint(
                    "⚠️ 想用 GPU 加速必须下载 GPU 版 (FP32) 模型！当前无 FP32 权重",
                    '#cc0000')
                return True

        # 正常状态
        self._set_device_selector_enabled(True)
        return False

    def _update_local_model_device_hint(self):
        """刷新模型运算设备提示"""
        if self._apply_model_device_constraint():
            return

        try:
            from translation.local_models import get_preferred_device
            pref = get_preferred_device()
        except Exception:
            pref = 'auto'

        actual = self._get_current_model_actual_device()
        variant_tag = self._get_current_model_variant_tag()

        if pref == 'auto':
            if actual:
                hint = f"💡 自动 → 当前实际：{actual}"
                if variant_tag:
                    hint += f"（{variant_tag}版）"
            else:
                hint = "💡 自动（CUDA > DirectML > CPU）"
            color = '#3a86ff'
        elif pref == 'cuda':
            hint = f"💡 强制 CUDA → 当前：{actual or '未加载'}"
            color = '#28a745' if actual == 'CUDA' else '#cc6600'
        elif pref == 'directml':
            try:
                from dml_device import describe_dml_device
                dml_tag = f"（{describe_dml_device()}）"
            except Exception:
                dml_tag = ''
            hint = f"💡 DirectML{dml_tag} → 当前：{actual or '未加载'}"
            color = ('#28a745' if actual == 'DirectML' else '#cc6600')
        else:
            hint = f"💡 强制 CPU → 当前：{actual or '未加载'}"
            if variant_tag == 'GPU':
                hint += "（FP32 模型）"
            elif variant_tag == 'CPU':
                hint += "（int8 模型）"
            color = '#666666'

        self._set_device_hint(hint, color)

    def _get_current_model_variant_tag(self) -> str:
        """读取当前选中模型的变体标签（'CPU' / 'GPU' / ''）"""
        label = getattr(self, '_current_model_label', None)
        if not label or label.startswith('（'):
            return ''
        mid = self._model_id_by_label.get(label)
        if not mid or mid.startswith('custom_'):
            return ''

        try:
            from translation.local_models.downloader import validate_model_files
            has_cpu, _ = validate_model_files(mid, 'cpu')
            has_gpu, _ = validate_model_files(mid, 'gpu')
            
            if has_gpu:
                return 'GPU'
            if has_cpu:
                return 'CPU'
        except Exception:
            pass
        return ''

    def _periodic_device_refresh(self):
        """每 2 秒刷新一次设备提示"""
        try:
            self._update_local_model_device_hint()
        except Exception:
            pass
        try:
            self.after(2000, self._periodic_device_refresh)
        except Exception:
            pass

    def _on_local_model_device_change(self, label):
        rev = {v: k for k, v in LOCAL_MODEL_DEVICES.items()}
        key = rev.get(label, DEFAULT_LOCAL_MODEL_DEVICE)

        # ⭐ 安全检测：如果目标设备不是纯 CPU，必须检查当前模型是否包含 FP32 权重
        if key != 'cpu':
            model_label = getattr(self, '_current_model_label', None)
            mid = self._model_id_by_label.get(model_label) if model_label else None

            if mid and not mid.startswith('custom_'):
                try:
                    from translation.local_models.downloader import validate_model_files
                    has_gpu, _ = validate_model_files(mid, 'gpu')
                    
                    if not has_gpu:
                        from tkinter import messagebox
                        messagebox.showwarning(
                            "不支持 GPU 加速", 
                            "当前选中的模型为 int8/q4 量化版，仅支持 CPU 运行！\n\n"
                            "如需使用显卡（DirectML/CUDA）加速，请先前往「管理」面板下载对应模型的 GPU 版 (FP32)。", 
                            parent=self
                        )
                        # 强行回滚设备选择器到原配置
                        current_pref = self.user_cfg.get('local_model_device', DEFAULT_LOCAL_MODEL_DEVICE)
                        if current_pref not in LOCAL_MODEL_DEVICES:
                            current_pref = DEFAULT_LOCAL_MODEL_DEVICE
                        self.local_model_device_selector.set(LOCAL_MODEL_DEVICES[current_pref])
                        self._update_local_model_device_hint()
                        return
                except Exception as e:
                    log.exception(f"[设备切换] 安全检测失败: {e}")
                    # 如果检测过程出错，为安全起见也回滚
                    current_pref = self.user_cfg.get('local_model_device', DEFAULT_LOCAL_MODEL_DEVICE)
                    if current_pref not in LOCAL_MODEL_DEVICES:
                        current_pref = DEFAULT_LOCAL_MODEL_DEVICE
                    self.local_model_device_selector.set(LOCAL_MODEL_DEVICES[current_pref])
                    return

        # ==========================================
        # 检测通过，执行正常的切换逻辑
        # ==========================================
        self.user_cfg.set('local_model_device', key)
        self.user_cfg.save()

        applied_now = True
        try:
            from translation.local_models import (
                set_preferred_device, unload_all_models)
            set_preferred_device(key)
            # ⭐ 运行期不销毁已加载的 DML 会话（会触发驱动级 0xC0000005）
            applied_now = unload_all_models()
        except Exception as e:
            log.exception(f"[本地模型] 切换设备失败：{e}")
            self.bottom_label.configure(text=f"切换失败：{e}")
            return

        try:
            self.translator.reload_config()
        except Exception:
            pass

        self._update_local_model_device_hint()

        log.info(f"[本地模型] 设备切换为 {key}"
                 f"{'' if applied_now else '（重启后生效）'}")
        if applied_now:
            self.bottom_label.configure(
                text=f"✅ 本地模型设备：{label}（下次翻译生效）")
        else:
            self.bottom_label.configure(
                text=f"✅ 本地模型设备：{label}（重启应用后生效，"
                     f"避免运行中切换显卡导致崩溃）")

    # ================= 本地模型 =================

    def _refresh_local_models(self):
        """⭐ 下拉显示区：明确区分 (int8) 和 (FP32)"""
        try:
            from translation.local_models import get_available_models
            all_models = get_available_models()
        except Exception as e:
            log.debug(f"[本地模型] 模块不可用：{e}")
            try:
                self.local_model_selector.configure(values=['（无本地模型支持）'])
                self.local_model_selector.set('（无本地模型支持）')
            except Exception:
                pass
            self._update_local_model_device_hint()
            return

        try:
            from translation.local_models import get_model_roots
            roots = [str(p) for p in get_model_roots()]
            log.debug(f"[本地模型] 扫描目录：{roots}")
        except Exception:
            pass

        labels = []
        self._model_id_by_label = {}

        for m in all_models:
            mid = m['id']
            is_custom = m.get('is_custom', False)
            
            if not is_custom:
                try:
                    from translation.local_models.downloader import validate_model_files
                    has_cpu, _ = validate_model_files(mid, 'cpu')
                    has_gpu, _ = validate_model_files(mid, 'gpu')
                    if not has_cpu and not has_gpu:
                        continue
                except Exception:
                    continue
            else:
                has_cpu = has_gpu = True

            tag = ''
            if has_gpu: tag = ' [FP32]'
            elif has_cpu: tag = ' [int8]'

            size = m.get('size_mb', 0)
            name = m.get('name', mid)
            
            if is_custom:
                label = f"{name} [自定义]"
            else:
                label = f"{name} ({size:.0f}MB){tag}"
                
            labels.append(label)
            self._model_id_by_label[label] = mid

        if labels:
            log.info(f"[本地模型] 检测到 {len(labels)} 个可用模型")
            try:
                from translation.api_config import sync_downloaded_models
                result = sync_downloaded_models()
                changed = result.get('added') or result.get('removed')
                if changed:
                    log.info(f"[本地模型] Provider 同步：{result}")
                    try:
                        self.translator.reload_config()
                    except Exception:
                        pass
            except Exception as e:
                log.debug(f"[本地模型] Provider 同步失败：{e}")
        else:
            log.info(
                "[本地模型] 未检测到可用模型（已扫描："
                f"{roots if 'roots' in locals() else '未知'}）")
        if not labels:
            try:
                self.local_model_selector.configure(values=['（未下载任何模型）'])
                self.local_model_selector.set('（未下载任何模型）')
            except Exception:
                pass
            self._current_model_label = None
            self._model_id_by_label = {}
            self._update_local_model_status()
            self._update_local_model_device_hint()
            return

        current = getattr(self, '_current_model_label', None)
        if current not in labels:
            current = labels[0]

        try:
            self.local_model_selector.configure(values=labels)
            self.local_model_selector.set(current)
        except Exception:
            pass

        self._current_model_label = current
        self._update_local_model_status()
        self._update_local_model_device_hint()

    def _on_local_model_selected(self, label):
        self._current_model_label = label
        self._update_local_model_status()
        self._update_local_model_device_hint()

        # 提示用户当前选择的类型
        if '[int8]' in label:
            self.bottom_label.configure(text="当前选择：int8 模型 (仅限 CPU 运行)")
        elif '[FP32]' in label:
            self.bottom_label.configure(text="当前选择：FP32 模型 (支持 GPU 加速)")

    def _update_local_model_status(self):
        label = getattr(self, '_current_model_label', None)

        if not label or label.startswith('（'):
            try:
                self.local_model_status.configure(
                    text="无已下载模型", text_color='#888888')
                self.local_model_btn.configure(
                    text="下载", state='normal',
                    fg_color='#16a085', hover_color='#138d75')
                self.local_model_del_btn.configure(state='disabled')
            except Exception:
                pass
            return

        mid = getattr(self, '_model_id_by_label', {}).get(label)
        if not mid:
            return

        is_custom = mid.startswith('custom_')

        try:
            if is_custom:
                self.local_model_status.configure(
                    text="✓ 已导入", text_color='#28a745')
                self.local_model_btn.configure(
                    text="下载", state='normal',
                    fg_color='#16a085', hover_color='#138d75')
                self.local_model_del_btn.configure(state='normal')
            else:
                self.local_model_status.configure(
                    text="✓ 已下载", text_color='#28a745')
                self.local_model_btn.configure(
                    text="下载更多", state='normal',
                    fg_color='#16a085', hover_color='#138d75')
                self.local_model_del_btn.configure(state='normal')
        except Exception:
            pass

    def _on_local_model_action(self):
        """点「下载」按钮：打开模型选择对话框"""
        if getattr(self, '_downloading', False):
            self.bottom_label.configure(
                text="⚠️ 已有下载任务进行中")
            return

        try:
            from ui.model_picker_dialog import ModelPickerDialog
            dlg = ModelPickerDialog(
                self, on_download=self._start_model_download)
            dlg.grab_set()
            dlg.bind('<Destroy>', lambda e: self._on_model_dialog_destroyed(
                e, dlg))
        except Exception as e:
            log.exception(f"[本地模型] 打开选择对话框失败：{e}")
            self.bottom_label.configure(
                text=f"打开选择对话框失败：{e}")

    def _on_model_dialog_destroyed(self, event, dialog):
        """只在对话框本身销毁时刷新，避免每个子控件触发一次刷新。"""
        if event.widget is not dialog:
            return
        try:
            if self.winfo_exists():
                self.after(300, self._refresh_local_models)
        except Exception:
            pass

    def _start_model_download(self, model_id: str, variant: str = 'cpu'):
        """从选择对话框触发的下载"""
        try:
            if getattr(self, '_downloading', False):
                self.bottom_label.configure(text="⚠️ 已有下载任务进行中")
                return

            from translation.local_models.downloader import (
                is_downloading, get_downloading_model)
            if is_downloading():
                other = get_downloading_model()
                self.bottom_label.configure(text=f"⚠️ 其他窗口正在下载：{other}")
                return

            from translation.local_models import ModelDownloader, get_available_models
            all_m = get_available_models()
            m = next((x for x in all_m if x['id'] == model_id), None)
            model_name = m['name'] if m else model_id
        except Exception as e:
            log.exception("[本地模型] 下载前置检查失败")
            self.bottom_label.configure(text=f"❌ 下载失败：{e}")
            return

        variant_label = 'GPU 版' if variant == 'gpu' else 'CPU 版'
        self._downloading = True
        self._downloading_model_id = model_id
        self._downloading_variant = variant
        self._download_ok = False
        self._download_status_text = ''
        self._downloader = ModelDownloader()

        try:
            self.local_model_btn.configure(text="下载中...", state='disabled')
            self.local_model_del_btn.configure(state='disabled')
            self.local_model_progress.set(0)
            self.local_model_status.configure(text=f"下载{variant_label} {model_name}", text_color='#3a86ff')
            self.bottom_label.configure(text=f"开始下载 {variant_label}：{model_name}")
        except Exception:
            pass

        downloader = self._downloader

        def work(progress_cb, status_cb):
            # 下载线程：只访问 downloader 和普通参数，不能碰 Tk。
            return downloader.download(
                model_id, variant=variant,
                progress_cb=progress_cb, status_cb=status_cb)

        task = DownloadTask(
            self,
            on_progress=self._update_model_download_progress,
            on_status=self._on_model_download_status,
            on_done=self._on_download_finished,
        )
        self._download_task = task

        if not task.start(work, cancel_cb=downloader.cancel):
            self._downloading = False
            self._download_task = None
            self._downloader = None
            try:
                self.local_model_btn.configure(
                    text="下载", state='normal')
                self.local_model_del_btn.configure(state='normal')
                self.local_model_progress.set(0)
                self.bottom_label.configure(
                    text="❌ 无法启动下载线程，请查看日志",
                    text_color='#cc0000')
            except Exception:
                pass

    def _update_model_download_progress(self, p, speed_mb=0.0,
                                        done_bytes=0, total_bytes=0):
        try:
            if not self.winfo_exists():
                return
            self.local_model_progress.set(p)
            if total_bytes > 0:
                detail = (f"{p * 100:5.1f}% · "
                          f"{done_bytes / 1024 / 1024:.0f}/"
                          f"{total_bytes / 1024 / 1024:.0f} MB · "
                          f"{speed_mb:.1f} MB/s")
            else:
                detail = f"{p * 100:5.1f}% · {speed_mb:.1f} MB/s"
            self.local_model_status.configure(text=detail)
        except Exception:
            pass

    def _on_model_download_status(self, text):
        self._download_status_text = text
        try:
            if self.winfo_exists():
                self.bottom_label.configure(text=text)
        except Exception:
            pass

    def _on_download_finished(self, ok=False, result=None,
                              error=None, cancelled=False):
        """下载完成或失败（始终在 Tk 主线程执行）"""
        self._downloading = False
        self._download_ok = bool(ok)
        self._download_task = None
        self._downloader = None
        if error:
            self._download_status_text = f"❌ 下载异常：{error[:120]}"

        try:
            if self.winfo_exists():
                self.local_model_btn.configure(state='normal')
                if self._download_ok:
                    self.local_model_progress.set(1.0)
        except Exception:
            pass

        if cancelled:
            try:
                if self.winfo_exists():
                    self.bottom_label.configure(text="下载已取消")
                    self.local_model_btn.configure(text="下载")
            except Exception:
                pass
            self._downloading_model_id = None
            self._downloading_variant = None
            if self.winfo_exists():
                self.after(500, self._refresh_local_models)
            return

        variant = getattr(self, '_downloading_variant', 'cpu')
        self._downloading_model_id = None
        self._downloading_variant = None

        if not getattr(self, '_download_ok', False):
            try:
                if self.winfo_exists():
                    self.bottom_label.configure(
                        text=self._download_status_text
                        or "❌ 下载失败，请查看日志",
                        text_color='#cc0000')
                    self.local_model_progress.set(0)
                    self.after(800, self._refresh_local_models)
            except Exception:
                pass
            return

        # ⭐ 下载完成后，根据 variant 自动设置设备
        try:
            if variant == 'cpu':
                self._set_device_strict('cpu')
                log.info("[本地模型] CPU 版下载完成，设备自动设为 CPU")
            elif variant == 'gpu':
                self._set_device_strict('auto')
                log.info("[本地模型] GPU 版下载完成，设备自动设为 auto")
        except Exception as e:
            log.exception(f"[本地模型] 设置设备失败：{e}")

        # 自动扫描同步 Provider
        try:
            from translation.api_config import sync_downloaded_models
            result = sync_downloaded_models()
            added = result.get('added', [])
            if added:
                log.info(f"[本地模型] 自动注册：{added}")
        except Exception as e:
            log.exception(f"[本地模型] 自动同步失败：{e}")

        try:
            self.translator.reload_config()
        except Exception:
            pass

        # 恢复 UI
        try:
            if self.winfo_exists():
                variant_label = 'GPU 版' if variant == 'gpu' else 'CPU 版'
                self.bottom_label.configure(text=f"✅ {variant_label}下载完成")
                self.after(500, self._refresh_local_models)
        except Exception:
            pass

    def _on_local_model_delete(self):
        from tkinter import messagebox

        label = getattr(self, '_current_model_label', None)
        if not label:
            return

        mid = getattr(self, '_model_id_by_label', {}).get(label)
        if not mid:
            return

        if mid.startswith('custom_'):
            if not messagebox.askyesno(
                    "确认",
                    "移除自定义模型注册？\n\n"
                    "（不会删除原文件）"):
                return
            try:
                from translation.local_models import (
                    remove_custom_model)
                remove_custom_model(mid)
                self.bottom_label.configure(
                    text=f"已移除注册：{label}")
            except Exception as e:
                log.exception(f"[本地模型] 移除失败：{e}")
                self.bottom_label.configure(text=f"移除失败：{e}")
            self.after(300, self._refresh_local_models)
            return

        if not messagebox.askyesno(
                "确认", f"删除本地模型 {mid}？\n\n"
                        f"（会删除 int8 和 FP32 两种版本）"):
            return

        try:
            from translation.local_models import ModelCacheManager
            ok = ModelCacheManager().delete_model(mid)
            if ok:
                try:
                    from translation.local_models import invalidate_model
                    invalidate_model(mid)
                except Exception:
                    pass
                try:
                    from translation.api_config import (
                        sync_downloaded_models)
                    result = sync_downloaded_models()
                    if result.get('removed'):
                        log.info(f"[本地模型] 已清理："
                                 f"{result['removed']}")
                except Exception:
                    pass

                self.translator.reload_config()
                self.bottom_label.configure(
                    text=f"已删除：{label}")
            else:
                self.bottom_label.configure(text="删除失败")
        except Exception as e:
            log.exception(f"[本地模型] 删除失败：{e}")
            self.bottom_label.configure(text=f"删除失败：{e}")

        self.after(300, self._refresh_local_models)

    def _open_model_manager(self):
        try:
            from ui.model_manager_dialog import ModelManagerDialog
            dlg = ModelManagerDialog(self)
            dlg.grab_set()
            dlg.bind('<Destroy>', lambda e: self._on_model_dialog_destroyed(
                e, dlg))
        except Exception as e:
            log.exception(f"[本地模型] 打开管理面板失败：{e}")

    def _open_models_dir(self):
        try:
            from translation.local_models import open_models_dir
            label = getattr(self, '_current_model_label', None)
            mid = getattr(self, '_model_id_by_label', {}).get(label)
            ok = open_models_dir(mid)
            if ok:
                self.bottom_label.configure(
                    text=f"已打开模型目录：{mid}" if mid else "已打开模型目录")
            else:
                self.bottom_label.configure(text="⚠️ 打开目录失败")
        except Exception as e:
            log.exception(f"[本地模型] 打开目录失败：{e}")
            self.bottom_label.configure(text=f"打开目录失败：{e}")

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
                lambda: self.switch_accel_mode(
                    'cpu_multi', force=True),
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

    # ================= OCR 开关 =================

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

        mode_tag = ("轻量型（PP-OCRv6 tiny）" if enabled
                    else "标准（server）")
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
            lambda: self._reload_ocr_safe(
                self.high_accuracy_ocr_switch),
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

        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
        self.bottom_label.configure(
            text=f"翻译策略：{label} | {status}")

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
                  self.vote_strategy_label,
                  self.vote_strategy_selector):
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
            self.vote_count_selector.set(
                str(cfg.get('vote_count', 3)))
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
            cfg['mode'] = getattr(
                self.app_state, 'translation_strategy',
                'single_translate')
            cfg['hybrid_order'] = self._hybrid_order_rev.get(
                self.hybrid_order_selector.get(), 'translate_first')
            try:
                cfg['vote_count'] = int(
                    self.vote_count_selector.get())
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

    # ================= 语言 =================

    def _on_source_lang_change(self, label):
        from config import SOURCE_LANG_TO_OCR

        rev = {v: k for k, v in SOURCE_LANGS.items()}
        key = rev.get(label, DEFAULT_SOURCE_LANG)

        old_ocr_lang = getattr(self, '_ocr_lang', 'CH')
        new_ocr_lang = SOURCE_LANG_TO_OCR.get(key, 'CH')

        self.app_state.source_lang = key
        try:
            self.translator.set_languages(source=key)
        except Exception:
            pass
        self.user_cfg.set('source_lang', key)
        self.user_cfg.save()
        log.info(f"[源语言] 切换为 {key} (OCR lang: {new_ocr_lang})")

        if new_ocr_lang != old_ocr_lang:
            if self.app_state.running:
                self.bottom_label.configure(
                    text="⚠️ 请先暂停翻译再切换源语言")
                return

            self._ocr_lang = new_ocr_lang
            self.user_cfg.set('ocr_lang', new_ocr_lang)
            self.user_cfg.save()

            self.bottom_label.configure(
                text=f"OCR 语言切换为 {new_ocr_lang}，正在重载...")
            self.after(200, self._reload_ocr_for_lang)

        self.bottom_label.configure(text=f"源语言：{label}")

    def _reload_ocr_for_lang(self):
        try:
            actual = self.ocr_engine.init(
                mode=self.ocr_engine.mode,
                pool_size=self.cpu_engine_count,
                dml_worker_count=self.dml_worker_count,
                model_size=self.ocr_model_size,
                high_accuracy=self.high_accuracy_ocr,
                speed_opt=self.enable_ocr_speed_opt,
                lang_name=getattr(self, '_ocr_lang', 'CH'),
            )
            try:
                self.pipeline.reset_debouncer()
            except Exception:
                pass
            self.overlay.clear()
            self.bottom_label.configure(
                text=f"✅ OCR 重载完成：{actual}")
        except Exception as e:
            log.exception(f"[OCR] 重载失败：{e}")
            self.bottom_label.configure(text=f"重载失败：{e}")

    def _on_target_lang_change(self, label):
        rev = {v: k for k, v in TARGET_LANGS.items()}
        key = rev.get(label, DEFAULT_TARGET_LANG)
        self.app_state.target_lang = key
        try:
            self.translator.set_languages(target=key)
        except Exception:
            pass
        self.user_cfg.set('target_lang', key)
        self.user_cfg.save()

        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()
        self.app_state.reset_signature()

        log.info(f"[目标语言] 切换为 {key}")
        self.bottom_label.configure(text=f"目标语言：{label}")

    def _sync_lang_ui(self):
        s = getattr(self.app_state, 'source_lang',
                     DEFAULT_SOURCE_LANG)
        t = getattr(self.app_state, 'target_lang',
                     DEFAULT_TARGET_LANG)
        try:
            self.source_lang_selector.set(
                SOURCE_LANGS.get(s, '自适应'))
        except Exception:
            pass
        try:
            self.target_lang_selector.set(
                TARGET_LANGS.get(t, '中文'))
        except Exception:
            pass

    # ================= 请求节流 =================

    def _on_throttle_toggle(self):
        enabled = bool(self.throttle_switch.get())
        try:
            from translation.throttle import get_limiter
            get_limiter().set_enabled(enabled)
        except Exception as e:
            log.exception(f"[节流] 应用失败：{e}")
        self.user_cfg.set('enable_throttle', enabled)
        self.user_cfg.save()
        msg = ("请求节流：已开启" if enabled
               else "请求节流：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[节流] {msg}")

    def _on_throttle_qps_change(self, value):
        try:
            qps = round(float(value), 1)
        except Exception:
            return
        qps = max(THROTTLE_QPS_MIN,
                  min(THROTTLE_QPS_MAX, qps))
        try:
            self.throttle_qps_value_label.configure(
                text=f"{qps:.1f}")
        except Exception:
            pass
        try:
            from translation.throttle import get_limiter
            get_limiter().set_qps(qps)
        except Exception:
            pass
        self.user_cfg.set('throttle_qps', qps)
        self.user_cfg.save()

    # ================= 显示模式 =================

    def _on_display_mode_change(self, label):
        key = self._display_label_to_key.get(
            label, DEFAULT_DISPLAY_MODE)
        self.app_state.display_mode = key
        self._sync_display_mode_ui()

        self.user_cfg.set('display_mode', key)
        self.user_cfg.save()

        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()
        if key == 'window':
            self.caption_window.show()
        else:
            self.caption_window.hide()

        self.app_state.reset_signature()
        log.info(f"[显示模式] 切换为 {key}")
        self.bottom_label.configure(text=f"显示模式：{label}")

    def _sync_display_mode_ui(self):
        key = getattr(self.app_state, 'display_mode',
                      DEFAULT_DISPLAY_MODE)
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

        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
                self.pipeline.reset_debouncer()
            except Exception:
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

    def open_latency_test(self):
        try:
            from ui.latency_dialog import LatencyTestDialog
            dlg = LatencyTestDialog(
                self,
                translator=self.translator,
                ocr_engine=self.ocr_engine,
                pipeline=self.pipeline,
                app_state=self.app_state,
            )
            dlg.grab_set()
        except Exception as e:
            log.exception(f"[延迟测试] 打开失败：{e}")

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
                        f"（内置 {stats['builtin']} + 其他 "
                        f"{stats['user']}）| 表：{fname}")
            else:
                text = (f"共 {stats['total']} 条"
                        f"（内置 {stats['builtin']} + 自定义 "
                        f"{stats['user']}）")

            self.phrase_count_label.configure(text=text)
        except Exception:
            pass

    # ================= 加速方式 =================

    def switch_accel_mode(self, mode, force=False):
        current_variant = check_onnxruntime_variant() or 'cpu'

        if mode == 'directml' and current_variant != 'directml':
            self._show_need_install(
                'DirectML', 'onnxruntime-directml', '50MB')
            return
        if mode == 'cuda' and current_variant != 'gpu':
            self._show_need_install(
                'NVIDIA CUDA', 'onnxruntime-gpu', '1.6GB')
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

        try:
            actual = self.ocr_engine.init(
                mode=mode,
                pool_size=self.cpu_engine_count,
                dml_worker_count=self.dml_worker_count,
                model_size=self.ocr_model_size,
                high_accuracy=self.high_accuracy_ocr,
                speed_opt=self.enable_ocr_speed_opt,
                lang_name=getattr(self, '_ocr_lang', 'CH'),
            )

            self.cache.clear()
            if hasattr(self.pipeline, 'inpaint_cache'):
                self.pipeline.inpaint_cache.clear()
            if hasattr(self.pipeline, 'fine_cache'):
                self.pipeline.fine_cache.clear()
            if hasattr(self.pipeline, 'color_cache'):
                self.pipeline.color_cache.clear()
            self.app_state.reset_signature()
            try:
                self.pipeline.reset_debouncer()
            except Exception:
                pass
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
        """弹窗询问是否立即安装；确认后启动自动安装流程"""
        from tkinter import messagebox

        answer = messagebox.askyesno(
            f"需要安装 {name}",
            f"当前 onnxruntime 版本不支持 {name}。\n\n"
            f"需要先安装：{pkg}（约 {size}）\n\n"
            f"是否现在自动安装？（需要联网，约几分钟）",
            parent=self,
        )
        if not answer:
            return

        self._do_install_accel(name)

    def _do_install_accel(self, name):
        """一键安装 DirectML / CUDA 环境"""
        try:
            from cuda_setup import (
                is_frozen, install_directml, install_cuda,
            )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror(
                "错误", f"无法加载安装模块：{e}", parent=self)
            return

        if is_frozen():
            from tkinter import messagebox
            messagebox.showinfo(
                "无法自动安装",
                "当前是打包版（exe），无法自动 pip 安装。\n\n"
                "请使用源码运行（python main.py）后安装，\n"
                "或下载对应 GPU 版本的 exe。",
                parent=self,
            )
            return

        # ============ 安装进度对话框 ============
        dlg = ctk.CTkToplevel(self)
        dlg.title(f"正在安装 {name}")
        dlg.geometry("500x280")
        dlg.attributes('-topmost', True)
        dlg.resizable(False, False)
        try:
            dlg.grab_set()
        except Exception:
            pass

        dlg.update_idletasks()
        sw = dlg.winfo_screenwidth()
        sh = dlg.winfo_screenheight()
        dlg.geometry(f"500x280+{(sw-500)//2}+{(sh-280)//2}")

        ctk.CTkLabel(
            dlg, text=f"正在安装 {name}",
            font=(FONT_NAME, 14, 'bold'),
        ).pack(pady=(20, 6))

        ctk.CTkLabel(
            dlg,
            text=f"安装包：{name}（可能需要几分钟，请耐心等待）",
            font=(FONT_NAME, 9), text_color='#888888',
        ).pack(pady=(0, 12))

        status_label = ctk.CTkLabel(
            dlg, text="准备中...",
            font=(FONT_NAME, 10), text_color='gray',
            wraplength=440, justify='left',
        )
        status_label.pack(pady=4, padx=20, fill='x')

        progress = ctk.CTkProgressBar(dlg, width=440, height=12)
        progress.set(0)
        progress.pack(pady=12, padx=30)

        hint_label = ctk.CTkLabel(
            dlg, text="安装过程中请勿关闭程序",
            font=(FONT_NAME, 9), text_color='#888888',
        )
        hint_label.pack(pady=(0, 6))

        close_btn = ctk.CTkButton(
            dlg, text="关闭", width=100, state='disabled',
            command=dlg.destroy,
        )
        close_btn.pack(pady=(6, 12))

        _state = {'v': 0.0}

        def _safe_ui(fn):
            try:
                if dlg.winfo_exists():
                    dlg.after(0, fn)
            except Exception:
                pass

        def progress_cb(msg):
            def _upd():
                try:
                    status_label.configure(
                        text=str(msg), text_color='gray')
                except Exception:
                    pass
                _state['v'] = min(0.9, _state['v'] + 0.12)
                try:
                    progress.set(_state['v'])
                except Exception:
                    pass
            _safe_ui(_upd)

        def done_cb(ok, msg):
            def _done():
                try:
                    progress.set(1.0)
                except Exception:
                    pass
                try:
                    if ok:
                        status_label.configure(
                            text=str(msg), text_color='#00a000')
                        hint_label.configure(
                            text="✅ 安装完成！请手动关闭并重新打开程序。")
                    else:
                        status_label.configure(
                            text=str(msg), text_color='#cc0000')
                        hint_label.configure(
                            text="❌ 安装失败，请检查网络或手动安装。")
                except Exception:
                    pass
                try:
                    close_btn.configure(state='normal')
                except Exception:
                    pass

                try:
                    if ok:
                        self.bottom_label.configure(
                            text=f"✅ {name} 安装完成，请重启程序")
                    else:
                        self.bottom_label.configure(
                            text=f"❌ {name} 安装失败")
                except Exception:
                    pass
            _safe_ui(_done)

        try:
            if name == 'DirectML':
                install_directml(
                    progress_cb=progress_cb, done_cb=done_cb)
            elif name == 'NVIDIA CUDA':
                install_cuda(
                    progress_cb=progress_cb, done_cb=done_cb)
            else:
                done_cb(False, f"未知的加速方式：{name}")
        except Exception as e:
            log.exception(f"[安装] {name} 异常")
            done_cb(False, f"异常：{e}")

        try:
            self.bottom_label.configure(
                text=f"正在安装 {name}，请稍候...")
        except Exception:
            pass

    def _sync_accel_ui(self):
        try:
            current_variant = check_onnxruntime_variant() or 'cpu'
            current_mode = self.ocr_engine.mode
            n_engines = len(self.ocr_engine.engines)

            model_tag = ("tiny" if self.ocr_model_size == 'tiny'
                         else "标准")
            ha_tag = " + 高精度" if self.high_accuracy_ocr else ""
            opt_tag = (" + 加速" if self.enable_ocr_speed_opt
                       else "")

            mode_map = {
                'cpu_multi': f'多线程 CPU（{n_engines} 引擎，'
                             f'{model_tag}{ha_tag}{opt_tag}）',
                'directml': f'DirectML（单引擎，'
                            f'{model_tag}{ha_tag}{opt_tag}）',
                'cuda': f'NVIDIA CUDA（{n_engines} 引擎，'
                        f'{model_tag}{ha_tag}{opt_tag}）',
            }
            self.ort_variant_label.configure(
                text=f"当前引擎："
                     f"{mode_map.get(current_mode, current_mode)}")

            self._highlight_buttons(current_mode, current_variant)

            if current_mode == 'cuda':
                ok, msg = check_cuda_ready()
                if ok:
                    self.accel_status_label.configure(
                        text=f"✅ CUDA 多引擎运行中"
                             f"（{n_engines} 引擎）",
                        text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            elif current_mode == 'directml':
                ok, msg = check_directml_ready()
                if ok:
                    self.accel_status_label.configure(
                        text=f"✅ DirectML 运行中（单引擎）",
                        text_color='#00a000')
                else:
                    self.accel_status_label.configure(
                        text=f"⚠️ {msg}", text_color='#cc6600')
            else:
                self.accel_status_label.configure(
                    text=f"✅ CPU 多线程运行中"
                         f"（{n_engines} 引擎）",
                    text_color='#00a000')
        except Exception as e:
            log.exception("[加速] 状态刷新失败")
            self.accel_status_label.configure(
                text=f"状态获取失败：{e}", text_color='#cc0000')

    def _highlight_buttons(self, current_mode, variant):
        """
        三个按钮永远可点击：
        - 已安装的变体：显示正常色（激活=绿，未激活=蓝）
        - 未安装的变体：灰色但可点击，点后弹安装提示
        """
        try:
            self.btn_cpu_multi.configure(
                fg_color=('#28a745' if current_mode == 'cpu_multi'
                          else '#6c757d'),
                state='normal',
            )

            dml_active = (current_mode == 'directml')
            if variant == 'directml':
                dml_color = '#28a745' if dml_active else '#3a86ff'
            else:
                dml_color = '#adb5bd'
            self.btn_directml.configure(
                fg_color=dml_color, state='normal')

            cuda_active = (current_mode == 'cuda')
            if variant == 'gpu':
                cuda_color = '#28a745' if cuda_active else '#3a86ff'
            else:
                cuda_color = '#adb5bd'
            self.btn_cuda.configure(
                fg_color=cuda_color, state='normal')
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
        self.scan_value_label.configure(
            text=f"{cfg['scan_interval']:.2f} s")
        self._update_mode_desc(mode_key)
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()

        fast_tag = ("（分段翻译）" if cfg.get('fast_mode')
                    else "（批量翻译）")
        self.bottom_label.configure(
            text=f"性能模式：{cfg['name']}{fast_tag}")
        log.info(f"[模式] 切换为 {cfg['name']} {fast_tag}")

        self.user_cfg.set('performance_mode', mode_key)
        self.user_cfg.set('scan_interval', cfg['scan_interval'])
        self.user_cfg.save()

    def _update_mode_desc(self, mode_key):
        cfg = PERFORMANCE_MODES[mode_key]
        mode_tag = ("分段翻译" if cfg.get('fast_mode')
                    else "批量翻译")
        desc = (
            f"扫描 {cfg['scan_interval']:.2f}s | "
            f"灵敏度 {cfg['change_ratio_threshold'] * 100:.3f}% | "
            f"外扩 {cfg['bbox_padding']}px | "
            f"OCR {cfg['max_ocr_width']}×{cfg['max_ocr_height']} | "
            f"{mode_tag}"
        )
        self.mode_desc_label.configure(text=desc)

    # ================= 开关 =================

    def _on_smart_ocr_toggle(self):
        enabled = bool(self.smart_ocr_switch.get())
        # 帧差检测与分块 OCR 必须同时开启才生效，合并为一个开关。
        self.app_state.enable_frame_diff = enabled
        self.app_state.enable_block_ocr = enabled
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        msg = ("智能帧差 + 分块 OCR：已开启"
               if enabled else "智能帧差 + 分块 OCR：已关闭（整屏识别）")
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        # 同时写入新旧配置键，兼容旧版本配置。
        self.user_cfg.set('enable_smart_ocr', enabled)
        self.user_cfg.set('enable_frame_diff', enabled)
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
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
        msg = ("标题/Logo 过滤：已开启" if enabled
               else "标题/Logo 过滤：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_title_filter', enabled)
        self.user_cfg.save()

    def _on_ocr_verify_toggle(self):
        enabled = bool(self.ocr_verify_switch.get())
        self.app_state.enable_ocr_verify = enabled
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()
        msg = ("二次 OCR 验证：已开启" if enabled
               else "二次 OCR 验证：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[开关] {msg}")
        self.user_cfg.set('enable_ocr_verify', enabled)
        self.user_cfg.save()

    def _on_fast_lookup_toggle(self):
        enabled = bool(self.fast_lookup_switch.get())
        self.app_state.enable_fast_lookup = enabled
        self.app_state.reset_signature()
        msg = ("快速响应层：已开启" if enabled
               else "快速响应层：已关闭")
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
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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

    def _on_bilingual_gap_change(self, value):
        value = int(round(value))
        try:
            self.bilingual_gap_value_label.configure(
                text=str(value))
        except Exception:
            pass
        self.overlay.set_bilingual_gap(value)
        self.overlay.clear()
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.user_cfg.set('bilingual_gap', value)
        self.user_cfg.save()

    def _on_inpaint_toggle(self):
        enabled = bool(self.inpaint_switch.get())
        self.app_state.use_inpainting = enabled
        self.app_state.reset_signature()
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
        self.overlay.clear()
        msg = ("图像修复（整框）：已启用" if enabled
               else "图像修复（整框）：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('use_inpainting', enabled)
        self.user_cfg.save()

    def _on_auto_clean_toggle(self):
        enabled = bool(self.auto_clean_switch.get())
        self.cache.set_auto_clean(enabled)
        msg = ("自动清理缓存：已开启" if enabled
               else "自动清理缓存：已关闭")
        self.bottom_label.configure(text=msg)
        log.info(f"[设置] {msg}")
        self.user_cfg.set('auto_clean', enabled)
        self.user_cfg.save()

    def _on_cache_limit_change(self, value):
        value = int(round(value / 50.0) * 50)
        value = max(CACHE_MIN, min(CACHE_MAX, value))
        self.cache_limit_label.configure(text=str(value))
        self.cache.max_size = value
        # 立即触发一次清理检查
        self.cache.set_auto_clean(self.cache.auto_clean)
        self._update_cache_label()
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
        self.bottom_label.configure(text="缓存已清空并释放了空间")
        log.info("[缓存] 手动清空")

    def _update_cache_label(self):
        try:
            n = self.cache.size()
            mb = self.cache.get_file_size_mb()
            self.cache_size_label.configure(
                text=f"已占用：{mb:.2f} MB ({n} 条)")
        except Exception:
            pass

    def _open_cache_dir(self):
        """打开缓存目录"""
        try:
            from cache import CACHE_DIR
            import os, sys
            os.makedirs(CACHE_DIR, exist_ok=True)
            if sys.platform == 'win32':
                os.startfile(CACHE_DIR)
            elif sys.platform == 'darwin':
                os.system(f'open "{CACHE_DIR}"')
            else:
                os.system(f'xdg-open "{CACHE_DIR}"')
        except Exception as e:
            log.error(f"[缓存] 打开目录失败：{e}")

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
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
        self.bottom_label.configure(
            text="请按下你想要的组合键...")

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
                self.bottom_label.configure(
                    text=f"{name} 热键为空")
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
            self.status_label.configure(
                text="状态：运行中", text_color='green')
            self.btn_toggle.configure(text="⏸ 暂停")
            self.bottom_label.configure(text="正在扫描屏幕...")
            log.info("状态：运行中")
        else:
            self.status_label.configure(
                text="状态：已暂停", text_color='gray')
            self.btn_toggle.configure(text="▶ 开始")
            self.bottom_label.configure(text="已暂停")
            self.scroll_detector.reset()
            self.app_state.scroll_boost = False
            log.info("状态：已暂停")

    def clear_overlay(self):
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
        try:
            self.pipeline.reset_debouncer()
        except Exception:
            pass
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
            is_window_mode = (
                self.app_state.display_mode == 'window')

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
                                    base_font_size=(
                                        self.app_state.base_font_size),
                                )
                        except Exception as e:
                            log.exception(
                                f"[UI] 分发译文失败：{e}")
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
                        log.info(f"[{mode}] 移除 {pruned} 个"
                                 f"超出边界的译文")
                except Exception as e:
                    log.exception(f"[UI] 边界裁剪失败：{e}")

            now = time.time()

            if now - self._last_cache_ui_update >= 5:
                try:
                    self._update_cache_label()
                    self._update_phrase_count()
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
                    if (is_follow and
                            self.scroll_detector.is_scrolling()):
                        interval = max(0.03, base * 0.2)
                        self.app_state.scroll_boost = True
                    else:
                        interval = base * 0.3
                        self.app_state.scroll_boost = False

                    if qsize >= 1:
                        interval *= 2.0
                else:
                    pending = getattr(self.app_state,
                                      'pending_change', False)

                    if (is_follow and
                            self.scroll_detector.is_scrolling()):
                        interval = base * 0.25
                        self.app_state.scroll_boost = True
                    elif pending:
                        interval = base * 0.3
                        self.app_state.scroll_boost = False
                    else:
                        interval = base
                        self.app_state.scroll_boost = False
                        if idle >= IDLE_THRESHOLD_FOR_SLOWDOWN:
                            interval = (base *
                                        IDLE_SLOWDOWN_FACTOR)
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