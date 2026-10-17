"""面板构建 Mixin"""

import customtkinter as ctk

from config import (
    FONT_NAME,
    PERFORMANCE_MODES, DEFAULT_MODE,
    SCAN_MIN, SCAN_MAX,
    FONT_MIN, FONT_MAX,
    INITIAL_FONT_SIZE,
    CPU_ENGINE_COUNT_MIN, CPU_ENGINE_COUNT_MAX, CPU_ENGINE_COUNT_DEFAULT,
    DML_WORKER_COUNT_MIN, DML_WORKER_COUNT_MAX, DML_WORKER_COUNT_DEFAULT,
    DEFAULT_ENABLE_SMART_OCR,
    DEFAULT_ENABLE_FILTER,
    DEFAULT_ENABLE_GENERAL_FILTER,
    DEFAULT_ENABLE_ENGLISH_ONLY,
    DEFAULT_ENABLE_TITLE_FILTER,
    DEFAULT_ENABLE_OCR_VERIFY,
    DEFAULT_ENABLE_FAST_LOOKUP,
    DEFAULT_ENABLE_ASYNC_TRANSLATE,
    DEFAULT_ENABLE_WINDOW_POOL,
    DEFAULT_USE_INPAINTING,
    DEFAULT_ENABLE_FINE_TEXT,
    DEFAULT_ENABLE_ASYNC_OVERLAY,
    FG_COLOR_PRESETS, DEFAULT_FG_COLOR,
    HOTKEY_LABELS,
)
from inpaint import CV2_AVAILABLE

from .constants import CACHE_MIN, CACHE_MAX, CACHE_DEFAULT


class DisplayPanelMixin:
    def _build_display_tab(self, parent):
        # ===== 性能模式 =====
        ctk.CTkLabel(
            parent, text="性能模式：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=0, column=0, sticky='w', padx=(15, 5), pady=(12, 6))

        self.mode_selector = ctk.CTkSegmentedButton(
            parent, values=["快", "中", "慢"],
            command=self._on_mode_change, font=(FONT_NAME, 11),
        )
        self.mode_selector.set(PERFORMANCE_MODES[DEFAULT_MODE]['name'])
        self.mode_selector.grid(
            row=0, column=1, columnspan=2,
            sticky='ew', padx=10, pady=(12, 6))

        self.mode_desc_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left',
        )
        self.mode_desc_label.grid(
            row=1, column=0, columnspan=3,
            sticky='w', padx=(15, 15), pady=(0, 8))
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
            accel_btns, text="多线程 CPU", width=140, height=48,
            font=(FONT_NAME, 10),
            fg_color='#6c757d', hover_color='#5a6268',
            command=lambda: self.switch_accel_mode('cpu_multi'),
        )
        self.btn_cpu_multi.pack(side='left', padx=3)

        self.btn_directml = ctk.CTkButton(
            accel_btns, text="DirectML GPU\n(单引擎)",
            width=140, height=48, font=(FONT_NAME, 10),
            fg_color='#3a86ff', hover_color='#2a6fd8',
            command=lambda: self.switch_accel_mode('directml'),
        )
        self.btn_directml.pack(side='left', padx=3)

        self.btn_cuda = ctk.CTkButton(
            accel_btns, text="NVIDIA CUDA\n(多引擎)",
            width=140, height=48, font=(FONT_NAME, 10),
            fg_color='#28a745', hover_color='#1e7e34',
            command=lambda: self.switch_accel_mode('cuda'),
        )
        self.btn_cuda.pack(side='left', padx=3)

        # ===== CPU 引擎数 =====
        cpu_row = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        cpu_row.grid(row=5, column=0, columnspan=3,
                     sticky='ew', padx=(25, 15), pady=(6, 2))
        cpu_row.pack_propagate(True)

        ctk.CTkLabel(cpu_row, text="CPU 引擎数：",
                     font=(FONT_NAME, 10), text_color='#333333',
                     anchor='w').pack(side='left')

        self.cpu_count_label = ctk.CTkLabel(
            cpu_row, text=str(CPU_ENGINE_COUNT_DEFAULT), width=26,
            font=(FONT_NAME, 10, 'bold'), text_color='#3a86ff',
        )
        self.cpu_count_label.pack(side='right')

        self.cpu_count_slider = ctk.CTkSlider(
            cpu_row, from_=CPU_ENGINE_COUNT_MIN,
            to=CPU_ENGINE_COUNT_MAX,
            number_of_steps=CPU_ENGINE_COUNT_MAX - CPU_ENGINE_COUNT_MIN,
            command=self._on_cpu_count_change,
            height=14, width=160,
        )
        self.cpu_count_slider.set(CPU_ENGINE_COUNT_DEFAULT)
        self.cpu_count_slider.pack(side='right', padx=4)

        # ===== CUDA 引擎数 =====
        dml_row = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        dml_row.grid(row=6, column=0, columnspan=3,
                     sticky='ew', padx=(25, 15), pady=(2, 4))
        dml_row.pack_propagate(True)

        ctk.CTkLabel(dml_row, text="CUDA 引擎数：",
                     font=(FONT_NAME, 10), text_color='#333333',
                     anchor='w').pack(side='left')

        self.dml_count_label = ctk.CTkLabel(
            dml_row, text=str(DML_WORKER_COUNT_DEFAULT), width=26,
            font=(FONT_NAME, 10, 'bold'), text_color='#3a86ff',
        )
        self.dml_count_label.pack(side='right')

        self.dml_count_slider = ctk.CTkSlider(
            dml_row, from_=DML_WORKER_COUNT_MIN,
            to=DML_WORKER_COUNT_MAX,
            number_of_steps=DML_WORKER_COUNT_MAX - DML_WORKER_COUNT_MIN,
            command=self._on_dml_count_change,
            height=14, width=160,
        )
        self.dml_count_slider.set(DML_WORKER_COUNT_DEFAULT)
        self.dml_count_slider.pack(side='right', padx=4)

        ctk.CTkLabel(
            parent,
            text="💡 CPU 引擎数用于「多线程 CPU」模式；"
                 "CUDA 引擎数用于「NVIDIA CUDA」模式；"
                 "DirectML 固定单引擎",
            font=(FONT_NAME, 9), text_color='#888888', anchor='w',
            wraplength=600, justify='left',
        ).grid(row=7, column=0, columnspan=3,
               sticky='w', padx=(25, 15), pady=(0, 6))

        self.accel_status_label = ctk.CTkLabel(
            parent, text="",
            font=(FONT_NAME, 9), text_color='#666666',
            anchor='w', justify='left', wraplength=600,
        )
        self.accel_status_label.grid(
            row=8, column=0, columnspan=3,
            sticky='w', padx=(25, 15), pady=(0, 8))

        # ===== 优化功能 =====
        ctk.CTkLabel(
            parent, text="优化功能：", font=(FONT_NAME, 11, 'bold'),
        ).grid(row=9, column=0, sticky='w', padx=(15, 5), pady=(6, 4))

        self.smart_ocr_switch = ctk.CTkSwitch(
            parent,
            text="智能帧差 + 分块 OCR（只识别变化区域，静止时跳过）",
            command=self._on_smart_ocr_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_SMART_OCR:
            self.smart_ocr_switch.select()
        self.smart_ocr_switch.grid(row=10, column=0, columnspan=3,
                                   sticky='w', padx=(25, 15), pady=(0, 3))

        self.filter_switch = ctk.CTkSwitch(
            parent, text="文字过滤（过滤异常框）",
            command=self._on_filter_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FILTER:
            self.filter_switch.select()
        self.filter_switch.grid(row=12, column=0, columnspan=3,
                                sticky='w', padx=(25, 15), pady=(0, 3))

        self.general_filter_switch = ctk.CTkSwitch(
            parent, text="通用过滤（只过滤数字 / 单位 / 图像）",
            command=self._on_general_filter_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_GENERAL_FILTER:
            self.general_filter_switch.select()
        self.general_filter_switch.grid(row=13, column=0, columnspan=3,
                                        sticky='w', padx=(25, 15),
                                        pady=(0, 3))

        self.fast_lookup_switch = ctk.CTkSwitch(
            parent, text="快速响应层（短语缓存，只处理单词和词组）",
            command=self._on_fast_lookup_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_FAST_LOOKUP:
            self.fast_lookup_switch.select()
        self.fast_lookup_switch.grid(row=14, column=0, columnspan=3,
                                     sticky='w', padx=(25, 15),
                                     pady=(0, 3))

        self.english_only_switch = ctk.CTkSwitch(
            parent,
            text="英语特化（字母占比 ≥ 50%，中文占比 ≤ 20%）",
            command=self._on_english_only_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_ENGLISH_ONLY:
            self.english_only_switch.select()
        self.english_only_switch.grid(row=15, column=0, columnspan=3,
                                      sticky='w', padx=(25, 15),
                                      pady=(0, 3))

        self.title_filter_switch = ctk.CTkSwitch(
            parent, text="标题/Logo 过滤（全大写 或 超大字号）",
            command=self._on_title_filter_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_TITLE_FILTER:
            self.title_filter_switch.select()
        self.title_filter_switch.grid(row=16, column=0, columnspan=3,
                                      sticky='w', padx=(25, 15),
                                      pady=(0, 3))

        self.ocr_verify_switch = ctk.CTkSwitch(
            parent, text="二次 OCR 验证（防一闪而过，慢约 1 倍）",
            command=self._on_ocr_verify_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_OCR_VERIFY:
            self.ocr_verify_switch.select()
        self.ocr_verify_switch.grid(row=17, column=0, columnspan=3,
                                    sticky='w', padx=(25, 15),
                                    pady=(0, 3))

        self.lightweight_ocr_switch = ctk.CTkSwitch(
            parent,
            text="轻量型 OCR（PP-OCRv6 tiny，速度极快，精度略低）",
            command=self._on_lightweight_ocr_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        self.lightweight_ocr_switch.grid(row=18, column=0, columnspan=3,
                                         sticky='w', padx=(25, 15),
                                         pady=(0, 3))

        self.high_accuracy_ocr_switch = ctk.CTkSwitch(
            parent, text="高精度 OCR（双引擎投票，慢约 2 倍）",
            command=self._on_high_accuracy_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        self.high_accuracy_ocr_switch.grid(row=19, column=0, columnspan=3,
                                           sticky='w', padx=(25, 15),
                                           pady=(0, 3))

        self.ocr_speed_opt_switch = ctk.CTkSwitch(
            parent,
            text="OCR 加速优化（关方向分类 + det 限长 + cv2 缩放）",
            command=self._on_ocr_speed_opt_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        self.ocr_speed_opt_switch.grid(row=20, column=0, columnspan=3,
                                       sticky='w', padx=(25, 15),
                                       pady=(0, 3))

        # ===== 短语表按钮 =====
        phrase_btns = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        phrase_btns.grid(row=21, column=0, columnspan=3,
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

        # ===== 异步翻译 / 窗口池 =====
        self.async_translate_switch = ctk.CTkSwitch(
            parent, text="异步翻译（不阻塞 OCR）",
            command=self._on_async_translate_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_ASYNC_TRANSLATE:
            self.async_translate_switch.select()
        self.async_translate_switch.grid(row=22, column=0, columnspan=3,
                                         sticky='w', padx=(25, 15),
                                         pady=(0, 3))

        self.window_pool_switch = ctk.CTkSwitch(
            parent, text="窗口池（复用译文窗口）",
            command=self._on_window_pool_toggle,
            onvalue=True, offvalue=False, font=(FONT_NAME, 10),
        )
        if DEFAULT_ENABLE_WINDOW_POOL:
            self.window_pool_switch.select()
        self.window_pool_switch.grid(row=23, column=0, columnspan=3,
                                    sticky='w', padx=(25, 15),
                                    pady=(0, 8))

        # ===== 刷新间隔 =====
        ctk.CTkLabel(
            parent, text="刷新间隔 (秒)：", font=(FONT_NAME, 11),
        ).grid(row=24, column=0, sticky='w', padx=(15, 5), pady=6)

        self.scan_value_label = ctk.CTkLabel(
            parent,
            text=f"{PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval']:.2f} s",
            width=60, font=(FONT_NAME, 11),
        )
        self.scan_value_label.grid(row=24, column=2, sticky='e',
                                    padx=(5, 15))

        n_steps = int((SCAN_MAX - SCAN_MIN) / 0.05)
        self.scan_slider = ctk.CTkSlider(
            parent, from_=SCAN_MIN, to=SCAN_MAX,
            number_of_steps=n_steps, command=self._on_scan_change,
        )
        self.scan_slider.set(PERFORMANCE_MODES[DEFAULT_MODE]['scan_interval'])
        self.scan_slider.grid(row=24, column=1, sticky='ew', padx=10)

        # ===== 文字大小 =====
        ctk.CTkLabel(
            parent, text="文字大小：", font=(FONT_NAME, 11),
        ).grid(row=25, column=0, sticky='w', padx=(15, 5), pady=6)

        self.font_value_label = ctk.CTkLabel(
            parent, text=str(INITIAL_FONT_SIZE), width=60,
            font=(FONT_NAME, 11),
        )
        self.font_value_label.grid(row=25, column=2, sticky='e',
                                    padx=(5, 15))

        self.font_slider = ctk.CTkSlider(
            parent, from_=FONT_MIN, to=FONT_MAX,
            number_of_steps=FONT_MAX - FONT_MIN,
            command=self._on_font_change,
        )
        self.font_slider.set(INITIAL_FONT_SIZE)
        self.font_slider.grid(row=25, column=1, sticky='ew', padx=10)

        # ===== 双语距离 =====
        ctk.CTkLabel(
            parent, text="双语距离 (px)：", font=(FONT_NAME, 11),
        ).grid(row=26, column=0, sticky='w', padx=(15, 5), pady=6)

        self.bilingual_gap_value_label = ctk.CTkLabel(
            parent, text="2", width=60, font=(FONT_NAME, 11),
        )
        self.bilingual_gap_value_label.grid(row=26, column=2,
                                             sticky='e', padx=(5, 15))

        self.bilingual_gap_slider = ctk.CTkSlider(
            parent, from_=-30, to=100, number_of_steps=130,
            command=self._on_bilingual_gap_change,
        )
        self.bilingual_gap_slider.set(2)
        self.bilingual_gap_slider.grid(row=26, column=1,
                                        sticky='ew', padx=10)

        ctk.CTkLabel(
            parent,
            text="💡 正数：译文在原文下方；负数：译文覆盖原文部分",
            font=(FONT_NAME, 9), text_color='#888888', anchor='w',
        ).grid(row=27, column=0, columnspan=3,
               sticky='w', padx=(25, 15), pady=(0, 6))

        # ===== 译文颜色 =====
        ctk.CTkLabel(
            parent, text="译文颜色：", font=(FONT_NAME, 11),
        ).grid(row=28, column=0, sticky='w', padx=(15, 5), pady=6)

        self.fg_color_selector = ctk.CTkOptionMenu(
            parent, values=list(FG_COLOR_PRESETS.values()),
            command=self._on_fg_color_change,
            font=(FONT_NAME, 10), width=200,
        )
        self.fg_color_selector.set(FG_COLOR_PRESETS[DEFAULT_FG_COLOR])
        self.fg_color_selector.grid(row=28, column=1, columnspan=2,
                                     sticky='w', padx=10, pady=6)

        # ===== 图像修复 =====
        ctk.CTkLabel(
            parent, text="图像修复（整框）：", font=(FONT_NAME, 11),
        ).grid(row=29, column=0, sticky='w', padx=(15, 5),
               pady=(12, 6))

        self.inpaint_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_inpaint_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_USE_INPAINTING and CV2_AVAILABLE:
            self.inpaint_switch.select()
        self.inpaint_switch.grid(row=29, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.inpaint_switch.configure(state='disabled')

        # ===== 精细文字替换 =====
        ctk.CTkLabel(
            parent, text="精细文字替换（保留背景）：",
            font=(FONT_NAME, 11),
        ).grid(row=30, column=0, sticky='w', padx=(15, 5),
               pady=(8, 6))

        self.fine_text_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_fine_text_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_FINE_TEXT and CV2_AVAILABLE:
            self.fine_text_switch.select()
        self.fine_text_switch.grid(row=30, column=1, sticky='w', padx=10)

        if not CV2_AVAILABLE:
            self.fine_text_switch.configure(state='disabled')

        # ===== 异步覆盖 =====
        ctk.CTkLabel(
            parent, text="异步覆盖（逐字显示）：",
            font=(FONT_NAME, 11),
        ).grid(row=31, column=0, sticky='w', padx=(15, 5),
               pady=(8, 6))

        self.async_overlay_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_async_overlay_toggle,
            onvalue=True, offvalue=False,
        )
        if DEFAULT_ENABLE_ASYNC_OVERLAY:
            self.async_overlay_switch.select()
        self.async_overlay_switch.grid(row=31, column=1,
                                        sticky='w', padx=10)

        # ===== 自动清理缓存 =====
        ctk.CTkLabel(
            parent, text="自动清理缓存：", font=(FONT_NAME, 11),
        ).grid(row=32, column=0, sticky='w', padx=(15, 5),
               pady=(12, 6))

        self.auto_clean_switch = ctk.CTkSwitch(
            parent, text="",
            command=self._on_auto_clean_toggle,
            onvalue=True, offvalue=False,
        )
        self.auto_clean_switch.select()
        self.auto_clean_switch.grid(row=32, column=1,
                                     sticky='w', padx=10)

        # ===== 缓存上限 =====
        ctk.CTkLabel(
            parent, text="缓存上限 (条)：", font=(FONT_NAME, 11),
        ).grid(row=33, column=0, sticky='w', padx=(15, 5), pady=6)

        self.cache_limit_label = ctk.CTkLabel(
            parent, text=str(CACHE_DEFAULT), width=60,
            font=(FONT_NAME, 11),
        )
        self.cache_limit_label.grid(row=33, column=2, sticky='e',
                                     padx=(5, 15))

        self.cache_limit_slider = ctk.CTkSlider(
            parent, from_=CACHE_MIN, to=CACHE_MAX,
            number_of_steps=(CACHE_MAX - CACHE_MIN) // 50,
            command=self._on_cache_limit_change,
        )
        self.cache_limit_slider.set(CACHE_DEFAULT)
        self.cache_limit_slider.grid(row=33, column=1,
                                      sticky='ew', padx=10)

        # ===== 缓存显示与操作 =====
        cache_row = ctk.CTkFrame(parent, fg_color='transparent', height=1)
        cache_row.grid(row=34, column=0, columnspan=3,
                       sticky='ew', padx=(15, 15), pady=(4, 12))
        cache_row.pack_propagate(True)

        self.cache_size_label = ctk.CTkLabel(
            cache_row, text="已占用：0.00 MB (0 条)",
            font=(FONT_NAME, 10), text_color='#3a86ff',
        )
        self.cache_size_label.pack(side='left')

        ctk.CTkButton(
            cache_row, text="打开缓存目录", width=100,
            font=(FONT_NAME, 10),
            fg_color='#17a2b8', hover_color='#138496',
            command=self._open_cache_dir,
        ).pack(side='right', padx=(6, 0))

        ctk.CTkButton(
            cache_row, text="清空缓存", width=80,
            font=(FONT_NAME, 10),
            fg_color='#dc3545', hover_color='#c82333',
            command=self._manual_clear_cache,
        ).pack(side='right')

        self._update_phrase_count()

        self._update_phrase_count()


class HotkeyPanelMixin:
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