"""延迟测试对话框"""

import threading
import customtkinter as ctk

from config import FONT_NAME
from logger import get_logger

log = get_logger('ui.latency')


class LatencyTestDialog(ctk.CTkToplevel):
    def __init__(self, master, translator=None, ocr_engine=None,
                 pipeline=None, app_state=None):
        super().__init__(master)
        self.title("延迟测试")
        self.geometry("920x700")
        self.attributes('-topmost', True)

        self.translator = translator
        self.ocr_engine = ocr_engine
        self.pipeline = pipeline
        self.app_state = app_state

        self._tester = None
        self._thread = None
        self._running = False
        self._was_running = False

        self._build_ui()

    def _build_ui(self):
        pad = 12

        # ============ 测试项 ============
        opt_frame = ctk.CTkFrame(self, height=1)
        opt_frame.pack(fill='x', padx=pad, pady=(pad, 6))
        opt_frame.pack_propagate(True)

        ctk.CTkLabel(
            opt_frame, text="测试项：",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left', padx=(6, 6))

        self.var_local = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opt_frame, text="本地阶段",
            variable=self.var_local,
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=4)

        self.var_ocr = ctk.BooleanVar(
            value=(self.ocr_engine is not None))
        ctk.CTkCheckBox(
            opt_frame, text="OCR",
            variable=self.var_ocr,
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=4)

        self.var_verify = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opt_frame, text="二次验证",
            variable=self.var_verify,
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=4)

        self.var_lm = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opt_frame, text="本地模型",
            variable=self.var_lm,
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=4)

        self.var_api = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opt_frame, text="API",
            variable=self.var_api,
            font=(FONT_NAME, 10),
        ).pack(side='left', padx=4)

        self.var_e2e = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opt_frame, text="★全链路",
            variable=self.var_e2e,
            font=(FONT_NAME, 10, 'bold'),
        ).pack(side='left', padx=4)

        # ⭐ 全链路冷启动选项
        self.var_e2e_clean = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opt_frame, text="冷启动(每轮清缓存)",
            variable=self.var_e2e_clean,
            font=(FONT_NAME, 9),
        ).pack(side='left', padx=4)

        # ============ 次数 ============
        opt2 = ctk.CTkFrame(self, height=1)
        opt2.pack(fill='x', padx=pad, pady=(0, 6))
        opt2.pack_propagate(True)

        ctk.CTkLabel(opt2, text="本地次数：",
                     font=(FONT_NAME, 10)).pack(side='left', padx=(6, 2))
        self.runs_entry = ctk.CTkEntry(opt2, width=55)
        self.runs_entry.insert(0, "20")
        self.runs_entry.pack(side='left', padx=2)

        ctk.CTkLabel(opt2, text="本地模型次数：",
                     font=(FONT_NAME, 10)).pack(side='left', padx=(12, 2))
        self.lm_runs_entry = ctk.CTkEntry(opt2, width=55)
        self.lm_runs_entry.insert(0, "5")
        self.lm_runs_entry.pack(side='left', padx=2)

        ctk.CTkLabel(opt2, text="API 次数：",
                     font=(FONT_NAME, 10)).pack(side='left', padx=(12, 2))
        self.api_runs_entry = ctk.CTkEntry(opt2, width=55)
        self.api_runs_entry.insert(0, "5")
        self.api_runs_entry.pack(side='left', padx=2)

        ctk.CTkLabel(opt2, text="全链路次数：",
                     font=(FONT_NAME, 10)).pack(side='left', padx=(12, 2))
        self.e2e_runs_entry = ctk.CTkEntry(opt2, width=55)
        self.e2e_runs_entry.insert(0, "5")
        self.e2e_runs_entry.pack(side='left', padx=2)

        # ============ 按钮 ============
        btn_frame = ctk.CTkFrame(self, height=1)
        btn_frame.pack(fill='x', padx=pad, pady=4)
        btn_frame.pack_propagate(True)

        self.btn_start = ctk.CTkButton(
            btn_frame, text="▶ 开始测试", width=110,
            command=self._on_start,
        )
        self.btn_start.pack(side='left', padx=3)

        self.btn_cancel = ctk.CTkButton(
            btn_frame, text="停止", width=80,
            fg_color='#dc3545', hover_color='#c82333',
            command=self._on_cancel,
            state='disabled',
        )
        self.btn_cancel.pack(side='left', padx=3)

        ctk.CTkButton(
            btn_frame, text="复制结果", width=90,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self._on_copy,
        ).pack(side='left', padx=3)

        # ============ 进度 ============
        self.progress_label = ctk.CTkLabel(
            self, text="就绪",
            font=(FONT_NAME, 10), anchor='w',
        )
        self.progress_label.pack(fill='x', padx=pad, pady=(2, 0))

        self.progress_bar = ctk.CTkProgressBar(self, height=10)
        self.progress_bar.pack(fill='x', padx=pad, pady=(2, 6))
        self.progress_bar.set(0)

        # ============ 日志 ============
        ctk.CTkLabel(
            self, text="日志：",
            font=(FONT_NAME, 10, 'bold'), anchor='w',
        ).pack(fill='x', padx=pad)

        self.log_text = ctk.CTkTextbox(
            self, height=90, wrap='word',
            font=(FONT_NAME, 10),
        )
        self.log_text.pack(fill='x', padx=pad, pady=(2, 6))

        # ============ 结果 ============
        ctk.CTkLabel(
            self, text="结果：",
            font=(FONT_NAME, 10, 'bold'), anchor='w',
        ).pack(fill='x', padx=pad)

        self.result_text = ctk.CTkTextbox(
            self, wrap='none',
            font=('Consolas', 10),
        )
        self.result_text.pack(fill='both', expand=True,
                              padx=pad, pady=(2, pad))

    # ================= 回调 =================

    def _on_start(self):
        if self._running:
            return

        try:
            runs = int(self.runs_entry.get())
            runs = max(1, min(1000, runs))
        except Exception:
            runs = 20

        try:
            api_runs = int(self.api_runs_entry.get())
            api_runs = max(1, min(100, api_runs))
        except Exception:
            api_runs = 5

        try:
            e2e_runs = int(self.e2e_runs_entry.get())
            e2e_runs = max(1, min(100, e2e_runs))
        except Exception:
            e2e_runs = 5

        try:
            local_runs = int(self.lm_runs_entry.get())
            local_runs = max(1, min(100, local_runs))
        except Exception:
            local_runs = 5

        do_ocr = self.var_ocr.get()
        do_verify = self.var_verify.get()
        do_api = self.var_api.get()
        do_local = self.var_lm.get()
        do_e2e = self.var_e2e.get()
        do_e2e_clean = self.var_e2e_clean.get()

        if do_api and self.translator is None:
            self._log("[错误] 未配置 translator")
            return
        if do_ocr and self.ocr_engine is None:
            self._log("[错误] 未配置 OCR 引擎")
            return
        if do_e2e and self.pipeline is None:
            self._log("[错误] 未配置 pipeline")
            return

        self._was_running = False
        try:
            if self.app_state is not None and self.app_state.running:
                self._was_running = True
                self.app_state.running = False
                self._log("[提示] 已自动暂停翻译")
        except Exception:
            pass

        self._running = True
        self.btn_start.configure(state='disabled', text='测试中...')
        self.btn_cancel.configure(state='normal')
        self.progress_bar.set(0)
        self.progress_label.configure(text="准备中...")
        self.log_text.delete('1.0', 'end')
        self.result_text.delete('1.0', 'end')

        self._log(f"[启动] runs={runs}, local_runs={local_runs}, "
                  f"api_runs={api_runs}, e2e_runs={e2e_runs}, "
                  f"e2e_clean={do_e2e_clean}")

        self._thread = threading.Thread(
            target=self._run_test,
            args=(runs, do_ocr, do_verify, do_api, api_runs,
                  do_local, local_runs, do_e2e, e2e_runs,
                  do_e2e_clean),
            daemon=True,
        )
        self._thread.start()

    def _on_cancel(self):
        if self._tester is not None:
            self._tester.cancel()
            self._log("[取消] 已发送停止信号")

    def _on_copy(self):
        try:
            text = self.result_text.get('1.0', 'end').strip()
            if not text:
                self._log("[复制] 结果为空")
                return
            self.clipboard_clear()
            self.clipboard_append(text)
            self._log("[复制] 结果已复制到剪贴板")
        except Exception as e:
            self._log(f"[复制] 失败：{e}")

    # ================= 线程内执行 =================

    def _run_test(self, runs, do_ocr, do_verify, do_api, api_runs,
                  do_local, local_runs, do_e2e, e2e_runs,
                  do_e2e_clean=False):
        from latency_tester import LatencyTester

        tester = LatencyTester(
            translator=self.translator,
            ocr_engine=self.ocr_engine if do_ocr else None,
            pipeline=self.pipeline if do_e2e else None,
            progress_cb=self._progress_cb,
            log_cb=self._log_cb,
        )
        self._tester = tester

        try:
            tester.run(
                runs=runs,
                warmup=2,
                do_ocr=do_ocr,
                do_api=do_api,
                api_runs=api_runs,
                do_local=do_local,
                local_runs=local_runs,
                do_verify=do_verify,
                do_e2e=do_e2e,
                e2e_runs=e2e_runs,
                e2e_clean_cache=do_e2e_clean,
            )
        except Exception as e:
            self.after(0, lambda: self._log(f"[异常] {e}"))

        report = tester.get_report_text()
        self.after(0, lambda: self._finish(report))

    def _progress_cb(self, cur, total):
        self.after(0, lambda: self._update_progress(cur, total))

    def _log_cb(self, msg):
        self.after(0, lambda: self._log(msg))

    def _update_progress(self, cur, total):
        try:
            self.progress_bar.set(cur / max(1, total))
            self.progress_label.configure(
                text=f"进度：{cur}/{total}")
        except Exception:
            pass

    def _log(self, msg):
        try:
            self.log_text.insert('end', msg + '\n')
            self.log_text.see('end')
        except Exception:
            pass

    def _finish(self, report):
        self._running = False
        try:
            self.btn_start.configure(state='normal', text='▶ 开始测试')
            self.btn_cancel.configure(state='disabled')
            self.progress_label.configure(text="完成")
            self.progress_bar.set(1.0)
        except Exception:
            pass

        try:
            self.result_text.delete('1.0', 'end')
            self.result_text.insert('end', report)
        except Exception:
            pass

        try:
            if self._was_running and self.app_state is not None:
                self.app_state.running = True
                self._log("[提示] 已恢复翻译运行")
        except Exception:
            pass

        log.info("延迟测试完成")