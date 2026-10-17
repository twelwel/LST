"""模型选择对话框：区分 CPU/GPU 变体，包含下载进度与实时速度"""

import threading
import customtkinter as ctk
from tkinter import messagebox
from config import FONT_NAME
from logger import get_logger

log = get_logger('ui.model_picker')

class ModelPickerDialog(ctk.CTkToplevel):
    def __init__(self, master, on_download=None):
        super().__init__(master)
        self.title("模型管理中心")
        self.geometry("920x680")
        self.attributes('-topmost', True)
        self.on_download = on_download
        self._closing = False
        self._downloading = False
        self._downloader = None
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._refresh_list()

    def _on_close(self):
        if self._downloading:
            if not messagebox.askyesno("确认", "正在下载中，强行关闭会中断下载，确定吗？", parent=self):
                return
            if self._downloader: self._downloader.cancel()
        self._closing = True
        self.destroy()

    def _build_ui(self):
        pad = 14
        top = ctk.CTkFrame(self, fg_color='transparent')
        top.pack(fill='x', padx=pad, pady=(pad, 6))

        ctk.CTkLabel(top, text="💡 int8/q4 版体积小，但【仅限 CPU】运行；FP32 版体积大，【支持 GPU 加速】。", 
                     font=(FONT_NAME, 11, 'bold'), text_color='#3a86ff').pack(side='left', padx=8)

        self.cache_label = ctk.CTkLabel(top, text="", font=(FONT_NAME, 10), text_color='#666666')
        self.cache_label.pack(side='right')

        self.list_frame = ctk.CTkScrollableFrame(self, fg_color='transparent')
        self.list_frame.pack(fill='both', expand=True, padx=pad, pady=4)

        ctk.CTkButton(self, text="关闭", width=100, fg_color='#6c757d', 
                      command=self._on_close).pack(pady=(4, pad))

    def _refresh_list(self):
        if self._closing: return
        for w in self.list_frame.winfo_children(): w.destroy()
        
        try:
            from translation.local_models import get_available_models, validate_model_files
            models = get_available_models()
        except Exception as e:
            log.exception(f"[模型选择] 读取失败：{e}")
            ctk.CTkLabel(self.list_frame, text=f"加载失败：{e}", text_color='#cc0000').pack(pady=20)
            return

        builtin = [m for m in models if not m.get('is_custom')]

        for m in builtin:
            try:
                has_cpu, _ = validate_model_files(m['id'], 'cpu')
                has_gpu, _ = validate_model_files(m['id'], 'gpu')
            except Exception:
                has_cpu = has_gpu = False
            
            self._render_model(m, has_cpu, has_gpu)

        self._update_cache_label()

    def _render_model(self, meta: dict, has_cpu: bool, has_gpu: bool):
        mid = meta['id']
        name = meta.get('name', mid)
        langs = ', '.join(f"{s}→{t}" for s, t in meta.get('languages', []))
        desc = meta.get('description', '')
        recommended = meta.get('recommended', False)
        quant_mb = meta.get('download_mb') or meta.get('size_mb') or 0
        fp32_mb = meta.get('fp32_mb') or 0

        card = ctk.CTkFrame(self.list_frame, fg_color=('gray90', 'gray18') if recommended else ('gray95', 'gray14'))
        card.pack(fill='x', padx=4, pady=4)

        left = ctk.CTkFrame(card, fg_color='transparent')
        left.pack(side='left', fill='both', expand=True, padx=12, pady=10)

        row1 = ctk.CTkFrame(left, fg_color='transparent')
        row1.pack(anchor='w', fill='x')
        ctk.CTkLabel(row1, text=name, font=(FONT_NAME, 12, 'bold')).pack(side='left')
        if recommended:
            ctk.CTkLabel(row1, text=" [推荐]", font=(FONT_NAME, 10, 'bold'), text_color='#28a745').pack(side='left')

        ctk.CTkLabel(left, text=f"🌐 {langs}  |  💡 {desc}", font=(FONT_NAME, 9), text_color='#666666').pack(anchor='w', pady=(2, 0))

        right = ctk.CTkFrame(card, fg_color='transparent')
        right.pack(side='right', padx=12, pady=10)

        status_frame = ctk.CTkFrame(left, fg_color='transparent', height=24)
        status_frame.pack(fill='x', pady=(5, 0))
        
        status_label = ctk.CTkLabel(status_frame, text="", font=(FONT_NAME, 9), text_color='#cc0000')
        status_label.pack(side='left')
        
        progress_bar = ctk.CTkProgressBar(status_frame, width=160, height=8)
        progress_bar.set(0)
        
        speed_label = ctk.CTkLabel(status_frame, text="", font=(FONT_NAME, 9), text_color='#3a86ff', width=70)

        # CPU 版按钮 (int8)
        btn_cpu = ctk.CTkButton(
            right, text=(f"下载 CPU版 (int8)\n({quant_mb:.0f}MB)" if not has_cpu else "✓ int8版已就绪"),
            width=120, height=44, font=(FONT_NAME, 9, 'bold'),
            fg_color='#16a085' if not has_cpu else 'gray50', hover_color='#138d75',
            state='normal' if not has_cpu else 'disabled',
            command=lambda: self._start_download(mid, name, 'cpu', btn_cpu, progress_bar, speed_label, status_label)
        )
        btn_cpu.pack(side='left', padx=3)

        # GPU 版按钮 (FP32)
        btn_gpu = ctk.CTkButton(
            right, text=(f"下载 GPU版 (FP32)\n({fp32_mb:.0f}MB)" if not has_gpu else "✓ FP32版已就绪"),
            width=130, height=44, font=(FONT_NAME, 9, 'bold'),
            fg_color='#e67e22' if not has_gpu else 'gray50', hover_color='#d35400',
            state='normal' if not has_gpu else 'disabled',
            command=lambda: self._start_download(mid, name, 'gpu', btn_gpu, progress_bar, speed_label, status_label)
        )
        btn_gpu.pack(side='left', padx=3)

    def _start_download(self, model_id, name, variant, btn, progress_bar, speed_label, status_label):
        if self._downloading:
            messagebox.showwarning("提示", "已有下载任务正在进行中", parent=self)
            return

        try:
            from translation.local_models import ModelDownloader
            self._downloader = ModelDownloader()
        except Exception as e:
            status_label.configure(text=f"初始化下载器失败：{e}")
            return

        self._downloading = True
        btn.configure(state='disabled', text="下载中...")
        progress_bar.pack(side='left', padx=(10, 0))
        speed_label.pack(side='left', padx=(6, 0))
        status_label.configure(text="准备连接...", text_color='#3a86ff')

        def progress_cb(p: float, speed_mb: float):
            if self._closing: return
            def update():
                progress_bar.set(p)
                if speed_mb > 0.01: speed_label.configure(text=f"{speed_mb:.1f} MB/s")
                elif p >= 1.0: speed_label.configure(text="")
            self.after(0, update)

        def status_cb(s: str):
            if self._closing: return
            color = '#cc0000' if '❌' in s or '失败' in s else ('#28a745' if '✅' in s else '#3a86ff')
            self.after(0, lambda: status_label.configure(text=s, text_color=color))

        def worker():
            try:
                self._downloader.download(model_id, variant, progress_cb, status_cb)
            except Exception as e:
                log.exception(f"[UI下载] 异常：{e}")
            finally:
                if not self._closing:
                    self.after(0, self._on_download_finish)

        threading.Thread(target=worker, daemon=True).start()

    def _on_download_finish(self):
        self._downloading = False
        self._update_cache_label()
        if not self._closing:
            self.after(1500, self._refresh_list)

    def _update_cache_label(self):
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            self.cache_label.configure(text=f"磁盘占用：{mgr.get_cache_size_mb():.0f} MB")
        except Exception: pass