"""模型选择对话框：区分 CPU/GPU 变体，包含下载进度与实时速度"""

import customtkinter as ctk
from tkinter import messagebox
from config import FONT_NAME
from logger import get_logger
from ui.download_task import DownloadTask

log = get_logger('ui.model_picker')


def _format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{max(0, int(num_bytes))} B"

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
        self._download_task = None
        self._last_download_ok = False
        self._last_download_model_id = ''
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._refresh_list()

    def _on_close(self):
        if self._closing:
            return
        if self._downloading:
            if not messagebox.askyesno(
                    "确认", "正在下载中，强行关闭会中断下载，确定吗？",
                    parent=self):
                return
        self._closing = True
        self.destroy()

    def destroy(self):
        """关闭时停止事件轮询，后台线程不会再触碰已销毁控件。"""
        self._closing = True
        task = getattr(self, '_download_task', None)
        self._download_task = None
        if task is not None:
            try:
                task.close()
            except Exception:
                pass
        super().destroy()

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
        if self._closing or self._downloading:
            return
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

        progress_text = ctk.CTkLabel(
            status_frame, text="", font=(FONT_NAME, 9),
            text_color='#444444', width=150, anchor='w')
        
        speed_label = ctk.CTkLabel(
            status_frame, text="", font=(FONT_NAME, 9),
            text_color='#3a86ff', width=78, anchor='w')

        # CPU 版按钮 (int8)
        btn_cpu = ctk.CTkButton(
            right, text=(f"下载 CPU版 (int8)\n({quant_mb:.0f}MB)" if not has_cpu else "✓ int8版已就绪"),
            width=120, height=44, font=(FONT_NAME, 9, 'bold'),
            fg_color='#16a085' if not has_cpu else 'gray50', hover_color='#138d75',
            state='normal' if not has_cpu else 'disabled',
            command=lambda: self._start_download(
                mid, name, 'cpu', btn_cpu, progress_bar, progress_text,
                speed_label, status_label)
        )
        btn_cpu.pack(side='left', padx=3)

        # GPU 版按钮 (FP32)
        btn_gpu = ctk.CTkButton(
            right, text=(f"下载 GPU版 (FP32)\n({fp32_mb:.0f}MB)" if not has_gpu else "✓ FP32版已就绪"),
            width=130, height=44, font=(FONT_NAME, 9, 'bold'),
            fg_color='#e67e22' if not has_gpu else 'gray50', hover_color='#d35400',
            state='normal' if not has_gpu else 'disabled',
            command=lambda: self._start_download(
                mid, name, 'gpu', btn_gpu, progress_bar, progress_text,
                speed_label, status_label)
        )
        btn_gpu.pack(side='left', padx=3)

    def _start_download(self, model_id, name, variant, btn, progress_bar,
                        progress_text, speed_label, status_label):
        if self._closing:
            return
        if self._downloading:
            messagebox.showwarning("提示", "已有下载任务正在进行中", parent=self)
            return

        try:
            from translation.local_models import ModelDownloader
            downloader = ModelDownloader()
        except Exception as e:
            status_label.configure(
                text=f"初始化下载器失败：{e}", text_color='#cc0000')
            return

        original_text = btn.cget('text')
        self._downloading = True
        self._last_download_model_id = model_id
        self._downloader = downloader
        btn.configure(state='disabled', text="下载中...")
        progress_bar.pack(side='left', padx=(10, 0))
        progress_text.pack(side='left', padx=(4, 0))
        speed_label.pack(side='left', padx=(4, 0))
        status_label.configure(text="准备连接...", text_color='#3a86ff')

        def work(progress_cb, status_cb):
            # 运行在下载线程：只访问 downloader 和普通参数，不能碰 Tk。
            return downloader.download(
                model_id, variant,
                progress_cb=progress_cb, status_cb=status_cb)

        task = DownloadTask(
            self,
            on_progress=lambda p, speed_mb=0.0, done_bytes=0, total_bytes=0:
                self._update_download_progress(
                    progress_bar, progress_text, speed_label,
                    p, speed_mb, done_bytes, total_bytes),
            on_status=lambda text: self._update_download_status(
                status_label, text),
            on_done=lambda ok, result, error, cancelled:
                self._on_download_finish(
                    ok, error, cancelled, status_label),
        )
        self._download_task = task

        if not task.start(work, cancel_cb=downloader.cancel):
            self._downloading = False
            self._download_task = None
            self._downloader = None
            try:
                btn.configure(state='normal', text=original_text)
                status_label.configure(
                    text="无法启动下载线程，请查看日志",
                    text_color='#cc0000')
            except Exception:
                pass

    def _update_download_progress(self, progress_bar, progress_text,
                                  speed_label, p, speed_mb=0.0,
                                  done_bytes=0, total_bytes=0):
        if self._closing:
            return
        try:
            progress_bar.set(p)
            if total_bytes > 0:
                progress_text.configure(
                    text=f"{p * 100:5.1f}% · "
                         f"{_format_bytes(done_bytes)} / "
                         f"{_format_bytes(total_bytes)}")
            else:
                progress_text.configure(text=f"{p * 100:5.1f}%")
            if p >= 1.0:
                speed_label.configure(text="完成")
            elif speed_mb >= 0.05:
                speed_label.configure(text=f"{speed_mb:.1f} MB/s")
            else:
                speed_label.configure(text="连接中...")
        except Exception:
            pass

    def _update_download_status(self, status_label, text):
        if self._closing:
            return
        try:
            color = ('#cc0000' if '❌' in text or '失败' in text
                     else ('#28a745' if '✅' in text else '#3a86ff'))
            status_label.configure(text=text, text_color=color)
        except Exception:
            pass

    def _on_download_finish(self, ok=False, error=None, cancelled=False,
                            status_label=None):
        if self._closing:
            return

        self._downloading = False
        self._last_download_ok = bool(ok)
        self._download_task = None
        self._downloader = None

        if status_label is not None:
            try:
                if cancelled:
                    status_label.configure(text="已取消", text_color='#888888')
                elif error:
                    status_label.configure(
                        text=f"❌ 下载异常：{error[:120]}",
                        text_color='#cc0000')
            except Exception:
                pass

        self._update_cache_label()
        if self._last_download_ok:
            self._sync_and_refresh_master(
                getattr(self, '_last_download_model_id', ''))

        if not self._closing:
            delay = 1200 if self._last_download_ok else 2500
            self.after(delay, self._refresh_list)

    def _sync_and_refresh_master(self, model_id=''):
        """下载成功后同步 Provider，并让主窗口重新扫描模型目录。"""
        try:
            from translation.api_config import sync_downloaded_models
            result = sync_downloaded_models()
            added = result.get('added', [])
            if added:
                log.info(f"[模型选择] 自动注册本地模型：{added}")
        except Exception as e:
            log.exception(f"[模型选择] 自动同步失败：{e}")

        try:
            master = self.master
            if (master is not None
                    and master.winfo_exists()
                    and hasattr(master, '_refresh_local_models')):
                if (model_id
                        and hasattr(master,
                                    '_select_downloaded_local_model')):
                    master._select_downloaded_local_model(model_id)
                else:
                    try:
                        master.translator.reload_config()
                    except Exception:
                        pass
                    master.after(0, master._refresh_local_models)
        except Exception:
            pass

    def _update_cache_label(self):
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            self.cache_label.configure(text=f"磁盘占用：{mgr.get_cache_size_mb():.0f} MB")
        except Exception: pass