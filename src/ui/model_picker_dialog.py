"""模型选择对话框：点击下载后弹出，列出所有可下载模型

⭐ 新手保护：
- 每个模型显示两个下载按钮：
    · 轻量版（int8，约 40MB）→ 只能 CPU 运行
    · GPU 版（FP32，约 800MB）→ 支持显卡加速
- 轻量版体积小、CPU 快，但无法用 DirectML 加速
- GPU 版体积大，但能避免量化模型在 DML 上的算子崩溃
"""

import customtkinter as ctk

from config import FONT_NAME
from logger import get_logger


log = get_logger('ui.model_picker')


class ModelPickerDialog(ctk.CTkToplevel):
    """下载模型选择对话框"""

    def __init__(self, master, on_download=None):
        super().__init__(master)
        self.title("选择要下载的模型")
        self.geometry("900x640")
        self.attributes('-topmost', True)

        self.on_download = on_download
        self._closing = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_list()

    def _on_close(self):
        self._closing = True
        try:
            self.destroy()
        except Exception:
            pass

    def _safe_after(self, delay, fn):
        if self._closing:
            return
        try:
            if self.winfo_exists():
                self.after(delay, fn)
        except Exception:
            pass

    def _build_ui(self):
        pad = 14

        # 顶部提示
        top = ctk.CTkFrame(self, fg_color='transparent')
        top.pack(fill='x', padx=pad, pady=(pad, 6))

        ctk.CTkLabel(
            top,
            text="💡 轻量版体积小、CPU 运行；GPU 版体积大，支持显卡加速",
            font=(FONT_NAME, 10, 'bold'),
            text_color='#3a86ff',
        ).pack(side='left', padx=8)

        self.cache_label = ctk.CTkLabel(
            top, text="", font=(FONT_NAME, 9),
            text_color='#666666',
        )
        self.cache_label.pack(side='right')

        self.list_frame = ctk.CTkScrollableFrame(
            self, fg_color='transparent',
        )
        self.list_frame.pack(fill='both', expand=True,
                              padx=pad, pady=4)

        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(fill='x', padx=pad, pady=(4, pad))

        ctk.CTkButton(
            btn_frame, text="关闭", width=80,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self._on_close,
        ).pack(side='right', padx=3)

    # ==================== 列表 ====================

    def _refresh_list(self):
        for w in self.list_frame.winfo_children():
            w.destroy()

        try:
            from translation.local_models import get_available_models
            models = get_available_models()
        except Exception as e:
            log.exception(f"[模型选择] 读取失败：{e}")
            ctk.CTkLabel(
                self.list_frame,
                text=f"加载失败：{e}",
                text_color='#cc0000',
            ).pack(pady=20)
            return

        builtin = [m for m in models if not m.get('is_custom')]

        # 分区
        not_downloaded = []
        quant_only = []
        fp32_ready = []

        for m in builtin:
            try:
                from translation.local_models import (
                    is_quantized_downloaded, is_fp32_downloaded)
                has_q = is_quantized_downloaded(m['id'])
                has_f = is_fp32_downloaded(m['id'])
            except Exception:
                has_q = has_f = False

            if not has_q and not has_f:
                not_downloaded.append(m)
            elif has_q and not has_f:
                quant_only.append(m)
            else:
                fp32_ready.append(m)

        if not_downloaded:
            ctk.CTkLabel(
                self.list_frame,
                text=f"—— 未下载（{len(not_downloaded)} 个）——",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#28a745',
            ).pack(pady=(6, 4))
            for m in not_downloaded:
                self._render_model(m, has_quant=False, has_fp32=False)

        if quant_only:
            ctk.CTkLabel(
                self.list_frame,
                text=f"—— 已下载轻量版（{len(quant_only)} 个）——",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#3a86ff',
            ).pack(pady=(12, 4))
            for m in quant_only:
                self._render_model(m, has_quant=True, has_fp32=False)

        if fp32_ready:
            ctk.CTkLabel(
                self.list_frame,
                text=f"—— 已下载 GPU 版（{len(fp32_ready)} 个）——",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#e67e22',
            ).pack(pady=(12, 4))
            for m in fp32_ready:
                self._render_model(m, has_quant=True, has_fp32=True)

        self._update_cache_label()

    def _render_model(self, meta: dict,
                       has_quant: bool, has_fp32: bool):
        mid = meta['id']
        name = meta.get('name', mid)
        size_mb = meta.get('size_mb', 0)
        langs = meta.get('languages', [])
        desc = meta.get('description', '')
        recommended = meta.get('recommended', False)

        # ⭐ 用注册表里的真实体积，不再硬编码 40MB / 800MB
        quant_mb = meta.get('download_mb') or size_mb or 0
        fp32_mb = meta.get('fp32_mb') or 0

        card = ctk.CTkFrame(
            self.list_frame,
            fg_color=('gray90', 'gray18') if recommended
            else ('gray95', 'gray14'),
        )
        card.pack(fill='x', padx=4, pady=4)

        left = ctk.CTkFrame(card, fg_color='transparent')
        left.pack(side='left', fill='both', expand=True,
                  padx=12, pady=10)

        row1 = ctk.CTkFrame(left, fg_color='transparent')
        row1.pack(anchor='w', fill='x')

        ctk.CTkLabel(
            row1, text=name,
            font=(FONT_NAME, 12, 'bold'), anchor='w',
        ).pack(side='left')

        if recommended:
            ctk.CTkLabel(
                row1, text="  [推荐]",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#28a745',
            ).pack(side='left')

        if langs:
            lang_str = ', '.join(f"{s}→{t}" for s, t in langs)
            ctk.CTkLabel(
                left, text=f"🌐 {lang_str}",
                font=(FONT_NAME, 10), text_color='#666666',
                anchor='w',
            ).pack(anchor='w', pady=(2, 0))

        info_parts = []
        if size_mb > 0:
            info_parts.append(f"📦 {size_mb:.0f} MB")
        if desc:
            info_parts.append(f"💡 {desc}")
        if info_parts:
            ctk.CTkLabel(
                left, text="  |  ".join(info_parts),
                font=(FONT_NAME, 9), text_color='#888888',
                anchor='w',
            ).pack(anchor='w', pady=(2, 0))

        # 已下载标签
        tags = []
        if has_quant:
            tags.append("✓ 轻量版")
        if has_fp32:
            tags.append("✓ GPU 版")
        if tags:
            ctk.CTkLabel(
                left, text='  '.join(tags),
                font=(FONT_NAME, 9, 'bold'),
                text_color='#28a745',
                anchor='w',
            ).pack(anchor='w', pady=(2, 0))

        # 右侧按钮
        right = ctk.CTkFrame(card, fg_color='transparent')
        right.pack(side='right', padx=12, pady=10)

        # 轻量版
        if not has_quant:
            ctk.CTkButton(
                right,
                text=(f"下载轻量版\n({quant_mb:.0f}MB, CPU)"
                      if quant_mb else "下载轻量版\n(CPU)"),
                width=110, height=44,
                font=(FONT_NAME, 9, 'bold'),
                fg_color='#16a085', hover_color='#138d75',
                command=lambda: self._on_download_clicked(
                    mid, name, 'quantized'),
            ).pack(side='left', padx=3)
        else:
            ctk.CTkLabel(
                right, text="✓ 轻量版\n已下载",
                font=(FONT_NAME, 9, 'bold'),
                text_color='#28a745', justify='center',
            ).pack(side='left', padx=3)

        # GPU 版
        if not has_fp32:
            ctk.CTkButton(
                right,
                text=(f"下载 GPU 版\n({fp32_mb:.0f}MB, 显卡)"
                      if fp32_mb else "下载 GPU 版\n(显卡)"),
                width=120, height=44,
                font=(FONT_NAME, 9, 'bold'),
                fg_color='#e67e22', hover_color='#d35400',
                command=lambda: self._on_download_clicked(
                    mid, name, 'fp32'),
            ).pack(side='left', padx=3)
        else:
            ctk.CTkLabel(
                right, text="✓ GPU 版\n已下载",
                font=(FONT_NAME, 9, 'bold'),
                text_color='#28a745', justify='center',
            ).pack(side='left', padx=3)

    def _update_cache_label(self):
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            size_mb = mgr.get_cache_size_mb()
            count = len(mgr.list_models())
            self.cache_label.configure(
                text=f"缓存：{count} 个 · {size_mb:.0f} MB")
        except Exception:
            pass

    def _on_download_clicked(self, model_id: str, name: str,
                              variant: str):
        if self._closing:
            return

        try:
            from translation.local_models.downloader import (
                is_downloading, get_downloading_model)
            if is_downloading():
                other = get_downloading_model()
                self._show_message(f"⚠️ 已有任务进行中：{other}")
                return
        except Exception:
            pass

        if self.on_download:
            try:
                self.on_download(model_id, variant)
            except Exception as e:
                log.exception(f"[模型选择] 触发下载失败：{e}")

        self._on_close()

    def _show_message(self, msg: str):
        try:
            from tkinter import messagebox
            messagebox.showinfo("提示", msg, parent=self)
        except Exception:
            pass