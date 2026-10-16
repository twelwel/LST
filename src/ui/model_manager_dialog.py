"""本地翻译模型管理对话框（安全版）

关键安全措施：
- 下载期间禁用所有操作按钮
- 关闭窗口时自动取消下载
- 所有 after 回调检查窗口存在性
- 全局下载锁，防止多窗口并发
"""

import os
import shutil
import threading
from pathlib import Path

import customtkinter as ctk

from config import FONT_NAME
from logger import get_logger


log = get_logger('ui.model_manager')


class ModelManagerDialog(ctk.CTkToplevel):

    def __init__(self, master):
        super().__init__(master)
        self.title("本地翻译模型管理")
        self.geometry("900x700")
        self.attributes('-topmost', True)

        self._downloader = None
        self._download_thread = None
        self._downloading = False
        self._closing = False
        self._current_download_id = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_list()

    # ==================== 生命周期 ====================

    def _safe_after(self, delay, fn):
        if self._closing:
            return
        try:
            if self.winfo_exists():
                self.after(delay, fn)
        except Exception:
            pass

    def _on_close(self):
        if self._closing:
            return
        self._closing = True

        if self._downloading and self._downloader:
            try:
                self._downloader.cancel()
            except Exception:
                pass

        if self._download_thread and self._download_thread.is_alive():
            try:
                self._download_thread.join(timeout=2.0)
            except Exception:
                pass

        try:
            self.destroy()
        except Exception:
            pass

    def _set_progress(self, text: str):
        if self._closing:
            return
        try:
            self.progress_label.configure(text=text)
        except Exception:
            pass

    # ==================== UI ====================

    def _build_ui(self):
        pad = 14

        top = ctk.CTkFrame(self, fg_color='transparent')
        top.pack(fill='x', padx=pad, pady=(pad, 6))

        ctk.CTkLabel(
            top,
            text="下载或导入本地翻译模型，离线使用无 API 费用",
            font=(FONT_NAME, 11, 'bold'),
        ).pack(side='left')

        self.cache_label = ctk.CTkLabel(
            top, text="", font=(FONT_NAME, 10),
            text_color='#666666',
        )
        self.cache_label.pack(side='right')

        # 提示行
        ctk.CTkLabel(
            self,
            text="💡 标准版（int8）体积小、CPU 快；"
                 "GPU 版（FP32）体积大，但可避免 DirectML 量化算子崩溃",
            font=(FONT_NAME, 9), text_color='#3a86ff',
            anchor='w',
        ).pack(fill='x', padx=pad, pady=(0, 4))

        self.list_frame = ctk.CTkScrollableFrame(
            self, fg_color='transparent',
        )
        self.list_frame.pack(fill='both', expand=True,
                              padx=pad, pady=4)

        self.progress_label = ctk.CTkLabel(
            self, text="", font=(FONT_NAME, 10),
            text_color='gray', anchor='w',
        )
        self.progress_label.pack(fill='x', padx=pad, pady=(4, 0))

        self.progress_bar = ctk.CTkProgressBar(self, height=10)
        self.progress_bar.pack(fill='x', padx=pad, pady=(2, 4))
        self.progress_bar.set(0)

        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(fill='x', padx=pad, pady=(4, pad))

        self.btn_refresh = ctk.CTkButton(
            btn_frame, text="刷新", width=80,
            command=self._refresh_list,
        )
        self.btn_refresh.pack(side='left', padx=3)

        self.btn_import = ctk.CTkButton(
            btn_frame, text="导入自定义模型", width=140,
            fg_color='#6f42c1', hover_color='#5a35a0',
            command=self._import_custom,
        )
        self.btn_import.pack(side='left', padx=3)

        self.btn_open_dir = ctk.CTkButton(
            btn_frame, text="打开模型目录", width=120,
            fg_color='#17a2b8', hover_color='#138496',
            command=self._open_models_dir,
        )
        self.btn_open_dir.pack(side='left', padx=3)

        self.btn_clear = ctk.CTkButton(
            btn_frame, text="清理缓存", width=90,
            fg_color='#dc3545', hover_color='#c82333',
            command=self._clear_cache,
        )
        self.btn_clear.pack(side='left', padx=3)

        self.btn_close = ctk.CTkButton(
            btn_frame, text="关闭", width=70,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self._on_close,
        )
        self.btn_close.pack(side='right', padx=3)

    def _set_ui_downloading(self, downloading: bool):
        if self._closing:
            return

        state = 'disabled' if downloading else 'normal'
        for btn in (self.btn_refresh, self.btn_import, self.btn_clear):
            try:
                btn.configure(state=state)
            except Exception:
                pass

        try:
            if downloading:
                self.btn_close.configure(
                    text="取消下载",
                    fg_color='#dc3545', hover_color='#c82333',
                    command=self._cancel_download,
                )
            else:
                self.btn_close.configure(
                    text="关闭",
                    fg_color='#6c757d', hover_color='#5a6268',
                    command=self._on_close,
                )
        except Exception:
            pass

    # ==================== 列表 ====================

    def _refresh_list(self):
        if self._closing or self._downloading:
            return

        for w in self.list_frame.winfo_children():
            w.destroy()

        try:
            from translation.local_models import get_available_models
            models = get_available_models()
        except Exception as e:
            log.exception(f"[模型管理] 读取失败：{e}")
            ctk.CTkLabel(
                self.list_frame,
                text=f"加载失败：{e}",
                text_color='#cc0000',
            ).pack(pady=20)
            return

        builtin = [m for m in models if not m.get('is_custom')]
        custom = [m for m in models if m.get('is_custom')]

        if builtin:
            ctk.CTkLabel(
                self.list_frame,
                text=f"—— 内置模型（{len(builtin)} 个）——",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#666666',
            ).pack(pady=(6, 4))
            for m in builtin:
                self._render_model_card(m)

        if custom:
            ctk.CTkLabel(
                self.list_frame,
                text=f"—— 自定义模型（{len(custom)} 个）——",
                font=(FONT_NAME, 10, 'bold'),
                text_color='#666666',
            ).pack(pady=(12, 4))
            for m in custom:
                self._render_model_card(m)

        self._update_cache_label()

    def _render_model_card(self, meta: dict):
        mid = meta['id']
        is_custom = meta.get('is_custom', False)

        if is_custom:
            has_quant = True
            has_fp32 = False
        else:
            try:
                from translation.local_models import (
                    is_quantized_downloaded, is_fp32_downloaded)
                has_quant = is_quantized_downloaded(mid)
                has_fp32 = is_fp32_downloaded(mid)
            except Exception:
                has_quant = False
                has_fp32 = False

        card = ctk.CTkFrame(self.list_frame)
        card.pack(fill='x', padx=4, pady=4)

        left = ctk.CTkFrame(card, fg_color='transparent')
        left.pack(side='left', fill='both', expand=True,
                  padx=10, pady=8)

        name_text = meta['name']
        if meta.get('recommended'):
            name_text += '  [推荐]'

        ctk.CTkLabel(
            left, text=name_text,
            font=(FONT_NAME, 12, 'bold'), anchor='w',
        ).pack(anchor='w')

        langs = meta.get('languages', [])
        if langs:
            lang_str = ', '.join(f"{s}→{t}" for s, t in langs)
            ctk.CTkLabel(
                left, text=f"支持：{lang_str}",
                font=(FONT_NAME, 10), text_color='#666666',
                anchor='w',
            ).pack(anchor='w')

        size = meta.get('size_mb', 0)
        desc = meta.get('description', '')
        info_text = f"{size:.0f} MB" if size > 0 else "自定义"
        if desc:
            info_text += f" · {desc}"

        ctk.CTkLabel(
            left, text=info_text,
            font=(FONT_NAME, 9), text_color='#888888',
            anchor='w',
        ).pack(anchor='w')

        # 状态标签
        tags = []
        if has_quant:
            tags.append("✓ int8")
        if has_fp32:
            tags.append("✓ FP32(GPU)")
        if tags:
            ctk.CTkLabel(
                left, text=' '.join(tags),
                font=(FONT_NAME, 9, 'bold'),
                text_color='#28a745', anchor='w',
            ).pack(anchor='w', pady=(2, 0))

        # 右侧按钮
        right = ctk.CTkFrame(card, fg_color='transparent')
        right.pack(side='right', padx=10, pady=8)

        if is_custom:
            ctk.CTkButton(
                right, text="移除注册", width=80,
                fg_color='#dc3545', hover_color='#c82333',
                command=lambda: self._remove_custom(mid),
            ).pack(side='left', padx=2)
            return

        # 内置模型：下载/删除按钮
        # 标准版下载
        if not has_quant:
            ctk.CTkButton(
                right, text="下载标准版", width=90,
                fg_color='#16a085', hover_color='#138d75',
                command=lambda: self._on_download_btn(
                    mid, 'quantized'),
            ).pack(side='left', padx=2)

        # GPU 版下载
        if not has_fp32:
            ctk.CTkButton(
                right, text="下载 GPU 版", width=100,
                fg_color='#e67e22', hover_color='#d35400',
                command=lambda: self._on_download_btn(
                    mid, 'fp32'),
            ).pack(side='left', padx=2)

        if has_quant or has_fp32:
            ctk.CTkButton(
                right, text="删除", width=60,
                fg_color='#dc3545', hover_color='#c82333',
                command=lambda: self._delete_model(mid),
            ).pack(side='left', padx=2)

    def _update_cache_label(self):
        try:
            from translation.local_models import (
                ModelCacheManager, get_models_dir)
            mgr = ModelCacheManager()
            size_mb = mgr.get_cache_size_mb()
            count = len(mgr.list_models())
            path = str(get_models_dir())
            if len(path) > 55:
                path = '...' + path[-52:]
            self.cache_label.configure(
                text=f"缓存：{count} 个 · {size_mb:.0f} MB · {path}")
        except Exception:
            pass

    # ==================== 下载 ====================

    def _on_download_btn(self, model_id: str, variant: str = 'quantized'):
        if self._downloading:
            self._set_progress("⚠️ 已有下载任务进行中")
            return

        try:
            from translation.local_models.downloader import (
                is_downloading, get_downloading_model)
            if is_downloading():
                other = get_downloading_model()
                self._set_progress(
                    f"⚠️ 其他窗口正在下载：{other}")
                return
        except Exception:
            pass

        self._start_download(model_id, variant)

    def _start_download(self, model_id: str, variant: str = 'quantized'):
        try:
            from translation.local_models import ModelDownloader
        except Exception as e:
            self._set_progress(f"下载器加载失败：{e}")
            return

        self._downloading = True
        self._current_download_id = model_id
        self._downloader = ModelDownloader()
        self.progress_bar.set(0)

        self._show_downloading_view(model_id, variant)
        self._set_ui_downloading(True)

        def progress_cb(p):
            self._safe_after(0, lambda: self._update_progress(p))

        def status_cb(s):
            self._safe_after(0, lambda: self._set_progress(s))

        def worker():
            try:
                self._downloader.download(
                    model_id, variant=variant,
                    progress_cb=progress_cb, status_cb=status_cb)
            except Exception as e:
                log.exception(f"[模型管理] 下载异常：{e}")
            self._safe_after(0, self._on_download_done)

        self._download_thread = threading.Thread(
            target=worker, daemon=True)
        self._download_thread.start()

    def _update_progress(self, p):
        if self._closing:
            return
        try:
            self.progress_bar.set(p)
        except Exception:
            pass

    def _show_downloading_view(self, model_id: str, variant: str):
        if self._closing:
            return
        for w in self.list_frame.winfo_children():
            w.destroy()

        # ⭐ 用注册表里的真实体积，不再硬编码 40MB / 800MB
        try:
            from translation.local_models.registry import (
                get_registry_entry)
            _meta = get_registry_entry(model_id) or {}
        except Exception:
            _meta = {}

        if variant == 'fp32':
            _mb = _meta.get('fp32_mb') or 0
            variant_label = (f'GPU 版（FP32，约 {_mb:.0f}MB）'
                             if _mb else 'GPU 版（FP32）')
        else:
            _mb = (_meta.get('download_mb')
                   or _meta.get('size_mb') or 0)
            variant_label = (f'标准版（int8，约 {_mb:.0f}MB）'
                             if _mb else '标准版（int8）')

        ctk.CTkLabel(
            self.list_frame,
            text="⏳ 正在下载...",
            font=(FONT_NAME, 16, 'bold'),
        ).pack(pady=(60, 10))

        ctk.CTkLabel(
            self.list_frame,
            text=f"{model_id}  ·  {variant_label}",
            font=(FONT_NAME, 12),
            text_color='#666666',
        ).pack(pady=4)

        ctk.CTkLabel(
            self.list_frame,
            text="下载期间请勿关闭窗口\n"
                 "如需中断，点右下角「取消下载」",
            font=(FONT_NAME, 10),
            text_color='#888888',
            justify='center',
        ).pack(pady=(20, 0))

    def _cancel_download(self):
        if not self._downloading:
            return

        if self._downloader:
            try:
                self._downloader.cancel()
            except Exception:
                pass

        self._set_progress("正在取消下载（后台会清理临时文件）...")
        self._downloading = False
        self._set_ui_downloading(False)
        self._safe_after(300, self._refresh_list)

    def _on_download_done(self):
        if self._closing:
            return
        if not self._downloading:
            return

        self._downloading = False
        self._current_download_id = None
        try:
            self.progress_bar.set(1.0)
        except Exception:
            pass
        self._set_ui_downloading(False)
        self._safe_after(300, self._refresh_list)

        try:
            master = self.master
            if master and hasattr(master, '_refresh_local_models'):
                self._safe_after(500, master._refresh_local_models)
        except Exception:
            pass

    # ==================== 删除 / 清理 ====================

    def _delete_model(self, model_id: str):
        if self._downloading:
            self._set_progress("⚠️ 下载中，无法删除")
            return

        from tkinter import messagebox
        if not messagebox.askyesno(
                "确认", f"删除模型 {model_id}？\n\n"
                        f"（会删除 int8 和 FP32 两种版本）",
                parent=self):
            return
        try:
            from translation.local_models import ModelCacheManager
            ModelCacheManager().delete_model(model_id)
            try:
                from translation.local_models import invalidate_model
                invalidate_model(model_id)
            except Exception:
                pass
            self._set_progress(f"已删除：{model_id}")
            self._refresh_list()
        except Exception as e:
            self._set_progress(f"删除失败：{e}")

    def _remove_custom(self, model_id: str):
        if self._downloading:
            self._set_progress("⚠️ 下载中，无法操作")
            return

        from tkinter import messagebox
        if not messagebox.askyesno(
                "确认",
                "移除自定义模型注册？\n\n"
                "（不会删除原文件）",
                parent=self):
            return
        try:
            from translation.local_models import remove_custom_model
            remove_custom_model(model_id)
            self._set_progress(f"已移除注册：{model_id}")
            self._refresh_list()
        except Exception as e:
            self._set_progress(f"移除失败：{e}")

    def _clear_cache(self):
        if self._downloading:
            self._set_progress("⚠️ 下载中，无法清理")
            return

        from tkinter import messagebox
        if not messagebox.askyesno(
                "确认",
                "清空所有已下载模型？\n\n"
                "（含下载失败留下的残留文件；"
                "自定义模型不受影响）",
                parent=self):
            return
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            for m in mgr.list_models():
                mgr.delete_model(m['id'])
            # ⭐ 顺带清掉「有残留但缺权重」的目录（失败/中断下载的残骸）
            invalid = mgr.list_invalid_models()
            for m in invalid:
                try:
                    shutil.rmtree(m['path'], ignore_errors=True)
                except Exception:
                    pass
            try:
                from translation.local_models import invalidate_model
                invalidate_model()
            except Exception:
                pass
            self._set_progress(
                f"缓存已清空（含 {len(invalid)} 个残留目录）")
            self._refresh_list()
        except Exception as e:
            self._set_progress(f"清理失败：{e}")

    # ==================== 自定义模型导入 ====================

    def _import_custom(self):
        if self._downloading:
            self._set_progress("⚠️ 下载中，无法导入")
            return

        from tkinter import filedialog, messagebox

        path = filedialog.askdirectory(
            title="选择 ONNX / CT2 模型目录",
            parent=self,
        )
        if not path:
            return

        try:
            from translation.local_models.custom import is_valid_ct2_model
            if not is_valid_ct2_model(path):
                messagebox.showerror(
                    "格式错误",
                    f"目录不是合法的模型：\n\n{path}",
                    parent=self,
                )
                return
        except Exception as e:
            messagebox.showerror("错误", f"验证失败：{e}", parent=self)
            return

        self._ask_model_info_then_import(path)

    def _ask_model_info_then_import(self, path: str):
        dlg = ctk.CTkToplevel(self)
        dlg.title("导入自定义模型")
        dlg.geometry("500x380")
        dlg.attributes('-topmost', True)
        dlg.grab_set()

        pad = 16

        ctk.CTkLabel(
            dlg, text="模型信息",
            font=(FONT_NAME, 13, 'bold'),
        ).pack(pady=(pad, 10))

        ctk.CTkLabel(
            dlg, text=f"路径：{path}",
            font=(FONT_NAME, 9), text_color='#888888',
            wraplength=460, justify='left',
        ).pack(padx=pad, pady=(0, 10))

        row1 = ctk.CTkFrame(dlg, fg_color='transparent')
        row1.pack(fill='x', padx=pad, pady=6)
        ctk.CTkLabel(row1, text="名称：",
                     width=70, anchor='w').pack(side='left')
        name_entry = ctk.CTkEntry(row1, width=360)
        name_entry.pack(side='left')
        name_entry.insert(0, os.path.basename(path) or "自定义模型")

        row2 = ctk.CTkFrame(dlg, fg_color='transparent')
        row2.pack(fill='x', padx=pad, pady=6)
        ctk.CTkLabel(row2, text="源语言：",
                     width=70, anchor='w').pack(side='left')

        from config import SOURCE_LANGS
        src_vals = list(SOURCE_LANGS.values())
        src_selector = ctk.CTkOptionMenu(
            row2, values=src_vals, width=200)
        src_selector.set('自适应')
        src_selector.pack(side='left')

        row3 = ctk.CTkFrame(dlg, fg_color='transparent')
        row3.pack(fill='x', padx=pad, pady=6)
        ctk.CTkLabel(row3, text="目标语言：",
                     width=70, anchor='w').pack(side='left')

        from config import TARGET_LANGS
        tgt_vals = list(TARGET_LANGS.values())
        tgt_selector = ctk.CTkOptionMenu(
            row3, values=tgt_vals, width=200)
        tgt_selector.set('中文')
        tgt_selector.pack(side='left')

        status = ctk.CTkLabel(
            dlg, text="", font=(FONT_NAME, 10),
            text_color='gray',
        )
        status.pack(pady=(10, 4))

        def do_import():
            name = name_entry.get().strip()
            if not name:
                status.configure(
                    text="❌ 名称不能为空", text_color='#cc0000')
                return

            from config import SOURCE_LANGS, TARGET_LANGS
            src_rev = {v: k for k, v in SOURCE_LANGS.items()}
            tgt_rev = {v: k for k, v in TARGET_LANGS.items()}
            src_code = src_rev.get(src_selector.get(), 'auto')
            tgt_code = tgt_rev.get(tgt_selector.get(), 'zh')

            try:
                from translation.local_models import import_custom_model
                entry = import_custom_model(
                    path=path, name=name,
                    languages=[[src_code, tgt_code]],
                )
                if entry:
                    status.configure(
                        text=f"✅ 导入成功：{name}",
                        text_color='#00a000')
                    dlg.after(600, dlg.destroy)
                    self._safe_after(600, self._refresh_list)
                else:
                    status.configure(
                        text="❌ 导入失败", text_color='#cc0000')
            except Exception as e:
                status.configure(
                    text=f"❌ 异常：{e}", text_color='#cc0000')

        btn_frame = ctk.CTkFrame(dlg, fg_color='transparent')
        btn_frame.pack(pady=16)

        ctk.CTkButton(
            btn_frame, text="导入", width=100,
            command=do_import,
        ).pack(side='left', padx=6)

        ctk.CTkButton(
            btn_frame, text="取消", width=80,
            fg_color='#6c757d', hover_color='#5a6268',
            command=dlg.destroy,
        ).pack(side='left', padx=6)

    # ==================== 辅助 ====================

    def _open_models_dir(self):
        try:
            from translation.local_models import open_models_dir
            ok = open_models_dir()
            if ok:
                self._set_progress("已打开模型目录")
            else:
                self._set_progress("⚠️ 打开目录失败")
        except Exception as e:
            self._set_progress(f"打开目录失败：{e}")