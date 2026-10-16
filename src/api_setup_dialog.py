"""API 配置对话框：多 Provider 管理 + 连接测试
注意：翻译策略由主界面顶部下拉框控制，本对话框只管 API 列表
"""

import threading
import customtkinter as ctk
from tkinter import messagebox

from translation import (
    load_providers, save_providers,
    new_provider_id,
)
from translation.providers import create_provider


# ==================== Provider 编辑器 ====================
class ProviderEditorDialog(ctk.CTkToplevel):
    ADAPTERS = {
        'translate': [
            ('baidu', '百度翻译'),
            ('openai_compat_ai', 'AI 翻译（推荐）'),
            ('local_model', '本地模型（离线）'),
            ('custom_rest', '自定义 REST'),
        ],
        'ai': [
            ('openai_compat', 'OpenAI 兼容（GPT/DeepSeek/Qwen/Kimi/ollama）'),
        ],
    }

    def __init__(self, master, provider_type, provider_cfg=None):
        super().__init__(master)
        self.title("编辑 API" if provider_cfg else "添加 API")
        self.provider_type = provider_type
        self.result = None
        self.existing = provider_cfg or {}

        self.geometry("640x620")
        self.attributes('-topmost', True)
        self.resizable(False, False)

        self._build_ui()
        self.grab_set()

    def _build_ui(self):
        pad = 16

        # ---- 名称 ----
        row1 = ctk.CTkFrame(self, fg_color='transparent')
        row1.pack(fill='x', padx=pad, pady=(pad, 8))
        ctk.CTkLabel(
            row1, text="名称：", width=80, anchor='w',
        ).pack(side='left')
        self.name_entry = ctk.CTkEntry(row1, width=460)
        self.name_entry.pack(side='left')
        self.name_entry.insert(0, self.existing.get('name', ''))

        # ---- 适配器 ----
        row2 = ctk.CTkFrame(self, fg_color='transparent')
        row2.pack(fill='x', padx=pad, pady=8)
        ctk.CTkLabel(
            row2, text="适配器：", width=80, anchor='w',
        ).pack(side='left')
        self.adapter_values = self.ADAPTERS[self.provider_type]
        self.adapter_map = {label: key for key, label in self.adapter_values}
        self.adapter_selector = ctk.CTkOptionMenu(
            row2,
            values=[lbl for _, lbl in self.adapter_values],
            width=460,
            command=lambda _: self._rebuild_form(),
        )
        self.adapter_selector.pack(side='left')

        # 当前适配器
        cur_adapter = self.existing.get('provider')
        cur_label = next(
            (lbl for key, lbl in self.adapter_values if key == cur_adapter),
            self.adapter_values[0][1])
        self.adapter_selector.set(cur_label)

        # ---- 主 API / 启用 ----
        row3 = ctk.CTkFrame(self, fg_color='transparent')
        row3.pack(fill='x', padx=pad, pady=8)
        ctk.CTkLabel(row3, text=" ", width=80).pack(side='left')
        self.primary_var = ctk.BooleanVar(
            value=bool(self.existing.get('is_primary', False)))
        ctk.CTkCheckBox(
            row3, text="设为主 API", variable=self.primary_var,
        ).pack(side='left')
        self.enabled_var = ctk.BooleanVar(
            value=bool(self.existing.get('enabled', True)))
        ctk.CTkCheckBox(
            row3, text="启用", variable=self.enabled_var,
        ).pack(side='left', padx=20)

        # ---- 动态表单 ----
        self.form_frame = ctk.CTkScrollableFrame(self, height=280)
        self.form_frame.pack(fill='both', expand=True,
                              padx=pad, pady=8)
        self.entries = {}
        self._rebuild_form()

        # ---- 状态 ----
        self.status_label = ctk.CTkLabel(
            self, text='', font=('Microsoft YaHei', 10),
            text_color='gray', anchor='w',
        )
        self.status_label.pack(fill='x', padx=pad, pady=(4, 0))

        # ---- 按钮 ----
        btns = ctk.CTkFrame(self, fg_color='transparent')
        btns.pack(pady=10)

        self.test_btn = ctk.CTkButton(
            btns, text="🔌 测试连接", width=110,
            fg_color='#17a2b8', hover_color='#138496',
            command=self._on_test,
        )
        self.test_btn.pack(side='left', padx=6)

        ctk.CTkButton(
            btns, text="确定", width=90,
            command=self._on_ok,
        ).pack(side='left', padx=6)

        ctk.CTkButton(
            btns, text="取消", width=90,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.destroy,
        ).pack(side='left', padx=6)

    # ==================== 表单构建 ====================

    def _rebuild_form(self):
        for w in self.form_frame.winfo_children():
            w.destroy()
        self.entries = {}

        adapter = self.adapter_map[self.adapter_selector.get()]
        existing_cfg = self.existing.get('config', {}) or {}
        fields = self._fields_for(adapter)

        for key, label, is_secret in fields:
            row = ctk.CTkFrame(self.form_frame, fg_color='transparent')
            row.pack(fill='x', pady=4)

            ctk.CTkLabel(
                row, text=label, width=110, anchor='w',
            ).pack(side='left')

            entry = ctk.CTkEntry(
                row, width=440,
                show='*' if is_secret else '',
            )
            entry.pack(side='left', fill='x', expand=True)

            if key in existing_cfg:
                entry.insert(0, str(existing_cfg[key]))

            self.entries[key] = entry

        # 本地模型额外提示 + 快捷选择按钮
        if adapter == 'local_model':
            hint = ctk.CTkFrame(
                self.form_frame, fg_color='transparent')
            hint.pack(fill='x', pady=(4, 0))

            ctk.CTkLabel(
                hint,
                text="💡 在「本地模型」面板下载后，把模型 ID 填入上方",
                font=('Microsoft YaHei', 9),
                text_color='#888888', anchor='w',
            ).pack(side='left')

            ctk.CTkButton(
                hint, text="查看已下载", width=100, height=24,
                font=('Microsoft YaHei', 9),
                fg_color='#16a085', hover_color='#138d75',
                command=self._show_downloaded_models,
            ).pack(side='right', padx=4)

            ctk.CTkButton(
                hint, text="选择模型", width=90, height=24,
                font=('Microsoft YaHei', 9),
                fg_color='#3a86ff', hover_color='#2a6fd8',
                command=self._pick_local_model,
            ).pack(side='right', padx=4)

        self._current_adapter = adapter

    def _fields_for(self, adapter):
        """根据适配器返回表单字段

        返回 [(key, label, is_secret), ...]
        """
        if adapter == 'baidu':
            return [
                ('appid', 'APPID', False),
                ('secret_key', '密钥', True),
            ]

        # ⭐ AI 翻译（推荐）：实际用 openai_compat 实现
        if adapter == 'openai_compat_ai':
            return [
                ('base_url', 'Base URL', False),
                ('api_key', 'API Key', True),
                ('model', '模型', False),
                ('system_prompt', 'System Prompt', False),
                ('temperature', 'Temperature', False),
                ('timeout', '超时(秒)', False),
            ]

        # ⭐ 本地模型（离线）
        if adapter == 'local_model':
            return [
                ('model_id', '模型 ID', False),
            ]

        if adapter == 'openai_compat':
            return [
                ('base_url', 'Base URL', False),
                ('api_key', 'API Key', True),
                ('model', '模型', False),
                ('system_prompt', 'System Prompt', False),
                ('temperature', 'Temperature', False),
                ('timeout', '超时(秒)', False),
            ]

        if adapter == 'custom_rest':
            return [
                ('url', 'URL 模板', False),
                ('method', '方法(GET/POST)', False),
                ('headers', 'Headers (JSON)', False),
                ('body', 'Body 模板', False),
                ('response_path', '响应路径', False),
            ]

        return []

    # ==================== 本地模型辅助 ====================

    def _show_downloaded_models(self):
        """弹窗显示已下载的本地模型"""
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            models = mgr.list_models()
        except Exception as e:
            messagebox.showerror(
                "错误",
                f"读取失败：{e}\n\n"
                f"请确认已安装：pip install ctranslate2 transformers "
                f"sentencepiece huggingface_hub",
                parent=self,
            )
            return

        if not models:
            messagebox.showinfo(
                "提示",
                "还没有下载任何本地模型。\n\n"
                "请到主窗口点「本地模型」按钮下载。",
                parent=self,
            )
            return

        text = "已下载的本地模型：\n\n"
        for m in models:
            text += f"· {m['id']}  ({m['size_mb']:.0f} MB)\n"
        text += "\n把 ID 填入上面的输入框即可。"

        messagebox.showinfo("已下载模型", text, parent=self)

    def _pick_local_model(self):
        """让用户从已下载模型中选一个"""
        try:
            from translation.local_models import ModelCacheManager
            mgr = ModelCacheManager()
            models = mgr.list_models()
        except Exception as e:
            messagebox.showerror("错误", f"读取失败：{e}", parent=self)
            return

        if not models:
            messagebox.showinfo(
                "提示",
                "还没有下载任何本地模型。\n"
                "请到主窗口点「本地模型」按钮下载。",
                parent=self,
            )
            return

        # 简单弹出选择
        picker = ctk.CTkToplevel(self)
        picker.title("选择模型")
        picker.geometry("420x360")
        picker.attributes('-topmost', True)

        ctk.CTkLabel(
            picker, text="选择一个本地模型：",
            font=('Microsoft YaHei', 11, 'bold'),
        ).pack(pady=10)

        sf = ctk.CTkScrollableFrame(picker)
        sf.pack(fill='both', expand=True, padx=10, pady=6)

        def _choose(mid):
            entry = self.entries.get('model_id')
            if entry is not None:
                entry.delete(0, 'end')
                entry.insert(0, mid)
            picker.destroy()

        for m in models:
            card = ctk.CTkFrame(sf)
            card.pack(fill='x', pady=3, padx=3)

            ctk.CTkLabel(
                card, text=m['id'],
                font=('Microsoft YaHei', 10, 'bold'),
                anchor='w',
            ).pack(side='left', padx=8)

            ctk.CTkLabel(
                card, text=f"{m['size_mb']:.0f} MB",
                font=('Microsoft YaHei', 9),
                text_color='#888888',
            ).pack(side='left', padx=4)

            ctk.CTkButton(
                card, text="选择", width=60,
                command=lambda mid=m['id']: _choose(mid),
            ).pack(side='right', padx=6, pady=4)

        picker.grab_set()

    # ==================== 收集 / 测试 / 保存 ====================

    def _collect_cfg(self):
        cfg = {k: e.get().strip() for k, e in self.entries.items()}
        # 本地模型：必须用 local_model 作为 provider
        adapter = self._current_adapter
        return {
            'id': self.existing.get('id') or 'tmp_test',
            'name': self.name_entry.get().strip() or 'test',
            'type': self.provider_type,
            'provider': adapter,
            'is_primary': False,
            'enabled': True,
            'config': cfg,
        }

    def _on_test(self):
        try:
            cfg = self._collect_cfg()
        except Exception as e:
            self.status_label.configure(
                text=f'❌ 收集配置失败：{e}', text_color='#cc0000')
            return

        # 本地模型的特殊提示
        if self._current_adapter == 'local_model':
            mid = (cfg.get('config', {}).get('model_id') or '').strip()
            if not mid:
                self.status_label.configure(
                    text='❌ 请先填写或选择模型 ID',
                    text_color='#cc0000')
                return

        self.test_btn.configure(state='disabled', text='测试中...')
        self.status_label.configure(
            text='⏳ 正在测试连接...', text_color='gray')

        def worker():
            try:
                p = create_provider(cfg)
                if self._current_adapter == 'local_model':
                    # 本地模型：测一句简单文本
                    r = p.translate('Hello' if
                                    cfg['config'].get('model_id', '')
                                    .startswith('opus-mt-en')
                                    else 'こんにちは')
                else:
                    r = p.translate('Hello')

                if r.success:
                    msg = f'✅ 成功：{r.text}  ({r.elapsed_ms}ms)'
                    color = '#00a000'
                else:
                    msg = f'❌ 失败：{r.error}'
                    color = '#cc0000'
            except Exception as e:
                msg = f'❌ 异常：{e}'
                color = '#cc0000'

            self.after(0, lambda: self._on_test_done(msg, color))

        threading.Thread(target=worker, daemon=True).start()

    def _on_test_done(self, msg, color):
        try:
            self.status_label.configure(text=msg, text_color=color)
            self.test_btn.configure(state='normal', text='🔌 测试连接')
        except Exception:
            pass

    def _on_ok(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("错误", "名称不能为空", parent=self)
            return

        cfg = {k: e.get().strip() for k, e in self.entries.items()}

        # 本地模型：model_id 必填
        if self._current_adapter == 'local_model':
            if not cfg.get('model_id'):
                messagebox.showerror(
                    "错误",
                    "请填写或选择模型 ID",
                    parent=self,
                )
                return

        self.result = {
            'id': self.existing.get('id') or new_provider_id(),
            'name': name,
            'type': self.provider_type,
            'provider': self._current_adapter,
            'is_primary': self.primary_var.get(),
            'enabled': self.enabled_var.get(),
            'config': cfg,
        }
        self.destroy()


# ==================== 主对话框 ====================
class APISetupDialog(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("翻译 API 配置")
        self.geometry("880x680")
        self.attributes('-topmost', True)

        self.result = False
        self.providers = load_providers()

        self._test_buttons = {}
        self._status_labels = {}

        self._build_ui()

    def _build_ui(self):
        pad = 14

        # 顶部提示 + 测试全部
        sf = ctk.CTkFrame(self, height=1)
        sf.pack(fill='x', padx=pad, pady=(pad, 6))
        sf.pack_propagate(True)

        ctk.CTkLabel(
            sf,
            text="💡 翻译策略请在主界面顶部选择；此处只管理 API 列表",
            font=('Microsoft YaHei', 10),
            text_color='#888888',
        ).pack(side='left', padx=8)

        ctk.CTkButton(
            sf, text="🔌 测试全部", width=100,
            fg_color='#17a2b8', hover_color='#138496',
            command=self._test_all,
        ).pack(side='right', padx=8)

        # Tab
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill='both', expand=True, padx=pad, pady=6)
        self.tabview.add("翻译 API")
        self.tabview.add("AI 模型")

        self._build_list_tab(self.tabview.tab("翻译 API"), 'translate')
        self._build_list_tab(self.tabview.tab("AI 模型"), 'ai')

        # 底部按钮
        btns = ctk.CTkFrame(self, fg_color='transparent', height=1)
        btns.pack(pady=10)
        btns.pack_propagate(True)

        ctk.CTkButton(
            btns, text="保存", width=120,
            command=self._on_save,
        ).pack(side='left', padx=6)

        ctk.CTkButton(
            btns, text="取消", width=100,
            fg_color='#6c757d',
            command=self.destroy,
        ).pack(side='left', padx=6)

        self._refresh_lists()

    def _build_list_tab(self, parent, ptype):
        ctk.CTkButton(
            parent, text="+ 添加", width=100,
            command=lambda: self._add_provider(ptype),
        ).pack(anchor='e', padx=6, pady=6)

        frame = ctk.CTkScrollableFrame(
            parent, fg_color='transparent')
        frame.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        setattr(self, f'_list_{ptype}', frame)

    def _refresh_lists(self):
        self._test_buttons.clear()
        self._status_labels.clear()

        for ptype in ('translate', 'ai'):
            frame = getattr(self, f'_list_{ptype}')
            for w in frame.winfo_children():
                w.destroy()
            for p in self.providers:
                if p.get('type') != ptype:
                    continue
                self._render_provider_row(frame, p)

    def _render_provider_row(self, parent, provider):
        wrapper = ctk.CTkFrame(parent, fg_color='transparent')
        wrapper.pack(fill='x', pady=3, padx=3)

        row = ctk.CTkFrame(wrapper)
        row.pack(fill='x')

        primary_tag = ' ⭐' if provider.get('is_primary') else ''
        enabled_tag = ('' if provider.get('enabled', True)
                       else ' [禁用]')

        ctk.CTkLabel(
            row,
            text=f"{provider.get('name', '')}{primary_tag}{enabled_tag}",
            anchor='w', width=180,
        ).pack(side='left', padx=6)

        adapter = provider.get('provider', '')
        adapter_text = adapter
        if adapter == 'local_model':
            mid = (provider.get('config', {}) or {}).get('model_id', '')
            adapter_text = f'本地模型 · {mid}' if mid else '本地模型'
        elif adapter == 'openai_compat_ai':
            adapter_text = 'AI 翻译'

        ctk.CTkLabel(
            row, text=adapter_text,
            text_color='gray', anchor='w',
        ).pack(side='left', padx=4, fill='x', expand=True)

        ctk.CTkButton(
            row, text="编辑", width=60,
            command=lambda p=provider: self._edit_provider(p),
        ).pack(side='right', padx=3)

        ctk.CTkButton(
            row, text="删除", width=60, fg_color='#dc3545',
            command=lambda p=provider: self._delete_provider(p),
        ).pack(side='right', padx=3)

        if not provider.get('is_primary'):
            ctk.CTkButton(
                row, text="设为主", width=70,
                command=lambda p=provider: self._set_primary(p),
            ).pack(side='right', padx=3)

        test_btn = ctk.CTkButton(
            row, text="🔌 测试", width=80,
            fg_color='#17a2b8', hover_color='#138496',
            command=lambda p=provider: self._test_provider(p),
        )
        test_btn.pack(side='right', padx=3)
        self._test_buttons[provider.get('id')] = test_btn

        status = ctk.CTkLabel(
            wrapper, text='', anchor='w',
            font=('Microsoft YaHei', 9),
            text_color='gray', wraplength=780, justify='left',
        )
        status.pack(fill='x', padx=(20, 6), pady=(2, 0))
        self._status_labels[provider.get('id')] = status

    # ==================== Provider 操作 ====================

    def _add_provider(self, ptype):
        dlg = ProviderEditorDialog(self, ptype)
        self.wait_window(dlg)
        if dlg.result:
            self.providers.append(dlg.result)
            self._refresh_lists()

    def _edit_provider(self, provider):
        dlg = ProviderEditorDialog(
            self, provider.get('type'), provider)
        self.wait_window(dlg)
        if dlg.result:
            idx = self.providers.index(provider)
            self.providers[idx] = dlg.result
            self._refresh_lists()

    def _delete_provider(self, provider):
        if not messagebox.askyesno(
                "确认",
                f"删除 {provider.get('name')}？",
                parent=self):
            return
        self.providers.remove(provider)
        self._refresh_lists()

    def _set_primary(self, provider):
        for p in self.providers:
            if p.get('type') == provider.get('type'):
                p['is_primary'] = False
        provider['is_primary'] = True
        self._refresh_lists()

    # ==================== 测试 ====================

    def _test_provider(self, provider):
        pid = provider.get('id')
        btn = self._test_buttons.get(pid)
        status = self._status_labels.get(pid)

        if btn:
            btn.configure(state='disabled', text='测试中...')
        if status:
            status.configure(text='⏳ 测试中...', text_color='gray')

        def worker():
            try:
                p = create_provider(provider)

                adapter = provider.get('provider', '')
                if adapter == 'local_model':
                    mid = (provider.get('config', {}) or {}).get(
                        'model_id', '')
                    test_text = ('Hello'
                                 if mid.startswith('opus-mt-en')
                                 else 'こんにちは')
                else:
                    test_text = 'Hello'

                r = p.translate(test_text)
                if r.success:
                    msg = (f'✅ 成功：{test_text} → {r.text}  '
                           f'({r.elapsed_ms}ms)')
                    color = '#00a000'
                else:
                    msg = f'❌ 失败：{r.error}'
                    color = '#cc0000'
            except Exception as e:
                msg = f'❌ 异常：{e}'
                color = '#cc0000'

            self.after(0, lambda: self._test_done(pid, msg, color))

        threading.Thread(target=worker, daemon=True).start()

    def _test_done(self, pid, msg, color):
        btn = self._test_buttons.get(pid)
        status = self._status_labels.get(pid)
        if btn:
            try:
                btn.configure(state='normal', text='🔌 测试')
            except Exception:
                pass
        if status:
            try:
                status.configure(text=msg, text_color=color)
            except Exception:
                pass

    def _test_all(self):
        enabled = [p for p in self.providers
                   if p.get('enabled', True)]
        if not enabled:
            messagebox.showinfo(
                "提示", "没有启用的 API", parent=self)
            return
        for p in enabled:
            self._test_provider(p)

    # ==================== 保存 ====================

    def _on_save(self):
        if not self.providers:
            messagebox.showerror(
                "错误", "至少需要一个 API", parent=self)
            return
        if not save_providers(self.providers):
            messagebox.showerror(
                "错误", "保存失败", parent=self)
            return
        # 注意：不修改 strategy.json（策略由主界面控制）
        self.result = True
        self.destroy()


def ask_api_config():
    dlg = APISetupDialog()
    dlg.mainloop()
    return dlg.result