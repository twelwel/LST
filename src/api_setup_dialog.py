"""API 配置对话框：策略选择 + 多 Provider 管理"""
import customtkinter as ctk
from tkinter import messagebox

from translation import (
    load_providers, save_providers,
    load_strategy, save_strategy,
    new_provider_id,
)
from translation.strategies import STRATEGY_LABELS
from translation.providers import create_provider


# ==================== Provider 编辑器 ====================
class ProviderEditorDialog(ctk.CTkToplevel):
    ADAPTERS = {
        'translate': [
            ('baidu', '百度翻译'),
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

        self.geometry("620x560")
        self.attributes('-topmost', True)
        self.resizable(False, False)

        self._build_ui()
        self.grab_set()

    def _build_ui(self):
        pad = 16
        # 名称
        row1 = ctk.CTkFrame(self, fg_color='transparent')
        row1.pack(fill='x', padx=pad, pady=(pad, 8))
        ctk.CTkLabel(row1, text="名称：", width=80, anchor='w').pack(side='left')
        self.name_entry = ctk.CTkEntry(row1, width=400)
        self.name_entry.pack(side='left')
        self.name_entry.insert(0, self.existing.get('name', ''))

        # 适配器
        row2 = ctk.CTkFrame(self, fg_color='transparent')
        row2.pack(fill='x', padx=pad, pady=8)
        ctk.CTkLabel(row2, text="适配器：", width=80, anchor='w').pack(side='left')
        self.adapter_values = self.ADAPTERS[self.provider_type]
        self.adapter_map = {label: key for key, label in self.adapter_values}
        self.adapter_selector = ctk.CTkOptionMenu(
            row2, values=[lbl for _, lbl in self.adapter_values],
            width=400, command=lambda _: self._rebuild_form())
        self.adapter_selector.pack(side='left')

        cur_adapter = self.existing.get('provider')
        cur_label = next(
            (lbl for key, lbl in self.adapter_values if key == cur_adapter),
            self.adapter_values[0][1])
        self.adapter_selector.set(cur_label)

        # 主 API 开关
        row3 = ctk.CTkFrame(self, fg_color='transparent')
        row3.pack(fill='x', padx=pad, pady=8)
        ctk.CTkLabel(row3, text=" ", width=80).pack(side='left')
        self.primary_var = ctk.BooleanVar(
            value=bool(self.existing.get('is_primary', False)))
        ctk.CTkCheckBox(
            row3, text="设为主 API", variable=self.primary_var
        ).pack(side='left')
        self.enabled_var = ctk.BooleanVar(
            value=bool(self.existing.get('enabled', True)))
        ctk.CTkCheckBox(
            row3, text="启用", variable=self.enabled_var
        ).pack(side='left', padx=20)

        # 动态表单
        self.form_frame = ctk.CTkScrollableFrame(self, height=260)
        self.form_frame.pack(fill='both', expand=True, padx=pad, pady=8)
        self.entries = {}
        self._rebuild_form()

        # 按钮
        btns = ctk.CTkFrame(self, fg_color='transparent')
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="确定", width=100,
                      command=self._on_ok).pack(side='left', padx=6)
        ctk.CTkButton(btns, text="取消", width=100,
                      fg_color='#6c757d',
                      command=self.destroy).pack(side='left', padx=6)

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
            ctk.CTkLabel(row, text=label, width=110, anchor='w').pack(side='left')
            entry = ctk.CTkEntry(row, width=420,
                                 show='*' if is_secret else '')
            entry.pack(side='left', fill='x', expand=True)
            if key in existing_cfg:
                entry.insert(0, str(existing_cfg[key]))
            self.entries[key] = entry
        self._current_adapter = adapter

    def _fields_for(self, adapter):
        if adapter == 'baidu':
            return [
                ('appid', 'APPID', False),
                ('secret_key', '密钥', True),
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

    def _on_ok(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("错误", "名称不能为空", parent=self)
            return
        cfg = {k: e.get().strip() for k, e in self.entries.items()}
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
        self.geometry("820x680")
        self.attributes('-topmost', True)

        self.result = False
        self.providers = load_providers()
        self.strategy = load_strategy()

        self._build_ui()

    def _build_ui(self):
        pad = 14
        # 策略
        sf = ctk.CTkFrame(self)
        sf.pack(fill='x', padx=pad, pady=(pad, 6))
        ctk.CTkLabel(sf, text="翻译策略：",
                     font=('Microsoft YaHei', 12, 'bold')).pack(
            side='left', padx=(10, 8))
        self.strategy_map = {v: k for k, v in STRATEGY_LABELS.items()}
        self.strategy_selector = ctk.CTkOptionMenu(
            sf, values=list(STRATEGY_LABELS.values()),
            command=self._on_strategy_change, width=240)
        cur = STRATEGY_LABELS.get(self.strategy.get('mode'),
                                  STRATEGY_LABELS['single_translate'])
        self.strategy_selector.set(cur)
        self.strategy_selector.pack(side='left')

        self.hybrid_order_selector = ctk.CTkOptionMenu(
            sf, values=['翻译优先', 'AI 优先', '并发取快'], width=120)
        order_map = {'翻译优先': 'translate_first', 'AI 优先': 'ai_first',
                     '并发取快': 'concurrent'}
        self.order_map = order_map
        rev = {v: k for k, v in order_map.items()}
        self.hybrid_order_selector.set(
            rev.get(self.strategy.get('hybrid_order', 'translate_first'),
                    '翻译优先'))
        self.hybrid_order_selector.pack(side='left', padx=8)

        # 两个列表
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill='both', expand=True, padx=pad, pady=6)
        self.tabview.add("翻译 API")
        self.tabview.add("AI 模型")

        self._build_list_tab(self.tabview.tab("翻译 API"), 'translate')
        self._build_list_tab(self.tabview.tab("AI 模型"), 'ai')

        # 按钮
        btns = ctk.CTkFrame(self, fg_color='transparent')
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="保存", width=120,
                      command=self._on_save).pack(side='left', padx=6)
        ctk.CTkButton(btns, text="取消", width=100,
                      fg_color='#6c757d',
                      command=self.destroy).pack(side='left', padx=6)

        self._refresh_lists()

    def _build_list_tab(self, parent, ptype):
        ctk.CTkButton(
            parent, text="+ 添加", width=100,
            command=lambda: self._add_provider(ptype),
        ).pack(anchor='e', padx=6, pady=6)

        frame = ctk.CTkScrollableFrame(parent, fg_color='transparent')
        frame.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        setattr(self, f'_list_{ptype}', frame)

    def _refresh_lists(self):
        for ptype in ('translate', 'ai'):
            frame = getattr(self, f'_list_{ptype}')
            for w in frame.winfo_children():
                w.destroy()
            for p in self.providers:
                if p.get('type') != ptype:
                    continue
                self._render_provider_row(frame, p)

    def _render_provider_row(self, parent, provider):
        row = ctk.CTkFrame(parent)
        row.pack(fill='x', pady=3, padx=3)

        primary_tag = ' ⭐' if provider.get('is_primary') else ''
        enabled_tag = '' if provider.get('enabled', True) else ' [禁用]'
        ctk.CTkLabel(
            row,
            text=f"{provider.get('name', '')}{primary_tag}{enabled_tag}",
            anchor='w', width=200,
        ).pack(side='left', padx=6)
        ctk.CTkLabel(
            row, text=provider.get('provider', ''),
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

    def _on_strategy_change(self, label):
        mode = self.strategy_map.get(label, 'single_translate')
        self.strategy['mode'] = mode

    # --- Provider 操作 ---
    def _add_provider(self, ptype):
        dlg = ProviderEditorDialog(self, ptype)
        self.wait_window(dlg)
        if dlg.result:
            self.providers.append(dlg.result)
            self._refresh_lists()

    def _edit_provider(self, provider):
        dlg = ProviderEditorDialog(self, provider.get('type'), provider)
        self.wait_window(dlg)
        if dlg.result:
            idx = self.providers.index(provider)
            self.providers[idx] = dlg.result
            self._refresh_lists()

    def _delete_provider(self, provider):
        if not messagebox.askyesno(
                "确认", f"删除 {provider.get('name')}？", parent=self):
            return
        self.providers.remove(provider)
        self._refresh_lists()

    def _set_primary(self, provider):
        for p in self.providers:
            if p.get('type') == provider.get('type'):
                p['is_primary'] = False
        provider['is_primary'] = True
        self._refresh_lists()

    # --- 保存 ---
    def _on_save(self):
        if not self.providers:
            messagebox.showerror("错误", "至少需要一个 API", parent=self)
            return
        if not save_providers(self.providers):
            messagebox.showerror("错误", "保存失败", parent=self)
            return
        self.strategy['hybrid_order'] = self.order_map.get(
            self.hybrid_order_selector.get(), 'translate_first')
        save_strategy(self.strategy)
        self.result = True
        self.destroy()


def ask_api_config():
    dlg = APISetupDialog()
    dlg.mainloop()
    return dlg.result