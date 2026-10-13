"""API 配置对话框：首次启动 / 主动打开时弹出"""

import customtkinter as ctk

from api_config import load_config, save_config
from translator import test_baidu_api


class APISetupDialog(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("配置百度翻译 API")
        self.resizable(False, False)
        self.attributes('-topmost', True)

        # 居中显示
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w, h = 540, 400
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        self.result = False   # 是否成功配置
        cfg = load_config()
        self._build_ui(cfg)

    def _build_ui(self, cfg):
        pad = 20

        # 标题
        ctk.CTkLabel(
            self, text="配置百度翻译 API",
            font=('Microsoft YaHei', 18, 'bold'),
        ).pack(pady=(pad, 5))

        ctk.CTkLabel(
            self,
            text="请前往百度翻译开放平台申请账号：\n"
                 "https://fanyi-api.baidu.com/choose",
            font=('Microsoft YaHei', 10),
            text_color='#3a86ff',
            justify='left',
        ).pack(pady=(0, 15))

        # 表单
        form = ctk.CTkFrame(self, fg_color='transparent')
        form.pack(fill='x', padx=pad)

        ctk.CTkLabel(
            form, text="APPID：", font=('Microsoft YaHei', 12),
        ).grid(row=0, column=0, sticky='w', pady=8)

        self.appid_entry = ctk.CTkEntry(form, width=320)
        self.appid_entry.grid(row=0, column=1, padx=(5, 0), pady=8)
        if cfg.get('baidu_appid'):
            self.appid_entry.insert(0, cfg['baidu_appid'])

        ctk.CTkLabel(
            form, text="密钥：", font=('Microsoft YaHei', 12),
        ).grid(row=1, column=0, sticky='w', pady=8)

        self.key_entry = ctk.CTkEntry(form, width=320, show='*')
        self.key_entry.grid(row=1, column=1, padx=(5, 0), pady=8)
        if cfg.get('baidu_secret_key'):
            self.key_entry.insert(0, cfg['baidu_secret_key'])

        # 显示/隐藏密钥
        self.show_key_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            form, text="显示密钥", variable=self.show_key_var,
            command=self._toggle_key_visible,
            font=('Microsoft YaHei', 10),
        ).grid(row=2, column=1, sticky='w', padx=(5, 0), pady=(0, 8))

        # 状态提示
        self.status_label = ctk.CTkLabel(
            self, text='', font=('Microsoft YaHei', 10),
            text_color='gray', wraplength=480,
        )
        self.status_label.pack(pady=(10, 0))

        # 按钮
        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(pady=20)

        ctk.CTkButton(
            btn_frame, text="测试并保存", width=140,
            command=self._test_and_save,
        ).pack(side='left', padx=8)

        ctk.CTkButton(
            btn_frame, text="取消", width=100,
            fg_color='#888888', hover_color='#666666',
            command=self._cancel,
        ).pack(side='left', padx=8)

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind('<Return>', lambda e: self._test_and_save())

    def _toggle_key_visible(self):
        self.key_entry.configure(show='' if self.show_key_var.get() else '*')

    def _test_and_save(self):
        appid = self.appid_entry.get().strip()
        key = self.key_entry.get().strip()

        if not appid or not key:
            self.status_label.configure(
                text='APPID 和密钥不能为空', text_color='#cc0000',
            )
            return

        self.status_label.configure(text='正在测试...', text_color='gray')
        self.update_idletasks()

        ok, msg = test_baidu_api(appid, key)
        if not ok:
            self.status_label.configure(
                text=f'❌ 测试失败：{msg}', text_color='#cc0000',
            )
            return

        # 保存
        cfg = load_config()
        cfg['baidu_appid'] = appid
        cfg['baidu_secret_key'] = key
        if not save_config(cfg):
            self.status_label.configure(
                text='❌ 配置保存失败', text_color='#cc0000',
            )
            return

        self.status_label.configure(
            text=f'✅ 配置成功！测试翻译：Hello → {msg}',
            text_color='#00a000',
        )
        self.result = True
        self.after(1200, self.destroy)

    def _cancel(self):
        self.result = False
        self.destroy()


def ask_api_config():
    """便捷函数：弹出对话框并返回是否成功"""
    dlg = APISetupDialog()
    dlg.mainloop()
    return dlg.result