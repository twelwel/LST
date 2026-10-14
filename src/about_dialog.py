"""关于 / 版本信息对话框"""

import sys
import platform
import customtkinter as ctk

from version import (
    __version__, __author__, __license__,
    APP_NAME, APP_NAME_CN, REPO_URL,
    VERSION_INFO,
)


class AboutDialog(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title(f"关于 {APP_NAME}")
        self.resizable(False, False)
        self.attributes('-topmost', True)

        w, h = 480, 520
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self._build_ui()

    def _build_ui(self):
        pad = 20

        # 标题
        ctk.CTkLabel(
            self,
            text=f"📖 {APP_NAME_CN}",
            font=('Microsoft YaHei', 22, 'bold'),
        ).pack(pady=(pad, 0))

        ctk.CTkLabel(
            self,
            text=f"{APP_NAME}  v{__version__}",
            font=('Microsoft YaHei', 14),
            text_color='#3a86ff',
        ).pack(pady=(0, 15))

        # 基础信息
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill='x', padx=pad, pady=(0, 10))

        self._add_row(info_frame, "版本号：", f"v{__version__}")
        self._add_row(info_frame, "作者：", __author__)
        self._add_row(info_frame, "许可证：", __license__)

        # 环境信息
        env_frame = ctk.CTkFrame(self)
        env_frame.pack(fill='x', padx=pad, pady=(0, 10))

        py_ver = (f"{sys.version_info.major}."
                  f"{sys.version_info.minor}."
                  f"{sys.version_info.micro}")
        self._add_row(env_frame, "Python：", py_ver)
        self._add_row(env_frame, "系统：",
                      platform.system() + " " + platform.release())

        # 版本历史
        ctk.CTkLabel(
            self, text="版本历史",
            font=('Microsoft YaHei', 12, 'bold'),
        ).pack(anchor='w', padx=pad, pady=(10, 5))

        history = ctk.CTkTextbox(
            self, height=100, wrap='word',
            font=('Microsoft YaHei', 10),
        )
        history.pack(fill='x', padx=pad, pady=(0, 10))
        for ver, desc in VERSION_INFO.items():
            history.insert('end', f"v{ver}  -  {desc}\n")
        history.configure(state='disabled')

        # 按钮
        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame, text="复制仓库地址", width=120,
            command=self._copy_repo,
        ).pack(side='left', padx=5)

        ctk.CTkButton(
            btn_frame, text="关闭", width=80,
            fg_color='#6c757d', hover_color='#5a6268',
            command=self.destroy,
        ).pack(side='left', padx=5)

        ctk.CTkLabel(
            self, text=f"仓库：{REPO_URL}",
            font=('Microsoft YaHei', 9), text_color='gray',
        ).pack(pady=(0, pad))

    def _add_row(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color='transparent')
        row.pack(fill='x', padx=10, pady=3)
        ctk.CTkLabel(
            row, text=label, width=80, anchor='w',
            font=('Microsoft YaHei', 10),
        ).pack(side='left')
        ctk.CTkLabel(
            row, text=value, anchor='w',
            font=('Microsoft YaHei', 10),
            text_color='#3a86ff',
        ).pack(side='left', padx=5)

    def _copy_repo(self):
        self.clipboard_clear()
        self.clipboard_append(REPO_URL)
        self.update()