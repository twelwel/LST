"""可复用 UI 控件"""

import customtkinter as ctk

from config import FONT_NAME


class CollapsibleFrame(ctk.CTkFrame):
    """可折叠面板（标题按钮 + 内容区）"""

    def __init__(self, master, title="标题", expanded=True, **kwargs):
        super().__init__(master, **kwargs)
        self._title = title
        self._expanded = expanded

        self.title_btn = ctk.CTkButton(
            self,
            text=self._make_text(),
            anchor='w',
            height=32,
            fg_color=('gray85', 'gray22'),
            hover_color=('gray75', 'gray32'),
            text_color=('black', 'white'),
            command=self.toggle,
            font=(FONT_NAME, 12, 'bold'),
        )
        self.title_btn.pack(fill='x', padx=2, pady=2)

        self.content = ctk.CTkFrame(self, fg_color='transparent', height=1)
        self.content.pack_propagate(True)
        if expanded:
            self.content.pack(fill='x', padx=2, pady=(0, 5))

    def _make_text(self):
        arrow = "▼" if self._expanded else "▶"
        return f"{arrow}   {self._title}"

    def toggle(self):
        self._expanded = not self._expanded
        self.title_btn.configure(text=self._make_text())
        if self._expanded:
            self.content.pack(fill='x', padx=2, pady=(0, 5))
        else:
            self.content.pack_forget()