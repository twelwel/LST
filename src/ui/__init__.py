"""UI 模块（拆分版）

对外只暴露 TranslatorApp，保持向后兼容：
    from ui import TranslatorApp
"""

from .main_window import TranslatorApp

__all__ = ['TranslatorApp']