"""热键管理模块"""

import keyboard


class HotkeyManager:
    """管理所有热键的注册、修改和注销"""

    def __init__(self):
        self.handles = {}   # name -> handle

    def register(self, name, hotkey, callback):
        """注册热键，同名旧热键会被替换"""
        self.unregister(name)
        try:
            self.handles[name] = keyboard.add_hotkey(hotkey, callback)
            print(f"[热键] 已注册 {name} = {hotkey}")
            return True
        except Exception as e:
            print(f"[热键] 注册 {name}={hotkey} 失败: {e}")
            return False

    def unregister(self, name):
        if name in self.handles:
            try:
                keyboard.remove_hotkey(self.handles[name])
            except Exception:
                pass
            del self.handles[name]

    def remove_all(self):
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        self.handles.clear()

    def capture(self, timeout=None):
        """
        阻塞读取用户按下的组合键
        返回：如 'ctrl+alt+t' 或 None（失败）
        """
        try:
            return keyboard.read_hotkey(suppress=False)
        except Exception as e:
            print(f"[热键] 捕获失败: {e}")
            return None