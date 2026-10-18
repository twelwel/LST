"""翻译模块 - 统一门面（兼容旧接口）"""
from translation import get_manager


class Translator:
    def __init__(self, source='auto', target='zh'):
        self.source = source
        self.target = target
        self._manager = get_manager()

    def set_languages(self, source=None, target=None):
        if source is not None:
            self.source = source
        if target is not None:
            self.target = target

    def reload_config(self):
        self._manager.reload()

    def is_configured(self):
        return self._manager.is_configured()

    def translate(self, text, source=None, target=None):
        """翻译单条文本

        source/target 为 None 时用实例默认值
        """
        src = source if source is not None else self.source
        tgt = target if target is not None else self.target
        return self._manager.translate(text, src, tgt)

    def translate_lines(self, lines, source=None, target=None):
        if not lines:
            return []
        src = source if source is not None else self.source
        tgt = target if target is not None else self.target
        return self._manager.translate_lines(lines, src, tgt)

    def get_strategy_name(self):
        return self._manager.get_strategy_name()

    def set_mode_override(self, mode):
        """临时切换策略（主界面「本地流式」按钮使用）。"""
        return self._manager.set_mode_override(mode)

    def clear_mode_override(self):
        return self._manager.clear_mode_override()

    def preload_local_model(self, model_id=''):
        """后台预热本地模型，返回 (ok, error)。"""
        return self._manager.preload_local_model(model_id)


def translate_text(text, source='auto', target='zh'):
    return Translator(source=source, target=target).translate(text)


def test_baidu_api(appid, secret_key, text='Hello'):
    from translation.providers import BaiduProvider
    cfg = {
        'id': 'test', 'name': 'test',
        'provider': 'baidu',
        'config': {'appid': appid, 'secret_key': secret_key},
    }
    p = BaiduProvider(cfg)
    r = p.translate(text)
    return (r.success, r.text if r.success else r.error)