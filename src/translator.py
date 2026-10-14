"""翻译模块 - 统一门面（兼容旧接口）"""
from translation import get_manager


class Translator:
    def __init__(self, source='en', target='zh'):
        self.source = source
        self.target = target
        self._manager = get_manager()

    def reload_config(self):
        self._manager.reload()

    def is_configured(self):
        return self._manager.is_configured()

    def translate(self, text):
        return self._manager.translate(text)

    def translate_lines(self, lines):
        if not lines:
            return []
        return self._manager.translate_lines(lines)

    def get_strategy_name(self):
        return self._manager.get_strategy_name()


def translate_text(text, source='en', target='zh'):
    return Translator(source=source, target=target).translate(text)


def test_baidu_api(appid, secret_key, text='Hello'):
    """保留旧接口，供 api_setup_dialog 使用"""
    from translation.providers import BaiduProvider
    cfg = {
        'id': 'test', 'name': 'test',
        'provider': 'baidu',
        'config': {'appid': appid, 'secret_key': secret_key},
    }
    p = BaiduProvider(cfg)
    r = p.translate(text)
    return (r.success, r.text if r.success else r.error)