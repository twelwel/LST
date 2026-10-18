"""本地模型 Provider / 翻译管理器回归测试（不加载真实模型）。

运行：
    python tools/test_local_model_provider.py
"""

import os
import sys
import threading
import unittest
from unittest import mock


_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from translation.manager import TranslationManager  # noqa: E402
from translation.providers import LocalModelProvider  # noqa: E402


class FakeModelInstance:
    def __init__(self, model_id, loaded=True, error=''):
        self._model_id = model_id
        self.is_loaded = loaded
        self.last_error = error
        self.calls = 0

    def supports(self, src, tgt):
        return True

    def translate(self, text, src, tgt):
        self.calls += 1
        return f'本地:{text}'


def make_provider(model_id='opus-mt-en-zh'):
    return LocalModelProvider({
        'id': 'local',
        'name': '本地模型',
        'type': 'translate',
        'provider': 'local_model',
        'enabled': True,
        'config': {'model_id': model_id},
    })


class LocalProviderTests(unittest.TestCase):
    def test_provider_reuses_loaded_instance(self):
        instance = FakeModelInstance('opus-mt-en-zh')
        with mock.patch(
                'translation.local_models.get_model_instance',
                return_value=instance) as factory:
            provider = make_provider()
            first = provider.translate('hello', 'en', 'zh')
            second = provider.translate('world', 'en', 'zh')

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(instance.calls, 2)
        self.assertEqual(factory.call_count, 1)

    def test_preload_warms_cache(self):
        instance = FakeModelInstance('opus-mt-en-zh', loaded=False,
                                     error='加载中')
        with mock.patch(
                'translation.local_models.get_model_instance',
                return_value=instance):
            provider = make_provider()
            ok, detail = provider.preload()
            self.assertFalse(ok)
            self.assertIn('加载中', detail)

            instance.is_loaded = True
            instance.last_error = ''
            ok, detail = provider.preload()

        self.assertTrue(ok)
        self.assertEqual(detail, '')

    def test_preload_requires_model_id(self):
        provider = make_provider('')
        ok, detail = provider.preload()
        self.assertFalse(ok)
        self.assertIn('model_id', detail)


class FakeBatchStrategy:
    def __init__(self):
        self.calls = []

    def translate(self, text, source='auto', target='zh'):
        return f'单:{text}'

    def translate_lines(self, lines, source='auto', target='zh'):
        self.calls.append(list(lines))
        return [f'批:{line}' for line in lines]


class ManagerBatchTests(unittest.TestCase):
    def _manager_with(self, strategy):
        manager = TranslationManager.__new__(TranslationManager)
        manager._lock = threading.RLock()
        manager._strategy = strategy
        return manager

    def _bare_manager(self):
        manager = TranslationManager.__new__(TranslationManager)
        manager._lock = threading.RLock()
        manager._providers = {}
        manager._strategy = None
        manager._strategy_mode = ''
        manager._strategy_cfg = {}
        manager._mode_override = None
        manager._first_load = False
        return manager

    def test_mode_override_is_runtime_only(self):
        manager = self._bare_manager()
        with mock.patch(
                'translation.manager.load_providers', return_value=[]), \
             mock.patch(
                'translation.manager.load_strategy',
                return_value={'mode': 'single_translate'}), \
             mock.patch(
                'translation.manager.create_strategy') as create:
            self.assertTrue(manager.set_mode_override('local_only'))
            self.assertEqual(manager.get_strategy_mode(), 'local_only')
            self.assertEqual(create.call_args[0][0], 'local_only')
            self.assertFalse(create.call_args[0][3]['local_fallback'])

            self.assertTrue(manager.clear_mode_override())
            self.assertEqual(manager.get_strategy_mode(),
                             'single_translate')
            self.assertEqual(create.call_args[0][0], 'single_translate')

    def test_translate_lines_uses_strategy_batch(self):
        strategy = FakeBatchStrategy()
        manager = self._manager_with(strategy)
        result = manager.translate_lines(['a', 'b'], 'en', 'zh')
        self.assertEqual(result, ['批:a', '批:b'])
        self.assertEqual(strategy.calls, [['a', 'b']])

    def test_translate_lines_falls_back_when_batch_length_wrong(self):
        class WrongLength(FakeBatchStrategy):
            def translate_lines(self, lines, source='auto', target='zh'):
                return ['only-one']

        manager = self._manager_with(WrongLength())
        result = manager.translate_lines(['a', 'b'], 'en', 'zh')
        self.assertEqual(result, ['单:a', '单:b'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
