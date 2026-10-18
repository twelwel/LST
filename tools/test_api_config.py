"""翻译配置层回归测试（使用临时目录，不触碰用户配置）。

运行：
    python tools/test_api_config.py
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock


_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import translation.api_config as api_config  # noqa: E402
from translation.local_models.registry import MODEL_REGISTRY  # noqa: E402


class ConfigSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='lst_cfg_')
        self._saved = (
            api_config.CONFIG_DIR,
            api_config.PROVIDERS_FILE,
            api_config.STRATEGY_FILE,
            api_config.LEGACY_CONFIG,
            api_config._sync_in_progress,
        )
        api_config.CONFIG_DIR = self.tmp
        api_config.PROVIDERS_FILE = os.path.join(
            self.tmp, 'api_providers.json')
        api_config.STRATEGY_FILE = os.path.join(
            self.tmp, 'translation_strategy.json')
        api_config.LEGACY_CONFIG = os.path.join(self.tmp, 'config.json')
        api_config._sync_in_progress = False

    def tearDown(self):
        (api_config.CONFIG_DIR,
         api_config.PROVIDERS_FILE,
         api_config.STRATEGY_FILE,
         api_config.LEGACY_CONFIG,
         api_config._sync_in_progress) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_providers(self, providers):
        with open(api_config.PROVIDERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(providers, f, ensure_ascii=False, indent=2)

    def test_select_persists_provider_and_strategy(self):
        self.assertTrue(
            api_config.select_local_model('opus-mt-en-zh', 'OPUS 英中'))

        providers = api_config.load_providers()
        local = [p for p in providers
                 if p.get('provider') == 'local_model']
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]['config']['model_id'], 'opus-mt-en-zh')
        self.assertTrue(local[0]['enabled'])
        self.assertEqual(
            api_config.load_strategy()['local_model_id'],
            'opus-mt-en-zh')

    def test_select_reenables_disabled_provider(self):
        self.assertTrue(
            api_config.ensure_local_provider('opus-mt-en-zh', 'OPUS'))
        providers = api_config.load_providers()
        for p in providers:
            if p.get('provider') == 'local_model':
                p['enabled'] = False
                p['type'] = 'ai'
        self._write_providers(providers)

        self.assertTrue(api_config.select_local_model('opus-mt-en-zh'))
        local = [p for p in api_config.load_providers()
                 if p.get('provider') == 'local_model']
        self.assertTrue(local[0]['enabled'])
        self.assertEqual(local[0]['type'], 'translate')

    def test_sync_keeps_custom_provider_and_clears_selection(self):
        self.assertIn('opus-mt-en-zh', MODEL_REGISTRY)
        self._write_providers([
            {
                'id': 'builtin_local',
                'name': '本地模型 · OPUS',
                'type': 'translate',
                'provider': 'local_model',
                'is_primary': True,
                'enabled': True,
                'config': {'model_id': 'opus-mt-en-zh'},
            },
            {
                'id': 'custom_local',
                'name': '本地模型 · 自定义',
                'type': 'translate',
                'provider': 'local_model',
                'is_primary': False,
                'enabled': True,
                'config': {'model_id': 'custom_abc'},
            },
        ])
        self.assertTrue(api_config.select_local_model('opus-mt-en-zh'))

        with mock.patch(
                'translation.local_models.downloader.list_downloaded_ids',
                return_value=[]):
            result = api_config.sync_downloaded_models()

        self.assertIn('opus-mt-en-zh', result['removed'])
        providers = api_config.load_providers()
        model_ids = {
            (p.get('config', {}) or {}).get('model_id')
            for p in providers
            if p.get('provider') == 'local_model'
        }
        self.assertEqual(model_ids, {'custom_abc'})
        self.assertEqual(
            api_config.load_strategy()['local_model_id'], '')

    def test_clear_selection_only_matching_model(self):
        self.assertTrue(
            api_config.select_local_model('opus-mt-en-zh'))
        self.assertFalse(
            api_config.clear_local_model_selection('other-model'))
        self.assertEqual(
            api_config.load_strategy()['local_model_id'],
            'opus-mt-en-zh')
        self.assertTrue(
            api_config.clear_local_model_selection('opus-mt-en-zh'))
        self.assertEqual(
            api_config.load_strategy()['local_model_id'], '')

    def test_remove_provider_clears_selection(self):
        self.assertTrue(
            api_config.select_local_model('opus-mt-en-zh'))
        self.assertTrue(
            api_config.remove_local_provider('opus-mt-en-zh'))
        self.assertEqual(
            api_config.load_strategy()['local_model_id'], '')
        self.assertFalse(any(
            p.get('provider') == 'local_model'
            for p in api_config.load_providers()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
