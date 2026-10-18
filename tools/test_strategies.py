"""翻译策略回归测试（不访问网络、不加载真实模型）。

运行：
    python tools/test_strategies.py
"""

import os
import sys
import unittest


_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from translation.base import ProviderResult  # noqa: E402
from translation.manager import _order_local_providers  # noqa: E402
from translation.strategies import (  # noqa: E402
    HybridStrategy,
    LocalAPIStrategy,
    LocalOnlyStrategy,
    MultiAIVoteStrategy,
    RoundRobinTranslateStrategy,
    SingleAIStrategy,
    SingleTranslateStrategy,
)


class FakeProvider:
    def __init__(self, name, text=None, error='失败', primary=False,
                 local=False, provider_type='translate'):
        self.id = name
        self.name = name
        self._text = text
        self._error = error
        self.is_primary = primary
        self.enabled = True
        self.is_local = local
        self.provider_type = provider_type
        self.params = {'model_id': name} if local else {}
        self.calls = 0

    def translate(self, text, source='auto', target='zh'):
        self.calls += 1
        if self._text is None:
            return ProviderResult(False, error=self._error,
                                  provider_id=self.id,
                                  provider_name=self.name)
        return ProviderResult(True, text=self._text,
                              provider_id=self.id,
                              provider_name=self.name)


class EchoProvider(FakeProvider):
    """把输入原样带回来，便于验证重复文本去重。"""

    def translate(self, text, source='auto', target='zh'):
        self.calls += 1
        return ProviderResult(True, text=f'译:{text}',
                              provider_id=self.id,
                              provider_name=self.name)


def _strategy(cls, translate=None, ai=None, local=None, options=None):
    return cls(translate or [], ai or [], options or {}, local or [])


class SingleTranslateTests(unittest.TestCase):
    def test_api_success_does_not_call_local(self):
        api = FakeProvider('api', 'API译文', primary=True)
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(SingleTranslateStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, 'API译文')
        self.assertEqual(local.calls, 0)

    def test_api_failure_falls_back_to_local(self):
        api = FakeProvider('api', error='401', primary=True)
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(SingleTranslateStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')
        self.assertEqual(local.calls, 1)

    def test_local_failure_keeps_api_error(self):
        api = FakeProvider('api', error='401', primary=True)
        local = FakeProvider('local', error='未加载', local=True)
        result = _strategy(SingleTranslateStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertIn('401', result)
        self.assertIn('未加载', result)

    def test_no_api_uses_local(self):
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(SingleTranslateStrategy, local=[local]).translate(
            'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')

    def test_remote_fallback_can_be_disabled(self):
        api = FakeProvider('api', error='401', primary=True)
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(
            SingleTranslateStrategy, translate=[api], local=[local],
            options={'local_fallback': False},
        ).translate('hello', 'en', 'zh')
        self.assertIn('401', result)
        self.assertEqual(local.calls, 0)


class LocalStrategyTests(unittest.TestCase):
    def test_local_api_prefers_local(self):
        local = FakeProvider('local', '本地译文', local=True)
        api = FakeProvider('api', 'API译文', primary=True)
        result = _strategy(LocalAPIStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')
        self.assertEqual(api.calls, 0)

    def test_local_api_falls_back_to_api(self):
        local = FakeProvider('local', error='模型未加载', local=True)
        api = FakeProvider('api', 'API译文', primary=True)
        result = _strategy(LocalAPIStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, 'API译文')

    def test_local_only_reports_real_error(self):
        local = FakeProvider('local', error='模型未加载', local=True)
        result = _strategy(LocalOnlyStrategy, local=[local]).translate(
            'hello', 'en', 'zh')
        self.assertIn('模型未加载', result)

    def test_local_only_batch_deduplicates_lines(self):
        local = EchoProvider('local', local=True)
        strategy = _strategy(LocalOnlyStrategy, local=[local])
        result = strategy.translate_lines(
            ['hello', 'hello', '  ', 'world'], 'en', 'zh')
        self.assertEqual(result, ['译:hello', '译:hello', '', '译:world'])
        self.assertEqual(local.calls, 2)


class OtherStrategyTests(unittest.TestCase):
    def test_round_robin_local_fallback(self):
        api = FakeProvider('api', error='网络错误', primary=True)
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(RoundRobinTranslateStrategy,
                           translate=[api], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')

    def test_hybrid_local_fallback(self):
        api = FakeProvider('api', error='API错误', primary=True)
        ai = FakeProvider('ai', error='AI错误', primary=True,
                          provider_type='ai')
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(HybridStrategy,
                           translate=[api], ai=[ai],
                           local=[local],
                           options={'hybrid_order': 'ai_first'}).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')

    def test_single_ai_local_fallback(self):
        ai = FakeProvider('ai', error='AI错误', primary=True,
                          provider_type='ai')
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(SingleAIStrategy,
                           ai=[ai], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')

    def test_multi_ai_local_fallback(self):
        ai = FakeProvider('ai', error='AI错误', primary=True,
                          provider_type='ai')
        local = FakeProvider('local', '本地译文', local=True)
        result = _strategy(MultiAIVoteStrategy,
                           ai=[ai], local=[local]).translate(
                               'hello', 'en', 'zh')
        self.assertEqual(result, '本地译文')


class LocalOrderTests(unittest.TestCase):
    def test_selected_local_model_moves_first(self):
        a = FakeProvider('opus-mt-en-zh', local=True)
        b = FakeProvider('gemma3-translator-270m', local=True)
        ordered = _order_local_providers([a, b], 'gemma3-translator-270m')
        self.assertEqual([p.id for p in ordered],
                         ['gemma3-translator-270m', 'opus-mt-en-zh'])

    def test_order_keeps_fallback_providers(self):
        a = FakeProvider('opus-mt-en-zh', local=True)
        b = FakeProvider('gemma3-translator-270m', local=True)
        ordered = _order_local_providers([a, b], 'missing-model')
        self.assertEqual([p.id for p in ordered],
                         ['opus-mt-en-zh', 'gemma3-translator-270m'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
