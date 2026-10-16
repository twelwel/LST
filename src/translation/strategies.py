"""翻译策略 —— 6 种模式（含请求节流）"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from logger import get_logger
from .throttle import get_limiter


log = get_logger('strategy')


def _similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _throttled_translate(provider, text, source, target):
    """带节流的 provider.translate 包装"""
    try:
        get_limiter().wait(provider.id)
    except Exception:
        pass
    return provider.translate(text, source, target)


class BaseStrategy:
    name = 'base'

    def __init__(self, translate_providers, ai_providers, options):
        self.translate_providers = list(translate_providers or [])
        self.ai_providers = list(ai_providers or [])
        self.options = options or {}
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def translate(self, text, source='auto', target='zh'):
        raise NotImplementedError

    def _primary_translate(self):
        for p in self.translate_providers:
            if p.is_primary:
                return p
        return self.translate_providers[0] if self.translate_providers else None

    def _primary_ai(self):
        for p in self.ai_providers:
            if p.is_primary:
                return p
        return self.ai_providers[0] if self.ai_providers else None

    def _has_translate(self):
        return bool(self.translate_providers)

    def _has_ai(self):
        return bool(self.ai_providers)


# ============ 离线 ============
class LocalOnlyStrategy(BaseStrategy):
    name = '离线'

    def translate(self, text, source='auto', target='zh'):
        return '[离线：本地未命中]'


# ============ 单翻译 API ============
class SingleTranslateStrategy(BaseStrategy):
    name = '单翻译 API'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_translate():
            return '[无翻译 API]'
        primary = self._primary_translate()
        r = _throttled_translate(primary, text, source, target)
        if r.success and r.text.strip():
            return r.text
        for other in self.translate_providers:
            if other is primary:
                continue
            r2 = _throttled_translate(other, text, source, target)
            if r2.success and r2.text.strip():
                return r2.text
        return f'[翻译失败 · {primary.name}: {r.error}]'


# ============ 多翻译 API 轮询 ============
class RoundRobinTranslateStrategy(BaseStrategy):
    name = '多翻译 API 轮询'

    def translate(self, text, source='auto', target='zh'):
        n = len(self.translate_providers)
        if n == 0:
            return '[无翻译 API]'
        with self._rr_lock:
            start = self._rr_index % n
            self._rr_index = (self._rr_index + 1) % max(1, n)
        last_err = ''
        for i in range(n):
            p = self.translate_providers[(start + i) % n]
            r = _throttled_translate(p, text, source, target)
            if r.success and r.text.strip():
                return r.text
            last_err = f'{p.name}: {r.error}'
        return f'[所有翻译 API 失败 · {last_err}]'


# ============ 翻译 API + AI 混合 ============
class HybridStrategy(BaseStrategy):
    name = '翻译 API + AI 混合'

    def translate(self, text, source='auto', target='zh'):
        order = self.options.get('hybrid_order', 'translate_first')
        if order == 'translate_first':
            return self._translate_first(text, source, target)
        if order == 'ai_first':
            return self._ai_first(text, source, target)
        if order == 'concurrent':
            return self._concurrent(text, source, target)
        return self._translate_first(text, source, target)

    def _translate_first(self, text, source, target):
        if self._has_translate():
            r = _throttled_translate(
                self._primary_translate(), text, source, target)
            if r.success and r.text.strip():
                return r.text
        if self._has_ai():
            r = _throttled_translate(
                self._primary_ai(), text, source, target)
            if r.success and r.text.strip():
                return r.text
        return '[混合翻译失败]'

    def _ai_first(self, text, source, target):
        if self._has_ai():
            r = _throttled_translate(
                self._primary_ai(), text, source, target)
            if r.success and r.text.strip():
                return r.text
        if self._has_translate():
            r = _throttled_translate(
                self._primary_translate(), text, source, target)
            if r.success and r.text.strip():
                return r.text
        return '[混合翻译失败]'

    def _concurrent(self, text, source, target):
        candidates = []
        if self._has_translate():
            candidates.append(self._primary_translate())
        if self._has_ai():
            candidates.append(self._primary_ai())
        if not candidates:
            return '[无可用 API]'

        ex = ThreadPoolExecutor(max_workers=len(candidates))
        try:
            futures = [ex.submit(_throttled_translate,
                                 p, text, source, target)
                       for p in candidates]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r.success and r.text.strip():
                        return r.text
                except Exception:
                    continue
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        return '[混合翻译失败]'


# ============ 单 AI ============
class SingleAIStrategy(BaseStrategy):
    name = '单 AI 翻译'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_ai():
            return '[无 AI 模型]'
        primary = self._primary_ai()
        r = _throttled_translate(primary, text, source, target)
        if r.success and r.text.strip():
            return r.text
        for other in self.ai_providers:
            if other is primary:
                continue
            r2 = _throttled_translate(other, text, source, target)
            if r2.success and r2.text.strip():
                return r2.text
        return f'[AI 失败 · {primary.name}: {r.error}]'


# ============ 多 AI 高精度投票 ============
class MultiAIVoteStrategy(BaseStrategy):
    name = '多 AI 高精度投票'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_ai():
            return '[无 AI 模型]'

        enabled = [p for p in self.ai_providers if p.enabled]

        if len(enabled) == 1:
            r = _throttled_translate(enabled[0], text, source, target)
            return r.text if r.success else f'[AI 失败: {r.error}]'

        vote_count = max(2, int(self.options.get('vote_count', 3)))
        vote_strategy = self.options.get('vote_strategy', 'similarity')

        sources = []
        idx = 0
        while len(sources) < vote_count:
            sources.append(enabled[idx % len(enabled)])
            idx += 1
            if idx > vote_count * 3:
                break

        results = []
        ex = ThreadPoolExecutor(max_workers=min(len(sources), 6))
        try:
            futures = [ex.submit(_throttled_translate,
                                 p, text, source, target)
                       for p in sources]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r.success and r.text.strip():
                        results.append(r.text.strip())
                except Exception:
                    continue
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        if not results:
            return '[所有 AI 翻译失败]'
        if len(results) == 1:
            return results[0]

        if vote_strategy == 'longest':
            return max(results, key=len)
        if vote_strategy == 'shortest':
            return min(results, key=len)

        best, best_score = results[0], -1.0
        for cand in results:
            scores = [_similarity(cand, o)
                      for o in results if o is not cand]
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg > best_score:
                best_score, best = avg, cand
        return best


STRATEGY_REGISTRY = {
    'local_only':          LocalOnlyStrategy,
    'single_translate':    SingleTranslateStrategy,
    'multi_translate_rr':  RoundRobinTranslateStrategy,
    'hybrid':              HybridStrategy,
    'single_ai':           SingleAIStrategy,
    'multi_ai_vote':       MultiAIVoteStrategy,
}

STRATEGY_LABELS = {
    'local_only':          '离线',
    'single_translate':    '单翻译 API',
    'multi_translate_rr':  '多翻译 API 轮询',
    'hybrid':              '翻译 API + AI 混合',
    'single_ai':           '单 AI 翻译',
    'multi_ai_vote':       '多 AI 高精度投票',
}

HYBRID_ORDER_LABELS = {
    'translate_first': '翻译优先',
    'ai_first':        'AI 优先',
    'concurrent':      '并发取快',
}

VOTE_STRATEGY_LABELS = {
    'similarity': '相似度居中',
    'longest':    '选最长',
    'shortest':   '选最短',
}


def create_strategy(mode, translate_providers, ai_providers, options):
    cls = STRATEGY_REGISTRY.get(mode, SingleTranslateStrategy)
    return cls(translate_providers, ai_providers, options)