"""5 种翻译策略"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher


def _similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class BaseStrategy:
    name = 'base'

    def __init__(self, translate_providers, ai_providers, options):
        self.translate_providers = translate_providers
        self.ai_providers = ai_providers
        self.options = options or {}
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def translate(self, text: str) -> str:
        raise NotImplementedError

    # --- 工具 ---
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


class SingleTranslateStrategy(BaseStrategy):
    name = '单翻译 API'

    def translate(self, text):
        p = self._primary_translate()
        if not p:
            return '[无翻译 API]'
        r = p.translate(text)
        if r.success and r.text.strip():
            return r.text
        # 兜底：尝试其他翻译 API
        for other in self.translate_providers:
            if other is p:
                continue
            r2 = other.translate(text)
            if r2.success and r2.text.strip():
                return r2.text
        return f'[翻译失败: {r.error}]'


class RoundRobinTranslateStrategy(BaseStrategy):
    name = '多翻译 API 轮询'

    def translate(self, text):
        n = len(self.translate_providers)
        if n == 0:
            return '[无翻译 API]'
        with self._rr_lock:
            start = self._rr_index % n
            self._rr_index += 1
        for i in range(n):
            p = self.translate_providers[(start + i) % n]
            r = p.translate(text)
            if r.success and r.text.strip():
                return r.text
        return '[所有翻译 API 失败]'


class HybridStrategy(BaseStrategy):
    name = '翻译 API + AI 混合'

    def translate(self, text):
        order = self.options.get('hybrid_order', 'translate_first')
        if order == 'translate_first':
            return self._translate_first(text)
        if order == 'ai_first':
            return self._ai_first(text)
        if order == 'concurrent':
            return self._concurrent(text)
        return self._translate_first(text)

    def _translate_first(self, text):
        p = self._primary_translate()
        if p:
            r = p.translate(text)
            if r.success and r.text.strip():
                return r.text
        ai = self._primary_ai()
        if ai:
            r = ai.translate(text)
            if r.success and r.text.strip():
                return r.text
        return '[混合翻译失败]'

    def _ai_first(self, text):
        ai = self._primary_ai()
        if ai:
            r = ai.translate(text)
            if r.success and r.text.strip():
                return r.text
        p = self._primary_translate()
        if p:
            r = p.translate(text)
            if r.success and r.text.strip():
                return r.text
        return '[混合翻译失败]'

    def _concurrent(self, text):
        candidates = []
        p1 = self._primary_translate()
        p2 = self._primary_ai()
        if p1:
            candidates.append(p1)
        if p2:
            candidates.append(p2)
        if not candidates:
            return '[无可用 API]'
        with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
            futures = [ex.submit(p.translate, text) for p in candidates]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r.success and r.text.strip():
                        return r.text
                except Exception:
                    continue
        return '[混合翻译失败]'


class SingleAIStrategy(BaseStrategy):
    name = '单 AI 翻译'

    def translate(self, text):
        ai = self._primary_ai()
        if not ai:
            return '[无 AI 模型]'
        r = ai.translate(text)
        if r.success and r.text.strip():
            return r.text
        for other in self.ai_providers:
            if other is ai:
                continue
            r2 = other.translate(text)
            if r2.success and r2.text.strip():
                return r2.text
        return f'[AI 翻译失败: {r.error}]'


class MultiAIVoteStrategy(BaseStrategy):
    """多 AI 高精度投票"""
    name = '多 AI 高精度投票'

    def translate(self, text):
        if not self.ai_providers:
            return '[无 AI 模型]'
        vote_count = max(1, int(self.options.get('vote_count', 3)))
        vote_strategy = self.options.get('vote_strategy', 'similarity')

        sources = self._collect_sources(vote_count)
        if not sources:
            return '[无可用 AI]'

        results = []
        with ThreadPoolExecutor(max_workers=min(len(sources), 6)) as ex:
            futures = [ex.submit(p.translate, text) for p in sources]
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r.success and r.text.strip():
                        results.append(r.text.strip())
                except Exception:
                    continue

        if not results:
            return '[所有 AI 翻译失败]'
        if len(results) == 1:
            return results[0]

        if vote_strategy == 'longest':
            return max(results, key=len)
        if vote_strategy == 'shortest':
            return min(results, key=len)

        # 默认：相似度居中
        best, best_score = results[0], -1.0
        for cand in results:
            scores = [_similarity(cand, o) for o in results if o is not cand]
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg > best_score:
                best_score, best = avg, cand
        return best

    def _collect_sources(self, count):
        enabled = [p for p in self.ai_providers if p.enabled]
        if not enabled:
            return []
        sources = list(enabled[:count])
        idx = 0
        while len(sources) < count:
            sources.append(enabled[idx % len(enabled)])
            idx += 1
        return sources


STRATEGY_REGISTRY = {
    'single_translate':    SingleTranslateStrategy,
    'multi_translate_rr':  RoundRobinTranslateStrategy,
    'hybrid':              HybridStrategy,
    'single_ai':           SingleAIStrategy,
    'multi_ai_vote':       MultiAIVoteStrategy,
}

STRATEGY_LABELS = {
    'single_translate':    '单翻译 API',
    'multi_translate_rr':  '多翻译 API 轮询',
    'hybrid':              '翻译 API + AI 混合',
    'single_ai':           '单 AI 翻译',
    'multi_ai_vote':       '多 AI 高精度投票',
}


def create_strategy(mode, translate_providers, ai_providers, options):
    cls = STRATEGY_REGISTRY.get(mode, SingleTranslateStrategy)
    return cls(translate_providers, ai_providers, options)