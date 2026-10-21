"""翻译策略 —— 7 种模式

层级：
    本地模型（离线） → 翻译 API → AI
"""

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

    def __init__(self, translate_providers, ai_providers,
                 options, local_providers=None):
        self.translate_providers = list(translate_providers or [])
        self.ai_providers = list(ai_providers or [])
        self.local_providers = list(local_providers or [])
        self.options = options or {}
        self._rr_lock = threading.Lock()
        self._rr_index = 0

    def translate(self, text, source='auto', target='zh'):
        raise NotImplementedError

    def translate_lines(self, lines, source='auto', target='zh'):
        """默认逐行翻译；单行异常不应影响同批其他文本。"""
        results = []
        for line in (lines or []):
            try:
                results.append(self.translate(line, source, target))
            except Exception as e:
                results.append(f'[翻译异常: {e}]')
        return results

    # ---------- 本地模型 ----------

    def _try_local_detail(self, text, source, target):
        """
        尝试所有本地模型，返回 (result_or_None, last_error_str)
        """
        last_err = ''
        for lp in self.local_providers:
            try:
                r = _throttled_translate(lp, text, source, target)
                if r.success and r.text and r.text.strip():
                    return r.text, ''
                last_err = r.error or '返回空结果'
            except Exception as e:
                last_err = str(e)
        return None, last_err

    def _try_local(self, text, source, target):
        """尝试所有本地模型，命中返回译文，否则 None"""
        result, _ = self._try_local_detail(text, source, target)
        return result

    def _has_local(self):
        return bool(self.local_providers)

    def _can_fallback_local(self):
        """远程策略是否允许自动回退本地模型。

        默认关闭：本地翻译由主界面「本地流式」按钮显式进入。
        「离线 + API」不经过这里，仍由用户主动选择本地优先。
        """
        return (self._has_local()
                and bool(self.options.get('local_fallback', True)))

    def _local_or(self, text, source, target, fallback):
        """尝试本地模型；全部失败时返回带本地原因的 fallback。"""
        if not self._can_fallback_local():
            return fallback
        result, err = self._try_local_detail(text, source, target)
        if result:
            return result
        if err:
            return f'{fallback}；本地模型：{err}'
        return fallback

    # ---------- 翻译 API ----------

    def _primary_translate(self):
        for p in self.translate_providers:
            if p.is_primary:
                return p
        return self.translate_providers[0] if self.translate_providers else None

    def _has_translate(self):
        return bool(self.translate_providers)

    # ---------- AI ----------

    def _primary_ai(self):
        for p in self.ai_providers:
            if p.is_primary:
                return p
        return self.ai_providers[0] if self.ai_providers else None

    def _has_ai(self):
        return bool(self.ai_providers)


# ============ 离线（只用本地模型） ============
class LocalOnlyStrategy(BaseStrategy):
    name = '离线'

    def translate(self, text, source='auto', target='zh'):
        # ⭐ 无本地模型时给明确引导
        if not self._has_local():
            return ('[离线：未配置本地模型。'
                    '请到「本地模型」下拉框选择并下载模型]')

        result, err = self._try_local_detail(text, source, target)
        if result:
            return result

        # ⭐ 显示真实错误原因（而不是笼统的"未命中"）
        if err:
            return f'[本地模型失败：{err}]'
        return '[离线：本地模型未命中]'

    def translate_lines(self, lines, source='auto', target='zh'):
        """本地流式批量翻译。

        跳过空行、同批去重，并**优先走模型的批量推理**（一次 generate 多条）；
        批量失败或个别失败时逐条兜底。
        """
        if not lines:
            return []
        results = [''] * len(lines)

        # 1) 收集需要翻译的唯一条目（保持首次出现顺序）
        uniq = {}
        for line in lines:
            text = (line or '').strip()
            if text and text not in uniq:
                uniq[text] = None
        if not uniq:
            return results

        texts = list(uniq.keys())

        # 2) 优先批量推理（本地 provider 才提供 translate_batch）
        for lp in self.local_providers:
            batch_fn = getattr(lp, 'translate_batch', None)
            if batch_fn is None:
                continue
            try:
                rs = batch_fn(texts, source, target)
            except Exception as e:
                log.warning(f"[离线] 批量翻译失败，回退逐条：{e}")
                continue
            if not rs or len(rs) != len(texts):
                continue
            for t, r in zip(texts, rs):
                if getattr(r, 'success', False) and r.text and r.text.strip():
                    uniq[t] = r.text
            if all(v is not None for v in uniq.values()):
                break

        # 3) 未命中的逐条兜底（保留原有的错误提示）
        for t in texts:
            if uniq[t] is None:
                try:
                    uniq[t] = self.translate(t, source, target)
                except Exception as e:
                    uniq[t] = f'[翻译异常: {e}]'

        for i, line in enumerate(lines):
            text = (line or '').strip()
            if text:
                results[i] = uniq.get(text, '')
        return results


# ============ 离线 + API（先本地，未命中走 API） ============
class LocalAPIStrategy(BaseStrategy):
    name = '离线 + API'

    def translate(self, text, source='auto', target='zh'):
        # 1. 先本地模型
        result, local_err = self._try_local_detail(text, source, target)
        if result:
            return result

        # 2. 未命中 → 走翻译 API
        api_err = ''
        if self._has_translate():
            primary = self._primary_translate()
            r = _throttled_translate(primary, text, source, target)
            if r.success and r.text.strip():
                return r.text
            api_err = f'{primary.name}: {r.error}'
            # 兜底其他 API
            for other in self.translate_providers:
                if other is primary:
                    continue
                r2 = _throttled_translate(other, text, source, target)
                if r2.success and r2.text.strip():
                    return r2.text
                api_err += f'；{other.name}: {r2.error}'

        # 3. 最后尝试 AI（避免本地模型或翻译 API 故障时整条链路失效）
        ai_err = ''
        if self._has_ai():
            for p in self.ai_providers:
                r = _throttled_translate(p, text, source, target)
                if r.success and r.text.strip():
                    return r.text
                if not ai_err:
                    ai_err = f'{p.name}: {r.error}'

        details = []
        if local_err:
            details.append(f'本地 {local_err}')
        if api_err:
            details.append(f'API {api_err}')
        if ai_err:
            details.append(f'AI {ai_err}')
        if details:
            return '[离线 + API 失败：' + '；'.join(details) + ']'
        return '[未配置本地模型和 API]'


# ============ 单翻译 API ============
class SingleTranslateStrategy(BaseStrategy):
    name = '单翻译 API'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_translate():
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无翻译 API]')
            return '[无翻译 API]'

        primary = self._primary_translate()
        r = _throttled_translate(primary, text, source, target)
        if r.success and r.text.strip():
            return r.text

        last_err = f'{primary.name}: {r.error}'
        for other in self.translate_providers:
            if other is primary:
                continue
            r2 = _throttled_translate(other, text, source, target)
            if r2.success and r2.text.strip():
                return r2.text
            last_err += f'；{other.name}: {r2.error}'

        # ⭐ API 全部失败时才回退本地模型。
        #    本地模型加载失败不会影响 API 正常路径，也不会让单 API 策略失效。
        if self._has_local():
            return self._local_or(
                text, source, target, f'[翻译失败 · {last_err}]')
        return f'[翻译失败 · {last_err}]'


# ============ 多翻译 API 轮询 ============
class RoundRobinTranslateStrategy(BaseStrategy):
    name = '多翻译 API 轮询'

    def translate(self, text, source='auto', target='zh'):
        n = len(self.translate_providers)
        if n == 0:
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无翻译 API]')
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

        if self._has_local():
            return self._local_or(
                text, source, target,
                f'[所有翻译 API 失败 · {last_err}]')
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
        last_err = ''
        if self._has_translate():
            p = self._primary_translate()
            r = _throttled_translate(p, text, source, target)
            if r.success and r.text.strip():
                return r.text
            last_err = f'{p.name}: {r.error}'
        if self._has_ai():
            p = self._primary_ai()
            r = _throttled_translate(p, text, source, target)
            if r.success and r.text.strip():
                return r.text
            last_err += f'；{p.name}: {r.error}'
        if self._has_local():
            return self._local_or(
                text, source, target, f'[混合翻译失败 · {last_err}]')
        return f'[混合翻译失败 · {last_err}]' if last_err else '[混合翻译失败]'

    def _ai_first(self, text, source, target):
        last_err = ''
        if self._has_ai():
            p = self._primary_ai()
            r = _throttled_translate(p, text, source, target)
            if r.success and r.text.strip():
                return r.text
            last_err = f'{p.name}: {r.error}'
        if self._has_translate():
            p = self._primary_translate()
            r = _throttled_translate(p, text, source, target)
            if r.success and r.text.strip():
                return r.text
            last_err += f'；{p.name}: {r.error}'
        if self._has_local():
            return self._local_or(
                text, source, target, f'[混合翻译失败 · {last_err}]')
        return f'[混合翻译失败 · {last_err}]' if last_err else '[混合翻译失败]'

    def _concurrent(self, text, source, target):
        candidates = []
        if self._has_translate():
            candidates.append(self._primary_translate())
        if self._has_ai():
            candidates.append(self._primary_ai())
        if not candidates:
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无可用 API]')
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
        if self._has_local():
            return self._local_or(
                text, source, target, '[混合翻译失败]')
        return '[混合翻译失败]'


# ============ 单 AI ============
class SingleAIStrategy(BaseStrategy):
    name = '单 AI 翻译'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_ai():
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无 AI 模型]')
            return '[无 AI 模型]'
        primary = self._primary_ai()
        r = _throttled_translate(primary, text, source, target)
        if r.success and r.text.strip():
            return r.text
        last_err = f'{primary.name}: {r.error}'
        for other in self.ai_providers:
            if other is primary:
                continue
            r2 = _throttled_translate(other, text, source, target)
            if r2.success and r2.text.strip():
                return r2.text
            last_err += f'；{other.name}: {r2.error}'
        if self._has_local():
            return self._local_or(
                text, source, target, f'[AI 失败 · {last_err}]')
        return f'[AI 失败 · {last_err}]'


# ============ 多 AI 高精度投票 ============
class MultiAIVoteStrategy(BaseStrategy):
    name = '多 AI 高精度投票'

    def translate(self, text, source='auto', target='zh'):
        if not self._has_ai():
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无 AI 模型]')
            return '[无 AI 模型]'

        enabled = [p for p in self.ai_providers if p.enabled]
        if not enabled:
            if self._has_local():
                return self._local_or(
                    text, source, target, '[无 AI 模型]')
            return '[无 AI 模型]'

        if len(enabled) == 1:
            r = _throttled_translate(enabled[0], text, source, target)
            if r.success and r.text.strip():
                return r.text
            if self._has_local():
                return self._local_or(
                    text, source, target, f'[AI 失败: {r.error}]')
            return f'[AI 失败: {r.error}]'

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
            if self._has_local():
                return self._local_or(
                    text, source, target, '[所有 AI 翻译失败]')
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
    'local_api':           LocalAPIStrategy,
    'single_translate':    SingleTranslateStrategy,
    'multi_translate_rr':  RoundRobinTranslateStrategy,
    'hybrid':              HybridStrategy,
    'single_ai':           SingleAIStrategy,
    'multi_ai_vote':       MultiAIVoteStrategy,
}

STRATEGY_LABELS = {
    'local_only':          '离线',
    'local_api':           '离线 + API',
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


def create_strategy(mode, translate_providers, ai_providers,
                    options, local_providers=None):
    cls = STRATEGY_REGISTRY.get(mode, SingleTranslateStrategy)
    return cls(translate_providers, ai_providers,
               options, local_providers)