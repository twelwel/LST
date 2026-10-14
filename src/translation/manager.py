"""翻译管理器：统一入口 + 策略调度"""
import threading

from .api_config import (
    load_providers, load_strategy,
)
from .providers import create_provider
from .strategies import create_strategy, STRATEGY_LABELS


class TranslationManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._providers = {}
        self._strategy = None
        self._strategy_mode = 'single_translate'
        self.reload()

    def reload(self):
        with self._lock:
            cfgs = load_providers()
            self._providers = {}
            for cfg in cfgs:
                try:
                    p = create_provider(cfg)
                    self._providers[p.id] = p
                except Exception as e:
                    print(f"[翻译管理] Provider 创建失败："
                          f"{cfg.get('name')} → {e}")

            strategy_cfg = load_strategy()
            self._strategy_mode = strategy_cfg.get(
                'mode', 'single_translate')

            translate_p = [p for p in self._providers.values()
                           if p.provider_type == 'translate' and p.enabled]
            ai_p = [p for p in self._providers.values()
                    if p.provider_type == 'ai' and p.enabled]

            self._strategy = create_strategy(
                self._strategy_mode,
                translate_p, ai_p, strategy_cfg)

    def is_configured(self) -> bool:
        with self._lock:
            return any(p.enabled for p in self._providers.values())

    def translate(self, text: str) -> str:
        with self._lock:
            strategy = self._strategy
        if strategy is None:
            return '[未配置翻译服务]'
        return strategy.translate(text)

    def translate_lines(self, lines: list) -> list:
        return [self.translate(line) for line in lines]

    def get_strategy_name(self) -> str:
        return STRATEGY_LABELS.get(self._strategy_mode, self._strategy_mode)

    def list_providers(self):
        with self._lock:
            return list(self._providers.values())


_manager = None
_manager_lock = threading.Lock()


def get_manager() -> TranslationManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TranslationManager()
        return _manager