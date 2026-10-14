"""Provider 抽象"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResult:
    success: bool
    text: str = ''
    error: str = ''
    provider_id: str = ''
    provider_name: str = ''
    elapsed_ms: int = 0


class BaseProvider(ABC):
    provider_type = 'unknown'   # 'translate' | 'ai'

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.id = cfg.get('id', '')
        self.name = cfg.get('name', 'Unnamed')
        self.is_primary = bool(cfg.get('is_primary', False))
        self.enabled = bool(cfg.get('enabled', True))
        self.params = cfg.get('config', {}) or {}

    @abstractmethod
    def translate(self, text: str,
                  from_lang: str = 'en',
                  to_lang: str = 'zh') -> ProviderResult:
        ...

    def test(self):
        r = self.translate('Hello')
        return (True, r.text) if r.success else (False, r.error)