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
    is_local = False            # ⭐ 是否是本地模型

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.id = cfg.get('id', '')
        self.name = cfg.get('name', 'Unnamed')
        self.is_primary = bool(cfg.get('is_primary', False))
        self.enabled = bool(cfg.get('enabled', True))
        self.params = cfg.get('config', {}) or {}

    @abstractmethod
    def translate(self, text: str,
                  from_lang: str = 'auto',
                  to_lang: str = 'zh') -> ProviderResult:
        ...

    def test(self):
        r = self.translate('Hello')
        return (True, r.text) if r.success else (False, r.error)

    # ---------- 子类共用工具（原先是每个适配器各复制一份）----------

    def _err(self, msg: str) -> ProviderResult:
        """构造失败结果"""
        return ProviderResult(False, error=msg,
                              provider_id=self.id,
                              provider_name=self.name)

    @staticmethod
    def _safe_float(v, default):
        try:
            if v is None or str(v).strip() == '':
                return default
            return float(v)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(v, default):
        try:
            if v is None or str(v).strip() == '':
                return default
            return int(v)
        except (ValueError, TypeError):
            return default