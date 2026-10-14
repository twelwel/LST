"""翻译后端"""
from .manager import TranslationManager, get_manager
from .api_config import (
    load_providers, save_providers,
    load_strategy, save_strategy,
    new_provider_id, is_configured,
)

__all__ = [
    'TranslationManager', 'get_manager',
    'load_providers', 'save_providers',
    'load_strategy', 'save_strategy',
    'new_provider_id', 'is_configured',
]