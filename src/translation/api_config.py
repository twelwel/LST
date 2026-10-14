"""翻译 API 配置存储（含旧配置迁移）"""
import os
import json
import uuid
import threading

CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
PROVIDERS_FILE = os.path.join(CONFIG_DIR, 'api_providers.json')
STRATEGY_FILE = os.path.join(CONFIG_DIR, 'translation_strategy.json')
LEGACY_CONFIG = os.path.join(CONFIG_DIR, 'config.json')

_lock = threading.Lock()


DEFAULT_PROVIDERS = [
    {
        "id": "baidu_default",
        "name": "百度翻译",
        "type": "translate",
        "provider": "baidu",
        "is_primary": True,
        "enabled": True,
        "config": {"appid": "", "secret_key": ""},
    },
]

DEFAULT_STRATEGY = {
    "mode": "single_translate",
    "hybrid_order": "translate_first",
    "vote_count": 3,
    "vote_strategy": "similarity",
}


def _ensure_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


def _migrate_legacy():
    """从旧 config.json 迁移百度密钥"""
    if os.path.exists(PROVIDERS_FILE):
        return
    if not os.path.exists(LEGACY_CONFIG):
        return
    try:
        with open(LEGACY_CONFIG, 'r', encoding='utf-8') as f:
            old = json.load(f)
    except Exception:
        return
    appid = (old.get('baidu_appid') or '').strip()
    key = (old.get('baidu_secret_key') or '').strip()
    if not (appid and key):
        return
    _ensure_dir()
    providers = [{
        "id": "baidu_legacy",
        "name": "百度翻译",
        "type": "translate",
        "provider": "baidu",
        "is_primary": True,
        "enabled": True,
        "config": {"appid": appid, "secret_key": key},
    }]
    with open(PROVIDERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(providers, f, ensure_ascii=False, indent=2)


def load_providers() -> list:
    with _lock:
        _migrate_legacy()
        if not os.path.exists(PROVIDERS_FILE):
            return [dict(p) for p in DEFAULT_PROVIDERS]
        try:
            with open(PROVIDERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
        return [dict(p) for p in DEFAULT_PROVIDERS]


def save_providers(providers: list) -> bool:
    with _lock:
        try:
            _ensure_dir()
            tmp = PROVIDERS_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(providers, f, ensure_ascii=False, indent=2)
            os.replace(tmp, PROVIDERS_FILE)
            return True
        except Exception as e:
            print(f"[API 配置] 保存失败：{e}")
            return False


def load_strategy() -> dict:
    with _lock:
        if not os.path.exists(STRATEGY_FILE):
            return dict(DEFAULT_STRATEGY)
        try:
            with open(STRATEGY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            result = dict(DEFAULT_STRATEGY)
            if isinstance(data, dict):
                result.update(data)
            return result
        except Exception:
            return dict(DEFAULT_STRATEGY)


def save_strategy(strategy: dict) -> bool:
    with _lock:
        try:
            _ensure_dir()
            tmp = STRATEGY_FILE + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STRATEGY_FILE)
            return True
        except Exception as e:
            print(f"[策略] 保存失败：{e}")
            return False


def new_provider_id() -> str:
    return uuid.uuid4().hex[:12]


def is_configured() -> bool:
    """是否至少有一个可用的 provider"""
    for p in load_providers():
        if not p.get('enabled', True):
            continue
        cfg = p.get('config', {}) or {}
        kind = p.get('provider', '')
        if kind == 'baidu' and cfg.get('appid') and cfg.get('secret_key'):
            return True
        if kind == 'openai_compat' and cfg.get('base_url') and cfg.get('model'):
            return True
        if kind == 'custom_rest' and cfg.get('url'):
            return True
    return False