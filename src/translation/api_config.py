"""翻译 API 配置存储（含旧配置迁移 + 本地模型自动同步）"""

import os
import json
import uuid
import threading


CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
PROVIDERS_FILE = os.path.join(CONFIG_DIR, 'api_providers.json')
STRATEGY_FILE = os.path.join(CONFIG_DIR, 'translation_strategy.json')
LEGACY_CONFIG = os.path.join(CONFIG_DIR, 'config.json')

_lock = threading.RLock()              # ⭐ 必须 RLock
_sync_in_progress = False


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
    # 主界面「本地模型」下拉框选中的模型；空 = 使用首个可用本地模型
    "local_model_id": "",
    # 远程策略失败后是否自动回退本地模型。
    # 默认关闭，本地翻译由主界面「本地流式」独立按钮显式触发。
    "local_fallback": False,
}


def _ensure_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


def _migrate_legacy():
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
            if isinstance(data, list):
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
    for p in load_providers():
        if not p.get('enabled', True):
            continue
        cfg = p.get('config', {}) or {}
        kind = p.get('provider', '')
        if kind == 'baidu' and cfg.get('appid') and cfg.get('secret_key'):
            return True
        if kind in ('openai_compat', 'openai_compat_ai'):
            if cfg.get('base_url') and cfg.get('model'):
                return True
        if kind == 'local_model' and cfg.get('model_id'):
            return True
        if kind == 'custom_rest' and cfg.get('url'):
            return True
    return False


# ==================== 本地模型注册 ====================

def ensure_local_provider(model_id: str,
                          model_name: str = '') -> bool:
    """确保本地模型已注册为 Provider（不触发 reload）"""
    if not model_id:
        return False

    with _lock:
        providers = load_providers()

        for p in providers:
            if p.get('provider') == 'local_model':
                cfg = p.get('config', {}) or {}
                if cfg.get('model_id') == model_id:
                    # 手动配置里可能把本地模型关闭了；从下拉框选中时应恢复启用。
                    changed = False
                    if not p.get('enabled', True):
                        p['enabled'] = True
                        changed = True
                    if p.get('type') != 'translate':
                        p['type'] = 'translate'
                        changed = True
                    if changed:
                        return save_providers(providers)
                    return True

        has_translate = any(
            p.get('type') == 'translate'
            and p.get('enabled', True)
            and p.get('provider') != 'local_model'
            for p in providers
        )

        new_p = {
            'id': new_provider_id(),
            'name': f'本地模型 · {model_name or model_id}',
            'type': 'translate',
            'provider': 'local_model',
            'is_primary': not has_translate,
            'enabled': True,
            'config': {'model_id': model_id},
        }
        providers.append(new_p)

        return save_providers(providers)


def select_local_model(model_id: str, model_name: str = '') -> bool:
    """主界面下拉框选中本地模型：注册 Provider 并持久化选择。"""
    model_id = (model_id or '').strip()
    if not model_id:
        return False

    try:
        ensure_local_provider(model_id, model_name)
    except Exception as e:
        print(f"[本地模型] 注册 Provider 失败：{e}")

    with _lock:
        strategy = load_strategy()
        if str(strategy.get('local_model_id') or '').strip() == model_id:
            return True
        strategy['local_model_id'] = model_id
        return save_strategy(strategy)


def clear_local_model_selection(model_id: str = '') -> bool:
    """清空本地模型选择；指定 model_id 时仅在匹配时清空。"""
    with _lock:
        strategy = load_strategy()
        selected = str(strategy.get('local_model_id') or '').strip()
        if not selected:
            return False
        if model_id and selected != (model_id or '').strip():
            return False
        strategy['local_model_id'] = ''
        return save_strategy(strategy)


def remove_local_provider(model_id: str) -> bool:
    """移除某个本地模型 Provider（不触发 reload）"""
    if not model_id:
        return False

    with _lock:
        providers = load_providers()
        new_list = []
        removed = False
        for p in providers:
            if p.get('provider') == 'local_model':
                cfg = p.get('config', {}) or {}
                if cfg.get('model_id') == model_id:
                    removed = True
                    continue
            new_list.append(p)

        if not removed:
            return False

        saved = save_providers(new_list)
        if saved:
            # 删除 Provider 时同步清空选择，避免策略继续指向已移除的模型。
            clear_local_model_selection(model_id)
        return saved


# ==================== 自动扫描同步 ====================

def sync_downloaded_models() -> dict:
    """
    扫描 models/ 目录，自动同步已下载模型为 Provider
    ⚠️ 不触发 manager.reload()，由调用方负责
    """
    global _sync_in_progress

    if _sync_in_progress:
        return {'added': [], 'removed': [], 'existing': []}

    _sync_in_progress = True
    try:
        return _do_sync_downloaded_models()
    finally:
        _sync_in_progress = False


def _do_sync_downloaded_models() -> dict:
    result = {'added': [], 'removed': [], 'existing': []}

    # ---- 1. 扫描已下载的内置模型（全部候选目录 + 校验真实权重）----
    try:
        from .local_models.downloader import list_downloaded_ids
        from .local_models.registry import MODEL_REGISTRY
        downloaded = list_downloaded_ids()
        builtin_ids = set(MODEL_REGISTRY)
    except Exception as e:
        print(f"[同步] 扫描失败：{e}")
        return result

    # ---- 2. 读取现有 providers ----
    with _lock:
        providers = load_providers()
        existing_local = {}

        for p in providers:
            if p.get('provider') == 'local_model':
                cfg = p.get('config', {}) or {}
                mid = (cfg.get('model_id') or '').strip()
                if mid:
                    existing_local[mid] = p

        need_save = False

        # ---- 3. 新增 ----
        for mid in downloaded:
            if mid in existing_local:
                result['existing'].append(mid)
                continue

            try:
                from .local_models.registry import get_registry_entry
                meta = get_registry_entry(mid)
                name = meta.get('name', mid) if meta else mid
            except Exception:
                name = mid

            has_other = any(
                p.get('type') == 'translate'
                and p.get('enabled', True)
                and p.get('provider') != 'local_model'
                for p in providers
            )

            new_p = {
                'id': new_provider_id(),
                'name': f'本地模型 · {name}',
                'type': 'translate',
                'provider': 'local_model',
                'is_primary': not has_other,
                'enabled': True,
                'config': {'model_id': mid},
            }
            providers.append(new_p)
            existing_local[mid] = new_p
            result['added'].append(mid)
            need_save = True

        # ---- 4. 移除 ----
        # 只清理「内置模型」中已经不存在的 Provider。
        # 自定义模型的 id 不在 MODEL_REGISTRY 里，不能因为扫描不到就删掉。
        to_remove = [
            mid for mid in existing_local
            if mid in builtin_ids and mid not in downloaded
        ]

        if to_remove:
            providers = [
                p for p in providers
                if not (
                    p.get('provider') == 'local_model'
                    and (p.get('config', {}) or {}).get('model_id')
                    in to_remove
                )
            ]
            result['removed'] = to_remove
            need_save = True

            # 删除的正好是当前选中模型时，清空选择，避免下拉框指向幽灵模型。
            try:
                strategy = load_strategy()
                selected = str(strategy.get('local_model_id') or '').strip()
                if selected in to_remove:
                    strategy['local_model_id'] = ''
                    save_strategy(strategy)
            except Exception:
                pass

        # ---- 5. 保存 ----
        if need_save:
            save_providers(providers)

    return result