"""翻译管理器：统一入口 + 策略调度（线程安全）"""
import threading

from .api_config import (
    load_providers, load_strategy,
    sync_downloaded_models,
)
from .providers import create_provider
from .strategies import create_strategy, STRATEGY_LABELS


def _order_local_providers(providers, selected_model_id):
    """把用户在「本地模型」下拉框选中的模型排到最前面。

    sorted() 是稳定排序，其余本地模型仍保持配置中的顺序，
    这样选中模型不可用（语言不支持 / 加载失败）时还能自动兜底。
    """
    selected = str(selected_model_id or '').strip()
    if not selected:
        return list(providers)

    def _key(provider):
        model_id = str(
            (getattr(provider, 'params', {}) or {}).get('model_id', '')
        ).strip()
        return 0 if model_id == selected else 1

    return sorted(providers, key=_key)


class TranslationManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._providers = {}
        self._strategy = None
        self._strategy_mode = 'single_translate'
        self._strategy_cfg = {}
        self._mode_override = None
        self._first_load = True
        self.reload()

    # ==================== 重载 ====================

    def reload(self):
        """从配置文件重新读取 providers 和 strategy"""
        with self._lock:
            # 0. ⭐ 首次加载时，同步已下载的本地模型
            if self._first_load:
                self._first_load = False
                try:
                    sync_downloaded_models()
                except Exception as e:
                    print(f"[翻译管理] 首次同步失败：{e}")

            # 1. 构造新 providers
            try:
                cfgs = load_providers()
            except Exception as e:
                print(f"[翻译管理] 读取 providers 失败：{e}")
                cfgs = []

            new_providers = {}
            for cfg in cfgs:
                try:
                    p = create_provider(cfg)
                    new_providers[p.id] = p
                except Exception as e:
                    print(f"[翻译管理] Provider 创建失败："
                          f"{cfg.get('name')} → {e}")

            # 2. 读取策略
            try:
                strategy_cfg = load_strategy()
            except Exception:
                strategy_cfg = {'mode': 'single_translate'}

            configured_mode = strategy_cfg.get('mode', 'single_translate')
            # 运行时覆盖用于主界面的「本地流式」按钮：只影响当前进程，
            # 不写入 translation_strategy.json，退出按钮后立即恢复原策略。
            mode = self._mode_override or configured_mode
            # 远程策略默认不自动加载本地模型；本地翻译由独立按钮显式触发。
            # 「离线 + API」仍然按用户选择优先调用本地模型。
            strategy_cfg.setdefault('local_fallback', False)
            self._strategy_mode = mode
            self._strategy_cfg = dict(strategy_cfg)

            # 3. 分类（本地模型单独拎出）
            local_p = [p for p in new_providers.values()
                       if p.provider_type == 'translate'
                       and p.enabled
                       and getattr(p, 'is_local', False)]
            local_p = _order_local_providers(
                local_p, strategy_cfg.get('local_model_id', ''))

            translate_p = [p for p in new_providers.values()
                           if p.provider_type == 'translate'
                           and p.enabled
                           and not getattr(p, 'is_local', False)]

            ai_p = [p for p in new_providers.values()
                    if p.provider_type == 'ai' and p.enabled]

            # 4. 构造新策略
            try:
                new_strategy = create_strategy(
                    mode, translate_p, ai_p, strategy_cfg, local_p)
            except Exception as e:
                print(f"[翻译管理] 策略创建失败：{e}")
                fallback_mode = (mode if mode in STRATEGY_LABELS
                                 else 'local_api')
                new_strategy = create_strategy(
                    fallback_mode, translate_p, ai_p,
                    strategy_cfg, local_p)

            # 5. 原子替换
            self._providers = new_providers
            self._strategy = new_strategy

            print(f"[翻译管理] 重载完成：策略={mode}, "
                  f"本地={len(local_p)}, "
                  f"翻译API={len(translate_p)}, AI={len(ai_p)}")

    # ==================== 运行时模式 ====================

    def set_mode_override(self, mode: str) -> bool:
        """临时覆盖策略模式（不写入配置文件）。

        主界面「本地流式」按钮使用 local_only；退出按钮时调用
        clear_mode_override() 恢复用户原本选择的策略。
        """
        mode = str(mode or '').strip()
        if mode not in STRATEGY_LABELS:
            return False
        with self._lock:
            if self._mode_override == mode:
                return True
            self._mode_override = mode
        self.reload()
        return True

    def clear_mode_override(self) -> bool:
        """清除临时策略覆盖，恢复配置文件中的策略。"""
        with self._lock:
            if self._mode_override is None:
                return False
            self._mode_override = None
        self.reload()
        return True

    def get_mode_override(self):
        with self._lock:
            return self._mode_override

    # ==================== 本地模型预热 ====================

    def preload_local_model(self, model_id: str = ''):
        """在后台线程预热本地模型，返回 (ok, error)。

        优先使用主界面下拉框选中的模型；调用方负责把耗时操作放到线程中，
        避免加载模型（数秒到数十秒）阻塞 Tk 主线程。
        """
        model_id = str(model_id or '').strip()
        with self._lock:
            strategy_cfg = dict(self._strategy_cfg)
            selected = (model_id
                        or str(strategy_cfg.get('local_model_id') or '').strip())
            local_p = [p for p in self._providers.values()
                       if p.provider_type == 'translate'
                       and p.enabled
                       and getattr(p, 'is_local', False)]
            local_p = _order_local_providers(local_p, selected)
            if selected:
                local_p = [
                    p for p in local_p
                    if str((getattr(p, 'params', {}) or {})
                           .get('model_id', '')).strip() == selected
                ] or local_p
            provider = local_p[0] if local_p else None

        if provider is None:
            return False, '未选择或未注册本地模型'
        preload = getattr(provider, 'preload', None)
        if not callable(preload):
            return False, '当前本地模型不支持预热'
        try:
            return preload()
        except Exception as e:
            return False, str(e)

    # ==================== 查询 ====================

    def is_configured(self) -> bool:
        with self._lock:
            return any(p.enabled for p in self._providers.values())

    def translate(self, text: str,
                  source: str = 'auto', target: str = 'zh') -> str:
        with self._lock:
            strategy = self._strategy
        if strategy is None:
            return '[未配置翻译服务]'
        try:
            return strategy.translate(text, source=source, target=target)
        except Exception as e:
            print(f"[翻译管理] 策略执行异常：{e}")
            return f'[翻译异常: {e}]'

    def translate_lines(self, lines: list,
                        source: str = 'auto', target: str = 'zh') -> list:
        if not lines:
            return []
        with self._lock:
            strategy = self._strategy
        if strategy is None:
            return ['[未配置翻译服务]' for _ in lines]

        # 策略可以覆盖批量翻译（例如本地流式对同批重复文本去重）。
        batch = getattr(strategy, 'translate_lines', None)
        if callable(batch):
            try:
                result = batch(lines, source=source, target=target)
                if isinstance(result, list) and len(result) == len(lines):
                    return result
                print("[翻译管理] 批量策略返回长度异常，回退逐行翻译")
            except Exception as e:
                print(f"[翻译管理] 批量策略执行异常：{e}")

        return [self.translate(line, source, target) for line in lines]

    def get_strategy_name(self) -> str:
        with self._lock:
            return STRATEGY_LABELS.get(self._strategy_mode,
                                       self._strategy_mode)

    def get_strategy_mode(self) -> str:
        with self._lock:
            return self._strategy_mode

    def get_strategy_cfg(self) -> dict:
        with self._lock:
            return dict(self._strategy_cfg)

    def get_status_text(self) -> str:
        """返回当前策略的详细状态（用于 UI 显示）"""
        with self._lock:
            mode = self._strategy_mode
            translate_p = [p for p in self._providers.values()
                           if p.provider_type == 'translate'
                           and p.enabled
                           and not getattr(p, 'is_local', False)]
            local_p = [p for p in self._providers.values()
                       if p.provider_type == 'translate'
                       and p.enabled
                       and getattr(p, 'is_local', False)]
            ai_p = [p for p in self._providers.values()
                    if p.provider_type == 'ai' and p.enabled]

            parts = [STRATEGY_LABELS.get(mode, mode)]
            if local_p:
                parts.append(f"本地 {len(local_p)}")
            parts.append(f"翻译API {len(translate_p)}")
            parts.append(f"AI {len(ai_p)}")
            return " | ".join(parts)

    def list_providers(self):
        with self._lock:
            return list(self._providers.values())


_manager = None
_manager_lock = threading.Lock()


_manager = None
_manager_lock = threading.RLock()    # ⭐ 可重入


def get_manager() -> TranslationManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TranslationManager()
        return _manager