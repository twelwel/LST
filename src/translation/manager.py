"""翻译管理器：统一入口 + 策略调度（线程安全）"""
import threading

from .api_config import (
    load_providers, load_strategy,
    sync_downloaded_models,
)
from .providers import create_provider
from .strategies import create_strategy, STRATEGY_LABELS


class TranslationManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._providers = {}
        self._strategy = None
        self._strategy_mode = 'single_translate'
        self._strategy_cfg = {}
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

            mode = strategy_cfg.get('mode', 'single_translate')
            self._strategy_mode = mode
            self._strategy_cfg = dict(strategy_cfg)

            # 3. 分类（本地模型单独拎出）
            local_p = [p for p in new_providers.values()
                       if p.provider_type == 'translate'
                       and p.enabled
                       and getattr(p, 'is_local', False)]

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
                new_strategy = create_strategy(
                    'local_api', translate_p, ai_p, {}, local_p)

            # 5. 原子替换
            self._providers = new_providers
            self._strategy = new_strategy

            print(f"[翻译管理] 重载完成：策略={mode}, "
                  f"本地={len(local_p)}, "
                  f"翻译API={len(translate_p)}, AI={len(ai_p)}")

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