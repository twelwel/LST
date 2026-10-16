"""翻译管理器：统一入口 + 策略调度（线程安全）"""
import threading

from .api_config import load_providers, load_strategy
from .providers import create_provider
from .strategies import create_strategy, STRATEGY_LABELS


class TranslationManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._providers = {}
        self._strategy = None
        self._strategy_mode = 'single_translate'
        self._strategy_cfg = {}
        self.reload()

    # ==================== 重载 ====================

    def reload(self):
        """从配置文件重新读取 providers 和 strategy"""
        with self._lock:
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

            # 3. 分类
            translate_p = [p for p in new_providers.values()
                           if p.provider_type == 'translate' and p.enabled]
            ai_p = [p for p in new_providers.values()
                    if p.provider_type == 'ai' and p.enabled]

            # 4. 构造新策略
            try:
                new_strategy = create_strategy(
                    mode, translate_p, ai_p, strategy_cfg)
            except Exception as e:
                print(f"[翻译管理] 策略创建失败：{e}")
                new_strategy = create_strategy(
                    'single_translate', translate_p, ai_p, {})

            # 5. 原子替换
            self._providers = new_providers
            self._strategy = new_strategy

            print(f"[翻译管理] 重载完成：策略={mode}, "
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
                           if p.provider_type == 'translate' and p.enabled]
            ai_p = [p for p in self._providers.values()
                    if p.provider_type == 'ai' and p.enabled]
            return (f"{STRATEGY_LABELS.get(mode, mode)} | "
                    f"翻译API {len(translate_p)} 个 | "
                    f"AI {len(ai_p)} 个")

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