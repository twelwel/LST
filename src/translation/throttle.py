"""请求节流器：按 provider 独立限速

设计：
- 全局单例，所有策略共享
- 按 provider.id 独立计算（多 API 轮询才能真正提吞吐）
- 关闭时完全无开销
"""

import threading
import time


class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_call = {}        # provider_id -> timestamp
        self._enabled = False
        self._min_interval = 1.0    # 秒（默认 1 QPS）

    # -------- 配置 --------

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)

    def is_enabled(self):
        return self._enabled

    def set_qps(self, qps):
        """qps: 每秒请求数（每个 provider 独立）"""
        try:
            qps = float(qps)
        except Exception:
            return
        if qps <= 0:
            return
        self._min_interval = 1.0 / qps

    def get_qps(self):
        if self._min_interval <= 0:
            return 999
        return 1.0 / self._min_interval

    # -------- 调用 --------

    def wait(self, provider_id):
        if not self._enabled:
            return

        if not provider_id:
            provider_id = '__default__'

        with self._lock:
            now = time.time()
            last = self._last_call.get(provider_id, 0)
            delta = now - last
            if delta < self._min_interval:
                sleep_time = self._min_interval - delta
            else:
                sleep_time = 0
            # 预留时间戳（放在锁内，避免并发时多个线程同时通过）
            self._last_call[provider_id] = now + sleep_time

        if sleep_time > 0:
            time.sleep(sleep_time)

    def reset(self):
        with self._lock:
            self._last_call.clear()


_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter