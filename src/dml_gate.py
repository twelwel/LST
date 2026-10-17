"""DirectML 推理闸门（进程级串行化）

背景
----
部分 Windows 驱动（典型：NVIDIA GeForce + nvwgf2umx.dll）在同一进程内
同时运行多个 DirectML 会话时，会触发原生访问违例（0xC0000005），
崩溃位置在驱动 / onnxruntime C++ 层，Python 无法 try/except 捕获，进程直接退出。
实测：OCR（RapidOCR/DML）与本地翻译模型（optimum/ORT + DmlExecutionProvider）
并发 Run 时 15~60 秒内必崩；所有 DML Run 串行后连续 2 分钟零崩溃。

用法
----
    from dml_gate import dml_guard, is_dml_provider

    if is_dml_provider(provider_name):
        with dml_guard():
            session.run(...)

约束
----
- 只用于 DirectML：CPU / CUDA 路径不要加锁，否则会拖慢多引擎并行。
- 可重入（RLock）：同一线程内嵌套获取不会死锁。
- 锁是「建议性」的：所有 DML 会话都必须经过它才有效。新增 DML 推理代码时
  务必包一层 dml_guard()，包括会话创建、预热和释放（析构会销毁 DML 资源）。
"""

import threading
import time

# 进程级可重入锁：所有 DirectML 会话共享
_lock = threading.RLock()

_stats_lock = threading.Lock()
_wait_count = 0
_wait_total = 0.0
_wait_max = 0.0

# 等待超过该阈值才打印提示，避免正常排队刷屏
SLOW_WAIT_WARN = 5.0


def is_dml_provider(provider_name) -> bool:
    """判断 provider 名称是否为 DirectML（兼容 'DmlExecutionProvider'）"""
    if not provider_name:
        return False
    return 'dml' in str(provider_name).lower()


class _DMLGuard:
    """dml_guard() 返回的上下文管理器"""

    __slots__ = ('_t0',)

    def __init__(self):
        self._t0 = 0.0

    def __enter__(self):
        global _wait_count, _wait_total, _wait_max
        self._t0 = time.perf_counter()
        _lock.acquire()
        waited = time.perf_counter() - self._t0
        with _stats_lock:
            _wait_count += 1
            _wait_total += waited
            if waited > _wait_max:
                _wait_max = waited
        if waited >= SLOW_WAIT_WARN:
            # 只在异常长的排队时提示，便于排查卡顿
            print(f"[DML] 推理闸门等待 {waited:.1f}s（已串行化，避免驱动崩溃）")
        return self

    def __exit__(self, exc_type, exc, tb):
        _lock.release()
        return False


def dml_guard() -> _DMLGuard:
    """获取/释放 DirectML 串行化锁（可重入，支持 with 语法）"""
    return _DMLGuard()


def dml_stats() -> dict:
    """闸门统计（调试用）：等待次数 / 总等待 / 最大等待"""
    with _stats_lock:
        return {
            'acquisitions': _wait_count,
            'total_wait_s': round(_wait_total, 3),
            'max_wait_s': round(_wait_max, 3),
        }


def reset_dml_stats():
    """重置统计（测试用）"""
    global _wait_count, _wait_total, _wait_max
    with _stats_lock:
        _wait_count = 0
        _wait_total = 0.0
        _wait_max = 0.0
