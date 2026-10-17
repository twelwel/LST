"""DirectML 显卡选择

问题背景
--------
onnxruntime-directml 在未显式指定 ``device_id`` 时固定使用 DXGI 适配器 0。
混合显卡笔记本上适配器 0 往往是 AMD/Intel 核显，而独显是 1：

    DML device 0 -> AMD Radeon(TM) Graphics (485 MB)
    DML device 1 -> NVIDIA GeForce RTX 3050 Ti Laptop GPU (3962 MB)

翻译模型的 FP32 权重约 650MB，放在核显上很容易显存不足，表现为
"DirectML 加载成功但推理崩溃 / 独显占用始终为 0"。

本模块通过 ``onnxruntime.get_ep_devices()`` 枚举真实 DML 适配器，
优先选择独立显卡 / 高性能适配器，并把 device_id 传给：

- optimum ORT 模型：``provider_options=[{"device_id": 1}]``
- RapidOCR：``EngineConfig.onnxruntime.dml_ep_cfg = {"device_id": 1}``

也可用环境变量 ``LST_DML_DEVICE_ID`` 强制指定。
"""

import os
import threading

from logger import get_logger

log = get_logger('dml_device')

_lock = threading.Lock()
_resolved = None
_devices_cache = None


def _parse_int(value, default=None):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _parse_vram_mb(value):
    """把 '3962 MB' / '4 GB' 解析为 MB。"""
    if value is None:
        return 0
    text = str(value).strip().upper()
    num = ''
    for ch in text:
        if ch.isdigit() or ch == '.':
            num += ch
        elif num:
            break
    try:
        val = float(num)
    except Exception:
        return 0
    if 'GB' in text:
        val *= 1024
    return int(val)


def list_dml_devices():
    """返回所有 DirectML 适配器信息（按 ORT 枚举顺序）。

    每项：device_id / name / discrete / vram_mb / high_perf
    获取失败时返回空列表。
    """
    global _devices_cache
    with _lock:
        if _devices_cache is not None:
            return list(_devices_cache)

    result = []
    try:
        import onnxruntime as ort
        get_ep_devices = getattr(ort, 'get_ep_devices', None)
        if get_ep_devices is None:
            return []
        for dev in get_ep_devices():
            if getattr(dev, 'ep_name', '') != 'DmlExecutionProvider':
                continue
            meta = dict(getattr(getattr(dev, 'device', None), 'metadata', {}) or {})
            ep_opts = dict(getattr(dev, 'ep_options', {}) or {})
            did = _parse_int(meta.get('DxgiAdapterNumber'),
                             _parse_int(ep_opts.get('device_id'), 0))
            result.append({
                'device_id': did,
                'name': meta.get('Description') or f'DML 设备 {did}',
                'discrete': str(meta.get('Discrete', '0')).strip() == '1',
                'vram_mb': _parse_vram_mb(meta.get('DxgiVideoMemory')),
                'high_perf': _parse_int(meta.get('DxgiHighPerformanceIndex'), 999),
            })
    except Exception as e:
        log.debug(f"[DML] 枚举显卡失败：{e}")
        return []

    result.sort(key=lambda d: d['device_id'])
    with _lock:
        _devices_cache = list(result)
    return list(result)


def resolve_dml_device():
    """选择 DirectML 设备，返回设备信息 dict。

    优先级：
    1. 环境变量 LST_DML_DEVICE_ID（显式覆盖）
    2. 独立显卡优先
    3. DXGI 高性能索引更小者优先
    4. 显存更大者优先
    """
    global _resolved
    with _lock:
        if _resolved is not None:
            return dict(_resolved)

    forced = os.environ.get('LST_DML_DEVICE_ID', '').strip()
    devices = list_dml_devices()

    chosen = None
    if forced != '':
        fid = _parse_int(forced)
        if fid is not None:
            for d in devices:
                if d['device_id'] == fid:
                    chosen = d
                    break
            if chosen is None:
                chosen = {
                    'device_id': fid,
                    'name': f'DML 设备 {fid}（环境变量指定）',
                    'discrete': False,
                    'vram_mb': 0,
                    'high_perf': 999,
                }

    if chosen is None and devices:
        chosen = min(devices, key=lambda d: (
            0 if d['discrete'] else 1,
            -d['vram_mb'],          # 显存大的优先（独显通常远大于核显）
            d['high_perf'],
            d['device_id'],
        ))

    if chosen is None:
        # 老版本 ORT 没有 get_ep_devices()：保持原行为
        chosen = {
            'device_id': 0,
            'name': 'DML 默认设备（无法枚举）',
            'discrete': False,
            'vram_mb': 0,
            'high_perf': 999,
        }

    with _lock:
        _resolved = dict(chosen)

    log.info(
        f"[DML] 选择显卡：{chosen['name']} "
        f"(device_id={chosen['device_id']}, "
        f"{'独显' if chosen['discrete'] else '核显/未知'}, "
        f"{chosen['vram_mb']} MB)"
    )
    return dict(chosen)


def get_dml_device_id() -> int:
    return int(resolve_dml_device()['device_id'])


def dml_provider_options():
    """optimum / onnxruntime 的 provider_options 列表。"""
    return [{'device_id': get_dml_device_id()}]


def dml_provider_option_dict() -> dict:
    """RapidOCR 的 dml_ep_cfg 字典。"""
    return {'device_id': get_dml_device_id()}


def describe_dml_device() -> str:
    d = resolve_dml_device()
    return f"{d['name']} (device_id={d['device_id']}, {d['vram_mb']} MB)"


def reset_cache():
    """测试用：清除缓存。"""
    global _resolved, _devices_cache
    with _lock:
        _resolved = None
        _devices_cache = None
