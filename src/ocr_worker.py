"""
OCR Worker 进程
用于 DirectML 多进程方案：
- 每个子进程独立创建一个 DML 引擎
- 通过 multiprocessing.Pool 分发任务
- 进程隔离规避 onnxruntime-directml 的多线程崩溃
"""

import gc

import numpy as np


# Worker 全局引擎（每个子进程独立一份）
_worker_engine = None
_worker_error = ''


def _worker_init():
    """
    Pool 初始化：每个子进程创建自己的 DML 引擎
    只有创建成功，worker 才会进入就绪状态
    """
    global _worker_engine, _worker_error
    _worker_error = ''

    try:
        from rapidocr import RapidOCR
        from dml_device import dml_provider_option_dict
        from ocr_engine import _patch_rapidocr_dml_options
        _patch_rapidocr_dml_options()

        params = {
            "EngineConfig.onnxruntime.use_dml": True,
            "EngineConfig.onnxruntime.dml_ep_cfg": dml_provider_option_dict(),
            "Det.use_dml": True,
            "Rec.use_dml": True,
            "Cls.use_dml": False,
        }

        engine = RapidOCR(params=params)

        # 预热
        dummy = np.zeros((32, 128, 3), dtype=np.uint8)
        _ = engine(dummy)

        _worker_engine = engine
        return True
    except Exception as e:
        _worker_engine = None
        _worker_error = str(e)
        return False


def worker_recognize(image_np):
    """
    Worker 端识别函数（供 Pool.map 调用）
    返回: [{'text': str, 'box': (x1, y1, x2, y2)}, ...]
    """
    global _worker_engine
    if _worker_engine is None:
        return []

    try:
        result = _worker_engine(image_np)
        if result is None or not result.txts:
            return []

        items = []
        for box, text in zip(result.boxes, result.txts):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            if text.strip():
                items.append({'text': text, 'box': (x1, y1, x2, y2)})
        return items
    except Exception as e:
        print(f"[OCR Worker] 识别错误：{e}")
        return []


def worker_ping(_=None):
    """健康检查：worker 是否就绪"""
    return _worker_engine is not None


def worker_cleanup():
    """Pool 关闭时的清理"""
    global _worker_engine
    _worker_engine = None
    try:
        gc.collect()
    except Exception:
        pass