"""OCR 引擎池
- DirectML 支持单 GPU 多引擎 / 多 GPU 并行
- 支持多线程 CPU 并行
"""

import numpy as np
import mss

from cuda_setup import (
    check_onnxruntime_variant,
    load_cuda_dlls,
)


class OCREngine:
    def __init__(self):
        self.engines = []
        self.pool_size = 1
        self.variant = 'cpu'      # 'cpu' / 'directml' / 'gpu'
        self.mode = 'cpu_multi'   # 'cpu_multi' / 'directml' / 'cuda'
        self.last_error = ''
        # 每个引擎对应的 GPU device_id（仅 directml 有意义）
        self.engine_device_ids = []
        # 本次生效的 gpu_engine_counts（列表）
        self.gpu_engine_counts = []

    # ================= 初始化 =================

    def init(self, mode='cpu_multi', pool_size=2, gpu_engine_counts=None):
        """
        初始化 OCR 引擎池
        mode:
          - 'cpu_multi'：多线程 CPU 并行
          - 'directml' ：DirectML GPU（支持单 GPU 多引擎 / 多 GPU 并行）
          - 'cuda'     ：NVIDIA CUDA 单引擎
        gpu_engine_counts:
          - list[int]，按 device_id 顺序的引擎数，如 [2, 1]
          - 也接受 dict {device_id: count}
          - None 时默认 [2]
        返回：实际使用的模式
        """
        self.last_error = ''
        self.engines = []
        self.engine_device_ids = []
        self.mode = mode

        # 归一化 gpu_engine_counts
        if mode == 'directml':
            self.gpu_engine_counts = self._normalize_gpu_counts(
                gpu_engine_counts)
        else:
            self.gpu_engine_counts = []

        # 检测 onnxruntime 变体
        try:
            self.variant = check_onnxruntime_variant() or 'cpu'
        except Exception:
            self.variant = 'cpu'

        if mode == 'directml' and self.variant != 'directml':
            self.last_error = "需要安装 onnxruntime-directml"
            mode = 'cpu_multi'
            self.mode = mode
        if mode == 'cuda' and self.variant != 'gpu':
            self.last_error = "需要安装 onnxruntime-gpu"
            mode = 'cpu_multi'
            self.mode = mode

        if mode == 'cuda':
            try:
                load_cuda_dlls()
            except Exception as e:
                print(f"[OCR] CUDA DLL 加载失败：{e}")

        # 计算 pool_size
        if mode == 'cuda':
            self.pool_size = 1
        elif mode == 'directml':
            self.pool_size = max(1, sum(self.gpu_engine_counts))
        else:
            self.pool_size = max(1, pool_size)

        try:
            self._init_engines()
            print(f"[OCR] 使用 {mode}，引擎数 {len(self.engines)}，"
                  f"设备分配 {self.engine_device_ids}")
            return mode
        except Exception as e:
            self.last_error = str(e)
            print(f"[OCR] 初始化失败：{e}")
            if mode != 'cpu_multi':
                self.mode = 'cpu_multi'
                self.pool_size = max(1, pool_size)
                self.engine_device_ids = []
                try:
                    self._init_engines()
                    return 'cpu_multi'
                except Exception as e2:
                    self.last_error = str(e2)
                    self.engines = []
            return 'cpu_multi'

    @staticmethod
    def _normalize_gpu_counts(raw):
        """把各种形式的入参归一化成 list[int]"""
        from config import (
            DIRECTML_ENGINE_COUNT_MIN,
            DIRECTML_ENGINE_COUNT_MAX,
            DIRECTML_ENGINE_COUNT_DEFAULT,
        )
        if raw is None:
            return [DIRECTML_ENGINE_COUNT_DEFAULT]

        if isinstance(raw, dict):
            max_id = max(raw.keys()) if raw else 0
            out = [0] * (max_id + 1)
            for k, v in raw.items():
                out[int(k)] = int(v)
        elif isinstance(raw, (list, tuple)):
            out = [int(x) for x in raw]
        else:
            out = [DIRECTML_ENGINE_COUNT_DEFAULT]

        # 裁剪
        out = [max(DIRECTML_ENGINE_COUNT_MIN,
                   min(DIRECTML_ENGINE_COUNT_MAX, x)) for x in out]

        # 至少有一个引擎
        if sum(out) < 1:
            out = [DIRECTML_ENGINE_COUNT_DEFAULT]
        return out

    def _init_engines(self):
        """按模式创建引擎"""
        from rapidocr import RapidOCR

        if self.mode == 'directml':
            self.engines = []
            self.engine_device_ids = []
            for device_id, count in enumerate(self.gpu_engine_counts):
                if count <= 0:
                    continue
                params = {
                    "EngineConfig.onnxruntime.use_dml": True,
                    # 指定 GPU（不同 RapidOCR 版本参数路径可能不同，
                    # 若无效请参考其文档 / 源码调整）
                    "EngineConfig.onnxruntime.dml_ep_cfg.device_id":
                        device_id,
                    "Det.use_dml": True,
                    "Rec.use_dml": True,
                    "Cls.use_dml": False,
                }
                for i in range(count):
                    try:
                        engine = RapidOCR(params=params)
                        dummy = np.zeros((32, 128, 3), dtype=np.uint8)
                        _ = engine(dummy)
                        self.engines.append(engine)
                        self.engine_device_ids.append(device_id)
                        print(f"[OCR] DirectML 设备 {device_id} "
                              f"引擎 {i+1}/{count} 就绪")
                    except Exception as e:
                        print(f"[OCR] DirectML 设备 {device_id} "
                              f"引擎 {i+1} 创建失败：{e}")
                        if not self.engines:
                            raise
                        break
            if not self.engines:
                raise RuntimeError("DirectML 引擎全部创建失败")

        elif self.mode == 'cuda':
            params = {
                "EngineConfig.onnxruntime.use_cuda": True,
                "EngineConfig.onnxruntime.cuda_ep_cfg."
                "cudnn_conv_algo_search": "DEFAULT",
                "Det.use_cuda": True,
                "Rec.use_cuda": True,
                "Cls.use_cuda": False,
            }
            engine = RapidOCR(params=params)
            dummy = np.zeros((32, 128, 3), dtype=np.uint8)
            _ = engine(dummy)
            self.engines = [engine]

        else:
            for _ in range(self.pool_size):
                self.engines.append(RapidOCR())

    # ================= 截图 =================

    def capture(self, region):
        with mss.mss() as sct:
            img = sct.grab(region)
            return np.array(img)[:, :, :3]

    # ================= 识别 =================

    def recognize(self, image_np):
        if not self.engines:
            return []
        return self._recognize_one(self.engines[0], image_np)

    def recognize_batch(self, images):
        if not self.engines or not images:
            return [[] for _ in images]

        n = len(self.engines)
        if n == 1 or len(images) == 1:
            return [self._recognize_one(self.engines[0], img)
                    for img in images]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {}
            for i, img in enumerate(images):
                engine = self.engines[i % n]
                fut = ex.submit(self._recognize_one, engine, img)
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[OCR] 并行识别错误: {e}")
                    results[idx] = []
        return results

    def _recognize_one(self, engine, image_np):
        try:
            result = engine(image_np)
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
            print(f"[OCR] 识别错误: {e}")
            return []

    # ================= 状态 =================

    def get_mode_text(self):
        if not self.engines:
            return "未初始化"
        n = len(self.engines)
        if self.mode == 'directml':
            detail = {}
            for d in self.engine_device_ids:
                detail[d] = detail.get(d, 0) + 1
            parts = [f"GPU{d}×{c}" for d, c in sorted(detail.items())]
            return f"DirectML [{', '.join(parts)}]"
        if self.mode == 'cuda':
            return 'CUDA'
        return f'CPU × {n}'

    def check_gpu_available(self):
        return self.mode in ('directml', 'cuda')