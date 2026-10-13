"""OCR 引擎池
- 自动检测 onnxruntime 变体（CPU / DirectML / GPU）
- 支持多引擎并行（仅 CPU 模式）
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
        self.variant = 'cpu'
        self.last_error = ''

    # ================= 初始化 =================

    def init(self, pool_size=1):
        """
        初始化 OCR 引擎池
        根据当前安装的 onnxruntime 变体自动选择加速方式
        """
        self.last_error = ''
        self.engines = []
        self.pool_size = max(1, pool_size)

        # 检测变体
        try:
            self.variant = check_onnxruntime_variant() or 'cpu'
        except Exception:
            self.variant = 'cpu'

        # GPU 模式下先加载 CUDA DLL
        if self.variant == 'gpu':
            try:
                load_cuda_dlls()
            except Exception as e:
                print(f"[OCR] CUDA DLL 加载失败：{e}")

        # DirectML / GPU 只创建一个引擎（显存限制）
        if self.variant in ('gpu', 'directml'):
            self.pool_size = 1

        # 创建引擎
        try:
            self._init_engines()
            print(f"[OCR] 使用 {self.variant}，池大小 {len(self.engines)}")
            return self.variant
        except Exception as e:
            self.last_error = str(e)
            print(f"[OCR] 初始化失败：{e}，回退 CPU")
            # 回退 CPU
            self.variant = 'cpu'
            try:
                self._init_engines()
                return 'cpu'
            except Exception as e2:
                self.last_error = str(e2)
                self.engines = []
                return 'cpu'

    def _init_engines(self):
        from rapidocr import RapidOCR

        params = {}

        if self.variant == 'gpu':
            params["EngineConfig.onnxruntime.use_cuda"] = True
            params["EngineConfig.onnxruntime.cuda_ep_cfg."
                   "cudnn_conv_algo_search"] = "DEFAULT"
            params["Det.use_cuda"] = True
            params["Rec.use_cuda"] = True
            params["Cls.use_cuda"] = False

        elif self.variant == 'directml':
            params["EngineConfig.onnxruntime.use_dml"] = True
            params["Det.use_dml"] = True
            params["Rec.use_dml"] = True
            params["Cls.use_dml"] = False

        # 创建
        if self.variant in ('gpu', 'directml'):
            engine = RapidOCR(params=params)
            # 自检
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
        return f"{self.variant} × {len(self.engines)}"

    def check_gpu_available(self):
        """兼容旧接口：是否使用非 CPU 加速"""
        return self.variant in ('gpu', 'directml')