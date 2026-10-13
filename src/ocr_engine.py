"""OCR 引擎池
- 支持 GPU/CPU 切换
- 支持多引擎并行（多线程 OCR）
"""

import threading
import numpy as np
import mss


class OCREngine:
    def __init__(self):
        self.engines = []          # 引擎池
        self.pool_size = 1         # 池大小
        self.use_gpu = False
        self.last_error = ''

    # ================= 初始化 =================

    def init(self, use_gpu=False, pool_size=1):
        """
        初始化 OCR 引擎池
        返回：实际模式（'gpu' 或 'cpu'）
        """
        self.last_error = ''
        self.engines = []

        # GPU 模式下池大小强制为 1（避免显存翻倍）
        if use_gpu and pool_size > 1:
            print(f"[OCR] GPU 模式下池大小自动降为 1（避免显存溢出）")
            pool_size = 1
        self.pool_size = max(1, pool_size)

        # 尝试 GPU
        if use_gpu:
            ok = self._init_engines(use_gpu=True)
            if ok:
                self.use_gpu = True
                print(f"[OCR] ✅ 使用 GPU（池大小 {len(self.engines)}）")
                return 'gpu'
            else:
                print(f"[OCR] ⚠️ GPU 失败：{self.last_error}，回退 CPU")

        # CPU
        self._init_engines(use_gpu=False)
        self.use_gpu = False
        print(f"[OCR] 使用 CPU（池大小 {len(self.engines)}）")
        return 'cpu'

    def _init_engines(self, use_gpu):
        """尝试创建 pool_size 个引擎"""
        try:
            from rapidocr import RapidOCR
            if use_gpu:
                from config import (
                    GPU_USE_CUDA, GPU_CUDNN_SEARCH, GPU_CLS_ON_CPU,
                )
                params = {}
                params["EngineConfig.onnxruntime.use_cuda"] = GPU_USE_CUDA
                params["EngineConfig.onnxruntime.cuda_ep_cfg."
                       "cudnn_conv_algo_search"] = GPU_CUDNN_SEARCH
                params["Det.use_cuda"] = True
                params["Rec.use_cuda"] = True
                params["Cls.use_cuda"] = not GPU_CLS_ON_CPU
                engine = RapidOCR(params=params)
                # GPU 只需一个
                dummy = np.zeros((32, 128, 3), dtype=np.uint8)
                _ = engine(dummy)
                self.engines = [engine]
                return True
            else:
                for _ in range(self.pool_size):
                    self.engines.append(RapidOCR())
                return True
        except Exception as e:
            self.last_error = str(e)
            self.engines = []
            return False

    # ================= 截图 =================

    def capture(self, region):
        with mss.mss() as sct:
            img = sct.grab(region)
            return np.array(img)[:, :, :3]

    # ================= 识别 =================

    def recognize(self, image_np):
        """单张识别（用第一个引擎）"""
        if not self.engines:
            return []
        return self._recognize_one(self.engines[0], image_np)

    def recognize_batch(self, images):
        """
        批量并行识别
        - pool_size==1 或单张：顺序处理
        - pool_size>1：多线程并行
        返回：list of results（顺序与输入对应）
        """
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
        """单引擎识别"""
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
        m = "GPU" if self.use_gpu else "CPU"
        return f"{m} × {len(self.engines)}"

    def check_gpu_available(self):
        try:
            import onnxruntime as ort
            return 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            return False