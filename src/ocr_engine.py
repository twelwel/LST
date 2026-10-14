"""OCR 引擎池
- CPU：多引擎（1-8 可调），线程池并行
- DirectML：单引擎（多引擎会触发 onnxruntime 原生崩溃）
- CUDA：单引擎
"""

import gc
import time

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
        self.mode = 'cpu_multi'
        self.last_error = ''

    # ================= 初始化 =================

    def init(self, mode='cpu_multi', pool_size=2):
        """
        mode:
          - 'cpu_multi'：多线程 CPU（pool_size 个引擎并行）
          - 'directml' ：DirectML 单引擎
          - 'cuda'     ：CUDA 单引擎
        """
        self.last_error = ''
        self.engines = []
        self.mode = mode

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

        # 引擎数量
        if mode == 'directml':
            self.pool_size = 1          # DML 强制单引擎
        elif mode == 'cuda':
            self.pool_size = 1          # CUDA 单引擎
        else:
            self.pool_size = max(1, int(pool_size))

        try:
            self._init_engines()
            print(f"[OCR] 使用 {mode}，引擎数 {len(self.engines)}")
            return mode
        except Exception as e:
            self.last_error = str(e)
            print(f"[OCR] 初始化失败：{e}")
            self._cleanup_engines()
            if mode != 'cpu_multi':
                self.mode = 'cpu_multi'
                self.pool_size = max(1, int(pool_size))
                try:
                    self._init_engines()
                    return 'cpu_multi'
                except Exception as e2:
                    self.last_error = str(e2)
                    self._cleanup_engines()
            return 'cpu_multi'

    def _cleanup_engines(self):
        try:
            for e in self.engines:
                try:
                    del e
                except Exception:
                    pass
        except Exception:
            pass
        self.engines = []
        try:
            gc.collect()
        except Exception:
            pass

    # ================= 引擎创建 =================

    def _init_engines(self):
        from rapidocr import RapidOCR

        if self.mode == 'directml':
            self._init_engines_directml()
        elif self.mode == 'cuda':
            self._init_engines_cuda()
        else:
            self._init_engines_cpu()

    def _init_engines_cpu(self):
        """CPU：多引擎并行"""
        from rapidocr import RapidOCR
        print(f"[OCR] 创建 {self.pool_size} 个 CPU 引擎...")
        for i in range(self.pool_size):
            try:
                t0 = time.time()
                engine = RapidOCR()
                self.engines.append(engine)
                print(f"[OCR] CPU 引擎 {i+1}/{self.pool_size} 就绪 "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[OCR] CPU 引擎 {i+1} 创建失败：{e}")
                if not self.engines:
                    raise
        if not self.engines:
            raise RuntimeError("所有 CPU 引擎创建失败")

    def _init_engines_directml(self):
        """DirectML：单引擎"""
        from rapidocr import RapidOCR

        params = {
            "EngineConfig.onnxruntime.use_dml": True,
            "Det.use_dml": True,
            "Rec.use_dml": True,
            "Cls.use_dml": False,
        }

        print("[OCR] 创建 DirectML 单引擎...")
        t0 = time.time()
        engine = RapidOCR(params=params)
        print(f"[OCR] session 创建完成 ({time.time()-t0:.1f}s)，预热中...")

        dummy = np.zeros((32, 128, 3), dtype=np.uint8)
        t0 = time.time()
        _ = engine(dummy)
        print(f"[OCR] 预热完成 ({time.time()-t0:.1f}s)")

        self.engines = [engine]
        print("[OCR] DirectML 单引擎就绪")

    def _init_engines_cuda(self):
        """CUDA：单引擎"""
        from rapidocr import RapidOCR

        params = {
            "EngineConfig.onnxruntime.use_cuda": True,
            "EngineConfig.onnxruntime.cuda_ep_cfg."
            "cudnn_conv_algo_search": "DEFAULT",
            "Det.use_cuda": True,
            "Rec.use_cuda": True,
            "Cls.use_cuda": False,
        }

        print("[OCR] 创建 CUDA 单引擎...")
        t0 = time.time()
        engine = RapidOCR(params=params)
        print(f"[OCR] session 创建完成 ({time.time()-t0:.1f}s)，预热中...")

        dummy = np.zeros((32, 128, 3), dtype=np.uint8)
        t0 = time.time()
        _ = engine(dummy)
        print(f"[OCR] 预热完成 ({time.time()-t0:.1f}s)")

        self.engines = [engine]
        print("[OCR] CUDA 单引擎就绪")

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
        """
        - CPU：多引擎线程池并行
        - DirectML / CUDA：单引擎串行
        """
        if not self.engines or not images:
            return [[] for _ in images]

        n = len(self.engines)

        # 单引擎 / 单图：串行
        if n == 1 or len(images) == 1:
            engine = self.engines[0]
            return [self._recognize_one(engine, img) for img in images]

        # CPU 多引擎：线程池并行
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
            return 'DirectML（单引擎）'
        if self.mode == 'cuda':
            return 'CUDA（单引擎）'
        return f'CPU × {n}'

    def check_gpu_available(self):
        return self.mode in ('directml', 'cuda')