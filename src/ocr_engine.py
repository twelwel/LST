"""OCR 引擎池
- DirectML / CPU 双模式切换（同一个包，无需重装）
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

    # ================= 初始化 =================

    def init(self, mode='cpu_multi', pool_size=2):
        """
        初始化 OCR 引擎池
        mode:
          - 'cpu_multi'：多线程 CPU 并行
          - 'directml' ：DirectML GPU 单引擎
          - 'cuda'     ：NVIDIA CUDA 单引擎（需 onnxruntime-gpu）
        返回：实际使用的模式
        """
        self.last_error = ''
        self.engines = []
        self.mode = mode

        # 检测已装的 onnxruntime 变体
        try:
            self.variant = check_onnxruntime_variant() or 'cpu'
        except Exception:
            self.variant = 'cpu'

        # 校验：DirectML 模式需要 directml 包
        if mode == 'directml' and self.variant != 'directml':
            self.last_error = "需要安装 onnxruntime-directml"
            mode = 'cpu_multi'
            self.mode = mode

        # 校验：CUDA 模式需要 gpu 包
        if mode == 'cuda' and self.variant != 'gpu':
            self.last_error = "需要安装 onnxruntime-gpu"
            mode = 'cpu_multi'
            self.mode = mode

        # CUDA 模式加载 DLL
        if mode == 'cuda':
            try:
                load_cuda_dlls()
            except Exception as e:
                print(f"[OCR] CUDA DLL 加载失败：{e}")

        # 引擎池大小
        if mode in ('directml', 'cuda'):
            self.pool_size = 1
        else:
            self.pool_size = max(1, pool_size)

        try:
            self._init_engines()
            print(f"[OCR] 使用 {mode}，引擎数 {len(self.engines)}")
            return mode
        except Exception as e:
            self.last_error = str(e)
            print(f"[OCR] 初始化失败：{e}")
            if mode != 'cpu_multi':
                self.mode = 'cpu_multi'
                self.pool_size = max(1, pool_size)
                try:
                    self._init_engines()
                    return 'cpu_multi'
                except Exception as e2:
                    self.last_error = str(e2)
                    self.engines = []
            return 'cpu_multi'

    def _init_engines(self):
        """按模式创建引擎"""
        from rapidocr import RapidOCR

        if self.mode == 'directml':
            params = {
                "EngineConfig.onnxruntime.use_dml": True,
                "Det.use_dml": True,
                "Rec.use_dml": True,
                "Cls.use_dml": False,
            }
            engine = RapidOCR(params=params)
            dummy = np.zeros((32, 128, 3), dtype=np.uint8)
            _ = engine(dummy)
            self.engines = [engine]

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
        mode_map = {
            'cpu_multi': f'CPU × {len(self.engines)}',
            'directml': 'DirectML',
            'cuda': 'CUDA',
        }
        return mode_map.get(self.mode, self.mode)

    def check_gpu_available(self):
        return self.mode in ('directml', 'cuda')