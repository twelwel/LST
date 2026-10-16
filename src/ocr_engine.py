"""OCR 引擎池
- CPU：多引擎并行（1-8 可调）
- CUDA：多引擎并行（1-4 可调）
- DirectML：单引擎
- 支持 server / tiny 两种模型（PP-OCRv6）
- 支持多语言（CH / EN / JAPAN / KOREAN / FRENCH / GERMAN / SPANISH / RUSSIAN）
- 支持高精度模式（双引擎投票融合）
- 支持 OCR 加速优化（关方向分类 + det 限长 + cv2 缩放）
- 热重载安全：旧引擎延时释放，重载期间 recognize 返回空
"""

import gc
import threading
import time
from collections import Counter

import numpy as np
import mss

from cuda_setup import (
    check_onnxruntime_variant,
    load_cuda_dlls,
)


# ==================== 枚举探测 ====================

_MODEL_TYPE_ENUM = None
_TINY_VALUE = None
_OCR_VERSION_ENUM = None
_PPOCRV6_VALUE = None
_LANG_DET_ENUM = None
_LANG_REC_ENUM = None


def _find_model_type():
    global _MODEL_TYPE_ENUM
    if _MODEL_TYPE_ENUM is not None:
        return _MODEL_TYPE_ENUM
    for path in ('rapidocr.utils.typings', 'rapidocr.utils',
                 'rapidocr.typings', 'rapidocr.config', 'rapidocr'):
        try:
            mod = __import__(path, fromlist=['ModelType'])
            mt = getattr(mod, 'ModelType', None)
            if mt is not None:
                _MODEL_TYPE_ENUM = mt
                return mt
        except Exception:
            continue
    return None


def _find_ocr_version():
    global _OCR_VERSION_ENUM
    if _OCR_VERSION_ENUM is not None:
        return _OCR_VERSION_ENUM
    for path in ('rapidocr.utils.typings', 'rapidocr.utils',
                 'rapidocr.typings', 'rapidocr.config', 'rapidocr'):
        try:
            mod = __import__(path, fromlist=['OCRVersion'])
            ov = getattr(mod, 'OCRVersion', None)
            if ov is not None:
                _OCR_VERSION_ENUM = ov
                return ov
        except Exception:
            continue
    return None


def _find_lang_enums():
    global _LANG_DET_ENUM, _LANG_REC_ENUM
    if _LANG_DET_ENUM is not None and _LANG_REC_ENUM is not None:
        return _LANG_DET_ENUM, _LANG_REC_ENUM
    for path in ('rapidocr.utils.typings', 'rapidocr.utils',
                 'rapidocr.typings', 'rapidocr'):
        try:
            mod = __import__(path, fromlist=['LangDet', 'LangRec'])
            if hasattr(mod, 'LangDet'):
                _LANG_DET_ENUM = mod.LangDet
            if hasattr(mod, 'LangRec'):
                _LANG_REC_ENUM = mod.LangRec
            if _LANG_DET_ENUM or _LANG_REC_ENUM:
                return _LANG_DET_ENUM, _LANG_REC_ENUM
        except Exception:
            continue
    return None, None


def _resolve_enum(enum_cls, names):
    """按候选名依次尝试解析枚举值"""
    if enum_cls is None:
        return None
    for name in names:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    return None


def _get_tiny_value():
    """解析 PP-OCRv6 的 TINY 模型枚举值"""
    global _TINY_VALUE
    if _TINY_VALUE is not None:
        return _TINY_VALUE
    mt = _find_model_type()
    _TINY_VALUE = _resolve_enum(mt, [
        'TINY', 'Tiny', 'tiny',
    ])
    return _TINY_VALUE


def _get_ppocrv6_value():
    """解析 PP-OCRv6 的版本枚举值"""
    global _PPOCRV6_VALUE
    if _PPOCRV6_VALUE is not None:
        return _PPOCRV6_VALUE
    ov = _find_ocr_version()
    _PPOCRV6_VALUE = _resolve_enum(ov, [
        'PPOCRV6', 'PP_OCRV6', 'PPOCRv6', 'PPOCR_V6', 'V6',
    ])
    return _PPOCRV6_VALUE


# ==================== 语言枚举解析 ====================

# 每种语言的候选枚举名（不同版本 rapidocr 命名不同）
_LANG_CANDIDATES = {
    'CH':      ['CH', 'Ch', 'ch', 'CHINESE', 'Chinese'],
    'EN':      ['EN', 'En', 'en', 'ENGLISH', 'English'],
    'JAPAN':   ['JAPAN', 'Japan', 'japan', 'JP', 'Jp',
                'JAPANESE', 'Japanese'],
    'KOREAN':  ['KOREAN', 'Korean', 'korean', 'KR', 'Kr',
                'KOREAN_KOR', 'Kor'],
    'FRENCH':  ['FRENCH', 'French', 'french', 'FR', 'Fr',
                'FRENCH_FRA', 'Fra'],
    'GERMAN':  ['GERMAN', 'German', 'german', 'DE', 'De',
                'GERMAN_DEU', 'Deu'],
    'SPANISH': ['SPANISH', 'Spanish', 'spanish', 'ES', 'Es',
                'SPANISH_SPA', 'Spa'],
    'RUSSIAN': ['RUSSIAN', 'Russian', 'russian', 'RU', 'Ru',
                'RUSSIAN_RUS', 'Rus'],
}


def _resolve_lang(enum_cls, lang_name):
    """根据语言名解析枚举值"""
    if enum_cls is None or not lang_name:
        return None
    candidates = _LANG_CANDIDATES.get(lang_name)
    if not candidates:
        candidates = [lang_name, lang_name.upper(), lang_name.lower()]
    return _resolve_enum(enum_cls, candidates)


def _build_model_params(model_size='server', lang_name='CH'):
    """
    构造 PP-OCRv6 模型参数

    model_size: 'server' / 'tiny'
    lang_name:  'CH' / 'EN' / 'JAPAN' / 'KOREAN' / ...
    """
    v6_v = _get_ppocrv6_value()
    if v6_v is None:
        return None

    params = {
        "Det.ocr_version": v6_v,
        "Rec.ocr_version": v6_v,
    }

    # tiny 模型
    if model_size == 'tiny':
        tiny_v = _get_tiny_value()
        if tiny_v is None:
            return None
        params["Det.model_type"] = tiny_v
        params["Rec.model_type"] = tiny_v

    # 语言类型
    ld, lr = _find_lang_enums()
    det_lv = _resolve_lang(ld, lang_name) or _resolve_lang(ld, 'CH')
    rec_lv = _resolve_lang(lr, lang_name) or _resolve_lang(lr, 'CH')

    if det_lv is not None:
        params["Det.lang_type"] = det_lv
    if rec_lv is not None:
        params["Rec.lang_type"] = rec_lv

    return params


# 兼容旧调用（其他模块可能还在用 _build_tiny_params）
def _build_tiny_params(lang_name='CH'):
    return _build_model_params('tiny', lang_name)


# ==================== IoU 工具 ====================

def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ==================== OCREngine ====================

class OCREngine:
    def __init__(self):
        self.engines = []
        self.aux_engines = []
        self.pool_size = 1
        self.variant = 'cpu'
        self.mode = 'cpu_multi'
        self.last_error = ''
        self.gpu_engine_count = 1
        self.model_size = 'server'
        self.high_accuracy = False
        self.speed_opt = True
        self.lang_name = 'CH'

        # ===== 热重载保护 =====
        self._lock = threading.Lock()
        self._reloading = False
        self._retiring = []

        # ===== mss 复用 =====
        self._sct = None

    # ================= 初始化 =================

    def init(self, mode='cpu_multi', pool_size=2, dml_worker_count=2,
             model_size='server', high_accuracy=False, speed_opt=True,
             lang_name='CH'):
        """
        mode: 'cpu_multi' / 'cuda' / 'directml'
        model_size: 'server' / 'tiny'
        high_accuracy: 是否启用多引擎投票
        speed_opt: 是否启用 OCR 加速优化
        lang_name: OCR 语言（CH / EN / JAPAN / KOREAN / ...）
        """
        with self._lock:
            self._reloading = True

        try:
            old_engines = self.engines
            old_aux = self.aux_engines
            self.engines = []
            self.aux_engines = []

            self._retire_engines(old_engines)
            self._retire_engines(old_aux)

            return self._do_init(
                mode=mode, pool_size=pool_size,
                dml_worker_count=dml_worker_count,
                model_size=model_size,
                high_accuracy=high_accuracy,
                speed_opt=speed_opt,
                lang_name=lang_name,
            )
        finally:
            with self._lock:
                self._reloading = False

    def _do_init(self, mode='cpu_multi', pool_size=2, dml_worker_count=2,
                 model_size='server', high_accuracy=False, speed_opt=True,
                 lang_name='CH'):
        """实际的初始化逻辑"""
        self.last_error = ''
        self.mode = mode

        if model_size not in ('server', 'tiny'):
            model_size = 'server'
        self.model_size = model_size
        self.high_accuracy = bool(high_accuracy)
        self.speed_opt = bool(speed_opt)
        self.lang_name = lang_name or 'CH'

        try:
            n = int(dml_worker_count)
        except Exception:
            n = 2
        from config import DML_WORKER_COUNT_MIN, DML_WORKER_COUNT_MAX
        n = max(DML_WORKER_COUNT_MIN, min(DML_WORKER_COUNT_MAX, n))
        self.gpu_engine_count = n

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
            self.pool_size = 1
        elif mode == 'cuda':
            self.pool_size = self.gpu_engine_count
        else:
            self.pool_size = max(1, int(pool_size))

        # tiny 模型可用性检查
        if self.model_size == 'tiny':
            if _build_model_params('tiny', self.lang_name) is None:
                print("[OCR] ⚠️ PP-OCRv6 tiny 不可用，回退 server")
                self.model_size = 'server'

        try:
            self._init_engines()
            if self.high_accuracy:
                self._init_aux_engines()
            print(f"[OCR] 使用 {mode}，主引擎 {len(self.engines)}，"
                  f"模型 {self.model_size}，语言 {self.lang_name}，"
                  f"高精度 {'开' if self.high_accuracy else '关'}，"
                  f"加速 {'开' if self.speed_opt else '关'}")
            return mode
        except Exception as e:
            self.last_error = str(e)
            print(f"[OCR] 初始化失败：{e}")
            self._cleanup_current()

            # 降级 1：tiny → server
            if self.model_size == 'tiny':
                print("[OCR] tiny 失败，回退 server")
                self.model_size = 'server'
                try:
                    self._init_engines()
                    if self.high_accuracy:
                        self._init_aux_engines()
                    return mode
                except Exception as e2:
                    print(f"[OCR] server 也失败：{e2}")
                    self._cleanup_current()

            # 降级 2：当前语言 → CH
            if self.lang_name != 'CH':
                print(f"[OCR] {self.lang_name} 失败，回退 CH")
                self.lang_name = 'CH'
                try:
                    self._init_engines()
                    if self.high_accuracy:
                        self._init_aux_engines()
                    return mode
                except Exception as e3:
                    print(f"[OCR] CH 也失败：{e3}")
                    self._cleanup_current()

            # 降级 3：GPU → CPU
            if mode != 'cpu_multi':
                print(f"[OCR] {mode} 失败，回退 CPU")
                self.mode = 'cpu_multi'
                self.pool_size = max(1, int(pool_size))
                try:
                    self._init_engines()
                    return 'cpu_multi'
                except Exception as e4:
                    self.last_error = str(e4)
                    self._cleanup_current()
            return 'cpu_multi'

    # ================= 引擎清理 =================

    def _cleanup_current(self):
        for e in self.engines + self.aux_engines:
            try:
                del e
            except Exception:
                pass
        self.engines = []
        self.aux_engines = []
        try:
            gc.collect()
        except Exception:
            pass

    def _retire_engines(self, engines_list):
        if not engines_list:
            return
        with self._lock:
            self._retiring.append(engines_list)

        def _gc_later():
            time.sleep(5.0)
            with self._lock:
                try:
                    self._retiring.remove(engines_list)
                except Exception:
                    pass
            try:
                gc.collect()
            except Exception:
                pass
            print(f"[OCR] 退休引擎已释放（{len(engines_list)} 个）")

        threading.Thread(target=_gc_later, daemon=True).start()

    # ================= 主引擎创建 =================

    def _build_model_params_for_current(self):
        """当前模型 + 语言对应的参数"""
        p = _build_model_params(self.model_size, self.lang_name)
        return p or {}

    def _speed_params(self):
        """OCR 加速优化参数（受 speed_opt 开关控制）"""
        if not self.speed_opt:
            return {}
        return {
            "Global.use_cls": False,
            "Det.limit_type": "max",
            "Det.limit_side_len": 1280,
        }

    def _init_engines(self):
        if self.mode == 'directml':
            self._init_engines_directml()
        elif self.mode == 'cuda':
            self._init_engines_cuda()
        else:
            self._init_engines_cpu()

    def _init_engines_cpu(self):
        from rapidocr import RapidOCR
        params = self._build_model_params_for_current()
        params.update(self._speed_params())
        print(f"[OCR] 创建 {self.pool_size} 个 CPU 引擎"
              f"（模型：{self.model_size}，语言：{self.lang_name}）...")
        for i in range(self.pool_size):
            try:
                t0 = time.time()
                engine = RapidOCR(params=params) if params else RapidOCR()
                self.engines.append(engine)
                print(f"[OCR] CPU 引擎 {i+1}/{self.pool_size} 就绪 "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"[OCR] CPU 引擎 {i+1} 失败：{e}")
                if not self.engines:
                    raise
        if not self.engines:
            raise RuntimeError("所有 CPU 引擎创建失败")

    def _init_engines_directml(self):
        from rapidocr import RapidOCR
        params = {
            "EngineConfig.onnxruntime.use_dml": True,
            "Det.use_dml": True,
            "Rec.use_dml": True,
            "Cls.use_dml": False,
        }
        params.update(self._build_model_params_for_current())
        params.update(self._speed_params())

        print(f"[OCR] 创建 DirectML 单引擎"
              f"（模型：{self.model_size}，语言：{self.lang_name}）...")
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
        from rapidocr import RapidOCR
        params = {
            "EngineConfig.onnxruntime.use_cuda": True,
            "EngineConfig.onnxruntime.cuda_ep_cfg."
            "cudnn_conv_algo_search": "HEURISTIC",
            "Det.use_cuda": True,
            "Rec.use_cuda": True,
            "Cls.use_cuda": False,
        }
        params.update(self._build_model_params_for_current())
        params.update(self._speed_params())

        n = self.pool_size
        print(f"[OCR] 创建 {n} 个 CUDA 引擎"
              f"（模型：{self.model_size}，语言：{self.lang_name}）...")

        self.engines = []
        for i in range(n):
            try:
                t0 = time.time()
                engine = RapidOCR(params=params)
                dummy = np.zeros((32, 128, 3), dtype=np.uint8)
                _ = engine(dummy)
                self.engines.append(engine)
                print(f"[OCR] CUDA 引擎 {i+1}/{n} 就绪 "
                      f"({time.time()-t0:.1f}s)")
                time.sleep(0.3)
            except Exception as e:
                print(f"[OCR] CUDA 引擎 {i+1} 失败：{e}")
                if not self.engines:
                    raise
                break

        if not self.engines:
            raise RuntimeError("所有 CUDA 引擎创建失败")
        print(f"[OCR] CUDA 就绪：{len(self.engines)} 个引擎")

    # ================= 辅助引擎创建（高精度模式）====================

    def _init_aux_engines(self):
        """
        辅助引擎：
        - 用 CPU + 与主引擎相反的模型（主 server → 辅 tiny，主 tiny → 辅 server）
        - 语言与主引擎相同
        - 不占用 GPU，避免与主引擎冲突
        """
        from rapidocr import RapidOCR

        aux_model = 'tiny' if self.model_size == 'server' else 'server'

        if aux_model == 'tiny':
            params = _build_model_params('tiny', self.lang_name)
            if params is None:
                print("[OCR] 辅助引擎：tiny 不可用，跳过")
                return
        else:
            params = _build_model_params('server', self.lang_name) or {}

        params.update(self._speed_params())

        print(f"[OCR] 高精度模式：创建 CPU 辅助引擎"
              f"（模型：{aux_model}，语言：{self.lang_name}）...")
        try:
            t0 = time.time()
            engine = RapidOCR(params=params) if params else RapidOCR()
            dummy = np.zeros((32, 128, 3), dtype=np.uint8)
            _ = engine(dummy)
            self.aux_engines = [engine]
            print(f"[OCR] 辅助引擎就绪 ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[OCR] 辅助引擎失败：{e}")
            self.aux_engines = []

    # ================= 截图 =================

    def capture(self, region):
        if self._sct is None:
            try:
                self._sct = mss.mss()
            except Exception:
                with mss.mss() as sct:
                    img = sct.grab(region)
                    return np.array(img)[:, :, :3]
        try:
            shot = self._sct.grab(region)
            return np.array(shot)[:, :, :3]
        except Exception:
            self._sct = None
            try:
                with mss.mss() as sct:
                    return np.array(sct.grab(region))[:, :, :3]
            except Exception as e:
                print(f"[OCR] 截图失败：{e}")
                return None

    # ================= 识别 =================

    def _get_engine_snapshot(self):
        with self._lock:
            if self._reloading:
                return None, None
            return self.engines, self.aux_engines

    def recognize(self, image_np):
        engines, _ = self._get_engine_snapshot()
        if not engines:
            return []
        return self._recognize_one(engines[0], image_np)

    def recognize_batch(self, images):
        engines, aux = self._get_engine_snapshot()
        if not engines or not images:
            return [[] for _ in images]

        main_results = self._recognize_main(images, engines)

        if self.high_accuracy and aux:
            aux_results = self._recognize_aux(images, aux)
            fused = []
            for r1, r2 in zip(main_results, aux_results):
                fused.append(self._fuse_results([r1, r2]))
            return fused

        return main_results

    def _recognize_main(self, images, engines):
        n = len(engines)
        if n == 1 or len(images) == 1:
            engine = engines[0]
            return [self._recognize_one(engine, img) for img in images]

        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(images)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {}
            for i, img in enumerate(images):
                engine = engines[i % n]
                fut = ex.submit(self._recognize_one, engine, img)
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[OCR] 主引擎错误：{e}")
                    results[idx] = []
        return results

    def _recognize_aux(self, images, aux):
        engine = aux[0]
        return [self._recognize_one(engine, img) for img in images]

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

    # ================= 结果融合 =================

    def _fuse_results(self, results_list, iou_threshold=0.5):
        all_items = []
        for src_idx, items in enumerate(results_list):
            for it in items:
                all_items.append({
                    'text': it['text'],
                    'box': it['box'],
                    '_src': src_idx,
                })

        if not all_items:
            return []

        all_items.sort(key=lambda x: (x['box'][1], x['box'][0]))

        clusters = []
        for it in all_items:
            matched = False
            for c in clusters:
                if _iou(it['box'], c['box']) > iou_threshold:
                    c['items'].append(it)
                    bx1 = min(c['box'][0], it['box'][0])
                    by1 = min(c['box'][1], it['box'][1])
                    bx2 = max(c['box'][2], it['box'][2])
                    by2 = max(c['box'][3], it['box'][3])
                    c['box'] = (bx1, by1, bx2, by2)
                    matched = True
                    break
            if not matched:
                clusters.append({
                    'box': it['box'],
                    'items': [it],
                })

        fused = []
        for c in clusters:
            texts = [it['text'].strip() for it in c['items']]
            texts = [t for t in texts if t]
            if not texts:
                continue

            counter = Counter(texts)
            most_common, count = counter.most_common(1)[0]

            if count >= 2:
                best_text = most_common
            else:
                best_text = max(texts, key=len)

            fused.append({
                'text': best_text,
                'box': c['box'],
            })

        fused.sort(key=lambda x: (x['box'][1], x['box'][0]))
        return fused

    # ================= 状态 =================

    def get_mode_text(self):
        if not self.engines:
            return "未初始化"
        n = len(self.engines)
        model_tag = "tiny" if self.model_size == 'tiny' else "标准"
        lang_tag = f" {self.lang_name}" if self.lang_name != 'CH' else ""
        ha_tag = " + 高精度" if self.high_accuracy else ""
        opt_tag = " + 加速" if self.speed_opt else ""
        if self.mode == 'directml':
            return f'DirectML 单引擎（{model_tag}{lang_tag}{ha_tag}{opt_tag}）'
        if self.mode == 'cuda':
            return f'CUDA × {n}（{model_tag}{lang_tag}{ha_tag}{opt_tag}）'
        return f'CPU × {n}（{model_tag}{lang_tag}{ha_tag}{opt_tag}）'

    def check_gpu_available(self):
        return self.mode in ('directml', 'cuda')

    def shutdown(self):
        with self._lock:
            self._reloading = True
        try:
            self._cleanup_current()
            with self._lock:
                self._retiring.clear()
        finally:
            with self._lock:
                self._reloading = False
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None