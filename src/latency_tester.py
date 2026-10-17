"""LST 延迟测试核心模块
供 CLI 入口 (tools/test_latency.py) 和 UI 对话框 (src/ui/latency_dialog.py) 共用
"""

import os
import queue
import sys
import time
import threading
import statistics
from typing import Callable, List, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_test_image(w: int = 800, h: int = 400) -> np.ndarray:
    img = Image.new('RGB', (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    font = None
    for name in ('msyh.ttc', 'simhei.ttf', 'arial.ttf'):
        try:
            font = ImageFont.truetype(name, 22)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    lines = [
        "Hello World",
        "This is a test message",
        "Settings",
        "Multiplayer",
        "Encyclopedia",
        "First Feudal",
        "100% complete",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(20, 20, 20), font=font)
        y += 45

    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def _add_noise(img: np.ndarray, seed: int, strength: int = 8) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randint(-strength, strength + 1, img.shape, dtype=np.int16)
    out = np.clip(img.astype(np.int16) + noise, 0, 255)
    return out.astype(np.uint8)


class _Stat:
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []

    def add(self, ms: float):
        self.times.append(ms)

    def summary(self) -> Dict:
        if not self.times:
            return {'name': self.name, 'n': 0, 'avg': 0,
                    'p50': 0, 'p95': 0, 'min': 0, 'max': 0, 'std': 0}
        arr = sorted(self.times)
        n = len(arr)
        return {
            'name': self.name,
            'n': n,
            'avg': statistics.mean(arr),
            'p50': arr[n // 2],
            'p95': arr[int(n * 0.95)] if n > 1 else arr[-1],
            'min': arr[0],
            'max': arr[-1],
            'std': statistics.pstdev(arr) if n > 1 else 0.0,
        }


class LatencyTester:
    def __init__(self, translator=None, ocr_engine=None,
                 pipeline=None,
                 progress_cb: Optional[Callable] = None,
                 log_cb: Optional[Callable] = None):
        self.translator = translator
        self.ocr_engine = ocr_engine
        self.pipeline = pipeline
        self.progress_cb = progress_cb or (lambda c, t: None)
        self.log_cb = log_cb or (lambda msg: None)

        self.stats: Dict[str, _Stat] = {}
        self._cancel = threading.Event()
        self._lock = threading.Lock()

        self.clean_cache_each = False
        self.e2e_warmup = 2

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    def reset(self):
        self.stats.clear()
        self._cancel.clear()

    def _get(self, name):
        with self._lock:
            if name not in self.stats:
                self.stats[name] = _Stat(name)
            return self.stats[name]

    def _run(self, name, fn, *args, **kwargs):
        if self.is_cancelled():
            return None
        t0 = time.perf_counter()
        try:
            r = fn(*args, **kwargs)
        except Exception as e:
            self.log_cb(f"[{name}] 异常：{e}")
            r = None
        t1 = time.perf_counter()
        self._get(name).add((t1 - t0) * 1000)
        return r

    def get_device_info(self) -> str:
        parts = []
        try:
            if self.ocr_engine is not None:
                parts.append(f"OCR={self.ocr_engine.mode}")
        except Exception:
            pass

        try:
            from translation.local_models import get_preferred_device
            pref = get_preferred_device()
            parts.append(f"本地模型偏好={pref}")

            try:
                from translation.local_models import TranslationModelFactory
                for mid, inst in (TranslationModelFactory._instances or {}).items():
                    dev = getattr(inst, 'device', '?')
                    parts.append(f"{mid}@{dev}")
                    break
            except Exception:
                pass
        except Exception:
            pass
        return " | ".join(parts) if parts else "未知"

    def get_report(self) -> List[Dict]:
        with self._lock:
            return [st.summary() for st in self.stats.values()]

    def get_report_text(self) -> str:
        lines = []
        lines.append("=" * 100)
        lines.append(f"设备：{self.get_device_info()}")
        lines.append("=" * 100)
        lines.append(f"{'阶段':<28} {'次数':>5} {'平均':>10} "
                     f"{'P50':>10} {'P95':>10} {'最小':>10} "
                     f"{'最大':>10} {'标准差':>10}")
        lines.append("-" * 100)

        local_total = 0.0
        api_avg = 0.0
        e2e_avg = 0.0
        local_model_avg = 0.0

        for s in self.get_report():
            lines.append(
                f"{s['name']:<28} "
                f"{s['n']:>5d} "
                f"{s['avg']:>9.2f}ms "
                f"{s['p50']:>9.2f}ms "
                f"{s['p95']:>9.2f}ms "
                f"{s['min']:>9.2f}ms "
                f"{s['max']:>9.2f}ms "
                f"{s['std']:>9.2f}")

            nm = s['name']
            if nm.startswith('★ 全链路'):
                e2e_avg = s['avg']
            elif nm == 'API 翻译':
                api_avg = s['avg']
            elif nm == '本地模型翻译':
                local_model_avg = s['avg']
            elif nm[0].isdigit():
                local_total += s['avg']

        lines.append("-" * 100)
        lines.append(f"{'本地阶段合计':<28} {local_total:>9.2f}ms")
        if local_model_avg > 0:
            lines.append(f"{'本地模型翻译平均':<28} {local_model_avg:>9.2f}ms")
        if api_avg > 0:
            lines.append(f"{'API 翻译平均':<28} {api_avg:>9.2f}ms")
        if e2e_avg > 0:
            lines.append(f"{'★ 实测全链路':<28} {e2e_avg:>9.2f}ms")
        lines.append("=" * 100)
        return '\n'.join(lines)

    # ==================== 核心测试方法 ====================

    def run(self, runs: int = 30, warmup: int = 3,
            do_ocr: bool = True,
            do_api: bool = False,
            api_runs: int = 5,
            do_local: bool = False,
            local_runs: int = 5,
            do_verify: bool = False,
            do_e2e: bool = False,
            e2e_runs: int = 5,
            e2e_clean_cache: bool = False):
        self.reset()
        self.clean_cache_each = bool(e2e_clean_cache)

        # ⭐ 强制刷新翻译策略
        if self.translator:
            try:
                self.translator.reload_config()
            except Exception:
                pass
        if self.pipeline and hasattr(self.pipeline, 'translator'):
            try:
                self.pipeline.translator.reload_config()
            except Exception:
                pass

        img = make_test_image(800, 400)
        prev_img = make_test_image(800, 400)
        img_mod = img.copy()
        img_mod[50:100, 50:300] = 128

        total_steps = runs
        if do_api: total_steps += api_runs
        if do_local: total_steps += local_runs
        if do_e2e: total_steps += e2e_runs

        # 1. 本地阶段预热与测试
        if warmup > 0 and not self.is_cancelled():
            self.log_cb(f"[预热] {warmup} 次...")
            for i in range(warmup):
                if self.is_cancelled(): return
                self._run_pipeline_once(img_mod, prev_img, do_ocr, do_verify, record=False)

        self.log_cb(f"[测试] 本地 {runs} 次")
        for i in range(runs):
            if self.is_cancelled():
                self.log_cb("[取消] 用户中止")
                return
            self._run_pipeline_once(img_mod, prev_img, do_ocr, do_verify, record=True)
            self.progress_cb(i + 1, total_steps)

        step = runs

        # 2. 本地模型测试
        if do_local:
            self.log_cb(f"[本地模型] 预热 + 测试 {local_runs} 次...")
            self._run_local_translate("warmup")
            for i in range(local_runs):
                if self.is_cancelled(): return
                self._run('本地模型翻译', self._run_local_translate, "Hello World")
                step += 1
                self.progress_cb(step, total_steps)

        # 3. API 测试
        if do_api and self.translator is not None:
            self.log_cb(f"[API] {api_runs} 次...")
            for i in range(api_runs):
                if self.is_cancelled(): return
                self._run('API 翻译', self.translator.translate, "Hello World", 'auto', 'zh')
                step += 1
                self.progress_cb(step, total_steps)

        # 4. 全链路
        if do_e2e and self.pipeline is not None:
            mode_tag = "（每轮清空缓存）" if self.clean_cache_each else "（复用缓存）"
            self.log_cb(f"[全链路] 预热 {self.e2e_warmup} 次 + 测试 {e2e_runs} 次 {mode_tag}")

            for i in range(self.e2e_warmup):
                if self.is_cancelled(): return
                test_img = _add_noise(img, seed=i * 7 + 1)
                self._run_e2e_once(test_img, record=False)

            for i in range(e2e_runs):
                if self.is_cancelled(): return
                test_img = _add_noise(img, seed=i * 7 + 1)
                ms = self._run_e2e_once(test_img, record=True)
                if ms is None:
                    self.log_cb(f"  第 {i+1} 次超时/未返回")
                step += 1
                self.progress_cb(step, total_steps)

        self.log_cb("[完成] 测试结束")

    def _run_local_translate(self, text):
        try:
            from translation.api_config import load_providers
            providers = load_providers()
            for p in providers:
                if p.get('provider') == 'local_model' and p.get('enabled', True):
                    mid = p.get('config', {}).get('model_id', '')
                    if not mid: continue
                    from translation.local_models import get_model_instance
                    inst = get_model_instance(mid)
                    if inst and inst.is_loaded:
                        return inst.translate(text, 'auto', 'zh')
        except Exception as e:
            self.log_cb(f"[本地模型] 异常：{e}")
        return ''

    def _run_e2e_once(self, img: np.ndarray, timeout: float = 30.0, record: bool = True) -> Optional[float]:
        if self.pipeline is None: return None

        try:
            while True: self.pipeline.result_queue.get_nowait()
        except queue.Empty: pass

        try:
            while not self.pipeline.task_queue.empty():
                try: self.pipeline.task_queue.get_nowait()
                except queue.Empty: break
        except Exception: pass

        try: self.pipeline.reset_debouncer()
        except Exception: pass

        if self.clean_cache_each:
            try: self.pipeline.cache.clear()
            except Exception: pass

        time.sleep(0.05)
        t0 = time.perf_counter()
        self.pipeline.submit(img)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_cancelled(): return None
            try: msg = self.pipeline.result_queue.get(timeout=0.3)
            except queue.Empty: continue
            if msg and msg[0] == 'items' and msg[1]:
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000
                if record: self._get('★ 全链路（截图→译文）').add(ms)
                return ms
        return None

    def _run_pipeline_once(self, img, prev_img, do_ocr: bool, do_verify: bool, record: bool):
        def timed(name, fn, *args, **kwargs):
            if not record:
                t0 = time.perf_counter()
                try: r = fn(*args, **kwargs)
                except Exception: r = None
                t1 = time.perf_counter()
                return r
            return self._run(name, fn, *args, **kwargs)

        def _frame_diff(img, prev):
            try:
                import cv2
                small = cv2.resize(img, (40, 40), interpolation=cv2.INTER_NEAREST)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                thumb = gray.astype(np.int16)

                prev_small = cv2.resize(prev, (40, 40), interpolation=cv2.INTER_NEAREST)
                prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                prev_thumb = prev_gray.astype(np.int16)

                diff = np.abs(thumb - prev_thumb)
                mask = diff > 20
                ratio = mask.sum() / diff.size

                if ratio < 0.03: return None
                if ratio > 0.25: return (0, 0, img.shape[1], img.shape[0])

                mask_u8 = (mask.astype(np.uint8) * 255)
                kernel = np.ones((3, 3), np.uint8)
                mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

                comps = []
                for k in range(1, n_labels):
                    area = stats[k, cv2.CC_STAT_AREA]
                    if area < 4: continue
                    comps.append((area, k))
                comps.sort(reverse=True)
                top_n = comps[:5]
                if not top_n: return None

                sx = img.shape[1] / 40
                sy = img.shape[0] / 40
                xs, ys = [], []
                for area, k in top_n:
                    cx = stats[k, cv2.CC_STAT_LEFT]
                    cy = stats[k, cv2.CC_STAT_TOP]
                    cw = stats[k, cv2.CC_STAT_WIDTH]
                    ch = stats[k, cv2.CC_STAT_HEIGHT]
                    xs.extend([cx, cx + cw])
                    ys.extend([cy, cy + ch])

                x1 = max(0, int(min(xs) * sx) - 60)
                y1 = max(0, int(min(ys) * sy) - 60)
                x2 = min(img.shape[1], int(max(xs) * sx) + 60)
                y2 = min(img.shape[0], int(max(ys) * sy) + 60)
                return (x1, y1, x2 - x1, y2 - y1)
            except Exception as e:
                self.log_cb(f"[frame_diff] {e}")
                return (0, 0, img.shape[1], img.shape[0])

        bbox = timed('1.帧差检测', _frame_diff, img, prev_img)
        if bbox is None: bbox = (0, 0, img.shape[1], img.shape[0])

        x, y, w, h = bbox
        if w < 10 or h < 10: return

        def _crop(img, x, y, w, h, mw=1400, mh=600):
            crop = img[y:y+h, x:x+w]
            ch, cw = crop.shape[:2]
            if cw <= mw and ch <= mh: return crop, 1.0
            try:
                import cv2
                scale = min(mw / cw, mh / ch)
                nw, nh = int(cw * scale), int(ch * scale)
                return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR), scale
            except Exception:
                return crop, 1.0

        result = timed('2.裁剪+缩放', _crop, img, x, y, w, h)
        if result is None: return
        crop_small, scale = result

        items = []
        if do_ocr and self.ocr_engine is not None:
            def _ocr():
                raw = self.ocr_engine.recognize(crop_small)
                inv = 1.0 / scale if scale > 0 else 1.0
                out = []
                for it in raw:
                    bx1, by1, bx2, by2 = it['box']
                    out.append({
                        'text': it['text'],
                        'box': (int(bx1 * inv) + x, int(by1 * inv) + y,
                                int(bx2 * inv) + x, int(by2 * inv) + y),
                    })
                return out
            items = timed('3.OCR 识别', _ocr) or []

        if not items: return

        from layout import (
            filter_invalid_items, filter_general_content, filter_english_only,
            filter_title_logo, merge_lines_to_paragraphs, paragraph_display_key,
            text_similarity,
        )

        def _filter_all(items):
            r = filter_invalid_items(items, w, h)
            r = filter_general_content(r)
            r = filter_english_only(r)
            r = filter_title_logo(r)
            return r

        items = timed('4.过滤', _filter_all, items) or items
        if not items: return

        paragraphs = timed('5.段落合并', merge_lines_to_paragraphs, items) or []
        if not paragraphs: return

        if do_verify and self.ocr_engine is not None:
            def _verify_one(p):
                x1, y1, x2, y2 = p['box']
                pad = 8
                cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                cx2 = min(img.shape[1], x2 + pad); cy2 = min(img.shape[0], y2 + pad)
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0: return True
                try: raw = self.ocr_engine.recognize(crop)
                except Exception: return True
                if not raw: return False
                new_text = ''.join(it['text'].strip() for it in raw)
                return text_similarity(p['text'], new_text) >= 0.6
            timed('6.二次OCR验证', lambda: [_verify_one(p) for p in paragraphs])

        timed('7.段落key生成', lambda: [paragraph_display_key(p) for p in paragraphs])


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description='LST 延迟测试')
    parser.add_argument('-n', '--runs', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--no-ocr', action='store_true')
    parser.add_argument('--api', action='store_true')
    parser.add_argument('--api-runs', type=int, default=5)
    parser.add_argument('--local', action='store_true', help='测试本地模型')
    parser.add_argument('--local-runs', type=int, default=5)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--e2e', action='store_true')
    parser.add_argument('--e2e-runs', type=int, default=5)
    parser.add_argument('--e2e-clean-cache', action='store_true', help='每轮全链路测试清空缓存')
    parser.add_argument('--device', type=str, default=None, help='本地模型设备：auto/cpu/cuda/directml')
    args = parser.parse_args()

    if args.device:
        try:
            from translation.local_models import set_preferred_device
            set_preferred_device(args.device)
            print(f"[准备] 本地模型设备：{args.device}")
        except Exception as e:
            print(f"[准备] 设置设备失败：{e}")

    ocr = None
    if not args.no_ocr:
        try:
            from ocr_engine import OCREngine
            ocr = OCREngine()
            ocr.init(mode='cpu_multi', pool_size=1, model_size='server')
            print(f"[准备] OCR 就绪：{ocr.mode}")
        except Exception as e:
            print(f"[准备] OCR 失败：{e}")

    translator = None
    if args.api or args.local or args.e2e:
        try:
            from translator import Translator
            translator = Translator()
            print("[准备] Translator 就绪")
        except Exception as e:
            print(f"[准备] Translator 失败：{e}")

    pipeline = None
    if args.e2e:
        try:
            from cache import ParagraphCache
            from pipeline import Pipeline, AppState
            from config import INITIAL_REGION

            cache = ParagraphCache(min_overlap=0.5, max_size=500)
            state = AppState(
                INITIAL_REGION, enable_frame_diff=True, enable_filter=True,
                enable_block_ocr=True, enable_async_translate=False, enable_multithread_ocr=False,
            )
            pipeline = Pipeline(ocr, cache, translator, state)
            pipeline.start()
            print("[准备] Pipeline 就绪")
        except Exception as e:
            print(f"[准备] Pipeline 失败：{e}")

    tester = LatencyTester(
        translator=translator, ocr_engine=ocr, pipeline=pipeline,
        log_cb=lambda msg: print(msg), progress_cb=lambda c, t: None,
    )

    tester.run(
        runs=args.runs, warmup=args.warmup, do_ocr=not args.no_ocr,
        do_api=args.api, api_runs=args.api_runs,
        do_local=args.local, local_runs=args.local_runs,
        do_verify=args.verify, do_e2e=args.e2e, e2e_runs=args.e2e_runs,
        e2e_clean_cache=args.e2e_clean_cache,
    )

    print()
    print(tester.get_report_text())


if __name__ == '__main__':
    _cli()