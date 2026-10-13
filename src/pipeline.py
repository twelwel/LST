"""后台处理管道：OCR + 翻译 + 图像修复
滚动检测已移到独立线程（scroll_detector.py）
"""

import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image as PILImage

from layout import (
    merge_lines_to_paragraphs,
    hash_text,
    hash_paragraph,
    paragraph_display_key,
    get_line_hashes,
    is_chinese_text,
    filter_invalid_items,
)
from inpaint import inpaint_region, CV2_AVAILABLE
from logger import get_logger


log = get_logger('pipeline')


THUMB_SIZE = 48
PIXEL_DIFF_THRESHOLD = 20
INPAINT_CACHE_MAX = 200

MULTITHREAD_OCR_MIN_AREA = 300 * 100
MULTITHREAD_OCR_OVERLAP = 30

# 滚动时跳过的帧数（由 UI 传入滚动标志，避免 OCR 阻塞跟随）
SCROLL_SKIP_AFTER = 2


class AppState:
    def __init__(self, initial_region,
                 frame_similarity_threshold=0.9,
                 position_quantize=10,
                 use_inpainting=False,
                 mode_cfg=None,
                 enable_frame_diff=True,
                 enable_filter=True,
                 enable_block_ocr=True,
                 enable_async_translate=True,
                 enable_window_pool=True,
                 enable_multithread_ocr=False):
        self.running = False
        self.region = dict(initial_region)
        self.base_font_size = 15
        self.use_inpainting = use_inpainting
        self.lock = threading.Lock()

        self.frame_similarity_threshold = frame_similarity_threshold
        self.position_quantize = position_quantize
        self.last_frame_keys = None
        self.last_frame_pixels = None

        self.idle_count = 0
        self.idle_skip_counter = 0

        self.performance_mode = 'medium'
        self.scan_interval = 0.30
        self.change_ratio_threshold = 0.03
        self.full_frame_ratio = 0.25
        self.bbox_padding = 60
        self.max_ocr_width = 1400
        self.max_ocr_height = 600

        self.enable_frame_diff = enable_frame_diff
        self.enable_filter = enable_filter
        self.enable_block_ocr = enable_block_ocr
        self.enable_async_translate = enable_async_translate
        self.enable_window_pool = enable_window_pool
        self.enable_multithread_ocr = enable_multithread_ocr
        self.freeze_overlay = False

        # 由 UI 设置：正在滚动时为 True
        self.scroll_boost = False

        if mode_cfg:
            self.apply_mode(mode_cfg)

    def apply_mode(self, cfg):
        self.scan_interval = cfg['scan_interval']
        self.change_ratio_threshold = cfg['change_ratio_threshold']
        self.full_frame_ratio = cfg['full_frame_ratio']
        self.bbox_padding = cfg['bbox_padding']
        self.max_ocr_width = cfg['max_ocr_width']
        self.max_ocr_height = cfg['max_ocr_height']

    def snapshot_region(self):
        with self.lock:
            return dict(self.region)

    def update_region(self, new_region):
        with self.lock:
            self.region.update(new_region)

    def reset_signature(self):
        self.last_frame_keys = None
        self.last_frame_pixels = None
        self.idle_count = 0
        self.idle_skip_counter = 0


class Pipeline:
    def __init__(self, ocr_engine, cache, translator, state):
        self.ocr = ocr_engine
        self.cache = cache
        self.translator = translator
        self.state = state

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.inpaint_cache = OrderedDict()
        self.inpaint_cache_max = INPAINT_CACHE_MAX
        self._inpaint_lock = threading.Lock()

        self.translate_pool = ThreadPoolExecutor(max_workers=2)

    def start(self):
        threading.Thread(target=self._process_loop, daemon=True).start()

    def submit(self, img):
        while self.task_queue.qsize() > 2:
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        self.task_queue.put(img)

    def _process_loop(self):
        while True:
            try:
                img = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process_one(img)
            except Exception as e:
                log.exception(f"[处理错误] {e}")

    # ==================== 帧差 ====================

    def _make_thumbnail(self, img):
        small = PILImage.fromarray(img).convert('L').resize(
            (THUMB_SIZE, THUMB_SIZE),
            PILImage.Resampling.NEAREST,
        )
        return np.asarray(small, dtype=np.int16)

    def _get_changed_bbox(self, img):
        full_frame = (0, 0, img.shape[1], img.shape[0])

        if not self.state.enable_frame_diff:
            return full_frame

        crt = self.state.change_ratio_threshold
        ffr = self.state.full_frame_ratio
        pad = self.state.bbox_padding

        thumb = self._make_thumbnail(img)
        prev = self.state.last_frame_pixels
        self.state.last_frame_pixels = thumb

        if prev is None:
            return full_frame

        diff = np.abs(thumb - prev)
        changed_ratio = (diff > PIXEL_DIFF_THRESHOLD).sum() / diff.size

        if changed_ratio < crt:
            return None

        if not self.state.enable_block_ocr:
            return full_frame

        if changed_ratio > ffr:
            return full_frame

        ys, xs = np.where(diff > PIXEL_DIFF_THRESHOLD)
        if len(xs) == 0:
            return None

        sx = img.shape[1] / THUMB_SIZE
        sy = img.shape[0] / THUMB_SIZE

        x1 = max(0, int(xs.min() * sx) - pad)
        y1 = max(0, int(ys.min() * sy) - pad)
        x2 = min(img.shape[1], int(xs.max() * sx) + pad)
        y2 = min(img.shape[0], int(ys.max() * sy) + pad)
        return (x1, y1, x2 - x1, y2 - y1)

    # ==================== 帧相似度 ====================

    def _make_frame_keys(self, items):
        q = self.state.position_quantize
        keys = set()
        for it in items:
            text = it['text'].strip()
            if not text:
                continue
            x1, y1, _, _ = it['box']
            keys.add((hash_text(text), x1 // q, y1 // q))
        return keys

    def _is_frame_unchanged(self, cur_keys):
        if not self.state.enable_frame_diff:
            return False
        prev = self.state.last_frame_keys
        if not prev or not cur_keys:
            return False
        inter = len(cur_keys & prev)
        union = len(cur_keys | prev)
        if union == 0:
            return True
        return inter / union >= self.state.frame_similarity_threshold

    def _shrink_for_ocr(self, crop):
        max_w = self.state.max_ocr_width
        max_h = self.state.max_ocr_height
        h, w = crop.shape[:2]
        if w <= max_w and h <= max_h:
            return crop, 1.0
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil = PILImage.fromarray(crop).resize(
            (new_w, new_h), PILImage.Resampling.BILINEAR,
        )
        return np.array(pil), scale

    # ==================== 多线程 OCR ====================

    def _ocr_multithread(self, img, x, y, w, h):
        n = self.ocr.pool_size
        if n <= 1:
            crop = img[y:y+h, x:x+w]
            crop_small, scale = self._shrink_for_ocr(crop)
            items = self.ocr.recognize(crop_small)
            return self._remap_items(items, x, y, scale)

        chunk_h = max(1, h // n)
        crops = []
        offsets = []
        for i in range(n):
            cy1 = y + i * chunk_h
            cy2 = y + h if i == n - 1 else y + (i + 1) * chunk_h
            cy1p = max(y, cy1 - MULTITHREAD_OCR_OVERLAP)
            cy2p = min(y + h, cy2 + MULTITHREAD_OCR_OVERLAP)
            chunk = img[cy1p:cy2p, x:x + w]
            crop_small, scale = self._shrink_for_ocr(chunk)
            crops.append(crop_small)
            offsets.append((x, cy1p, scale))

        results = self.ocr.recognize_batch(crops)

        merged = []
        for items, (ox, oy, scale) in zip(results, offsets):
            merged.extend(self._remap_items(items, ox, oy, scale))

        return self._dedupe(merged)

    def _remap_items(self, items, ox, oy, scale):
        inv = 1.0 / scale if scale > 0 else 1.0
        out = []
        for it in items:
            bx1, by1, bx2, by2 = it['box']
            out.append({
                'text': it['text'],
                'box': (
                    int(bx1 * inv) + ox,
                    int(by1 * inv) + oy,
                    int(bx2 * inv) + ox,
                    int(by2 * inv) + oy,
                ),
            })
        return out

    def _dedupe(self, items):
        seen = set()
        result = []
        for it in items:
            x1, y1, _, _ = it['box']
            key = (it['text'].strip(), x1 // 8, y1 // 8)
            if key in seen:
                continue
            seen.add(key)
            result.append(it)
        return result

    # ==================== 主处理 ====================

    def _process_one(self, img):
        if self.state.freeze_overlay:
            return

        # 1) 帧差检测
        bbox = self._get_changed_bbox(img)
        if bbox is None:
            self.state.idle_count += 1
            return
        self.state.idle_count = 0
        self.state.idle_skip_counter = 0

        x, y, w, h = bbox
        if w < 10 or h < 10:
            return

        # 2) OCR
        use_mt = (self.state.enable_multithread_ocr and
                  self.ocr.pool_size > 1 and
                  w * h >= MULTITHREAD_OCR_MIN_AREA)

        if use_mt:
            items = self._ocr_multithread(img, x, y, w, h)
        else:
            crop = img[y:y+h, x:x+w]
            crop_small, scale = self._shrink_for_ocr(crop)
            raw = self.ocr.recognize(crop_small)
            items = self._remap_items(raw, x, y, scale)

        if not items:
            return

        region = self.state.snapshot_region()

        # 3) 过滤
        if self.state.enable_filter:
            items = filter_invalid_items(
                items, region['width'], region['height']
            )
            if not items:
                return

        # 4) 帧相似度
        cur_keys = self._make_frame_keys(items)
        if self._is_frame_unchanged(cur_keys):
            return
        self.state.last_frame_keys = cur_keys

        # 5) 段落合并
        paragraphs = merge_lines_to_paragraphs(items)
        to_translate = [p for p in paragraphs if not is_chinese_text(p['text'])]
        if not to_translate:
            return

        # 6) 缓存
        need_api = []
        for p in to_translate:
            p_hash = hash_paragraph(p)
            line_hashes = get_line_hashes(p)

            if self.cache.get(p_hash) is not None:
                continue

            reused = self.cache.find_reusable(line_hashes)
            if reused is not None:
                self.cache.put(p_hash, reused, line_hashes)
                continue

            need_api.append((p, p_hash, line_hashes))

        # 7) 立即推入已有
        self._emit_paragraphs(to_translate, img, region)

        # 8) API 翻译
        if need_api:
            mt_tag = " [多线程OCR]" if use_mt else ""
            log.info(f"[段落] 共 {len(to_translate)} 段 | "
                     f"需 API {len(need_api)} | 变化区 {w}x{h}{mt_tag}")

            if self.state.enable_async_translate:
                self.translate_pool.submit(
                    self._translate_and_emit,
                    need_api, to_translate, img, region,
                )
            else:
                self._translate_and_emit(
                    need_api, to_translate, img, region,
                )

    def _translate_and_emit(self, need_api, all_paragraphs, img, region):
        if self.state.freeze_overlay:
            return
        try:
            texts = [p['text'] for p, _, _ in need_api]
            translations = self.translator.translate_lines(texts)
            for (p, p_hash, lh), zh in zip(need_api, translations):
                self.cache.put(p_hash, zh, lh)

            if self.state.freeze_overlay:
                return
            self._emit_paragraphs(all_paragraphs, img, region)
        except Exception as e:
            log.exception(f"[异步翻译错误] {e}")

    def _emit_paragraphs(self, paragraphs, img, region):
        if not paragraphs or self.state.freeze_overlay:
            return

        use_inpaint = self.state.use_inpainting and CV2_AVAILABLE
        final = []

        for p in paragraphs:
            p_hash = hash_paragraph(p)
            zh = self.cache.get(p_hash)
            if not zh:
                continue

            x1, y1, x2, y2 = p['box']

            avg = self._crop_avg_color(img, x1, y1, x2, y2)
            bg_hex = f'#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}'
            luminance = avg[0]*0.299 + avg[1]*0.587 + avg[2]*0.114
            fg = '#FFFFFF' if luminance < 128 else '#000000'

            bg_image = None
            if use_inpaint:
                bg_image = self._get_inpaint_cached(p_hash, img, x1, y1, x2, y2)

            final.append({
                'x1': x1 + region['left'],
                'y1': y1 + region['top'],
                'x2': x2 + region['left'],
                'y2': y2 + region['top'],
                'text': zh,
                'bg_hex': bg_hex,
                'fg': fg,
                'bg_image': bg_image,
                'key': paragraph_display_key(p),
            })

        if final:
            self.result_queue.put(('items', final))

    def _get_inpaint_cached(self, p_hash, img, x1, y1, x2, y2):
        with self._inpaint_lock:
            if p_hash in self.inpaint_cache:
                item = self.inpaint_cache[p_hash]
                self.inpaint_cache.move_to_end(p_hash)
                return item
            bg_image = inpaint_region(img, x1, y1, x2, y2)
            self.inpaint_cache[p_hash] = bg_image
            if (self.cache.auto_clean and
                    len(self.inpaint_cache) > self.inpaint_cache_max):
                self.inpaint_cache.popitem(last=False)
            return bg_image

    @staticmethod
    def _crop_avg_color(img, x1, y1, x2, y2):
        img_h, img_w = img.shape[:2]
        cx1 = max(0, min(img_w - 1, x1))
        cy1 = max(0, min(img_h - 1, y1))
        cx2 = max(cx1 + 1, min(img_w, x2))
        cy2 = max(cy1 + 1, min(img_h, y2))
        patch = img[cy1:cy2, cx1:cx2]
        if patch.size == 0:
            return (255, 255, 255)
        avg = patch.mean(axis=(0, 1)).astype(int)
        return (int(avg[0]), int(avg[1]), int(avg[2]))