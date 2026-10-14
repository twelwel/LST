"""后台处理管道
- 三种翻译模式：本地 / 混合 / 联网
- 短语缓存：预填段落缓存
- 本地词典：查词
- API：兜底
"""

import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from layout import (
    merge_lines_to_paragraphs,
    hash_text,
    hash_paragraph,
    paragraph_display_key,
    get_line_hashes,
    is_chinese_text,
    filter_invalid_items,
    filter_general_content,
)
from inpaint import (
    inpaint_region,
    inpaint_text_only,
    sample_text_color,
    is_fine_text_suitable,
    CV2_AVAILABLE,
)
from phrase_cache import PhraseCache
from local_dict import LocalDictionary
from logger import get_logger


log = get_logger('pipeline')


THUMB_SIZE = 40
PIXEL_DIFF_THRESHOLD = 20
INPAINT_CACHE_MAX = 200

MULTITHREAD_OCR_MIN_AREA = 400 * 120
MULTITHREAD_OCR_OVERLAP = 15
MULTITHREAD_OCR_MIN_CHUNK_H = 300
MULTITHREAD_OCR_MAX_CHUNKS = 3

SLOW_THRESHOLD_MS = 300


class AppState:
    def __init__(self, initial_region,
                 frame_similarity_threshold=0.9,
                 position_quantize=10,
                 use_inpainting=False,
                 enable_fine_text=False,
                 enable_general_filter=True,
                 enable_phrase_cache=True,
                 translation_mode='hybrid',
                 local_dict_path='',
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
        self.enable_fine_text = enable_fine_text
        self.enable_general_filter = enable_general_filter
        self.enable_phrase_cache = enable_phrase_cache
        self.translation_mode = translation_mode
        self.local_dict_path = local_dict_path
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
        self.fast_mode = False

        self.enable_frame_diff = enable_frame_diff
        self.enable_filter = enable_filter
        self.enable_block_ocr = enable_block_ocr
        self.enable_async_translate = enable_async_translate
        self.enable_window_pool = enable_window_pool
        self.enable_multithread_ocr = enable_multithread_ocr

        self.follow_mode = 'follow'
        self.freeze_overlay = False
        self.scroll_boost = False
        self.pending_change = False

        if mode_cfg:
            self.apply_mode(mode_cfg)

    def apply_mode(self, cfg):
        self.scan_interval = cfg['scan_interval']
        self.change_ratio_threshold = cfg['change_ratio_threshold']
        self.full_frame_ratio = cfg['full_frame_ratio']
        self.bbox_padding = cfg['bbox_padding']
        self.max_ocr_width = cfg['max_ocr_width']
        self.max_ocr_height = cfg['max_ocr_height']
        self.fast_mode = cfg.get('fast_mode', False)

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
        self.pending_change = False


class Pipeline:
    def __init__(self, ocr_engine, cache, translator, state):
        self.ocr = ocr_engine
        self.cache = cache
        self.translator = translator
        self.state = state

        self.task_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()

        self.inpaint_cache = OrderedDict()
        self.fine_cache = OrderedDict()
        self.color_cache = {}
        self.inpaint_cache_max = INPAINT_CACHE_MAX
        self._inpaint_lock = threading.Lock()

        # 短语缓存
        self.phrase_cache = PhraseCache()
        # 👇 本地词典
        self.local_dict = LocalDictionary()
        if state.local_dict_path:
            try:
                self.local_dict.load(state.local_dict_path)
            except Exception as e:
                log.error(f"[本地词典] 初始化失败：{e}")

        self.translate_pool = ThreadPoolExecutor(max_workers=4)
        self._pending_translates = set()

    def start(self):
        threading.Thread(target=self._process_loop, daemon=True).start()

    def submit(self, img):
        try:
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except queue.Empty:
                    break
            self.task_queue.put_nowait(img)
        except queue.Full:
            pass

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
        if _CV2_OK:
            try:
                small = cv2.resize(
                    img, (THUMB_SIZE, THUMB_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                return gray.astype(np.int16)
            except Exception:
                pass

        try:
            h, w = img.shape[:2]
            sy = max(1, h // THUMB_SIZE)
            sx = max(1, w // THUMB_SIZE)
            sampled = img[::sy, ::sx][:THUMB_SIZE, :THUMB_SIZE]
            gray = (sampled[:, :, 2] * 0.299 +
                    sampled[:, :, 1] * 0.587 +
                    sampled[:, :, 0] * 0.114)
            return gray.astype(np.int16)
        except Exception:
            pass

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
        if thumb.shape != prev.shape:
            return full_frame

        diff = np.abs(thumb - prev)
        mask = diff > PIXEL_DIFF_THRESHOLD
        changed_ratio = mask.sum() / diff.size

        if changed_ratio < crt:
            return None

        if not self.state.enable_block_ocr:
            return full_frame

        if changed_ratio > ffr:
            return full_frame

        ys, xs = np.where(mask)
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

    # ==================== 分块 OCR ====================

    def _decide_chunks(self, h):
        by_size = max(1, h // MULTITHREAD_OCR_MIN_CHUNK_H)
        n = min(self.ocr.pool_size, by_size, MULTITHREAD_OCR_MAX_CHUNKS)
        return max(1, n)

    def _ocr_multithread(self, img, x, y, w, h):
        n = self._decide_chunks(h)

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

    # ==================== 短语缓存 ====================

    def _prefill_from_phrases(self, items):
        """短语缓存命中 → 预填段落缓存"""
        if not self.state.enable_phrase_cache:
            return 0

        filled = 0
        for it in items:
            text = it['text'].strip()
            if not text:
                continue

            zh = self.phrase_cache.lookup_fuzzy(text)
            if not zh:
                continue

            fake_para = {
                'text': text,
                'box': it['box'],
                'lines': [it],
            }
            p_hash = hash_paragraph(fake_para)

            if self.cache.get(p_hash) is None:
                line_hashes = get_line_hashes(fake_para)
                self.cache.put(p_hash, zh, line_hashes)
                filled += 1

        if filled:
            log.info(f"[短语缓存] 预填 {filled} 条段落")
        return filled

    # ==================== 本地词典 ====================

    def _lookup_local(self, text):
        """
        本地查询（短语缓存 → 本地词典）
        使用 smart_lookup 多级匹配，大幅提升本地模式命中率
        """
        # 1) 短语缓存（整句优先）
        if self.state.enable_phrase_cache:
            zh = self.phrase_cache.lookup_fuzzy(text)
            if zh:
                return zh

        # 2) 本地词典（智能多级查询）
        zh = self.local_dict.smart_lookup(text)
        if zh:
            return zh

        return None

    # ==================== 主处理 ====================

    def _process_one(self, img):
        if self.state.freeze_overlay:
            return

        t_start = time.time()

        bbox = self._get_changed_bbox(img)
        if bbox is None:
            self.state.idle_count += 1
            return
        self.state.idle_count = 0
        self.state.idle_skip_counter = 0
        self.state.pending_change = True

        t_frame_diff = time.time()

        x, y, w, h = bbox
        if w < 10 or h < 10:
            return

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

        t_ocr = time.time()

        if not items:
            return

        region = self.state.snapshot_region()

        if self.state.enable_filter:
            items = filter_invalid_items(
                items, region['width'], region['height']
            )
            if not items:
                return

        if self.state.enable_general_filter:
            items = filter_general_content(items)
            if not items:
                return

        # 短语缓存预填
        self._prefill_from_phrases(items)

        cur_keys = self._make_frame_keys(items)
        if self._is_frame_unchanged(cur_keys):
            return
        self.state.last_frame_keys = cur_keys

        paragraphs = merge_lines_to_paragraphs(items)
        to_translate = [p for p in paragraphs if not is_chinese_text(p['text'])]
        if not to_translate:
            return

        # 👇 按模式处理
        mode = self.state.translation_mode

        need_api = []
        for p in to_translate:
            p_hash = hash_paragraph(p)
            line_hashes = get_line_hashes(p)

            # 已缓存（含短语预填）
            if self.cache.get(p_hash) is not None:
                continue

            # 复用
            reused = self.cache.find_reusable(line_hashes)
            if reused is not None:
                self.cache.put(p_hash, reused, line_hashes)
                continue

            # 👇 按模式决定是否查本地
            if mode in ('local', 'hybrid'):
                text = p['text']
                zh = self._lookup_local(text)
                if zh:
                    self.cache.put(p_hash, zh, line_hashes)
                    continue

            # 本地模式不命中的：不翻译，跳过
            if mode == 'local':
                continue

            # 混合 / 联网模式：走 API
            need_api.append((p, p_hash, line_hashes))

        # 立即推送已有译文
        self._emit_paragraphs(to_translate, img, region)

        t_emit = time.time()

        total_ms = (t_emit - t_start) * 1000
        if total_ms > SLOW_THRESHOLD_MS:
            log.warning(
                f"[处理慢] 总 {total_ms:.0f}ms | "
                f"帧差 {(t_frame_diff-t_start)*1000:.0f}ms | "
                f"OCR {(t_ocr-t_frame_diff)*1000:.0f}ms | "
                f"推送 {(t_emit-t_ocr)*1000:.0f}ms | "
                f"{len(to_translate)} 段 / {len(need_api)} 需API | "
                f"模式 {mode}"
            )

        if need_api:
            mt_tag = " [多线程OCR]" if use_mt else ""
            fast_tag = " [快档]" if self.state.fast_mode else ""
            log.info(f"[段落] 共 {len(to_translate)} 段 | "
                     f"需 API {len(need_api)} | 模式 {mode} | "
                     f"变化区 {w}x{h}{mt_tag}{fast_tag}")

            if self.state.fast_mode:
                for p, p_hash, lh in need_api:
                    if p_hash in self._pending_translates:
                        continue
                    self._pending_translates.add(p_hash)
                    self.translate_pool.submit(
                        self._translate_one_and_emit,
                        p, p_hash, lh, to_translate, img, region,
                    )
            else:
                if self.state.enable_async_translate:
                    self.translate_pool.submit(
                        self._translate_and_emit,
                        need_api, to_translate, img, region,
                    )
                else:
                    self._translate_and_emit(
                        need_api, to_translate, img, region,
                    )

    def _translate_one_and_emit(self, p, p_hash, lh, all_paragraphs, img, region):
        try:
            text = p['text']
            zh = self.translator.translate_lines([text])[0]
            self.cache.put(p_hash, zh, lh)

            if len(text) <= 30:
                self.phrase_cache.put(text, zh)

            if self.state.freeze_overlay:
                return

            self._emit_paragraphs(all_paragraphs, img, region)
        except Exception as e:
            log.exception(f"[单段翻译错误] {e}")
        finally:
            self._pending_translates.discard(p_hash)

    def _translate_and_emit(self, need_api, all_paragraphs, img, region):
        if self.state.freeze_overlay:
            return
        try:
            texts = [p['text'] for p, _, _ in need_api]
            translations = self.translator.translate_lines(texts)
            for (p, p_hash, lh), zh in zip(need_api, translations):
                self.cache.put(p_hash, zh, lh)

                t = p['text']
                if len(t) <= 30:
                    self.phrase_cache.put(t, zh)

            if self.state.freeze_overlay:
                return
            self._emit_paragraphs(all_paragraphs, img, region)
        except Exception as e:
            log.exception(f"[异步翻译错误] {e}")

    # ==================== 组装并推入 ====================

    def _emit_paragraphs(self, paragraphs, img, region):
        if not paragraphs or self.state.freeze_overlay:
            return

        use_inpaint = self.state.use_inpainting and CV2_AVAILABLE
        fine_enabled = self.state.enable_fine_text and CV2_AVAILABLE
        base_font = self.state.base_font_size

        final = []
        fine_used = 0
        inpaint_used = 0

        for p in paragraphs:
            p_hash = hash_paragraph(p)
            zh = self.cache.get(p_hash)
            if not zh:
                continue

            x1, y1, x2, y2 = p['box']

            use_fine_this = (fine_enabled and
                             is_fine_text_suitable(img, x1, y1, x2, y2))

            if use_fine_this:
                fine_used += 1

                with self._inpaint_lock:
                    if p_hash not in self.fine_cache:
                        self.fine_cache[p_hash] = inpaint_text_only(
                            img, x1, y1, x2, y2
                        )
                        if (self.cache.auto_clean and
                                len(self.fine_cache) > self.inpaint_cache_max):
                            self.fine_cache.popitem(last=False)
                    bg_image = self.fine_cache[p_hash]

                    if p_hash not in self.color_cache:
                        self.color_cache[p_hash] = sample_text_color(
                            img, x1, y1, x2, y2
                        )
                    fg = self.color_cache[p_hash]

                bg_hex = '#FFFFFF'
                item_extra = {'force_font_size': base_font}
            else:
                avg = self._crop_avg_color(img, x1, y1, x2, y2)
                bg_hex = f'#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}'
                luminance = avg[0]*0.299 + avg[1]*0.587 + avg[2]*0.114
                fg = '#FFFFFF' if luminance < 128 else '#000000'

                bg_image = None
                if use_inpaint:
                    inpaint_used += 1
                    bg_image = self._get_inpaint_cached(
                        p_hash, img, x1, y1, x2, y2
                    )
                item_extra = {}

            item = {
                'x1': x1 + region['left'],
                'y1': y1 + region['top'],
                'x2': x2 + region['left'],
                'y2': y2 + region['top'],
                'text': zh,
                'bg_hex': bg_hex,
                'fg': fg,
                'bg_image': bg_image,
                'key': paragraph_display_key(p),
            }
            item.update(item_extra)
            final.append(item)

        if fine_enabled:
            log.info(f"[精细模式] 使用 {fine_used} 段 | "
                     f"回退整框 {inpaint_used} 段")

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