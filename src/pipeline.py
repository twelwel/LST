"""后台处理管道"""

import queue
import re
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
    is_target_language_text,
    filter_invalid_items,
    filter_general_content,
    filter_english_only,
    filter_title_logo,
    text_similarity,
)
from inpaint import (
    inpaint_region,
    inpaint_text_only,
    sample_text_color,
    is_fine_text_suitable,
    CV2_AVAILABLE,
)
from phrase_cache import PhraseCache
from logger import get_logger
from config import (
    LOCAL_LOOKUP_MAX_WORDS,
    LOCAL_LOOKUP_MAX_CHARS,
)
from translation.strategies import STRATEGY_LABELS
from emit_debouncer import EmitDebouncer


log = get_logger('pipeline')


# ==================== 帧差检测 ====================
# 固定尺寸灰度缩略图 + 面积平均缩放。
# 旧实现用 NEAREST 抽样到 40x40，小字 / 细线经常一个采样点都命中不了，
# 导致帧差“不灵敏”；现在改为 160x90 + INTER_AREA，任何像素变化都会被
# 平均进单元格，再配合更低的像素阈值与变化比例阈值。
FRAME_THUMB_WIDTH = 160
FRAME_THUMB_HEIGHT = 90
FRAME_DIFF_PIXEL_THRESHOLD = 6
FRAME_DIFF_MIN_CHANGED_PIXELS = 4
FRAME_DIFF_MIN_COMPONENT_AREA = 6
FRAME_DIFF_PADDING_RATIO = 0.5
FRAME_DIFF_MAX_COMPONENTS = 6
FRAME_DIFF_MAX_ROIS = 4
FRAME_DIFF_ROI_MERGE_GAP = 24
FRAME_DIFF_MAX_ROI_AREA_RATIO = 0.55

# 分块 OCR 的兜底：每隔一段时间强制整屏补扫一次，
# 修正漏识别 / 切屏后未被分块覆盖的文本。
FULL_OCR_REFRESH_SEC = 8.0

# 兼容旧名称（外部脚本可能引用）
THUMB_SIZE = FRAME_THUMB_WIDTH
PIXEL_DIFF_THRESHOLD = FRAME_DIFF_PIXEL_THRESHOLD

INPAINT_CACHE_MAX = 200

MULTITHREAD_OCR_MIN_AREA = 600 * 150
MULTITHREAD_OCR_OVERLAP = 40
MULTITHREAD_OCR_MIN_CHUNK_H = 400
MULTITHREAD_OCR_MAX_CHUNKS = 4

SLOW_THRESHOLD_MS = 300
PERF_REPORT_INTERVAL = 10.0

POSITION_MATCH_TOLERANCE = 40
TRACKER_TIMEOUT = 1.5

OCR_VERIFY_PADDING = 8
OCR_VERIFY_SIMILARITY = 0.6
OCR_VERIFY_MIN_W = 32
OCR_VERIFY_MIN_H = 32
OCR_VERIFY_MIN_AREA = 32 * 32
OCR_VERIFY_MAX_ITEMS = 20
OCR_VERIFY_FAIL_OPEN = True


# ==================== 文本归一化（稳定 key / 同帧去重 共用）=================

# 需要剔除的标点/符号集合。
# 注意：字符类里的 '-' 必须放在最后，否则会被解析成区间；
# 另把 ASCII 单引号单独放进双引号 raw string，避免 `''` 意外截断字面量
# （历史写法被截断后，后半段不再是 raw string，会产生 SyntaxWarning）。
_PUNCT_RE = re.compile(
    r'[.,;:!?"\'`~@#$%^&*()\[\]{}<>|\\/'
    r'《》【】。，；：！？、'
    r'""'
    r"''"
    r'…—_-]+'
)


def normalize_key_text(text):
    """归一化文本：去空白 → 去标点 → 转小写

    用于「稳定 key 跟踪」与「同帧去重」两处比较。
    """
    if not text:
        return ''
    t = re.sub(r'\s+', '', text)
    t = _PUNCT_RE.sub('', t)
    return t.lower()


class _PerfStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.frames = 0
        self.t_frame_diff = 0.0
        self.t_ocr = 0.0
        self.t_filter = 0.0
        self.t_merge = 0.0
        self.t_verify = 0.0
        self.t_emit = 0.0
        self.t_total = 0.0
        self.frames_with_ocr = 0
        self.frames_skipped = 0
        self.verify_pass = 0
        self.verify_fail = 0

    def add(self, key, ms):
        with self._lock:
            setattr(self, key, getattr(self, key) + ms)

    def tick(self, has_ocr=False, skipped=False):
        with self._lock:
            self.frames += 1
            if has_ocr:
                self.frames_with_ocr += 1
            if skipped:
                self.frames_skipped += 1

    def verify(self, passed):
        with self._lock:
            if passed:
                self.verify_pass += 1
            else:
                self.verify_fail += 1

    def summary_and_reset(self):
        with self._lock:
            n = max(1, self.frames)
            no = max(1, self.frames_with_ocr)
            result = {
                'frames': self.frames,
                'frames_with_ocr': self.frames_with_ocr,
                'frames_skipped': self.frames_skipped,
                'frame_diff': self.t_frame_diff / n,
                'ocr': self.t_ocr / no,
                'filter': self.t_filter / n,
                'merge': self.t_merge / n,
                'verify': self.t_verify / n,
                'emit': self.t_emit / n,
                'total': self.t_total / n,
                'verify_pass': self.verify_pass,
                'verify_fail': self.verify_fail,
            }
            self.frames = 0
            self.t_frame_diff = 0.0
            self.t_ocr = 0.0
            self.t_filter = 0.0
            self.t_merge = 0.0
            self.t_verify = 0.0
            self.t_emit = 0.0
            self.t_total = 0.0
            self.frames_with_ocr = 0
            self.frames_skipped = 0
            self.verify_pass = 0
            self.verify_fail = 0
            return result


class AppState:
    def __init__(self, initial_region,
                 frame_similarity_threshold=0.9,
                 position_quantize=10,
                 use_inpainting=False,
                 enable_fine_text=False,
                 enable_general_filter=True,
                 enable_english_only=False,
                 enable_title_filter=False,
                 enable_ocr_verify=False,
                 enable_fast_lookup=True,
                 mode_cfg=None,
                 enable_frame_diff=True,
                 enable_filter=True,
                 enable_block_ocr=True,
                 enable_async_translate=True,
                 enable_window_pool=True,
                 enable_multithread_ocr=False,
                 enable_ocr_speed_opt=True,
                 source_lang='auto',
                 target_lang='zh'):
        self.running = False
        self.region = dict(initial_region)
        self.base_font_size = 15
        self.use_inpainting = use_inpainting
        self.enable_fine_text = enable_fine_text
        self.enable_general_filter = enable_general_filter
        self.enable_english_only = enable_english_only
        self.enable_title_filter = enable_title_filter
        self.enable_ocr_verify = enable_ocr_verify
        self.enable_fast_lookup = enable_fast_lookup
        self.enable_ocr_speed_opt = enable_ocr_speed_opt
        self.translation_strategy = 'single_translate'
        self.display_mode = 'overlay'
        self.fg_color = 'auto'
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.lock = threading.Lock()
        self.frame_similarity_threshold = frame_similarity_threshold
        self.position_quantize = position_quantize
        self.last_frame_keys = None
        self.last_frame_pixels = None
        self.last_full_scan_ts = 0.0
        self.scene_reset_pending = False
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
        self._scroll_until = 0.0
        if mode_cfg:
            self.apply_mode(mode_cfg)

    def mark_scroll(self):
        self._scroll_until = time.time() + 0.5

    def is_scrolling(self):
        return time.time() < self._scroll_until

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
        self.last_full_scan_ts = 0.0
        self.scene_reset_pending = False
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

        self.phrase_cache = PhraseCache()

        self.translate_pool = ThreadPoolExecutor(max_workers=4)
        self._pending_translates = set()

        self.emit_debouncer = EmitDebouncer()
        self._known_items = {}
        self._known_lock = threading.Lock()

        self._generation = 0
        self._tracker_entries = {}
        self._tracker_lock = threading.Lock()

        self._verify_engine = None
        self._verify_engine_lock = threading.Lock()

        self._perf = _PerfStats()
        self._perf_last_report = time.time()

    def start(self):
        threading.Thread(target=self._process_loop, daemon=True).start()

    def reset_debouncer(self):
        with self._known_lock:
            self._known_items.clear()
            self._generation += 1

        with self._tracker_lock:
            self._tracker_entries.clear()

        self.emit_debouncer.reset()

        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

    def reset_perf(self):
        self._perf = _PerfStats()
        self._perf_last_report = time.time()

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

    def _assign_stable_key(self, para_box, fallback_key,
                            used_keys=None, text=None):
        now = time.time()
        x1, y1, x2, y2 = para_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cur_text_norm = normalize_key_text(text)

        with self._tracker_lock:
            expired = [k for k, e in self._tracker_entries.items()
                       if now - e['last_seen'] > TRACKER_TIMEOUT]
            for k in expired:
                del self._tracker_entries[k]

            best_key = None
            best_dist = float('inf')

            for k, e in self._tracker_entries.items():
                if used_keys and k in used_keys:
                    continue

                ex1, ey1, ex2, ey2 = e['box']
                ecx = (ex1 + ex2) / 2
                ecy = (ey1 + ey2) / 2
                d = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5

                e_text_norm = normalize_key_text(e.get('text', ''))

                if (cur_text_norm and e_text_norm and
                        d < POSITION_MATCH_TOLERANCE):
                    from difflib import SequenceMatcher as _SM
                    sim = _SM(None, cur_text_norm, e_text_norm).ratio()
                    if sim < 0.3:
                        continue

                same_text = (cur_text_norm and e_text_norm and
                             cur_text_norm == e_text_norm)

                if same_text and d < POSITION_MATCH_TOLERANCE * 2:
                    self._tracker_entries[k]['box'] = (x1, y1, x2, y2)
                    self._tracker_entries[k]['last_seen'] = now
                    if text:
                        self._tracker_entries[k]['text'] = text
                    if used_keys is not None:
                        used_keys.add(k)
                    return k

                if d < best_dist:
                    best_dist = d
                    best_key = k

            if best_key is not None and best_dist < POSITION_MATCH_TOLERANCE:
                self._tracker_entries[best_key]['box'] = (x1, y1, x2, y2)
                self._tracker_entries[best_key]['last_seen'] = now
                if text:
                    self._tracker_entries[best_key]['text'] = text
                if used_keys is not None:
                    used_keys.add(best_key)
                return best_key

            if used_keys and fallback_key in used_keys:
                fallback_key = hash_text(
                    f"{fallback_key}|{int(x1)}|{int(y1)}")
                while used_keys and fallback_key in used_keys:
                    fallback_key = hash_text(fallback_key + "|salt")

            self._tracker_entries[fallback_key] = {
                'box': (x1, y1, x2, y2),
                'last_seen': now,
                'text': text or '',
            }
            if used_keys is not None:
                used_keys.add(fallback_key)
            return fallback_key

    def _dedupe_same_frame(self, cur_items):
        def center(item):
            return ((item['x1'] + item['x2']) / 2,
                    (item['y1'] + item['y2']) / 2)

        def area(item):
            return (item['x2'] - item['x1']) * (item['y2'] - item['y1'])

        from difflib import SequenceMatcher

        items_list = list(cur_items.values())
        result = {}
        processed = set()

        for i, item in enumerate(items_list):
            if i in processed:
                continue
            text_i = normalize_key_text(item.get('text', ''))
            if not text_i:
                result[item['key']] = item
                processed.add(i)
                continue

            group = [item]
            cx_i, cy_i = center(item)
            for j in range(i + 1, len(items_list)):
                if j in processed:
                    continue
                other = items_list[j]
                text_j = normalize_key_text(other.get('text', ''))
                if not text_j:
                    continue

                if text_i == text_j:
                    group.append(other)
                    processed.add(j)
                    continue

                sim = SequenceMatcher(None, text_i, text_j).ratio()
                if sim < 0.9:
                    continue
                cx_j, cy_j = center(other)
                dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
                if dist < 60:
                    group.append(other)
                    processed.add(j)

            best = max(group, key=area)
            result[best['key']] = best
            processed.add(i)

        return result

    def _resolve_fg_color(self, avg_rgb):
        user_fg = getattr(self.state, 'fg_color', 'auto')
        if user_fg and user_fg != 'auto':
            return user_fg
        luminance = (avg_rgb[0] * 0.299 +
                     avg_rgb[1] * 0.587 +
                     avg_rgb[2] * 0.114)
        return '#FFFFFF' if luminance < 128 else '#000000'

    def _make_thumbnail(self, img):
        """生成帧差用的灰度缩略图。

        使用固定 160x90 + INTER_AREA / BOX 缩放，而不是跳点抽样：
        即使只有几个像素的文字变化，也会被平均进缩略图单元格，
        避免小字、细线、单行字幕变化被漏检。
        """
        if img is None or getattr(img, 'size', 0) == 0:
            return np.zeros((FRAME_THUMB_HEIGHT, FRAME_THUMB_WIDTH),
                            dtype=np.int16)

        if _CV2_OK:
            try:
                if img.ndim == 3 and img.shape[2] >= 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                small = cv2.resize(
                    gray, (FRAME_THUMB_WIDTH, FRAME_THUMB_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                return small.astype(np.int16)
            except Exception:
                pass

        try:
            if img.ndim == 3 and img.shape[2] >= 3:
                pil = PILImage.fromarray(img[:, :, :3]).convert('L')
            else:
                pil = PILImage.fromarray(img).convert('L')
            small = pil.resize(
                (FRAME_THUMB_WIDTH, FRAME_THUMB_HEIGHT),
                PILImage.Resampling.BOX,
            )
            return np.asarray(small, dtype=np.int16)
        except Exception:
            return np.zeros((FRAME_THUMB_HEIGHT, FRAME_THUMB_WIDTH),
                            dtype=np.int16)

    def _get_changed_rois(self, img):
        """返回本帧需要 OCR 的区域列表；None 表示画面静止可跳过。

        与旧实现相比：
        - 使用面积平均缩略图，小文本变化也能被检测到；
        - 支持返回多个变化区块（而不是一个巨大的外接框）；
        - 大范围变化 / 定期补扫时直接返回整屏，保证切屏后不漏文本。
        """
        h, w = img.shape[:2]
        full_frame = [(0, 0, w, h)]
        now = time.monotonic()

        if not self.state.enable_frame_diff:
            self.state.last_full_scan_ts = now
            return full_frame

        crt = self.state.change_ratio_threshold
        ffr = self.state.full_frame_ratio
        base_pad = self.state.bbox_padding

        thumb = self._make_thumbnail(img)
        prev = self.state.last_frame_pixels

        if prev is None or thumb.shape != prev.shape:
            self.state.last_frame_pixels = thumb
            self.state.last_full_scan_ts = now
            return full_frame

        diff = np.abs(thumb - prev)
        mask = diff > FRAME_DIFF_PIXEL_THRESHOLD
        changed_pixels = int(mask.sum())

        # 分块 OCR 兜底：无论当前有没有变化，到时间就整屏补扫一次，
        # 修正上一次分块漏识别 / 未覆盖的文本。
        if self.state.enable_block_ocr:
            last_full = getattr(self.state, 'last_full_scan_ts', 0.0)
            if now - last_full >= FULL_OCR_REFRESH_SEC:
                self.state.last_frame_pixels = thumb
                self.state.last_full_scan_ts = now
                log.debug(f"[帧差] 距上次整屏 {now - last_full:.1f}s，"
                          "执行整屏补扫")
                return full_frame

        if changed_pixels < FRAME_DIFF_MIN_CHANGED_PIXELS:
            # 不更新基准帧：很小的变化会累积到下一次比较，
            # 避免慢速渐变 / 逐帧轻微变化永远达不到阈值。
            return None

        changed_ratio = changed_pixels / float(diff.size)
        if changed_ratio < crt:
            return None

        if not self.state.enable_block_ocr:
            self.state.last_frame_pixels = thumb
            self.state.last_full_scan_ts = now
            return full_frame

        # 大范围变化（切屏 / 换页 / 场景切换）直接整屏 OCR，
        # 避免分块只覆盖一部分，导致新出现的英文没有被翻译。
        if changed_ratio > ffr:
            self.state.last_frame_pixels = thumb
            self.state.last_full_scan_ts = now
            self.state.scene_reset_pending = True
            log.debug(f"[帧差] 大范围变化 {changed_ratio:.1%} > "
                      f"{ffr:.1%}，整屏 OCR")
            return full_frame

        rois = self._build_changed_rois(mask, w, h, base_pad)
        if not rois:
            return None

        if rois == full_frame:
            self.state.last_frame_pixels = thumb
            self.state.last_full_scan_ts = now
            log.debug("[帧差] 变化区块过多，整屏 OCR")
            return full_frame

        total = sum(rw * rh for _, _, rw, rh in rois)
        if total >= FRAME_DIFF_MAX_ROI_AREA_RATIO * w * h:
            self.state.last_frame_pixels = thumb
            self.state.last_full_scan_ts = now
            log.debug(f"[帧差] 变化区域占 {total / (w * h):.1%}，"
                      "整屏 OCR")
            return full_frame

        self.state.last_frame_pixels = thumb
        log.debug(f"[帧差] 变化 {changed_ratio:.2%}，分块 {len(rois)} 区")
        return rois

    def _build_changed_rois(self, mask, img_w, img_h, base_pad):
        """把变化 mask 转成若干带外扩的 OCR 区块。"""
        th, tw = mask.shape[:2]
        sx = img_w / float(tw)
        sy = img_h / float(th)

        if not _CV2_OK:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return []
            x1 = int(xs.min()) * sx
            y1 = int(ys.min()) * sy
            x2 = (int(xs.max()) + 1) * sx
            y2 = (int(ys.max()) + 1) * sy
            roi = self._pad_roi(x1, y1, x2 - x1, y2 - y1,
                                base_pad, base_pad, img_w, img_h)
            return [roi] if roi else []

        try:
            mask_u8 = (mask.astype(np.uint8) * 255)
            kernel = np.ones((3, 3), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                mask_u8, connectivity=8)
        except Exception as e:
            log.debug(f"[帧差] 连通域分析失败：{e}")
            return []

        comps = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= FRAME_DIFF_MIN_COMPONENT_AREA:
                comps.append((area, i))

        if not comps:
            return []

        comps.sort(reverse=True)
        if len(comps) > FRAME_DIFF_MAX_COMPONENTS:
            # 变化点太分散，逐块 OCR 反而更慢，直接整屏。
            return [(0, 0, img_w, img_h)]

        rois = []
        for _area, i in comps:
            cx = int(stats[i, cv2.CC_STAT_LEFT]) * sx
            cy = int(stats[i, cv2.CC_STAT_TOP]) * sy
            cw = max(1.0, int(stats[i, cv2.CC_STAT_WIDTH]) * sx)
            ch = max(1.0, int(stats[i, cv2.CC_STAT_HEIGHT]) * sy)
            pad_x = max(base_pad, int(cw * 0.2))
            pad_y = max(base_pad,
                        int(ch * FRAME_DIFF_PADDING_RATIO))
            roi = self._pad_roi(cx, cy, cw, ch,
                                pad_x, pad_y, img_w, img_h)
            if roi:
                rois.append(roi)

        merged = self._merge_rois(rois, FRAME_DIFF_ROI_MERGE_GAP)
        if not merged:
            return []
        if len(merged) > FRAME_DIFF_MAX_ROIS:
            return [(0, 0, img_w, img_h)]
        return merged


    @staticmethod
    def _pad_roi(x, y, w, h, pad_x, pad_y, img_w, img_h):
        x1 = max(0, int(round(x - pad_x)))
        y1 = max(0, int(round(y - pad_y)))
        x2 = min(img_w, int(round(x + w + pad_x)))
        y2 = min(img_h, int(round(y + h + pad_y)))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _merge_rois(rois, gap=FRAME_DIFF_ROI_MERGE_GAP):
        """合并重叠或间距小于 gap 的区块，减少 OCR 调用次数。"""
        boxes = [[r[0], r[1], r[0] + r[2], r[1] + r[3]]
                 for r in rois if r]
        merged = True
        while merged:
            merged = False
            out = []
            while boxes:
                a = boxes.pop()
                i = 0
                while i < len(boxes):
                    b = boxes[i]
                    overlap = not (
                        a[2] + gap < b[0] or b[2] + gap < a[0] or
                        a[3] + gap < b[1] or b[3] + gap < a[1]
                    )
                    if overlap:
                        a = [min(a[0], b[0]), min(a[1], b[1]),
                             max(a[2], b[2]), max(a[3], b[3])]
                        boxes.pop(i)
                        merged = True
                    else:
                        i += 1
                out.append(a)
            boxes = out
        return [(b[0], b[1], max(1, b[2] - b[0]),
                 max(1, b[3] - b[1])) for b in boxes]

    def _get_changed_bbox(self, img):
        """兼容旧接口：始终返回单个外接框。"""
        rois = self._get_changed_rois(img)
        if rois is None:
            return None
        if len(rois) == 1:
            return rois[0]
        x1 = min(r[0] for r in rois)
        y1 = min(r[1] for r in rois)
        x2 = max(r[0] + r[2] for r in rois)
        y2 = max(r[1] + r[3] for r in rois)
        return (x1, y1, x2 - x1, y2 - y1)


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
        # 只要有新文本就必须继续处理；仅当本帧文本是上一帧的子集时才算未变化。
        # 旧实现用 Jaccard >= 0.9，切屏后“大量相同文本 + 少量新英文”会被
        # 误判为未变化，导致新增英文没有被翻译。
        return not (cur_keys - prev)

    def _shrink_for_ocr(self, crop):
        max_w = self.state.max_ocr_width
        max_h = self.state.max_ocr_height
        h, w = crop.shape[:2]
        if w <= max_w and h <= max_h:
            return crop, 1.0
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if _CV2_OK and self.state.enable_ocr_speed_opt:
            return cv2.resize(crop, (new_w, new_h),
                              interpolation=cv2.INTER_LINEAR), scale
        pil = PILImage.fromarray(crop).resize(
            (new_w, new_h), PILImage.Resampling.BILINEAR,
        )
        return np.array(pil), scale

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

        chunks = []
        for i in range(n):
            cy1 = y + (h * i) // n
            cy2 = y + (h * (i + 1)) // n
            if cy2 <= cy1:
                continue
            chunks.append((cy1, cy2))

        if not chunks:
            crop = img[y:y+h, x:x+w]
            crop_small, scale = self._shrink_for_ocr(crop)
            items = self.ocr.recognize(crop_small)
            return self._remap_items(items, x, y, scale)

        crops = []
        offsets = []
        for cy1, cy2 in chunks:
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
        if not items:
            return []
        result = []
        for it in items:
            text = it['text'].strip()
            if not text:
                continue
            bx1, by1, bx2, by2 = it['box']
            a1 = (bx2 - bx1) * (by2 - by1)
            dup = False
            for idx, kept in enumerate(result):
                if kept['text'].strip() != text:
                    continue
                kx1, ky1, kx2, ky2 = kept['box']
                ix1 = max(bx1, kx1)
                iy1 = max(by1, ky1)
                ix2 = min(bx2, kx2)
                iy2 = min(by2, ky2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                a2 = (kx2 - kx1) * (ky2 - ky1)
                union = a1 + a2 - inter
                if union <= 0:
                    continue
                if inter / union > 0.5:
                    dup = True
                    if a1 > a2:
                        result[idx] = it
                    break
            if not dup:
                result.append(it)
        return result

    def _prefill_from_phrases(self, items):
        if not self.state.enable_fast_lookup:
            return 0
        filled = 0
        for it in items:
            text = it['text'].strip()
            if not text:
                continue
            if len(text) > LOCAL_LOOKUP_MAX_CHARS:
                continue
            wc = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
            if wc > LOCAL_LOOKUP_MAX_WORDS:
                continue
            zh = self.phrase_cache.lookup_fuzzy(text)
            if not zh:
                continue
            fake_para = {'text': text, 'box': it['box'], 'lines': [it]}
            p_hash = hash_paragraph(fake_para)
            if self.cache.get(p_hash) is None:
                line_hashes = get_line_hashes(fake_para)
                self.cache.put(p_hash, zh, line_hashes)
                filled += 1
        if filled:
            log.info(f"[快速响应] 预填 {filled} 条段落")
        return filled

    def _lookup_local(self, text):
        if not self.state.enable_fast_lookup:
            return None
        text = (text or '').strip()
        if not text:
            return None
        if len(text) > LOCAL_LOOKUP_MAX_CHARS:
            return None

        has_kana = any('\u3040' <= ch <= '\u30ff' for ch in text)

        if not has_kana:
            word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",
                                        text))
            if word_count > LOCAL_LOOKUP_MAX_WORDS:
                return None

        zh = self.phrase_cache.lookup_fuzzy(text)
        if zh:
            return zh
        return None

    def _get_verify_engine(self):
        try:
            if self.ocr.aux_engines:
                return self.ocr.aux_engines[0]
        except Exception:
            pass

        if self._verify_engine is not None:
            return self._verify_engine

        with self._verify_engine_lock:
            if self._verify_engine is not None:
                return self._verify_engine
            try:
                from rapidocr import RapidOCR
                log.info("[验证] 创建独立 CPU 引擎...")
                self._verify_engine = RapidOCR()
                log.info("[验证] 独立 CPU 引擎就绪")
            except Exception as e:
                log.error(f"[验证] 引擎创建失败：{e}")
                self._verify_engine = None
        return self._verify_engine

    def _verify_paragraph(self, p, img, region):
        x1, y1, x2, y2 = p['box']
        w = x2 - x1
        h = y2 - y1
        if w < OCR_VERIFY_MIN_W or h < OCR_VERIFY_MIN_H:
            return True
        if w * h < OCR_VERIFY_MIN_AREA:
            return True

        pad = OCR_VERIFY_PADDING
        img_h, img_w = img.shape[:2]
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img_w, x2 + pad)
        cy2 = min(img_h, y2 + pad)

        if cx2 - cx1 < OCR_VERIFY_MIN_W or cy2 - cy1 < OCR_VERIFY_MIN_H:
            return True

        crop = img[cy1:cy2, cx1:cx2]
        if crop is None or crop.size == 0:
            return True

        try:
            if crop.dtype != np.uint8:
                crop = crop.astype(np.uint8)
            if crop.ndim == 2:
                if _CV2_OK:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                else:
                    crop = np.stack([crop] * 3, axis=-1)
            elif crop.ndim == 3:
                if crop.shape[2] == 4:
                    if _CV2_OK:
                        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR)
                    else:
                        crop = crop[:, :, :3]
                elif crop.shape[2] == 1:
                    if _CV2_OK:
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    else:
                        crop = np.repeat(crop, 3, axis=2)
            else:
                return True
        except Exception:
            return True

        crop_h, crop_w = crop.shape[:2]
        if crop_h < OCR_VERIFY_MIN_H or crop_w < OCR_VERIFY_MIN_W:
            return True

        try:
            crop_small, _ = self._shrink_for_ocr(crop)
        except Exception:
            return True

        engine = self._get_verify_engine()
        if engine is None:
            return True

        try:
            result = engine(crop_small)
        except Exception:
            return True

        if result is None or not result.txts:
            return False

        new_text = ''.join(
            (t or '').strip() for t in result.txts if t
        )
        if not new_text:
            return False

        old_text = (p.get('text') or '').strip()
        if not old_text:
            return False

        sim = text_similarity(old_text, new_text)
        return sim >= OCR_VERIFY_SIMILARITY

    def _process_one(self, img):
        if self.state.freeze_overlay:
            return

        t_start = time.time()
        rois = self._get_changed_rois(img)
        t_frame_diff = time.time()

        if rois is None:
            self.state.idle_count += 1
            self._perf.tick(skipped=True)
            self._maybe_report_perf()
            return

        self.state.idle_count = 0
        self.state.idle_skip_counter = 0
        self.state.pending_change = True
        if getattr(self.state, 'scene_reset_pending', False):
            # 切屏 / 大范围变化：清空防抖状态和旧译文，避免旧屏幕的内容
            # 遮挡新屏幕，或旧异步翻译任务回填到新画面。
            self.state.scene_reset_pending = False
            self.reset_debouncer()
        scan_generation = self._generation
        self._scan_generation = scan_generation
        region = self.state.snapshot_region()

        items = []
        used_mt = False
        for x, y, w, h in rois:
            if w < 10 or h < 10:
                continue
            use_mt = (self.state.enable_multithread_ocr and
                      self.ocr.pool_size > 1 and
                      w * h >= MULTITHREAD_OCR_MIN_AREA)
            if use_mt:
                used_mt = True
                items.extend(self._ocr_multithread(img, x, y, w, h))
            else:
                crop = img[y:y+h, x:x+w]
                crop_small, scale = self._shrink_for_ocr(crop)
                raw = self.ocr.recognize(crop_small)
                items.extend(self._remap_items(raw, x, y, scale))

        # 多个变化区块可能有重叠，统一去重，避免同一段被翻译两次
        items = self._dedupe(items)
        t_ocr = time.time()

        if not items:
            # OCR 没有识别到文本：清掉本次扫描区域内的旧译文，
            # 防止切屏后旧覆盖层停留在新画面上。
            self._emit_paragraphs([], img, region, rois)
            self._perf.tick(has_ocr=True)
            self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
            self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
            self._perf.add('t_total', (t_ocr - t_start) * 1000)
            self._maybe_report_perf()
            return

        if self.state.enable_filter:
            items = filter_invalid_items(
                items, region['width'], region['height'])
            if not items:
                self._emit_paragraphs([], img, region, rois)
                self._perf.tick(has_ocr=True)
                self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
                self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
                self._perf.add('t_total', (t_ocr - t_start) * 1000)
                self._maybe_report_perf()
                return

        if self.state.enable_general_filter:
            items = filter_general_content(items)
            if not items:
                self._emit_paragraphs([], img, region, rois)
                self._perf.tick(has_ocr=True)
                self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
                self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
                self._perf.add('t_total', (t_ocr - t_start) * 1000)
                self._maybe_report_perf()
                return

        target_lang = getattr(self.state, 'target_lang', 'zh')
        english_only_on = getattr(self.state, 'enable_english_only', False)
        if english_only_on and target_lang != 'en':
            items = filter_english_only(items)
            if not items:
                self._emit_paragraphs([], img, region, rois)
                self._perf.tick(has_ocr=True)
                self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
                self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
                self._perf.add('t_total', (t_ocr - t_start) * 1000)
                self._maybe_report_perf()
                return

        if getattr(self.state, 'enable_title_filter', False):
            items = filter_title_logo(items)
            if not items:
                self._emit_paragraphs([], img, region, rois)
                self._perf.tick(has_ocr=True)
                self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
                self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
                self._perf.add('t_total', (t_ocr - t_start) * 1000)
                self._maybe_report_perf()
                return

        t_filter = time.time()

        self._prefill_from_phrases(items)

        cur_keys = self._make_frame_keys(items)
        self.state.last_frame_keys = cur_keys

        paragraphs = merge_lines_to_paragraphs(items)
        to_translate = [
            p for p in paragraphs
            if not is_target_language_text(p['text'], target_lang)
        ]

        t_merge = time.time()

        if not to_translate:
            # 即使没有需要翻译的文本，也要按本次扫描区域更新 known_items，
            # 否则分块 OCR 下已经消失的旧译文会一直残留。
            self._emit_paragraphs(to_translate, img, region, rois)
            self._perf.tick(has_ocr=True)
            self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
            self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
            self._perf.add('t_filter', (t_filter - t_ocr) * 1000)
            self._perf.add('t_merge', (t_merge - t_filter) * 1000)
            self._perf.add('t_total', (t_merge - t_start) * 1000)
            self._maybe_report_perf()
            return

        strategy = getattr(self.state, 'translation_strategy',
                           'single_translate')
        strategy_name = STRATEGY_LABELS.get(strategy, strategy)

        # ⭐ 所有未命中的段落都交给策略层（含离线策略）
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
            text = p['text']
            zh = self._lookup_local(text)
            if zh:
                self.cache.put(p_hash, zh, line_hashes)
                continue
            need_api.append((p, p_hash, line_hashes))

        self._emit_paragraphs(to_translate, img, region, rois)

        t_emit = time.time()
        total_ms = (t_emit - t_start) * 1000

        self._perf.tick(has_ocr=True)
        self._perf.add('t_frame_diff', (t_frame_diff - t_start) * 1000)
        self._perf.add('t_ocr', (t_ocr - t_frame_diff) * 1000)
        self._perf.add('t_filter', (t_filter - t_ocr) * 1000)
        self._perf.add('t_merge', (t_merge - t_filter) * 1000)
        self._perf.add('t_emit', (t_emit - t_merge) * 1000)
        self._perf.add('t_total', total_ms)
        self._maybe_report_perf()

        if total_ms > SLOW_THRESHOLD_MS:
            log.warning(
                f"[处理慢] 总 {total_ms:.0f}ms | "
                f"帧差 {(t_frame_diff-t_start)*1000:.0f}ms | "
                f"OCR {(t_ocr-t_frame_diff)*1000:.0f}ms | "
                f"过滤 {(t_filter-t_ocr)*1000:.0f}ms | "
                f"合并 {(t_merge-t_filter)*1000:.0f}ms | "
                f"推送 {(t_emit-t_merge)*1000:.0f}ms | "
                f"{len(to_translate)} 段 / {len(need_api)} 需API | "
                f"策略 {strategy_name}"
            )

        if need_api:
            mt_tag = " [多线程OCR]" if used_mt else ""
            fast_tag = " [快档]" if self.state.fast_mode else ""
            roi_tag = " / ".join(f"{rw}x{rh}"
                                 for _, _, rw, rh in rois)
            log.info(f"[段落] 共 {len(to_translate)} 段 | "
                     f"需 API {len(need_api)} | "
                     f"策略 {strategy_name} ({strategy}) | "
                     f"变化区 {roi_tag}{mt_tag}{fast_tag}")

            if self.state.fast_mode:
                for p, p_hash, lh in need_api:
                    if p_hash in self._pending_translates:
                        continue
                    self._pending_translates.add(p_hash)
                    self.translate_pool.submit(
                        self._translate_one_and_emit,
                        p, p_hash, lh, to_translate, img, region, rois,
                        scan_generation,
                    )
            else:
                if self.state.enable_async_translate:
                    self.translate_pool.submit(
                        self._translate_and_emit,
                        need_api, to_translate, img, region, rois,
                        scan_generation,
                    )
                else:
                    self._translate_and_emit(
                        need_api, to_translate, img, region, rois,
                        scan_generation,
                    )

    def _maybe_report_perf(self):
        now = time.time()
        if now - self._perf_last_report < PERF_REPORT_INTERVAL:
            return
        self._perf_last_report = now
        s = self._perf.summary_and_reset()
        if s['frames'] == 0:
            return

        verify_tag = ''
        if self.state.enable_ocr_verify:
            verify_tag = (f" | 验证 通过 {s['verify_pass']} / "
                          f"丢弃 {s['verify_fail']} | "
                          f"验证 {s['verify']:.1f}ms")

        log.info(
            f"[性能] 最近 {PERF_REPORT_INTERVAL:.0f}s | "
            f"帧数 {s['frames']}（OCR {s['frames_with_ocr']} / "
            f"跳过 {s['frames_skipped']}）| "
            f"帧差 {s['frame_diff']:.1f}ms | "
            f"OCR {s['ocr']:.1f}ms | "
            f"过滤 {s['filter']:.1f}ms | "
            f"合并 {s['merge']:.1f}ms | "
            f"推送 {s['emit']:.1f}ms | "
            f"总计 {s['total']:.1f}ms"
            f"{verify_tag}"
        )

    def _translate_one_and_emit(self, p, p_hash, lh, all_paragraphs,
                                img, region, scan_rois=None,
                                expected_generation=None):
        try:
            text = p['text']
            zh = self.translator.translate_lines([text])[0]
            self.cache.put(p_hash, zh, lh)
            if len(text) <= 30:
                self.phrase_cache.put(text, zh)
            if self.state.freeze_overlay:
                return
            self._emit_paragraphs(all_paragraphs, img, region, scan_rois,
                                  expected_generation)
        except Exception as e:
            log.exception(f"[单段翻译错误] {e}")
        finally:
            self._pending_translates.discard(p_hash)

    def _translate_and_emit(self, need_api, all_paragraphs, img, region,
                            scan_rois=None, expected_generation=None):
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
            self._emit_paragraphs(all_paragraphs, img, region, scan_rois,
                                  expected_generation)
        except Exception as e:
            log.exception(f"[异步翻译错误] {e}")

    def _emit_paragraphs(self, paragraphs, img, region, scan_rois=None,
                         expected_generation=None):
        if self.state.freeze_overlay:
            return
        if expected_generation is None:
            expected_generation = getattr(self, '_scan_generation', None)
        if expected_generation is not None:
            with self._known_lock:
                if expected_generation != self._generation:
                    # 切屏 / 区域变更后，旧的异步翻译结果不能再回填。
                    return

        use_inpaint = self.state.use_inpainting and CV2_AVAILABLE
        fine_enabled = self.state.enable_fine_text and CV2_AVAILABLE
        base_font = self.state.base_font_size
        display_mode = getattr(self.state, 'display_mode', 'overlay')
        verify_on = getattr(self.state, 'enable_ocr_verify', False)

        try:
            scroll_mode = self.state.is_scrolling()
        except Exception:
            scroll_mode = False

        cur_items = {}
        fine_used = 0
        inpaint_used = 0
        verify_failed = 0
        verify_count = 0
        t_verify_total = 0.0
        used_keys = set()

        for p in paragraphs:
            p_hash = hash_paragraph(p)
            zh = self.cache.get(p_hash)
            if not zh:
                continue

            x1, y1, x2, y2 = p['box']

            if verify_on and verify_count < OCR_VERIFY_MAX_ITEMS:
                verify_count += 1
                t_v0 = time.time()
                passed = self._verify_paragraph(p, img, region)
                t_v1 = time.time()
                t_verify_total += (t_v1 - t_v0) * 1000
                self._perf.verify(passed)
                if not passed:
                    verify_failed += 1
                    continue

            fallback_key = paragraph_display_key(p)
            display_key = self._assign_stable_key(
                (x1, y1, x2, y2), fallback_key, used_keys,
                text=p.get('text', ''))

            if display_mode in ('bilingual', 'window'):
                avg = self._crop_avg_color(img, x1, y1, x2, y2)
                bg_hex = f'#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}'
                fg = self._resolve_fg_color(avg)

                subtitle = False
                if display_mode == 'bilingual':
                    try:
                        from layout import is_subtitle_text
                        subtitle = is_subtitle_text(p['text'], p['box'])
                    except Exception:
                        subtitle = False

                cur_items[display_key] = {
                    'x1': x1 + region['left'],
                    'y1': y1 + region['top'],
                    'x2': x2 + region['left'],
                    'y2': y2 + region['top'],
                    'orig_y2': y2 + region['top'],
                    'text': zh,
                    'bg_hex': bg_hex,
                    'fg': fg,
                    'bg_image': None,
                    'key': display_key,
                    'bilingual': (display_mode == 'bilingual'),
                    'subtitle': subtitle,
                    'orig_w': x2 - x1,
                    'orig_h': y2 - y1,
                }
                continue

            use_fine_this = (fine_enabled and
                             is_fine_text_suitable(img, x1, y1, x2, y2))

            if use_fine_this:
                fine_used += 1
                with self._inpaint_lock:
                    if p_hash not in self.fine_cache:
                        self.fine_cache[p_hash] = inpaint_text_only(
                            img, x1, y1, x2, y2)
                        if (self.cache.auto_clean and
                                len(self.fine_cache) > self.inpaint_cache_max):
                            self.fine_cache.popitem(last=False)
                    bg_image = self.fine_cache[p_hash]

                    if p_hash not in self.color_cache:
                        self.color_cache[p_hash] = sample_text_color(
                            img, x1, y1, x2, y2)
                    user_fg = getattr(self.state, 'fg_color', 'auto')
                    if user_fg and user_fg != 'auto':
                        fg = user_fg
                    else:
                        fg = self.color_cache[p_hash]
                bg_hex = '#FFFFFF'
                item_extra = {'force_font_size': base_font}
            else:
                avg = self._crop_avg_color(img, x1, y1, x2, y2)
                bg_hex = f'#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}'
                fg = self._resolve_fg_color(avg)
                bg_image = None
                if use_inpaint:
                    inpaint_used += 1
                    bg_image = self._get_inpaint_cached(
                        p_hash, img, x1, y1, x2, y2)
                item_extra = {}

            item = {
                'x1': x1 + region['left'],
                'y1': y1 + region['top'],
                'x2': x2 + region['left'],
                'y2': y2 + region['top'],
                'orig_y2': y2 + region['top'],
                'text': zh,
                'bg_hex': bg_hex,
                'fg': fg,
                'bg_image': bg_image,
                'key': display_key,
                'bilingual': False,
                'subtitle': False,
                'orig_w': x2 - x1,
                'orig_h': y2 - y1,
            }
            item.update(item_extra)
            cur_items[display_key] = item

        self._perf.add('t_verify', t_verify_total)

        if scroll_mode or display_mode == 'bilingual':
            before = len(cur_items)
            cur_items = self._dedupe_same_frame(cur_items)
            after = len(cur_items)
            if before != after:
                log.info(f"[去重] {display_mode} {before} → {after}")

        with self._known_lock:
            if (expected_generation is not None and
                    expected_generation != self._generation):
                return
            gen_at_start = self._generation
            if scan_rois and not self._is_full_scan(scan_rois, region):
                # 分块 OCR 只重新识别变化区域：保留扫描区域之外的旧译文，
                # 避免 overlay 因 cur_items 只包含局部段落而删掉其他区域，
                # 造成“切屏后有些英文没有翻译/覆盖层消失”。
                kept = {}
                for key, old in self._known_items.items():
                    if not self._item_in_rois(old, scan_rois, region):
                        kept[key] = old
                kept.update(cur_items)
                self._known_items = kept
            else:
                # 整屏扫描（含整屏补扫）时完全替换，清掉旧屏幕残留的译文。
                self._known_items = dict(cur_items)
            all_known = list(self._known_items.values())

        stable = self.emit_debouncer.process(all_known)
        if stable is None:
            if all_known:
                return
            # 整屏扫描后没有任何译文：显式发送空列表，
            # 让覆盖层清掉旧屏幕残留的窗口。
            stable = []

        with self._known_lock:
            if self._generation != gen_at_start:
                return
        self.result_queue.put(('items', stable))

    @staticmethod
    def _is_full_scan(rois, region):
        if not rois or len(rois) != 1:
            return False
        rx, ry, rw, rh = rois[0]
        return (rx <= 0 and ry <= 0 and
                rw >= region.get('width', 0) and
                rh >= region.get('height', 0))

    @staticmethod
    def _item_in_rois(item, rois, region):
        """判断一个 overlay item（绝对坐标）是否落在本次 OCR 区块内。"""
        ax1 = item.get('x1', 0)
        ay1 = item.get('y1', 0)
        ax2 = item.get('x2', 0)
        ay2 = item.get('y2', 0)
        left = region.get('left', 0)
        top = region.get('top', 0)
        for rx, ry, rw, rh in rois:
            bx1 = left + rx
            by1 = top + ry
            bx2 = bx1 + rw
            by2 = by1 + rh
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                return True
        return False

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