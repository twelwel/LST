"""自适应输出防抖模块（视频字幕专用 + 日语优化）

核心机制：
1. **位置即身份**：用「中心点距离」判断同段，避免文本变长导致误判
2. **多帧投票**：窗口内文本聚类，选最大簇的最长文本
3. **显示节流**：同一段的最小更新间隔
4. **位置防抖**：位置变化 < 8px 视为 OCR 抖动，不更新窗口
5. **消失容忍**：连续 N 帧未出现才移除
6. **日语模式**：更激进的节流和消失容忍
"""

import threading
import time
from collections import deque
from difflib import SequenceMatcher


def _has_kana(text):
    if not text:
        return False
    return any('\u3040' <= ch <= '\u30ff' for ch in text)


class EmitDebouncer:
    # ============ 通用参数 ============

    VOTE_WINDOW = 4
    MIN_UPDATE_INTERVAL = 0.12

    # 中心点容差：判断是否同一段（容忍文本变长）
    CENTER_TOLERANCE = 30

    # ⭐ 位置更新阈值
    # 文本未变时，位置偏差超过此值才更新窗口（防 OCR 抖动）
    # - 太小：窗口跟着 OCR 抖动
    # - 太大：真移动不跟随
    # 8px 是经验值：> POSITION_QUANTIZE(6)，< 一行字号
    POS_UPDATE_THRESHOLD = 8

    MISS_TOLERANCE = 2

    TEXT_SIMILARITY = 0.7
    TEXT_MIN_LEN_RATIO = 0.5

    # ============ 日语专用参数 ============
    JP_VOTE_WINDOW = 3
    JP_MIN_UPDATE_INTERVAL = 0.06
    JP_MISS_TOLERANCE = 1
    JP_TEXT_SIMILARITY = 0.8

    def __init__(self):
        self._lock = threading.Lock()
        self._states = {}

    def reset(self):
        with self._lock:
            self._states.clear()

    def process(self, items):
        with self._lock:
            now = time.time()
            incoming = {it['key']: it for it in items}
            dirty = False

            # 1. 已存在的 key
            for key in list(self._states.keys()):
                st = self._states[key]

                if key not in incoming:
                    st['miss'] += 1
                    miss_tol = st.get('miss_tol', self.MISS_TOLERANCE)
                    if st['miss'] > miss_tol:
                        del self._states[key]
                        dirty = True
                    continue

                st['miss'] = 0
                item = incoming[key]

                if self._update_state(st, item, now):
                    dirty = True

            # 2. 新 key
            for key, item in incoming.items():
                if key in self._states:
                    continue

                text = item.get('text', '')
                is_jp = _has_kana(text)

                self._states[key] = {
                    'stable': item,
                    'texts': deque(
                        [text],
                        maxlen=self.JP_VOTE_WINDOW if is_jp
                        else self.VOTE_WINDOW,
                    ),
                    'anchor': self._box(item),
                    'last_update_ts': now,
                    'miss': 0,
                    'is_jp': is_jp,
                    'miss_tol': (self.JP_MISS_TOLERANCE if is_jp
                                 else self.MISS_TOLERANCE),
                    'update_interval': (self.JP_MIN_UPDATE_INTERVAL
                                        if is_jp
                                        else self.MIN_UPDATE_INTERVAL),
                    'text_sim': (self.JP_TEXT_SIMILARITY if is_jp
                                 else self.TEXT_SIMILARITY),
                }
                dirty = True

            if not dirty:
                return None

            return [
                s['stable'] for s in self._states.values()
                if s.get('stable')
            ]

    # ==================== 内部 ====================

    def _update_state(self, st, item, now):
        new_text = item.get('text', '')
        st['texts'].append(new_text)

        new_box = self._box(item)

        # 位置大变 → 视为新段落
        if not self._center_close(
                st['anchor'], new_box, self.CENTER_TOLERANCE):
            is_jp = _has_kana(new_text)
            st['stable'] = item
            st['texts'] = deque(
                [new_text],
                maxlen=(self.JP_VOTE_WINDOW if is_jp
                        else self.VOTE_WINDOW),
            )
            st['anchor'] = new_box
            st['last_update_ts'] = now
            st['is_jp'] = is_jp
            st['miss_tol'] = (self.JP_MISS_TOLERANCE if is_jp
                              else self.MISS_TOLERANCE)
            st['update_interval'] = (self.JP_MIN_UPDATE_INTERVAL
                                     if is_jp
                                     else self.MIN_UPDATE_INTERVAL)
            st['text_sim'] = (self.JP_TEXT_SIMILARITY if is_jp
                              else self.TEXT_SIMILARITY)
            return True

        # ⭐ anchor 保留最新位置（不 blend，避免累积误差）
        st['anchor'] = new_box

        # 多帧投票
        voted = self._vote(
            st['texts'], st.get('text_sim', self.TEXT_SIMILARITY))
        cur_text = st['stable'].get('text', '')

        # ---------- 文本未变 ----------
        if not voted or voted == cur_text:
            # 比较最新位置 vs 上次显示位置
            cur_pos = self._box(st['stable'])
            drift = self._pos_drift(new_box, cur_pos)

            if drift < self.POS_UPDATE_THRESHOLD:
                # 抖动 → 不动
                return False

            # 真移动（>8px）→ 更新位置
            new_stable = dict(st['stable'])
            (new_stable['x1'], new_stable['y1'],
             new_stable['x2'], new_stable['y2']) = new_box
            if 'orig_y2' in new_stable:
                new_stable['orig_y2'] = new_box[3]
            st['stable'] = new_stable
            return True

        # ---------- 文本有变化：节流 ----------
        interval = st.get('update_interval', self.MIN_UPDATE_INTERVAL)
        if now - st['last_update_ts'] < interval:
            return False

        # 采纳投票结果 + 用最新位置
        new_stable = dict(item)
        new_stable['text'] = voted
        new_stable['key'] = st['stable'].get('key',
                                              item.get('key', ''))
        (new_stable['x1'], new_stable['y1'],
         new_stable['x2'], new_stable['y2']) = new_box
        if 'orig_y2' in new_stable:
            new_stable['orig_y2'] = new_box[3]
        st['stable'] = new_stable
        st['last_update_ts'] = now
        return True

    # ==================== 工具 ====================

    @staticmethod
    def _box(it):
        return (it['x1'], it['y1'], it['x2'], it['y2'])

    @staticmethod
    def _center_close(a, b, tol):
        acx = (a[0] + a[2]) / 2
        acy = (a[1] + a[3]) / 2
        bcx = (b[0] + b[2]) / 2
        bcy = (b[1] + b[3]) / 2
        return abs(acx - bcx) <= tol and abs(acy - bcy) <= tol

    @staticmethod
    def _pos_drift(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]),
                   abs(a[2] - b[2]), abs(a[3] - b[3]))

    def _vote(self, texts, similarity=None):
        if similarity is None:
            similarity = self.TEXT_SIMILARITY
        valid = [t for t in texts if t]
        if not valid:
            return ''

        clusters = []
        for t in valid:
            placed = False
            for c in clusters:
                if self._text_sim(c[0], t) >= similarity:
                    c.append(t)
                    placed = True
                    break
            if not placed:
                clusters.append([t])

        best = max(clusters,
                   key=lambda c: (len(c), sum(len(t) for t in c)))

        return max(best, key=len)

    def _text_sim(self, a, b):
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        if min(la, lb) / max(la, lb) < self.TEXT_MIN_LEN_RATIO:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()