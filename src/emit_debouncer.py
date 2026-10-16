"""自适应输出防抖模块（视频字幕专用 + 日语优化 + 防非法留存）

核心机制：
1. **位置即身份**：用「中心点距离」判断同段
2. **多帧投票**：窗口内文本聚类，选最大簇的最长文本
3. **快速换句**：最近 2 帧一致 → 立即采纳（不等 4 帧）
4. **剧变检测**：新文本与当前稳定文本相似度 < 0.3 → 视为新段，重置窗口
5. **显示节流**：同一段的最小更新间隔
6. **位置防抖**：位置变化 < 8px 视为 OCR 抖动
7. **消失容忍**：连续 N 帧未出现才移除
8. **日语模式**：更激进的节流和消失容忍
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

    # 中心点容差：判断是否同一段
    CENTER_TOLERANCE = 30

    # 位置更新阈值（文本未变时）
    POS_UPDATE_THRESHOLD = 8

    # 消失容忍
    MISS_TOLERANCE = 2

    # 文本相似度阈值
    TEXT_SIMILARITY = 0.7
    TEXT_MIN_LEN_RATIO = 0.5

    # ⭐ 剧变阈值：新文本与稳定文本相似度 < 此值 → 视为新段
    TEXT_RESET_THRESHOLD = 0.3

    # ⭐ 快速换句：最近 2 帧相似度 >= 此值 → 立即采纳
    TEXT_FAST_MATCH = 0.9

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
                        maxlen=(self.JP_VOTE_WINDOW if is_jp
                                else self.VOTE_WINDOW),
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

        st['anchor'] = new_box

        # ⭐ 文本剧变检测：新文本与稳定文本相似度过低 → 视为新段
        cur_text = st['stable'].get('text', '')
        if new_text and cur_text:
            sim_cur = self._text_sim_raw(new_text, cur_text)
            if sim_cur < self.TEXT_RESET_THRESHOLD:
                # 重置文本窗口（视为新段）
                is_jp = _has_kana(new_text)
                st['texts'].clear()
                st['texts'].append(new_text)
                st['is_jp'] = is_jp
                st['update_interval'] = (self.JP_MIN_UPDATE_INTERVAL
                                         if is_jp
                                         else self.MIN_UPDATE_INTERVAL)

        st['texts'].append(new_text)

        # 多帧投票
        voted = self._vote(
            st['texts'], st.get('text_sim', self.TEXT_SIMILARITY))

        cur_text = st['stable'].get('text', '')

        # ---------- 文本未变 ----------
        if not voted or voted == cur_text:
            cur_pos = self._box(st['stable'])
            drift = self._pos_drift(new_box, cur_pos)

            if drift < self.POS_UPDATE_THRESHOLD:
                return False

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

    @staticmethod
    def _text_sim_raw(a, b):
        """不带长度比例限制的相似度（用于剧变检测）"""
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _vote(self, texts, similarity=None):
        """
        投票：
        1. ⭐ 最近 2 帧一致 → 立即采纳（快速换句）
        2. 否则聚类选最大簇的最长文本
        """
        if similarity is None:
            similarity = self.TEXT_SIMILARITY
        valid = [t for t in texts if t]
        if not valid:
            return ''

        # ⭐ 快速通道：最近 2 帧一致
        if len(valid) >= 2:
            if valid[-1] == valid[-2]:
                return valid[-1]
            if self._text_sim_raw(valid[-1], valid[-2]) >= \
                    self.TEXT_FAST_MATCH:
                return valid[-1]

        # 常规聚类投票
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