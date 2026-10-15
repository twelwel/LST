"""自适应输出防抖模块（视频字幕专用）

针对视频字幕场景的核心机制：

1. **位置即身份**：同一位置（容差内）视为同一段，文本变化不影响 identity
2. **多帧投票**：窗口内文本聚类，选最大簇的最长文本（字幕逐字出现时最完整）
3. **显示节流**：同一段的最小更新间隔，避免逐字出现时 UI 高频刷新
4. **位置平滑**：anchor 缓慢跟随，位置小抖动不传到 UI
5. **消失容忍**：连续 N 帧未出现才移除

对外接口：
    reset()          → 清空状态
    process(items)   → 处理一帧全量输入，返回 None（无变化）或 list（全量稳定列表）
"""

import threading
import time
from collections import deque
from difflib import SequenceMatcher


class EmitDebouncer:
    # ============ 参数（视频字幕场景调优） ============

    # 投票窗口大小（帧）—— 越大越稳，但字幕换句响应越慢
    VOTE_WINDOW = 4

    # 同一段的最小更新间隔（秒）
    # 字幕逐字出现时，本段内每帧文本都不同
    # 此值控制更新频率：0.12s ≈ 8次/秒，人眼看起来流畅且不抖
    MIN_UPDATE_INTERVAL = 0.12

    # 位置容差（像素）—— 低于此值视为同段
    # 视频字幕位置固定，抖动一般在 ±10px 内
    POS_TOLERANCE = 16

    # 位置平滑系数（0=不动，1=完全跟随）
    # 0.35：稳定但能缓慢适应
    POS_BLEND = 0.35

    # 消失容忍帧数 —— 字幕短暂识别失败不立即消失
    MISS_TOLERANCE = 2

    # 文本相似度阈值 —— 用于投票聚类
    TEXT_SIMILARITY = 0.7

    # 文本长度相似下限（低于此值视为不同文本，不参与聚类）
    TEXT_MIN_LEN_RATIO = 0.5

    # ============ 状态 ============

    def __init__(self):
        self._lock = threading.Lock()
        self._states = {}   # key -> state dict

    def reset(self):
        """清空所有状态（切换区域/模式/策略时调用）"""
        with self._lock:
            self._states.clear()

    def process(self, items):
        """
        参数：
            items: pipeline 认为「当前应该显示」的全部 item 列表

        返回：
            None  → 无变化，UI 无需更新
            list  → 有变化，返回完整的稳定 item 列表（全量替换）
        """
        with self._lock:
            now = time.time()
            incoming = {it['key']: it for it in items}
            dirty = False

            # ============ 1. 处理已存在的 key ============
            for key in list(self._states.keys()):
                st = self._states[key]

                if key not in incoming:
                    # 消失：容忍几帧
                    st['miss'] += 1
                    if st['miss'] > self.MISS_TOLERANCE:
                        del self._states[key]
                        dirty = True
                    continue

                st['miss'] = 0
                item = incoming[key]

                if self._update_state(st, item, now):
                    dirty = True

            # ============ 2. 处理新 key ============
            for key, item in incoming.items():
                if key in self._states:
                    continue
                # 新 key 直接采纳（字幕新句出现时立即可见）
                self._states[key] = {
                    'stable': item,
                    'texts': deque([item.get('text', '')],
                                   maxlen=self.VOTE_WINDOW),
                    'anchor': self._box(item),
                    'last_update_ts': now,
                    'miss': 0,
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
        """更新已有 key 的状态，返回 True 表示 dirty"""
        st['texts'].append(item.get('text', ''))

        new_box = self._box(item)
        same_pos = self._pos_close(st['anchor'], new_box)

        # ============ 位置大变：视为新段落 ============
        if not same_pos:
            st['stable'] = item
            st['texts'] = deque([item.get('text', '')],
                                maxlen=self.VOTE_WINDOW)
            st['anchor'] = new_box
            st['last_update_ts'] = now
            return True

        # ============ 同一段：位置平滑 ============
        st['anchor'] = self._blend_pos(
            st['anchor'], new_box, self.POS_BLEND)

        # ============ 多帧投票 ============
        voted = self._vote(st['texts'])
        cur_text = st['stable'].get('text', '')

        # ---- 文本未变：只更新位置（如果位置有明显偏差）----
        if not voted or voted == cur_text:
            if self._pos_far_from_stable(st['stable'], st['anchor']):
                new_stable = dict(st['stable'])
                (new_stable['x1'], new_stable['y1'],
                 new_stable['x2'], new_stable['y2']) = st['anchor']
                if 'orig_y2' in new_stable:
                    new_stable['orig_y2'] = st['anchor'][3]
                st['stable'] = new_stable
                return True
            return False

        # ---- 文本有变化：节流 ----
        if now - st['last_update_ts'] < self.MIN_UPDATE_INTERVAL:
            # 节流窗口内，暂不更新（投票窗口继续收集）
            return False

        # ---- 通过节流：采纳投票结果 + 平滑位置 ----
        new_stable = dict(item)
        new_stable['text'] = voted
        new_stable['key'] = st['stable'].get('key', item.get('key', ''))
        (new_stable['x1'], new_stable['y1'],
         new_stable['x2'], new_stable['y2']) = st['anchor']
        if 'orig_y2' in new_stable:
            new_stable['orig_y2'] = st['anchor'][3]
        st['stable'] = new_stable
        st['last_update_ts'] = now
        return True

    # ==================== 工具 ====================

    @staticmethod
    def _box(it):
        return (it['x1'], it['y1'], it['x2'], it['y2'])

    def _pos_close(self, a, b):
        return (abs(a[0] - b[0]) <= self.POS_TOLERANCE and
                abs(a[1] - b[1]) <= self.POS_TOLERANCE and
                abs(a[2] - b[2]) <= self.POS_TOLERANCE and
                abs(a[3] - b[3]) <= self.POS_TOLERANCE)

    def _pos_far_from_stable(self, item, anchor):
        tol = self.POS_TOLERANCE // 2
        return (abs(item['x1'] - anchor[0]) > tol or
                abs(item['y1'] - anchor[1]) > tol or
                abs(item['x2'] - anchor[2]) > tol or
                abs(item['y2'] - anchor[3]) > tol)

    @staticmethod
    def _blend_pos(old, new, alpha):
        return tuple(int(o * (1 - alpha) + n * alpha)
                     for o, n in zip(old, new))

    def _vote(self, texts):
        """
        多帧投票：
        1. 按相似度聚类
        2. 最大簇（平局取总长度大者）
        3. 簇内取最长文本（字幕逐字出现时最长=最完整）
        """
        valid = [t for t in texts if t]
        if not valid:
            return ''

        clusters = []
        for t in valid:
            placed = False
            for c in clusters:
                if self._text_sim(c[0], t) >= self.TEXT_SIMILARITY:
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