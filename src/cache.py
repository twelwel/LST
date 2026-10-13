"""段落缓存：翻译结果与段落复用，支持 LRU 自动清理"""

from collections import OrderedDict


class ParagraphCache:
    def __init__(self, min_overlap=0.5, max_size=500):
        self.min_overlap = min_overlap
        self.max_size = max_size
        self.auto_clean = True           # 👈 自动清理开关（默认开）
        # 段落哈希 -> {'translation':..., 'line_hashes': frozenset}
        self.paragraph_cache = OrderedDict()
        # 行哈希 -> [段落哈希, ...]
        self.line_index = {}

    def get(self, p_hash):
        item = self.paragraph_cache.get(p_hash)
        if item is None:
            return None
        # LRU：命中时移到末尾
        self.paragraph_cache.move_to_end(p_hash)
        return item['translation']

    def find_reusable(self, line_hashes):
        if not line_hashes:
            return None

        overlap_count = {}
        for lh in line_hashes:
            for cached_h in self.line_index.get(lh, []):
                overlap_count[cached_h] = overlap_count.get(cached_h, 0) + 1

        if not overlap_count:
            return None

        best_h = max(overlap_count, key=overlap_count.get)
        ratio = overlap_count[best_h] / len(line_hashes)
        if ratio >= self.min_overlap:
            return self.paragraph_cache[best_h]['translation']
        return None

    def put(self, p_hash, translation, line_hashes):
        # 已存在：直接更新
        if p_hash in self.paragraph_cache:
            self.paragraph_cache[p_hash]['translation'] = translation
            self.paragraph_cache.move_to_end(p_hash)
            return

        self.paragraph_cache[p_hash] = {
            'translation': translation,
            'line_hashes': line_hashes,
        }
        for lh in line_hashes:
            self.line_index.setdefault(lh, []).append(p_hash)

        # 自动清理
        if self.auto_clean and len(self.paragraph_cache) > self.max_size:
            self._evict_oldest()

    def _evict_oldest(self):
        """淘汰最旧条目（大约淘汰 20%）"""
        to_remove = max(1, int(self.max_size * 0.2))
        removed = 0
        while removed < to_remove and self.paragraph_cache:
            old_hash, old_item = self.paragraph_cache.popitem(last=False)
            # 清理反向索引
            for lh in old_item['line_hashes']:
                lst = self.line_index.get(lh)
                if lst and old_hash in lst:
                    lst.remove(old_hash)
                    if not lst:
                        del self.line_index[lh]
            removed += 1
        print(f"[缓存] 自动清理了 {removed} 条")

    def set_auto_clean(self, enabled):
        self.auto_clean = bool(enabled)

    def size(self):
        return len(self.paragraph_cache)

    def clear(self):
        self.paragraph_cache.clear()
        self.line_index.clear()