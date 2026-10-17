"""段落缓存：翻译结果与段落复用（本地记忆 / SQLite 持久化）"""

import os
import time  # 👈 加上这一行
import sqlite3
import threading
from collections import OrderedDict

CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
CACHE_DIR = os.path.join(CONFIG_DIR, 'cache')
DB_PATH = os.path.join(CACHE_DIR, 'translation_cache.db')


class ParagraphCache:
    def __init__(self, min_overlap=0.5, max_size=500):
        self.min_overlap = min_overlap
        self.max_size = max_size
        self.auto_clean = True
        
        # 内存级 LRU（加速高频查询）
        self.paragraph_cache = OrderedDict()
        self.line_index = {}
        
        self._lock = threading.Lock()
        
        # 初始化本地数据库
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._init_db()
        self._load_recent_to_memory()

    def _init_db(self):
        """创建表结构和索引"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    p_hash TEXT PRIMARY KEY,
                    translation TEXT,
                    line_hashes TEXT,
                    last_access REAL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_access ON cache(last_access)')
            # 开启 WAL 模式提升并发性能
            conn.execute('PRAGMA journal_mode=WAL')
            conn.commit()

    def _load_recent_to_memory(self):
        """启动时加载最近的部分数据到内存，避免全部加载卡顿"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute(
                    'SELECT p_hash, translation, line_hashes FROM cache ORDER BY last_access DESC LIMIT ?',
                    (self.max_size,)
                )
                for p_hash, translation, line_hashes_str in cursor:
                    line_hashes = frozenset(line_hashes_str.split(',')) if line_hashes_str else frozenset()
                    self.paragraph_cache[p_hash] = {
                        'translation': translation,
                        'line_hashes': line_hashes
                    }
                    for lh in line_hashes:
                        self.line_index.setdefault(lh, []).append(p_hash)
        except Exception as e:
            print(f"[缓存] 加载历史数据失败：{e}")

    def get(self, p_hash):
        with self._lock:
            # 1. 查内存
            item = self.paragraph_cache.get(p_hash)
            if item is not None:
                self.paragraph_cache.move_to_end(p_hash)
                return item['translation']
                
            # 2. 查数据库
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cur = conn.execute('SELECT translation, line_hashes FROM cache WHERE p_hash=?', (p_hash,))
                    row = cur.fetchone()
                    if row:
                        translation, line_hashes_str = row
                        line_hashes = frozenset(line_hashes_str.split(',')) if line_hashes_str else frozenset()
                        # 更新访问时间
                        conn.execute('UPDATE cache SET last_access=? WHERE p_hash=?', (time.time(), p_hash))
                        conn.commit()
                        # 放入内存
                        self.paragraph_cache[p_hash] = {
                            'translation': translation,
                            'line_hashes': line_hashes
                        }
                        for lh in line_hashes:
                            self.line_index.setdefault(lh, []).append(p_hash)
                        return translation
            except Exception:
                pass
            return None

    def find_reusable(self, line_hashes):
        if not line_hashes:
            return None

        with self._lock:
            overlap_count = {}
            for lh in line_hashes:
                for cached_h in self.line_index.get(lh, []):
                    overlap_count[cached_h] = overlap_count.get(cached_h, 0) + 1

            if not overlap_count:
                return None

            best_h = max(overlap_count, key=overlap_count.get)
            ratio = overlap_count[best_h] / len(line_hashes)
            if ratio >= self.min_overlap:
                item = self.paragraph_cache.get(best_h)
                if item:
                    return item['translation']
        return None

    def put(self, p_hash, translation, line_hashes):
        with self._lock:
            # 内存更新
            if p_hash in self.paragraph_cache:
                self.paragraph_cache[p_hash]['translation'] = translation
                self.paragraph_cache.move_to_end(p_hash)
            else:
                self.paragraph_cache[p_hash] = {
                    'translation': translation,
                    'line_hashes': line_hashes,
                }
                for lh in line_hashes:
                    self.line_index.setdefault(lh, []).append(p_hash)

            # 写入数据库
            try:
                lh_str = ','.join(line_hashes) if line_hashes else ''
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache (p_hash, translation, line_hashes, last_access)
                        VALUES (?, ?, ?, ?)
                    ''', (p_hash, translation, lh_str, time.time()))
                    conn.commit()
            except Exception as e:
                print(f"[缓存] 写入失败：{e}")

            # 自动清理
            if self.auto_clean:
                self._check_and_evict()

    def _check_and_evict(self):
        """检查内存和数据库的总容量，淘汰最旧数据"""
        # 内存清理
        while len(self.paragraph_cache) > self.max_size:
            old_hash, old_item = self.paragraph_cache.popitem(last=False)
            for lh in old_item['line_hashes']:
                lst = self.line_index.get(lh)
                if lst and old_hash in lst:
                    lst.remove(old_hash)
                    if not lst:
                        del self.line_index[lh]

        # 数据库清理（保留最近的 max_size 条）
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.execute('SELECT COUNT(*) FROM cache')
                count = cur.fetchone()[0]
                if count > self.max_size:
                    to_delete = count - self.max_size
                    conn.execute('''
                        DELETE FROM cache WHERE p_hash IN (
                            SELECT p_hash FROM cache ORDER BY last_access ASC LIMIT ?
                        )
                    ''', (to_delete,))
                    conn.commit()
                    # 收缩数据库
                    if to_delete > 100:
                        conn.execute('VACUUM')
        except Exception:
            pass

    def set_auto_clean(self, enabled):
        self.auto_clean = bool(enabled)
        if self.auto_clean:
            self._check_and_evict()

    def size(self):
        """返回当前总条数（包含数据库）"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.execute('SELECT COUNT(*) FROM cache')
                return cur.fetchone()[0]
        except Exception:
            return len(self.paragraph_cache)

    def get_file_size_mb(self):
        """获取数据库文件实际占用空间 (MB)"""
        try:
            size_bytes = os.path.getsize(DB_PATH)
            # 加上 WAL 文件的大小
            wal_path = DB_PATH + '-wal'
            if os.path.exists(wal_path):
                size_bytes += os.path.getsize(wal_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def clear(self):
        with self._lock:
            self.paragraph_cache.clear()
            self.line_index.clear()
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('DELETE FROM cache')
                    conn.commit()
                    conn.execute('VACUUM')
            except Exception:
                pass