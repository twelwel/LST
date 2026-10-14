"""本地词典（智能多级查询版）
- 文件夹 / 单文件自动识别
- 多级查询：整句 → 去标点 → 词组 → 逐词 → 变体
- 智能拼接：处理空格、大小写、复数
- 部分命中：允许部分单词命中
"""

import os
import re
import json
import threading

from logger import get_logger


log = get_logger('dict')


SUPPORTED_EXTS = {'.txt', '.csv', '.json', '.dict', '.tsv'}

IGNORE_DIRS = {
    '__pycache__', '.git', '.svn', 'node_modules',
    '.idea', '.vscode', 'build', 'dist',
}


# ==================== 编码检测 ====================
def _detect_encoding(path, sample_size=64 * 1024):
    try:
        with open(path, 'rb') as f:
            raw = f.read(4)
        if raw.startswith(b'\xef\xbb\xbf'):
            return 'utf-8-sig'
        if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
            return 'utf-16'
    except Exception:
        pass

    try:
        with open(path, 'rb') as f:
            raw = f.read(sample_size)
    except Exception:
        return 'utf-8'

    for enc in ('utf-8', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1'):
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return 'utf-8'


# ==================== 分隔符 ====================
SEPARATORS = [
    ('\t\t', 'DoubleTab'),
    ('\t', 'Tab'),
    (' --- ', 'Dashes'),
    ('---', 'Dashes'),
    (' | ', 'Pipe'),
    ('|', 'Pipe'),
    (' = ', 'Equals'),
    ('=', 'Equals'),
    ('：', 'CNColon'),
    (':', 'Colon'),
]

COMMA_PATTERN = re.compile(r'^([^,]+),\s*(.+)$')


def _clean_translation(zh):
    if not zh:
        return ''
    zh = zh.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', '')
    zh = re.sub(r'\s+', ' ', zh).strip()
    return zh


def _clean_word(en):
    if not en:
        return ''
    en = en.strip()
    if en.startswith('"') and en.endswith('"') and len(en) > 2:
        en = en[1:-1]
    return en


def _is_valid_entry(en, zh):
    if not en or not zh:
        return False
    if len(en) > 200 or len(zh) > 2000:
        return False
    if not re.search(r'[A-Za-z]', en):
        return False
    if '\n' in en or '\n' in zh:
        return False
    return True


def _parse_line(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    for sep, _ in SEPARATORS:
        if sep in line:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                en = _clean_word(parts[0])
                zh = _clean_translation(parts[1])
                if _is_valid_entry(en, zh):
                    return (en, zh)

    m = COMMA_PATTERN.match(line)
    if m:
        en = _clean_word(m.group(1))
        zh = _clean_translation(m.group(2))
        if _is_valid_entry(en, zh) and not re.search(r'[\u4e00-\u9fff]', en):
            return (en, zh)

    if re.search(r'\s{2,}', line):
        parts = re.split(r'\s{2,}', line, 1)
        if len(parts) == 2:
            en = _clean_word(parts[0])
            zh = _clean_translation(parts[1])
            if _is_valid_entry(en, zh):
                return (en, zh)

    return None


# ==================== 单词变体生成 ====================
def _word_variants(word):
    """
    生成单词的常见变体：
    - 原词
    - 复数（+s, +es, -y+ies）
    - 过去式（+ed, +d）
    - 进行时（+ing）
    - 大小写变体
    """
    w = word.lower()
    variants = {w}

    # 复数
    if w.endswith('y') and len(w) > 1:
        variants.add(w[:-1] + 'ies')
    variants.add(w + 's')
    if w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        variants.add(w + 'es')

    # 过去式
    if w.endswith('e'):
        variants.add(w + 'd')
    variants.add(w + 'ed')

    # 进行时
    if w.endswith('e'):
        variants.add(w[:-1] + 'ing')
    variants.add(w + 'ing')

    return variants


# ==================== 中英间空格规则 ====================
# 中文内部不加空格，中文和英文之间加空格
_CJK = '\u4e00-\u9fff'


def _smart_join(pieces):
    """
    智能拼接多个中文/英文片段：
    - 中文 + 中文：直接连接
    - 中文 + 英文：加空格
    - 英文 + 中文：加空格
    - 英文 + 英文：加空格
    """
    if not pieces:
        return ''

    result = pieces[0]
    for piece in pieces[1:]:
        if not piece:
            continue
        prev = result[-1] if result else ''
        cur = piece[0] if piece else ''

        prev_cjk = bool(re.match(f'[{_CJK}]', prev)) if prev else False
        cur_cjk = bool(re.match(f'[{_CJK}]', cur)) if cur else False

        if prev_cjk and cur_cjk:
            result += piece
        else:
            result += ' ' + piece

    return result


class LocalDictionary:
    """本地词典（智能多级查询）"""

    # 部分命中阈值：命中率 >= 此值就用逐词结果
    PARTIAL_HIT_THRESHOLD = 0.7
    # 长句跳过阈值：超过此长度的句子不走逐词
    MAX_WORDS_FOR_PARTIAL = 10

    def __init__(self, path=''):
        self._lock = threading.Lock()
        self._dict = {}
        self._path = path or ''
        self._stats = {
            'total_files': 0,
            'total_lines': 0,
            'valid_entries': 0,
            'skipped': 0,
            'is_folder': False,
            'encodings': set(),
            'formats': set(),
        }

        if self._path:
            self.load(self._path)

    # ============ 归一化 ============

    def _normalize(self, text):
        if not text:
            return ''
        return ' '.join(text.strip().lower().split())

    def _normalize_strip_punct(self, text):
        """去标点后归一化"""
        if not text:
            return ''
        cleaned = re.sub(r"[^\w\s]", ' ', text)
        return self._normalize(cleaned)

    # ============ 加载 ============

    def load(self, path):
        if not path or not os.path.exists(path):
            log.warning(f"[本地词典] 路径不存在：{path}")
            return 0

        if os.path.isdir(path):
            return self._load_folder(path)
        else:
            return self._load_single(path)

    def _load_folder(self, folder):
        files = self._scan_folder(folder)
        if not files:
            log.warning(f"[本地词典] 文件夹内没有词典文件：{folder}")
            return 0

        log.info(f"[本地词典] 扫描到 {len(files)} 个文件，开始加载...")

        merged = {}
        total_lines = 0
        total_skipped = 0
        encodings = set()

        for i, file_path in enumerate(files):
            try:
                sub_dict, lines, skipped, enc = self._parse_file(file_path)
                total_lines += lines
                total_skipped += skipped
                encodings.add(enc)

                for k, v in sub_dict.items():
                    if k not in merged:
                        merged[k] = v

                if (i + 1) % 20 == 0:
                    log.info(f"[本地词典] 已加载 {i+1}/{len(files)} 个文件，"
                             f"累计 {len(merged)} 条")
            except Exception as e:
                log.warning(f"[本地词典] 跳过 {os.path.basename(file_path)}：{e}")
                continue

        with self._lock:
            self._dict = merged
            self._path = folder
            self._stats = {
                'total_files': len(files),
                'total_lines': total_lines,
                'valid_entries': len(merged),
                'skipped': total_skipped,
                'is_folder': True,
                'encodings': encodings,
                'formats': {'TXT/CSV'},
            }

        log.info(
            f"[本地词典] 文件夹加载完成："
            f"{len(files)} 个文件 → {len(merged)} 条"
            f" | 总行 {total_lines} | 跳过 {total_skipped}"
            f" | 编码 {', '.join(sorted(encodings))}"
        )
        return len(merged)

    def _scan_folder(self, folder):
        files = []
        try:
            for root, dirs, filenames in os.walk(folder):
                dirs[:] = [d for d in dirs
                           if d not in IGNORE_DIRS and not d.startswith('.')]
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        files.append(os.path.join(root, fn))
        except Exception as e:
            log.error(f"[本地词典] 扫描失败：{e}")

        files.sort()
        return files

    def _load_single(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == '.json':
            return self._load_json(path)

        new_dict, total_lines, skipped, enc = self._parse_file(path)

        with self._lock:
            self._dict = new_dict
            self._path = path
            self._stats = {
                'total_files': 1,
                'total_lines': total_lines,
                'valid_entries': len(new_dict),
                'skipped': skipped,
                'is_folder': False,
                'encodings': {enc},
                'formats': {'TXT/CSV'},
            }

        log.info(
            f"[本地词典] 单文件加载：{len(new_dict)} 条"
            f" | 编码 {enc} | 总行 {total_lines} | 跳过 {skipped}"
        )
        return len(new_dict)

    def _parse_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.json':
            return self._parse_json_file(path)

        enc = _detect_encoding(path)
        new_dict = {}
        total_lines = 0
        skipped = 0

        try:
            with open(path, 'r', encoding=enc, errors='ignore') as f:
                for raw_line in f:
                    total_lines += 1
                    parsed = _parse_line(raw_line)
                    if parsed is None:
                        skipped += 1
                        continue
                    en, zh = parsed
                    key = self._normalize(en)
                    if key not in new_dict:
                        new_dict[key] = zh
        except Exception as e:
            log.error(f"[本地词典] 读取 {os.path.basename(path)} 失败：{e}")
            return {}, 0, 0, enc

        return new_dict, total_lines, skipped, enc

    def _parse_json_file(self, path):
        enc = _detect_encoding(path)
        try:
            with open(path, 'r', encoding=enc) as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"[本地词典] JSON 解析失败：{e}")
            return {}, 0, 0, enc

        new_dict = {}
        if isinstance(data, dict):
            for k, v in data.items():
                en = _clean_word(str(k))
                zh = _clean_translation(str(v))
                if _is_valid_entry(en, zh):
                    new_dict[self._normalize(en)] = zh
        return new_dict, len(data), 0, enc

    def _load_json(self, path):
        new_dict, total, _, enc = self._parse_json_file(path)
        with self._lock:
            self._dict = new_dict
            self._path = path
            self._stats = {
                'total_files': 1,
                'total_lines': total,
                'valid_entries': len(new_dict),
                'skipped': 0,
                'is_folder': False,
                'encodings': {enc},
                'formats': {'JSON'},
            }
        log.info(f"[本地词典] JSON：{len(new_dict)} 条")
        return len(new_dict)

    def reload(self):
        if not self._path:
            return 0
        return self.load(self._path)

    def set_path(self, path):
        self._path = path or ''

    def get_path(self):
        return self._path

    def is_folder(self):
        with self._lock:
            return self._stats.get('is_folder', False)

    # ============ 核心查询 ============

    def _dict_get(self, key):
        """内部查字典（加锁）"""
        if not key:
            return None
        with self._lock:
            return self._dict.get(key)

    def lookup(self, text):
        """
        多级查询：
        1. 精确匹配
        2. 去标点后匹配
        3. 无大写敏感
        """
        if not text or not text.strip():
            return None

        # 1) 直接归一化
        key = self._normalize(text)
        zh = self._dict_get(key)
        if zh:
            return zh

        # 2) 去标点
        key2 = self._normalize_strip_punct(text)
        if key2 and key2 != key:
            zh = self._dict_get(key2)
            if zh:
                return zh

        return None

    def lookup_with_variants(self, text):
        """
        带词形变化的查询：
        - 整句查不到时，用变体再试
        - 比如 "Saving" → "save"
        """
        zh = self.lookup(text)
        if zh:
            return zh

        # 生成变形组合（只对单词或短句）
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if not words:
            return None

        # 逐个单词尝试变体
        all_ok = True
        results = []
        for w in words:
            zh_w = self.lookup(w)
            if not zh_w:
                # 尝试变体
                for v in _word_variants(w):
                    if v == w.lower():
                        continue
                    zh_w = self.lookup(v)
                    if zh_w:
                        break
            if zh_w:
                results.append(zh_w)
            else:
                results.append(None)
                all_ok = False

        if all_ok:
            return _smart_join(results)

        return None

    def lookup_partial(self, text):
        """
        部分命中：
        - 只对短句（≤ MAX_WORDS_FOR_PARTIAL 词）
        - 命中率 ≥ PARTIAL_HIT_THRESHOLD 才返回
        - 未命中的词保留原文
        """
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if not words:
            return None
        if len(words) > self.MAX_WORDS_FOR_PARTIAL:
            return None

        results = []
        hit_count = 0

        for w in words:
            zh_w = self.lookup(w)
            if not zh_w:
                for v in _word_variants(w):
                    if v == w.lower():
                        continue
                    zh_w = self.lookup(v)
                    if zh_w:
                        break

            if zh_w:
                results.append(zh_w)
                hit_count += 1
            else:
                results.append(w)   # 未命中的保留原样

        hit_ratio = hit_count / len(words)
        if hit_ratio >= self.PARTIAL_HIT_THRESHOLD:
            return _smart_join(results)

        return None

    def lookup_word_by_word(self, text):
        """严格逐词：所有词都命中才返回"""
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if not words:
            return None

        results = []
        for w in words:
            zh = self.lookup(w)
            if not zh:
                return None
            results.append(zh)

        return _smart_join(results)

    def smart_lookup(self, text):
        """
        综合查询（推荐用这个）：
        优先级：
        1. 整句精确
        2. 带变体
        3. 部分命中（容错）
        4. 逐词严格
        """
        # 1) 整句
        zh = self.lookup(text)
        if zh:
            return zh

        # 2) 变体
        zh = self.lookup_with_variants(text)
        if zh:
            return zh

        # 3) 部分命中
        zh = self.lookup_partial(text)
        if zh:
            return zh

        return None

    # ============ 状态 ============

    def size(self):
        with self._lock:
            return len(self._dict)

    def stats(self):
        with self._lock:
            s = dict(self._stats)
            if isinstance(s.get('encodings'), set):
                s['encodings'] = ', '.join(sorted(s['encodings']))
            if isinstance(s.get('formats'), set):
                s['formats'] = ', '.join(sorted(s['formats']))
            return s

    def clear(self):
        with self._lock:
            self._dict.clear()
            self._stats = {
                'total_files': 0,
                'total_lines': 0,
                'valid_entries': 0,
                'skipped': 0,
                'is_folder': False,
                'encodings': set(),
                'formats': set(),
            }
        log.info("[本地词典] 已清空")