"""短语缓存池
- 内置短语（150+ 条常见 UI 短语）
- 用户自定义短语（json 持久化）
- 外部短语表（用户指定路径，支持 TXT / CSV / JSON）
- 线程安全
"""

import os
import json
import re
import threading

from logger import get_logger


log = get_logger('phrase')


# ==================== 路径 ====================
CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
USER_PHRASE_FILE = os.path.join(CONFIG_DIR, 'phrase_cache.json')
# 外部短语表路径由用户指定，不再固定
# =================================================


# ==================== 内置短语表 ====================
BUILTIN_PHRASES = {
        # ===== 日语常用短语 =====
    'こんにちは': '你好',
    'おはよう': '早上好',
    'こんばんは': '晚上好',
    'ありがとう': '谢谢',
    'ありがとうございます': '非常感谢',
    'すみません': '对不起 / 不好意思',
    'ごめんなさい': '对不起',
    'はい': '是',
    'いいえ': '不',
    'お願いします': '拜托了',
    'よろしくお願いします': '请多关照',
    'さようなら': '再见',
    'じゃあ': '那么',
    'それでは': '那么',
    'どうぞ': '请',
    'どうも': '谢谢 / 你好',
    'お疲れ様': '辛苦了',
    'いただきます': '我开动了',
    'ごちそうさまでした': '多谢款待',
    'おやすみ': '晚安',
    'いらっしゃいませ': '欢迎光临',
    'かしこまりました': '明白了',
    'わかりました': '明白了',
    '了解': '了解',
    'ちょっと待って': '稍等一下',
    '大丈夫': '没关系',
    '問題ない': '没问题',
    '始めましょう': '开始吧',
    'やめて': '停下',
    'すごい': '厉害',
    'かわいい': '可爱',
    '美味しい': '好吃',
    '楽しい': '开心',
    '嬉しい': '高兴',
    '悲しい': '伤心',
    '痛い': '疼',
    '寒い': '冷',
    '暑い': '热',
    '綺麗': '漂亮',
    '美しい': '美丽',
    '新しい': '新的',
    '古い': '旧的',
    '大きい': '大的',
    '小さい': '小的',
    '早い': '快的',
    '遅い': '慢的',
    '高い': '高的 / 贵的',
    '安い': '便宜的',
    '好き': '喜欢',
    '嫌い': '讨厌',
    '欲しい': '想要',
    'できる': '能 / 会',
    'できない': '不能 / 不会',
    'ある': '有',
    'ない': '没有',
    'いる': '在',
    '行く': '去',
    '来る': '来',
    '帰る': '回',
    # 通用按钮
    'ok': '确定', 'yes': '是', 'no': '否', 'cancel': '取消',
    'close': '关闭', 'open': '打开', 'save': '保存', 'delete': '删除',
    'edit': '编辑', 'add': '添加', 'remove': '移除', 'apply': '应用',
    'confirm': '确认', 'continue': '继续', 'back': '返回',
    'next': '下一步', 'previous': '上一步', 'finish': '完成',
    'submit': '提交', 'reset': '重置', 'retry': '重试',

    # 菜单 / 导航
    'menu': '菜单', 'settings': '设置', 'option': '选项',
    'options': '选项', 'help': '帮助', 'about': '关于',
    'exit': '退出', 'quit': '退出', 'home': '主页',
    'start': '开始', 'stop': '停止', 'pause': '暂停', 'play': '播放',
    'search': '搜索', 'find': '查找', 'filter': '筛选',
    'refresh': '刷新', 'reload': '重新加载', 'update': '更新',
    'install': '安装', 'uninstall': '卸载',
    'login': '登录', 'logout': '登出', 'sign in': '登录', 'sign up': '注册',
    'register': '注册', 'password': '密码', 'username': '用户名',

    # 状态
    'loading': '加载中', 'loading...': '加载中...',
    'error': '错误', 'warning': '警告', 'success': '成功',
    'failed': '失败', 'done': '完成', 'ready': '就绪',
    'enabled': '已启用', 'disabled': '已禁用',
    'on': '开启', 'off': '关闭',
    'new': '新建', 'create': '创建', 'copy': '复制', 'paste': '粘贴',
    'cut': '剪切', 'undo': '撤销', 'redo': '重做', 'select': '选择',
    'select all': '全选', 'clear': '清除', 'clean': '清理',

    # 游戏 / 3D
    'position': '位置', 'rotation': '旋转', 'scale': '缩放',
    'structure': '结构', 'properties': '属性', 'material': '材质',
    'color': '颜色', 'texture': '纹理', 'light': '灯光',
    'camera': '摄像机', 'scene': '场景', 'level': '关卡',

    # 时间 / 数字
    'today': '今天', 'yesterday': '昨天', 'tomorrow': '明天',
    'time': '时间', 'date': '日期', 'name': '名称',

    # 常见短句
    'are you sure': '你确定吗',
    'do you want to continue': '是否继续',
    'please wait': '请稍候',
    'loading please wait': '加载中请稍候',
    'no data': '无数据',
    'not found': '未找到',
    'access denied': '拒绝访问',
    'permission denied': '权限不足',
    'unknown error': '未知错误',
    'try again': '重试',
    'more': '更多', 'less': '收起', 'show': '显示', 'hide': '隐藏',
    'expand': '展开', 'collapse': '收起',
    'all': '全部', 'none': '无',
    'default': '默认', 'custom': '自定义',
    'apply changes': '应用更改',
    'save changes': '保存更改',
    'discard changes': '放弃更改',
    'unsaved changes': '未保存的更改',
    'are you sure you want to exit': '确定要退出吗',
    'are you sure you want to delete': '确定要删除吗',
    'operation successful': '操作成功',
    'operation failed': '操作失败',
    'network error': '网络错误',
    'connection failed': '连接失败',
    'server error': '服务器错误',
    'not available': '不可用',
    'coming soon': '即将推出',
    'beta': '测试版',
    'new version': '新版本',
    'update available': '有新版本',
    'download': '下载', 'upload': '上传',
    'browse': '浏览', 'choose file': '选择文件',
    'drag and drop': '拖放',
    'enter text': '输入文字',
    'type here': '在此输入',
    'click here': '点击这里',
    'learn more': '了解更多',
    'show more': '显示更多',
    'show less': '收起',
    'read more': '阅读更多',
    'see all': '查看全部',
    'view all': '查看全部',
    'back to top': '回到顶部',
    'scroll down': '向下滚动',
    'scroll up': '向上滚动',
    'powered by': '技术支持',
    'all rights reserved': '版权所有',
}
# =================================================


class PhraseCache:
    """短语缓存池（线程安全 + 外部表路径由用户指定）"""

    def __init__(self,
                 storage_path=None,
                 external_path=None,
                 load_disk=True):
        self._lock = threading.Lock()
        self._cache = {}
        self._storage_path = storage_path or USER_PHRASE_FILE
        self._external_path = external_path or ''    # 空 = 未指定
        self._builtin = dict(BUILTIN_PHRASES)

        # 加载内置
        self._cache.update(self._builtin)

        # 加载用户自定义
        if load_disk:
            self._load_user_phrases()

        # 加载外部短语表（仅当用户指定了路径）
        if self._external_path:
            self._load_external()

        log.info(f"[短语缓存] 初始化：{len(self._cache)} 条"
                 f"（外部表：{self._external_path or '未指定'}）")

    # ============ 归一化 ============

    def _normalize(self, text):
        if not text:
            return ''
        return ' '.join(text.strip().lower().split())

    # ============ 查询 ============

    def lookup(self, text):
        key = self._normalize(text)
        if not key:
            return None
        with self._lock:
            return self._cache.get(key)

    def lookup_fuzzy(self, text):
        result = self.lookup(text)
        if result is not None:
            return result

        cleaned = re.sub(r'[^\w\s]', '', text)
        if cleaned != text:
            return self.lookup(cleaned)
        return None

    # ============ 写入 ============

    def put(self, text, translation):
        key = self._normalize(text)
        if not key or not translation:
            return
        with self._lock:
            self._cache[key] = translation

    def put_many(self, mapping):
        with self._lock:
            for k, v in mapping.items():
                key = self._normalize(k)
                if key and v:
                    self._cache[key] = v

    # ============ 用户短语（自动学习） ============

    def _load_user_phrases(self):
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                with self._lock:
                    for k, v in data.items():
                        key = self._normalize(k)
                        if key and v:
                            self._cache[key] = v
                log.info(f"[短语缓存] 用户短语：{len(data)} 条")
        except Exception as e:
            log.error(f"[短语缓存] 加载用户短语失败：{e}")

    def save_user_phrases(self):
        """保存用户自定义短语（不含内置和外部）"""
        try:
            os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
            with self._lock:
                user_only = {
                    k: v for k, v in self._cache.items()
                    if k not in self._builtin
                }
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(user_only, f, ensure_ascii=False, indent=2)
            log.info(f"[短语缓存] 已保存 {len(user_only)} 条用户短语")
            return True
        except Exception as e:
            log.error(f"[短语缓存] 保存失败：{e}")
            return False

    # ============ 外部短语表（用户指定路径） ============

    def set_external_path(self, path):
        """设置外部短语表路径（可为空）"""
        self._external_path = path or ''
        log.info(f"[短语缓存] 外部表路径：{self._external_path or '已清空'}")

    def get_external_path(self):
        return self._external_path

    def load_external(self):
        """加载外部短语表（如已指定路径）"""
        if not self._external_path:
            return 0
        if not os.path.exists(self._external_path):
            log.warning(f"[短语缓存] 外部表不存在：{self._external_path}")
            return 0

        try:
            ext = os.path.splitext(self._external_path)[1].lower()

            if ext == '.json':
                loaded = self._load_external_json(self._external_path)
            elif ext == '.csv':
                loaded = self._load_external_csv(self._external_path)
            else:
                loaded = self._load_external_txt(self._external_path)

            if loaded:
                with self._lock:
                    self._cache.update(loaded)
                log.info(f"[短语缓存] 外部表加载：{len(loaded)} 条")
            return len(loaded)

        except Exception as e:
            log.error(f"[短语缓存] 加载外部表失败：{e}")
            return 0

    def reload_external(self):
        """重载外部短语表（先移除旧的，再加载新的）"""
        if not self._external_path:
            return 0

        # 先移除旧的外部短语（保留内置 + 用户）
        try:
            ext = os.path.splitext(self._external_path)[1].lower()
            if ext == '.json':
                old = self._load_external_json(self._external_path)
            elif ext == '.csv':
                old = self._load_external_csv(self._external_path)
            else:
                old = self._load_external_txt(self._external_path)

            with self._lock:
                for k in old.keys():
                    # 只清理非内置的
                    if k in self._cache and k not in self._builtin:
                        self._cache.pop(k, None)
        except Exception:
            pass

        return self.load_external()

    def _load_external_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            self._normalize(k): v
            for k, v in data.items()
            if k and v
        }

    def _load_external_csv(self, path):
        result = {}
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',', 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    result[self._normalize(parts[0])] = parts[1]
        return result

    def _load_external_txt(self, path):
        """TXT：每行 '英文<TAB>中文' 或 '英文|中文' 或 '英文=中文'"""
        result = {}
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.rstrip('\n\r')
                if not line.strip() or line.strip().startswith('#'):
                    continue

                for sep in ('\t', '|', '=', '：', ':'):
                    if sep in line:
                        parts = line.split(sep, 1)
                        if len(parts) == 2:
                            en = parts[0].strip()
                            zh = parts[1].strip()
                            if en and zh:
                                result[self._normalize(en)] = zh
                            break
        return result

    # ============ 统计 ============

    def size(self):
        with self._lock:
            return len(self._cache)

    def stats(self):
        with self._lock:
            total = len(self._cache)
            builtin = len(self._builtin)
            return {
                'total': total,
                'builtin': builtin,
                'user': total - builtin,
            }

    def clear_user(self):
        """清空用户自定义（保留内置和外部）"""
        with self._lock:
            self._cache = dict(self._builtin)
        log.info("[短语缓存] 已清空用户自定义")