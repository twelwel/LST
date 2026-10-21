"""内置 Provider 适配器"""

import time
import json
import random
import hashlib
import threading
import requests

from .base import BaseProvider, ProviderResult


# ============ 百度翻译 ============
class BaiduProvider(BaseProvider):
    provider_type = 'translate'
    API_URL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

    ERROR_MAP = {
        '52001': '请求超时', '52002': '服务端错误',
        '52003': '未授权（检查 APPID/服务）', '54001': '签名错误',
        '54003': '频率受限', '54004': '余额不足', '58001': '语言方向不支持',
    }

    def translate(self, text, from_lang='auto', to_lang='zh'):
        from config import BAIDU_LANG_MAP
        t0 = time.time()
        appid = (self.params.get('appid') or '').strip()
        key = (self.params.get('secret_key') or '').strip()
        if not appid or not key:
            return self._err('未配置 APPID 或密钥')

        api_from = BAIDU_LANG_MAP.get(from_lang, 'auto')
        api_to = BAIDU_LANG_MAP.get(to_lang, 'zh')

        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5(
            (appid + text + salt + key).encode('utf-8')
        ).hexdigest()
        params = {
            'q': text, 'from': api_from, 'to': api_to,
            'appid': appid, 'salt': salt, 'sign': sign,
        }
        try:
            r = requests.get(self.API_URL, params=params, timeout=8)
            data = r.json()
            if 'error_code' in data:
                code = data['error_code']
                return self._err(self.ERROR_MAP.get(code, f'错误码 {code}'))
            if 'trans_result' in data:
                dst = '\n'.join(it['dst'] for it in data['trans_result'])
                return ProviderResult(
                    True, text=dst, provider_id=self.id,
                    provider_name=self.name,
                    elapsed_ms=int((time.time() - t0) * 1000))
            return self._err('返回数据异常')
        except Exception as e:
            return self._err(str(e)[:200])


# ============ OpenAI 兼容 ============
class OpenAICompatProvider(BaseProvider):
    provider_type = 'ai'

    DEFAULT_SYSTEM = (
        "你是一个专业的翻译引擎。请将用户提供的文本翻译为简体中文。\n\n"
        "要求：\n"
        "1. 只输出译文，不要任何解释、前缀或后缀\n"
        "2. 保持原文的格式和标点\n"
        "3. 专业术语保持准确"
    )

    def _endpoint(self):
        base = (self.params.get('base_url') or '').strip().rstrip('/')
        if not base:
            return ''
        if base.endswith('/chat/completions'):
            return base
        return base + '/chat/completions'

    def translate(self, text, from_lang='auto', to_lang='zh'):
        from config import TARGET_LANG_NAME
        t0 = time.time()
        url = self._endpoint()
        model = (self.params.get('model') or '').strip()
        if not url or not model:
            return self._err('未配置 base_url 或 model')

        api_key = (self.params.get('api_key') or '').strip()

        user_system = (self.params.get('system_prompt') or '').strip()
        if user_system:
            system = user_system
        else:
            target_name = TARGET_LANG_NAME.get(to_lang, '简体中文')
            system = (
                f"你是一个专业的翻译引擎。请将用户提供的文本翻译为"
                f"{target_name}。\n\n"
                "要求：\n"
                "1. 只输出译文，不要任何解释、前缀或后缀\n"
                "2. 保持原文的格式和标点\n"
                "3. 专业术语保持准确"
            )

        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        body = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': text},
            ],
            'temperature': self._safe_float(
                self.params.get('temperature'), 0.2),
            'stream': False,
        }

        timeout = self._safe_int(self.params.get('timeout'), 30)

        try:
            r = requests.post(
                url, json=body, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            return self._err(f'请求超时（{timeout}s）')
        except requests.exceptions.ConnectionError as e:
            return self._err(f'连接失败：{str(e)[:150]}')
        except Exception as e:
            return self._err(f'请求异常：{str(e)[:150]}')

        if r.status_code != 200:
            snippet = ''
            try:
                snippet = r.text[:200]
            except Exception:
                pass
            return self._err(f'HTTP {r.status_code}：{snippet}')

        try:
            data = r.json()
        except Exception as e:
            return self._err(f'响应解析失败：{str(e)[:150]}')

        if 'error' in data:
            err = data['error']
            if isinstance(err, dict):
                msg = err.get('message') or str(err)
            else:
                msg = str(err)
            return self._err(msg[:200])

        try:
            content = data['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, AttributeError) as e:
            return self._err(f'响应结构异常：{str(e)[:150]}')

        return ProviderResult(
            True, text=content, provider_id=self.id,
            provider_name=self.name,
            elapsed_ms=int((time.time() - t0) * 1000))


# ============ 自定义 REST ============
class CustomRestProvider(BaseProvider):
    provider_type = 'translate'

    def translate(self, text, from_lang='auto', to_lang='zh'):
        t0 = time.time()
        url_tpl = (self.params.get('url') or '').strip()
        if not url_tpl:
            return self._err('未配置 URL')

        method = (self.params.get('method') or 'GET').upper()
        url = (url_tpl
               .replace('{text}', requests.utils.quote(text))
               .replace('{from}', from_lang)
               .replace('{to}', to_lang))

        headers = self.params.get('headers') or {}
        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except Exception:
                headers = {}

        body = None
        body_tpl = self.params.get('body') or ''
        if method == 'POST' and body_tpl:
            try:
                body_str = (body_tpl
                            .replace('{text}', text.replace('"', '\\"'))
                            .replace('{from}', from_lang)
                            .replace('{to}', to_lang))
                body = json.loads(body_str)
            except Exception:
                body = body_tpl

        try:
            if method == 'GET':
                r = requests.get(url, headers=headers, timeout=10)
            elif isinstance(body, dict):
                r = requests.post(url, headers=headers, json=body, timeout=10)
            else:
                r = requests.post(url, headers=headers, data=body, timeout=10)

            data = r.json()
            path = (self.params.get('response_path') or '').strip()
            result = data
            if path:
                for seg in path.split('.'):
                    if seg.isdigit() and isinstance(result, list):
                        result = result[int(seg)]
                    elif isinstance(result, dict):
                        result = result[seg]
                    else:
                        return self._err(f'路径 {path} 无法解析')

            if isinstance(result, str):
                return ProviderResult(
                    True, text=result, provider_id=self.id,
                    provider_name=self.name,
                    elapsed_ms=int((time.time() - t0) * 1000))
            return self._err(f'响应不是字符串：{type(result).__name__}')
        except Exception as e:
            return self._err(str(e)[:200])


# ============ 本地模型 ============
class LocalModelProvider(BaseProvider):
    """本地翻译模型 Provider"""
    provider_type = 'translate'
    is_local = True                      # ⭐ 关键

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # 缓存已加载实例，避免每行译文都走一次 Factory 查找与加锁。
        self._instance = None
        self._instance_lock = threading.Lock()

    def _get_instance(self, model_id: str):
        instance = self._instance
        if (instance is not None
                and getattr(instance, 'is_loaded', False)
                and getattr(instance, '_model_id', model_id) == model_id):
            return instance

        from .local_models import get_model_instance
        with self._instance_lock:
            instance = self._instance
            if (instance is not None
                    and getattr(instance, 'is_loaded', False)
                    and getattr(instance, '_model_id', model_id) == model_id):
                return instance
            instance = get_model_instance(model_id)
            self._instance = instance
            return instance

    def preload(self):
        """预加载当前模型，供 UI 在后台线程预热。"""
        model_id = (self.params.get('model_id') or '').strip()
        if not model_id:
            return False, '未配置 model_id'
        try:
            instance = self._get_instance(model_id)
        except ImportError as e:
            return False, f'本地模型模块加载失败：{e}'
        except Exception as e:
            return False, f'模型创建失败：{e}'

        if instance is None:
            return False, f'未知模型：{model_id}'
        if instance.is_loaded:
            return True, ''

        detail = (getattr(instance, 'last_error', '') or '').strip()
        if detail:
            return False, f'本地模型加载失败：{model_id}（{detail}）'
        return False, f'模型未加载：{model_id}'

    def translate(self, text, from_lang='auto', to_lang='zh'):
        t0 = time.time()

        model_id = (self.params.get('model_id') or '').strip()
        if not model_id:
            return self._err('未配置 model_id')

        try:
            instance = self._get_instance(model_id)
        except ImportError as e:
            return self._err(f'本地模型模块加载失败：{e}')
        except Exception as e:
            return self._err(f'模型创建失败：{e}')

        if instance is None:
            return self._err(f'未知模型：{model_id}')

        if not instance.is_loaded:
            # 已下载但加载失败时，不能误导用户去重新下载；
            # 把 Factory / 模型类的真实原因带到界面。
            detail = (getattr(instance, 'last_error', '') or '').strip()
            if detail:
                return self._err(
                    f'本地模型加载失败：{model_id}（{detail}）')
            return self._err(
                f'模型未加载：{model_id}。'
                f'请检查模型文件是否完整（config.json 与 onnx 权重）')

        if not instance.supports(from_lang, to_lang):
            return self._err(
                f'模型 {instance.name} 不支持 {from_lang}→{to_lang}')

        try:
            result = instance.translate(text, from_lang, to_lang)
            # ⭐ 过滤错误标记
            if not result or result.startswith('['):
                return self._err(result or '空结果')
            return ProviderResult(
                True,
                text=result,
                provider_id=self.id,
                provider_name=self.name,
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return self._err(f'翻译异常：{e}')

    def translate_batch(self, texts, from_lang='auto', to_lang='zh'):
        """批量翻译：返回与 texts 等长的 ProviderResult 列表。

        本地模型支持一次推理多条，可摊薄逐条推理的固定开销；
        其它适配器没有此方法，调用方需自行回退。
        """
        texts = list(texts or [])
        n = len(texts)
        if n == 0:
            return []
        t0 = time.time()

        model_id = (self.params.get('model_id') or '').strip()
        if not model_id:
            return [self._err('未配置 model_id') for _ in texts]

        try:
            instance = self._get_instance(model_id)
        except ImportError as e:
            return [self._err(f'本地模型模块加载失败：{e}') for _ in texts]
        except Exception as e:
            return [self._err(f'模型创建失败：{e}') for _ in texts]

        if instance is None:
            return [self._err(f'未知模型：{model_id}') for _ in texts]

        if not instance.is_loaded:
            detail = (getattr(instance, 'last_error', '') or '').strip()
            msg = (f'本地模型加载失败：{model_id}（{detail}）' if detail
                   else f'模型未加载：{model_id}')
            return [self._err(msg) for _ in texts]

        if not instance.supports(from_lang, to_lang):
            msg = f'模型 {instance.name} 不支持 {from_lang}→{to_lang}'
            return [self._err(msg) for _ in texts]

        try:
            batch_fn = getattr(instance, 'translate_batch', None)
            outs = (batch_fn(texts, from_lang, to_lang) if batch_fn
                    else [instance.translate(t, from_lang, to_lang)
                          for t in texts])
        except Exception as e:
            return [self._err(f'翻译异常：{e}') for _ in texts]

        if not outs or len(outs) != n:
            return [self._err('批量返回数量不匹配') for _ in texts]

        elapsed = int((time.time() - t0) * 1000)
        results = []
        for s in outs:
            s = (s or '').strip()
            if not s or s.startswith('['):
                results.append(self._err(s or '空结果'))
            else:
                results.append(ProviderResult(
                    True, text=s, provider_id=self.id,
                    provider_name=self.name, elapsed_ms=elapsed))
        return results


# ============ 注册表 ============
PROVIDER_REGISTRY = {
    'baidu':            BaiduProvider,
    'openai_compat':    OpenAICompatProvider,
    'openai_compat_ai': OpenAICompatProvider,
    'custom_rest':      CustomRestProvider,
    'local_model':      LocalModelProvider,     # ⭐ 必须
}


def create_provider(cfg: dict) -> BaseProvider:
    kind = cfg.get('provider', '')
    cls = PROVIDER_REGISTRY.get(kind)
    if not cls:
        raise ValueError(f"未知的适配器：{kind}")
    return cls(cfg)