"""内置 Provider 适配器"""
import time
import json
import random
import hashlib
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

    def translate(self, text, from_lang='en', to_lang='zh'):
        t0 = time.time()
        appid = (self.params.get('appid') or '').strip()
        key = (self.params.get('secret_key') or '').strip()
        if not appid or not key:
            return self._err('未配置 APPID 或密钥')
        salt = str(random.randint(32768, 65536))
        sign = hashlib.md5(
            (appid + text + salt + key).encode('utf-8')
        ).hexdigest()
        params = {
            'q': text, 'from': from_lang, 'to': to_lang,
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

    def _err(self, msg):
        return ProviderResult(False, error=msg,
                              provider_id=self.id, provider_name=self.name)


# ============ OpenAI 兼容（GPT/DeepSeek/Qwen/Kimi/智谱/ollama） ============
class OpenAICompatProvider(BaseProvider):
    provider_type = 'ai'

    DEFAULT_SYSTEM = (
        "你是一个专业的翻译引擎。请将用户提供的文本翻译为简体中文。\n\n"
        "要求：\n"
        "1. 只输出译文，不要任何解释、前缀或后缀\n"
        "2. 保持原文的格式和标点\n"
        "3. 专业术语保持准确"
    )

    # ---------- 参数安全解析 ----------
    @staticmethod
    def _safe_float(v, default):
        try:
            if v is None or str(v).strip() == '':
                return default
            return float(v)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(v, default):
        try:
            if v is None or str(v).strip() == '':
                return default
            return int(v)
        except (ValueError, TypeError):
            return default

    # ---------- 端点拼接 ----------
    def _endpoint(self):
        """
        base_url 兼容多种写法：
        - https://api.deepseek.com         → /chat/completions
        - https://api.deepseek.com/v1      → /v1/chat/completions
        - https://api.deepseek.com/v1/     → /v1/chat/completions
        - 已经带 /chat/completions 的完整路径 → 直接用
        """
        base = (self.params.get('base_url') or '').strip().rstrip('/')
        if not base:
            return ''
        if base.endswith('/chat/completions'):
            return base
        return base + '/chat/completions'

    # ---------- 翻译 ----------
    def translate(self, text, from_lang='en', to_lang='zh'):
        t0 = time.time()
        url = self._endpoint()
        model = (self.params.get('model') or '').strip()
        if not url or not model:
            return self._err('未配置 base_url 或 model')

        api_key = (self.params.get('api_key') or '').strip()
        system = (self.params.get('system_prompt') or '').strip() \
                 or self.DEFAULT_SYSTEM

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

        # HTTP 状态检查
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

    def _err(self, msg):
        return ProviderResult(False, error=msg,
                              provider_id=self.id, provider_name=self.name)


# ============ 自定义 REST ============
class CustomRestProvider(BaseProvider):
    """URL 模板 + JSON 路径提取"""
    provider_type = 'translate'

    def translate(self, text, from_lang='en', to_lang='zh'):
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

    def _err(self, msg):
        return ProviderResult(False, error=msg,
                              provider_id=self.id, provider_name=self.name)


PROVIDER_REGISTRY = {
    'baidu': BaiduProvider,
    'openai_compat': OpenAICompatProvider,
    'custom_rest': CustomRestProvider,
}


def create_provider(cfg: dict) -> BaseProvider:
    kind = cfg.get('provider', '')
    cls = PROVIDER_REGISTRY.get(kind)
    if not cls:
        raise ValueError(f"未知的适配器：{kind}")
    return cls(cfg)