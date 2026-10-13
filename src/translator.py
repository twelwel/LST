"""翻译模块 - 百度翻译 API 版（从本地配置读取密钥）"""

import hashlib
import random
import requests

from api_config import load_config

SOURCE_LANG = 'en'
TARGET_LANG = 'zh'

BAIDU_API_URL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

ERROR_CODE_MAP = {
    '52001': '请求超时',
    '52002': '服务端系统错误',
    '52003': '未授权用户，请检查 APPID 或服务是否开通',
    '54000': '必填参数为空',
    '54001': '签名错误，请检查密钥',
    '54003': '访问频率受限',
    '54004': '账户余额不足',
    '54005': '长文本请求过于频繁',
    '58000': '客户端 IP 非法',
    '58001': '译文语言方向不支持',
    '58002': '服务当前已关闭',
    '90107': '认证未通过或未生效',
}


def _make_sign(appid, query, salt, secret_key):
    s = appid + query + str(salt) + secret_key
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def test_baidu_api(appid, secret_key, text='Hello'):
    """
    测试百度翻译 API 是否可用
    返回 (bool, 消息)：True 时 msg 是译文，False 时 msg 是错误原因
    """
    if not appid or not secret_key:
        return False, 'APPID 或密钥为空'

    salt = str(random.randint(32768, 65536))
    sign = _make_sign(appid, text, salt, secret_key)

    params = {
        'q': text, 'from': 'en', 'to': 'zh',
        'appid': appid, 'salt': salt, 'sign': sign,
    }
    try:
        r = requests.get(BAIDU_API_URL, params=params, timeout=5)
        result = r.json()
        if 'error_code' in result:
            msg = ERROR_CODE_MAP.get(
                result['error_code'], f"错误码 {result['error_code']}"
            )
            return False, msg
        if 'trans_result' in result:
            return True, result['trans_result'][0]['dst']
        return False, '返回数据异常'
    except requests.exceptions.Timeout:
        return False, '请求超时，请检查网络'
    except requests.exceptions.ConnectionError:
        return False, '网络连接失败'
    except Exception as e:
        return False, f'异常: {e}'


class Translator:
    def __init__(self, source=SOURCE_LANG, target=TARGET_LANG):
        self.source = source
        self.target = target
        self.appid = ''
        self.secret_key = ''
        self.reload_config()

    def reload_config(self):
        """从本地配置读取 APPID 和密钥"""
        cfg = load_config()
        self.appid = (cfg.get('baidu_appid') or '').strip()
        self.secret_key = (cfg.get('baidu_secret_key') or '').strip()

    def is_configured(self):
        return bool(self.appid and self.secret_key)

    def translate(self, text):
        r = self.translate_lines([text])
        return r[0] if r else ''

    def translate_lines(self, lines):
        if not lines:
            return []
        if not self.is_configured():
            return ['[请先配置百度翻译 API]' for _ in lines]

        query = '\n'.join(lines)
        salt = str(random.randint(32768, 65536))
        sign = _make_sign(self.appid, query, salt, self.secret_key)

        params = {
            'q': query, 'from': self.source, 'to': self.target,
            'appid': self.appid, 'salt': salt, 'sign': sign,
        }

        try:
            r = requests.get(BAIDU_API_URL, params=params, timeout=5)
            result = r.json()

            if 'error_code' in result:
                msg = ERROR_CODE_MAP.get(
                    result['error_code'], f"错误码 {result['error_code']}"
                )
                return [f'[翻译错误] {msg}' for _ in lines]

            if 'trans_result' in result:
                trans = [it['dst'] for it in result['trans_result']]
                if len(trans) == len(lines):
                    return trans
                out = [''] * len(lines)
                for i, t in enumerate(trans):
                    if i < len(out):
                        out[i] = t
                return out

            return ['[返回数据异常]' for _ in lines]

        except requests.exceptions.Timeout:
            return ['[翻译超时]' for _ in lines]
        except requests.exceptions.ConnectionError:
            return ['[网络连接失败]' for _ in lines]
        except Exception as e:
            return [f'[翻译错误] {e}' for _ in lines]


def translate_text(text, source=SOURCE_LANG, target=TARGET_LANG):
    return Translator(source=source, target=target).translate(text)