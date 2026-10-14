"""API 配置存储：读写本地 JSON 配置文件"""

import os
import json

CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.screen_translator')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')

DEFAULT_CONFIG = {
    'baidu_appid': '',
    'baidu_secret_key': '',
}


def load_config():
    """读取配置，不存在则返回默认值"""
    if not os.path.exists(CONFIG_FILE):
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(data)
        return cfg
    except Exception as e:
        print(f"[配置] 读取失败: {e}")
        return dict(DEFAULT_CONFIG)


def save_config(cfg):
    """保存配置到文件"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"[配置] 已保存到 {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"[配置] 保存失败: {e}")
        return False


def is_configured():
    """检查是否已配置 API"""
    cfg = load_config()
    return bool(cfg.get('baidu_appid') and cfg.get('baidu_secret_key'))