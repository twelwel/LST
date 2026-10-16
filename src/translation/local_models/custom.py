"""自定义模型注册表

支持 ONNX / CT2 两种格式
注册信息保存在「软件目录/models/custom_models.json」
"""

import os
import json
import time
import threading
from pathlib import Path

from logger import get_logger


log = get_logger('local_models.custom')


_lock = threading.Lock()


def _get_custom_file() -> str:
    from .downloader import get_models_dir
    return str(get_models_dir() / 'custom_models.json')


# ==================== 读写 ====================

def load_custom_models() -> list:
    with _lock:
        path = _get_custom_file()
        if not os.path.isfile(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                valid = []
                for m in data:
                    p = m.get('path', '')
                    if p and os.path.isdir(p):
                        valid.append(m)
                    else:
                        log.warning(f"[自定义模型] 路径不存在：{p}")
                return valid
        except Exception as e:
            log.error(f"[自定义模型] 读取失败：{e}")
        return []


def save_custom_models(models: list) -> bool:
    with _lock:
        try:
            path = _get_custom_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(models, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return True
        except Exception as e:
            log.error(f"[自定义模型] 保存失败：{e}")
            return False


# ==================== 验证 ====================

def is_valid_model(path: str) -> bool:
    """检查是否是合法的模型目录（支持 ONNX / CT2）"""
    if not path or not os.path.isdir(path):
        return False

    p = Path(path)

    # ONNX 格式（首选）
    onnx_patterns = [
        'encoder_model.onnx',
        'encoder_model_quantized.onnx',
        'onnx/encoder_model.onnx',
        'onnx/encoder_model_quantized.onnx',
        'model.onnx',
    ]
    for pat in onnx_patterns:
        if (p / pat).exists():
            return True

    # CT2 格式（兼容旧模型）
    if (p / 'model.bin').exists():
        return True

    return False


# 兼容旧名字
is_valid_ct2_model = is_valid_model


# ==================== 导入 / 移除 ====================

def import_custom_model(path: str, name: str = '',
                        languages: list = None) -> dict:
    path = os.path.abspath(path)

    if not os.path.isdir(path):
        log.error(f"[自定义模型] 路径不存在：{path}")
        return None

    if not is_valid_model(path):
        log.error(f"[自定义模型] 不是合法的模型：{path}")
        return None

    existing = load_custom_models()
    for m in existing:
        if os.path.abspath(m.get('path', '')) == path:
            log.info(f"[自定义模型] 已存在：{path}")
            return m

    model_id = f"custom_{int(time.time())}"

    if not name:
        name = os.path.basename(path) or model_id

    if not languages:
        languages = [['auto', 'zh']]

    entry = {
        'id': model_id,
        'name': name,
        'path': path,
        'format': 'onnx',
        'languages': languages,
        'imported_at': int(time.time()),
    }

    existing.append(entry)
    if not save_custom_models(existing):
        return None

    log.info(f"[自定义模型] 已导入：{name} → {path}")
    return entry


def remove_custom_model(model_id: str) -> bool:
    models = load_custom_models()
    new_models = [m for m in models if m.get('id') != model_id]
    if len(new_models) == len(models):
        return False
    return save_custom_models(new_models)


# ==================== 查询 ====================

def get_custom_model(model_id: str):
    for m in load_custom_models():
        if m.get('id') == model_id:
            return m
    return None


def list_custom_models() -> list:
    return load_custom_models()