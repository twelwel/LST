"""本地翻译模型（可插拔）

模块功能：
- 模型抽象接口 + 工厂
- 从 HuggingFace 下载模型（int8 / FP32）
- 从本地目录导入自定义模型
- 磁盘缓存管理
- 执行设备选择（CPU / CUDA / DirectML）
"""

from .base import BaseTranslationModel
from .factory import (
    TranslationModelFactory,
    get_model_instance,
    get_available_models,
)
from .registry import MODEL_REGISTRY
from .downloader import (
    ModelDownloader,
    get_models_dir,
    get_app_models_dir,
    get_user_models_dir,
    iter_models_dirs,
    get_model_path,
    find_model_dir,
    has_model_weights,
    list_downloaded_ids,
    is_downloaded,
    is_quantized_downloaded,
    is_fp32_downloaded,
    open_models_dir,
    get_app_root,
)
from .cache import ModelCacheManager
from .custom import (
    import_custom_model,
    remove_custom_model,
    get_custom_model,
    list_custom_models,
    load_custom_models,
    is_valid_ct2_model,
)

# ⭐ 设备控制：Gemma3 是主推 GPU 模型，Opus-MT 强制 CPU
from .gemma3_onnx import (
    set_preferred_device as _set_gemma3_device,
    get_preferred_device as _get_gemma3_device,
)
from . import hf_onnx as _hf_onnx_module


def set_preferred_device(device):
    """设置设备偏好（Gemma3 生效，Opus-MT 忽略）"""
    _set_gemma3_device(device)
    # Opus-MT 也记录但实际不使用
    try:
        _hf_onnx_module.set_preferred_device(device)
    except Exception:
        pass


def get_preferred_device() -> str:
    """返回当前设备偏好（以 Gemma3 为准）"""
    return _get_gemma3_device()


def unload_all_models():
    """卸载所有已加载的本地模型"""
    try:
        TranslationModelFactory.unload_all()
        return True
    except Exception:
        return False


def invalidate_model(model_id=None):
    """丢弃模型缓存（下载完成 / 删除 / 换设备后调用）

    不这样做的话，工厂会一直返回旧实例（可能是未加载状态），
    导致「刚下载完却无法调用」。
    """
    try:
        TranslationModelFactory.invalidate(model_id)
        return True
    except Exception:
        return False


__all__ = [
    'BaseTranslationModel',
    'TranslationModelFactory',
    'get_model_instance',
    'get_available_models',
    'MODEL_REGISTRY',

    'ModelDownloader',
    'get_models_dir',
    'get_app_models_dir',
    'get_user_models_dir',
    'iter_models_dirs',
    'get_model_path',
    'find_model_dir',
    'has_model_weights',
    'list_downloaded_ids',
    'is_downloaded',
    'is_quantized_downloaded',
    'is_fp32_downloaded',
    'open_models_dir',
    'get_app_root',

    'ModelCacheManager',

    'import_custom_model',
    'remove_custom_model',
    'get_custom_model',
    'list_custom_models',
    'load_custom_models',
    'is_valid_ct2_model',

    'set_preferred_device',
    'get_preferred_device',
    'unload_all_models',
    'invalidate_model',
]