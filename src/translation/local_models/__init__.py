"""本地翻译模型（可插拔）"""

from .base import BaseTranslationModel
from .factory import (
    TranslationModelFactory,
    get_model_instance,
    get_available_models,
)
from .registry import MODEL_REGISTRY
from .downloader import (
    # 路径获取
    get_models_dir,
    get_user_models_dir,
    get_app_models_dir,
    get_model_path,
    iter_models_dirs,
    open_models_dir,
    get_app_root,
    # 状态检查
    is_downloaded,
    list_downloaded_ids,
    is_cpu_variant_downloaded,
    is_gpu_variant_downloaded,
    is_quantized_downloaded,
    is_fp32_downloaded,
    validate_model_files,  # ⭐ 就是漏了这个
    # 下载器与缓存
    ModelDownloader,
    is_downloading,
    get_downloading_model,
    ModelCacheManager,
)
from .custom import (
    import_custom_model,
    remove_custom_model,
    get_custom_model,
    list_custom_models,
    load_custom_models,
    is_valid_ct2_model,
)

# 设备控制
from .hf_onnx import (
    set_preferred_device as _set_hf_device,
    get_preferred_device as _get_hf_device,
)
from .gemma3_onnx import (
    set_preferred_device as _set_gemma3_device,
    get_preferred_device as _get_gemma3_device,
)

def set_preferred_device(device):
    _set_hf_device(device)
    _set_gemma3_device(device)

def get_preferred_device() -> str:
    return _get_gemma3_device()

def unload_all_models(force: bool = False):
    """卸载全部本地模型。

    默认 force=False：运行期存在已加载的 GPU 模型时跳过卸载，返回 False。
    避免 OCR 的 DML 会话存活时销毁翻译 DML 会话导致驱动崩溃；
    CPU 模型可安全卸载，因此 CPU -> GPU 切换仍可立即生效。
    """
    try:
        return bool(TranslationModelFactory.unload_all(force=force))
    except Exception:
        return False


def invalidate_model(model_id=None, force: bool = False):
    """丢弃模型缓存。

    默认 force=False：有已加载模型时跳过，避免运行期销毁 DML 会话
    导致驱动崩溃；重启应用后新模型/新设备生效。
    """
    try:
        return bool(TranslationModelFactory.invalidate(model_id, force=force))
    except Exception:
        return False


def has_loaded_models() -> bool:
    try:
        return TranslationModelFactory.has_loaded_models()
    except Exception:
        return False


__all__ = [
    'BaseTranslationModel',
    'TranslationModelFactory',
    'get_model_instance',
    'get_available_models',
    'MODEL_REGISTRY',
    'get_models_dir',
    'get_user_models_dir',
    'get_app_models_dir',
    'get_model_path',
    'iter_models_dirs',
    'open_models_dir',
    'get_app_root',
    'is_downloaded',
    'list_downloaded_ids',
    'is_cpu_variant_downloaded',
    'is_gpu_variant_downloaded',
    'is_quantized_downloaded',
    'is_fp32_downloaded',
    'validate_model_files',
    'ModelDownloader',
    'is_downloading',
    'get_downloading_model',
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
    'has_loaded_models',
]