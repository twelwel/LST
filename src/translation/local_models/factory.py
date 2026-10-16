"""模型工厂：内置 + 自定义 统一管理（ONNX 版）

根据 registry 的 loader 字段选择不同的模型类：
- 'hf_onnx' → HFONNXModel（Opus-MT，Seq2Seq）
- 'gemma3'  → Gemma3ONNXModel（Gemma3，CausalLM）
"""

import threading
import time
from pathlib import Path
from typing import Optional

from logger import get_logger
from .base import BaseTranslationModel
from .registry import MODEL_REGISTRY, get_registry_entry
from .hf_onnx import HFONNXModel
from .gemma3_onnx import Gemma3ONNXModel
from .custom import get_custom_model, list_custom_models


log = get_logger('local_models.factory')


_LOADER_MAP = {
    'hf_onnx': HFONNXModel,
    'gemma3':  Gemma3ONNXModel,
}


class TranslationModelFactory:
    _instances: dict = {}
    _load_locks: dict = {}
    _load_failures: dict = {}
    _global_lock = threading.Lock()

    # 加载失败后的重试冷却（秒）：避免每次翻译都重跑一次昂贵的加载
    RETRY_COOLDOWN = 5.0

    @classmethod
    def create(cls, model_id: str,
               model_path: Optional[Path] = None
               ) -> Optional[BaseTranslationModel]:
        meta = get_registry_entry(model_id)
        is_custom = False

        if not meta:
            custom = get_custom_model(model_id)
            if custom:
                meta = {
                    'id': custom['id'],
                    'name': custom['name'],
                    'size_mb': 0.0,
                    'download_mb': 0.0,
                    'languages': [tuple(x) for x in
                                  custom.get('languages', [])],
                    'hf_repo': '',
                    'description': '自定义模型',
                    'recommended': False,
                    'path': custom['path'],
                    'loader': 'hf_onnx',
                }
                is_custom = True

        if not meta:
            log.warning(f"[Factory] 未知模型：{model_id}")
            return None

        # ⭐ 只有「已加载」的实例才直接复用。
        #    未加载的占位实例必须在下次调用时重新解析路径并尝试加载，
        #    否则模型下载完成后会永远返回未加载实例（=无法调用本地模型）。
        with cls._global_lock:
            cached = cls._instances.get(model_id)
            if cached is not None and cached.is_loaded:
                return cached
            if model_id not in cls._load_locks:
                cls._load_locks[model_id] = threading.Lock()

        lock = cls._load_locks[model_id]
        with lock:
            with cls._global_lock:
                cached = cls._instances.get(model_id)
                if cached is not None and cached.is_loaded:
                    return cached

            # ⭐ 根据 loader 字段选择模型类
            loader = meta.get('loader', 'hf_onnx')
            ModelClass = _LOADER_MAP.get(loader, HFONNXModel)

            # 解析模型路径（每次重新解析，模型可能刚下载好）
            if model_path is None:
                if is_custom:
                    model_path = Path(meta['path'])
                else:
                    from .downloader import get_model_path
                    model_path = get_model_path(model_id)

            if model_path is None or not Path(model_path).exists():
                log.warning(f"[Factory] 模型文件不存在：{model_id}")
                instance = cached or ModelClass(model_id, meta)
                with cls._global_lock:
                    cls._instances[model_id] = instance
                return instance

            # 同一路径近期加载失败过 → 冷却期内不再重试
            fail = cls._load_failures.get(model_id)
            if (fail is not None
                    and fail[0] == str(model_path)
                    and (time.monotonic() - fail[1]) < cls.RETRY_COOLDOWN):
                instance = cached or ModelClass(model_id, meta)
                with cls._global_lock:
                    cls._instances[model_id] = instance
                return instance

            instance = ModelClass(model_id, meta)
            ok = instance.load(Path(model_path))
            if not ok:
                log.error(f"[Factory] 加载失败：{model_id}"
                          f"（{getattr(instance, 'last_error', '')}）")
                cls._load_failures[model_id] = (
                    str(model_path), time.monotonic(),
                    getattr(instance, 'last_error', ''))
            else:
                cls._load_failures.pop(model_id, None)

            with cls._global_lock:
                cls._instances[model_id] = instance

            return instance

    @classmethod
    def invalidate(cls, model_id: str = None):
        """丢弃缓存（下载完成 / 删除 / 切换设备后调用），强制下次重新加载"""
        with cls._global_lock:
            if model_id is None:
                items = list(cls._instances.items())
                cls._instances.clear()
                cls._load_failures.clear()
            else:
                inst = cls._instances.pop(model_id, None)
                cls._load_failures.pop(model_id, None)
                items = [(model_id, inst)] if inst is not None else []

        for _, inst in items:
            try:
                inst.unload()
            except Exception:
                pass

    @classmethod
    def get_if_loaded(cls, model_id: str):
        with cls._global_lock:
            return cls._instances.get(model_id)

    @classmethod
    def unload(cls, model_id: str):
        with cls._global_lock:
            inst = cls._instances.pop(model_id, None)
        if inst is not None:
            try:
                inst.unload()
            except Exception:
                pass

    @classmethod
    def unload_all(cls):
        with cls._global_lock:
            items = list(cls._instances.items())
            cls._instances.clear()
        for _, inst in items:
            try:
                inst.unload()
            except Exception:
                pass


def get_model_instance(model_id: str,
                       model_path: Optional[Path] = None):
    return TranslationModelFactory.create(model_id, model_path)


def get_available_models():
    """所有可用模型元信息（供 UI 展示）"""
    result = []
    for mid, meta in MODEL_REGISTRY.items():
        result.append({
            'id': mid,
            'name': meta.get('name', mid),
            'size_mb': meta.get('size_mb', 0),
            'download_mb': meta.get('download_mb',
                                    meta.get('size_mb', 0)),
            'languages': meta.get('languages', []),
            'description': meta.get('description', ''),
            'recommended': meta.get('recommended', False),
            'is_custom': False,
            'loader': meta.get('loader', 'hf_onnx'),
        })
    for m in list_custom_models():
        result.append({
            'id': m['id'],
            'name': m.get('name', m['id']),
            'size_mb': 0,
            'download_mb': 0,
            'languages': [tuple(x) for x in
                          m.get('languages', [])],
            'description': f"自定义 · {m.get('path', '')}",
            'recommended': False,
            'is_custom': True,
            'path': m.get('path', ''),
            'loader': 'hf_onnx',
        })
    return result