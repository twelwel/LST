"""模型缓存与磁盘管理"""

import os
import shutil
from pathlib import Path

from logger import get_logger
from .downloader import get_models_dir, iter_models_dirs, has_model_weights


log = get_logger('local_models.cache')


class ModelCacheManager:
    def __init__(self, max_size_gb: float = 2.0):
        self.cache_dir = get_models_dir()
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)

    def _dirs(self):
        dirs = list(iter_models_dirs())
        return dirs or [self.cache_dir]

    def get_cache_size(self) -> int:
        total = 0
        for base in self._dirs():
            try:
                for f in base.rglob('*'):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except Exception:
                            pass
            except Exception:
                continue
        return total

    def get_cache_size_mb(self) -> float:
        return self.get_cache_size() / (1024 * 1024)

    def list_models(self) -> list:
        """列出已下载的模型（扫描全部候选目录，要求含真实权重）"""
        result = []
        seen = set()

        for base in self._dirs():
            try:
                children = list(base.iterdir())
            except Exception:
                continue

            for d in children:
                if not d.is_dir() or d.name.startswith('.'):
                    continue
                if d.name in seen:
                    continue
                if not has_model_weights(d):
                    continue

                try:
                    size = sum(
                        f.stat().st_size for f in d.rglob('*')
                        if f.is_file()
                    )
                    result.append({
                        'id': d.name,
                        'path': str(d),
                        'base': str(base),
                        'size_mb': size / (1024 * 1024),
                        'mtime': d.stat().st_mtime,
                    })
                    seen.add(d.name)
                except Exception:
                    continue

        return result

    def list_invalid_models(self) -> list:
        """列出「有残留但缺权重」的模型目录

        下载中途失败/被取消会留下 config.json 之类的残骸，
        它们不会被当成可用模型，但会白占磁盘，需要能清理掉。
        """
        from .registry import MODEL_REGISTRY

        result = []
        seen = set()

        for base in self._dirs():
            try:
                children = list(base.iterdir())
            except Exception:
                continue

            for d in children:
                if not d.is_dir() or d.name.startswith('.'):
                    continue
                if d.name in seen:
                    continue
                if has_model_weights(d):
                    continue

                looks_like_model = (
                    d.name in MODEL_REGISTRY
                    or (d / 'config.json').exists()
                    or any(d.rglob('*.onnx'))
                )
                if not looks_like_model:
                    continue

                try:
                    size = sum(
                        f.stat().st_size for f in d.rglob('*')
                        if f.is_file()
                    )
                except Exception:
                    size = 0

                result.append({
                    'id': d.name,
                    'path': str(d),
                    'base': str(base),
                    'size_mb': size / (1024 * 1024),
                })
                seen.add(d.name)

        return result

    def evict_oldest(self):
        """缓存超限时删除最久未使用的模型"""
        if self.get_cache_size() <= self.max_size_bytes:
            return 0
        models = sorted(
            self.list_models(), key=lambda m: m['mtime'])
        removed = 0
        for m in models:
            if self.get_cache_size() <= self.max_size_bytes:
                break
            try:
                shutil.rmtree(m['path'], ignore_errors=True)
                removed += 1
                log.info(f'[缓存] 已删除：{m["id"]}')
            except Exception as e:
                log.warning(f'[缓存] 删除失败 {m["id"]}: {e}')
        return removed

    def delete_model(self, model_id: str) -> bool:
        """删除所有候选目录中的该模型（下载目录可能不止一个）"""
        removed_any = False
        for base in self._dirs():
            path = base / model_id
            if not path.exists():
                continue
            try:
                shutil.rmtree(path, ignore_errors=True)
                removed_any = True
                log.info(f'[缓存] 已删除：{path}')
            except Exception as e:
                log.error(f'[缓存] 删除失败：{e}')
        return removed_any