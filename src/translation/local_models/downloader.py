"""模型下载器（ONNX 版，只下载，不转换，含完整校验和实时速度监控）"""

import os
import sys
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional, List

from logger import get_logger
from .registry import get_registry_entry

log = get_logger('local_models.downloader')

# ==================== 文件白名单 ====================
_COMMON_PATTERNS = [
    'config.json', 'generation_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json',
    'source.spm', 'target.spm', '*.model',
]

OPUS_INT8_PATTERNS = _COMMON_PATTERNS + [
    'onnx/encoder_model_quantized.onnx',
    'onnx/decoder_model_quantized.onnx',
    'onnx/decoder_with_past_model_quantized.onnx',
]
OPUS_FP32_PATTERNS = _COMMON_PATTERNS + [
    'onnx/encoder_model.onnx',
    'onnx/decoder_model.onnx',
    'onnx/decoder_with_past_model.onnx',
]

GEMMA3_Q4_PATTERNS = _COMMON_PATTERNS + [
    'onnx/model_q4.onnx', 'onnx/model_q4.onnx_data*',
]
GEMMA3_FP32_PATTERNS = _COMMON_PATTERNS + [
    'onnx/model.onnx', 'onnx/model.onnx_data*',
]


# ==================== 全局锁 ====================
_download_lock = threading.Lock()
_downloading_model: Optional[str] = None

def is_downloading() -> bool:
    with _download_lock: return _downloading_model is not None

def get_downloading_model() -> Optional[str]:
    with _download_lock: return _downloading_model

def _set_downloading(model_id: Optional[str]):
    global _downloading_model
    with _download_lock: _downloading_model = model_id


# ==================== 路径管理 ====================
def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent.parent

def get_user_models_dir() -> Path:
    fallback = Path(os.path.expanduser('~/.screen_translator/models'))
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

def get_models_dir() -> Path:
    base = get_app_root()
    models_dir = base / 'models'
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        test_file = models_dir / '.write_test'
        test_file.write_text('ok')
        test_file.unlink()
        return models_dir
    except Exception:
        return get_user_models_dir()

def get_app_models_dir() -> Path: return get_models_dir()

def iter_models_dirs() -> List[Path]:
    base = get_models_dir()
    if not base.exists(): return []
    return [d for d in base.iterdir() if d.is_dir() and not d.name.startswith('.')]

def get_model_path(model_id: str) -> Optional[Path]:
    path = get_models_dir() / model_id
    if path.exists() and (path / 'config.json').exists():
        return path
    return None

# ==================== 状态检测 ====================
def is_downloaded(model_id: str) -> bool:
    return get_model_path(model_id) is not None


def list_downloaded_ids() -> List[str]:
    """已下载且包含真实 ONNX 权重的内置模型 id（供 API 配置自动同步）。"""
    from .registry import MODEL_REGISTRY
    result = []
    for mid in MODEL_REGISTRY:
        path = get_models_dir() / mid
        if not path.exists() or not (path / 'config.json').exists():
            continue
        try:
            if has_model_weights(path):
                result.append(mid)
        except Exception:
            continue
    return result

def is_cpu_variant_downloaded(model_id: str) -> bool:
    p = get_models_dir() / model_id
    if not p.exists(): return False
    return ((p / 'onnx' / 'encoder_model_quantized.onnx').exists() or 
            (p / 'onnx' / 'model_q4.onnx').exists())

def is_gpu_variant_downloaded(model_id: str) -> bool:
    p = get_models_dir() / model_id
    if not p.exists(): return False
    return ((p / 'onnx' / 'encoder_model.onnx').exists() or 
            (p / 'onnx' / 'model.onnx').exists())

def is_quantized_downloaded(model_id: str) -> bool: return is_cpu_variant_downloaded(model_id)
def is_fp32_downloaded(model_id: str) -> bool: return is_gpu_variant_downloaded(model_id)

def open_models_dir() -> bool:
    try:
        path = str(get_models_dir())
        if sys.platform == 'win32': os.startfile(path)
        elif sys.platform == 'darwin': os.system(f'open "{path}"')
        else: os.system(f'xdg-open "{path}"')
        return True
    except Exception: return False


# ==================== 完整性校验 ====================
def validate_model_files(model_id: str, variant: str) -> tuple[bool, str]:
    p = get_models_dir() / model_id
    if not p.exists(): return False, "目录不存在"

    if not (p / 'config.json').exists():
        return False, "缺少配置文件 (config.json)"

    if variant == 'cpu':
        if (p / 'onnx' / 'encoder_model_quantized.onnx').exists():
            if (p / 'onnx' / 'encoder_model_quantized.onnx').stat().st_size < 1024:
                return False, "文件损坏"
        elif (p / 'onnx' / 'model_q4.onnx').exists():
            if (p / 'onnx' / 'model_q4.onnx').stat().st_size < 1024:
                return False, "文件损坏"
        else:
            return False, "未找到 CPU 模型权重"
            
    elif variant == 'gpu':
        if (p / 'onnx' / 'encoder_model.onnx').exists():
            if (p / 'onnx' / 'encoder_model.onnx').stat().st_size < 1024:
                return False, "文件损坏"
        elif (p / 'onnx' / 'model.onnx').exists():
            if (p / 'onnx' / 'model.onnx').stat().st_size < 1024:
                return False, "文件损坏"
        else:
            return False, "未找到 GPU 模型权重"

    return True, "完整"


# ==================== 辅助：判断目录是否含真实权重 ====================
def has_model_weights(model_dir) -> bool:
    """目录内是否存在真实 ONNX 权重（用于区分完整模型与下载残骸）"""
    try:
        d = Path(model_dir)
        if not d.is_dir():
            return False
        # 下载残骸也可能留下几百 KB 的 .onnx 占位文件，按 1MB 过滤
        return any(f.is_file() and f.stat().st_size > 1024 * 1024
                   for f in d.rglob('*.onnx'))
    except Exception:
        return False


# ==================== 辅助：计算目录大小 ====================
def _get_dir_size(path: Path) -> int:
    total = 0
    if not path.exists(): return 0
    for f in path.rglob('*'):
        if f.is_file():
            try: total += f.stat().st_size
            except: pass
    return total


# ==================== 缓存管理 ====================
class ModelCacheManager:
    def __init__(self, max_size_gb: float = 5.0):
        self.cache_dir = get_models_dir()
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)

    def get_cache_size(self) -> int:
        return _get_dir_size(self.cache_dir)

    def get_cache_size_mb(self) -> float:
        return self.get_cache_size() / (1024 * 1024)

    def list_models(self) -> list:
        result = []
        if not self.cache_dir.exists(): return result
        for d in self.cache_dir.iterdir():
            if not d.is_dir() or d.name.startswith('.'): continue
            has_model = ((d / 'config.json').exists() or any(d.rglob('*.onnx')))
            if not has_model: continue
            try:
                size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                result.append({
                    'id': d.name, 'path': str(d),
                    'size_mb': size / (1024 * 1024),
                    'mtime': d.stat().st_mtime,
                })
            except Exception: continue
        return result

    def delete_model(self, model_id: str) -> bool:
        path = self.cache_dir / model_id
        if not path.exists(): return False
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception as e:
            log.error(f'[缓存] 删除失败：{e}')
            return False


# ==================== 下载器 ====================
class ModelDownloader:
    def __init__(self):
        self._cancel = threading.Event()
        self._model_id: Optional[str] = None

    def cancel(self): self._cancel.set()
    def is_cancelled(self): return self._cancel.is_set()

    def download(self, model_id: str, variant: str = 'cpu',
                 progress_cb: Optional[Callable[[float, float], None]] = None,
                 status_cb: Optional[Callable[[str], None]] = None) -> Optional[Path]:
        """
        progress_cb(p: float, speed_mb: float) -> None
        """
        meta = get_registry_entry(model_id)
        if not meta:
            if status_cb: status_cb(f'未知模型：{model_id}')
            return None
        if is_downloading():
            if status_cb: status_cb(f'已有任务进行中：{get_downloading_model()}')
            return None

        if variant not in ('cpu', 'gpu'):
            variant = 'cpu'

        self._cancel.clear()
        self._model_id = model_id
        _set_downloading(f"{model_id} ({variant})")

        try:
            return self._do_download(model_id, meta, variant, progress_cb, status_cb)
        finally:
            _set_downloading(None)

    def _do_download(self, model_id, meta, variant, progress_cb, status_cb):
        dest = get_models_dir() / model_id
        dest.mkdir(parents=True, exist_ok=True)
        
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            if status_cb: status_cb('缺少 huggingface_hub')
            return None

        os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
        loader = meta.get('loader', 'hf_onnx')

        if loader == 'gemma3':
            if variant == 'gpu':
                patterns, size_mb, v_label = GEMMA3_FP32_PATTERNS, 1024, 'GPU版 (FP32)'
            else:
                patterns, size_mb, v_label = GEMMA3_Q4_PATTERNS, 764, 'CPU版 (q4)'
        else:
            if variant == 'gpu':
                patterns, size_mb, v_label = OPUS_FP32_PATTERNS, 800, 'GPU版 (FP32)'
            else:
                patterns, size_mb, v_label = OPUS_INT8_PATTERNS, 40, 'CPU版 (int8)'

        # ⭐ 启动监控线程
        stop_monitor = threading.Event()
        
        def monitor_task():
            last_size = _get_dir_size(dest)
            last_time = time.time()
            total_bytes = max(size_mb * 1024 * 1024, last_size)

            while not stop_monitor.is_set():
                time.sleep(0.8) # 每 0.8 秒刷新一次
                if stop_monitor.is_set(): break
                
                current_size = _get_dir_size(dest)
                now = time.time()
                dt = now - last_time
                
                if dt > 0:
                    speed_bytes = (current_size - last_size) / dt
                    speed_mb = speed_bytes / (1024 * 1024)
                    
                    # 动态更新 total_bytes，防止预估偏小导致进度条超出
                    if current_size > total_bytes: total_bytes = current_size
                    
                    progress = current_size / total_bytes if total_bytes > 0 else 0
                    if progress > 0.99: progress = 0.99 # 预留 1% 给校验
                    
                    if progress_cb:
                        try:
                            progress_cb(progress, speed_mb)
                        except Exception:
                            pass
                            
                    last_size = current_size
                    last_time = now
                    
        mon_thread = threading.Thread(target=monitor_task, daemon=True)
        mon_thread.start()

        try:
            if status_cb: status_cb(f"下载 {meta['name']} · {v_label}...")
            log.info(f"[下载] {model_id} ({variant}) ← {meta['hf_repo']}")
            
            snapshot_download(
                repo_id=meta['hf_repo'],
                local_dir=str(dest),
                allow_patterns=patterns,
                resume_download=True,
                max_workers=4,
            )
            
            stop_monitor.set() # 停止监控
            
            if self._cancel.is_set():
                if status_cb: status_cb('已取消')
                return None

            if status_cb: status_cb("正在校验文件完整性...")
            if progress_cb: progress_cb(1.0, 0.0)
            
            is_ok, reason = validate_model_files(model_id, variant)
            if is_ok:
                if status_cb: status_cb(f'✅ 下载成功，文件校验通过')
                log.info(f"[下载] 成功：{model_id} ({variant})")
                return dest
            else:
                if status_cb: status_cb(f'❌ 校验失败：{reason}')
                log.error(f"[下载] 校验失败：{model_id} - {reason}")
                return None

        except Exception as e:
            stop_monitor.set()
            log.error(f"[下载] 网络或IO异常：{e}")
            if status_cb: status_cb(f'❌ 下载失败：{str(e)[:50]}...')
            return None