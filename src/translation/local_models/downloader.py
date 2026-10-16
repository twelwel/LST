"""模型下载器（ONNX 版，只下载，不转换）

支持两种变体：
- quantized / q4（默认，约 40MB~764MB）
- fp32（约 800MB~1GB，GPU 友好）
"""

import os
import sys
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

from logger import get_logger
from .registry import get_registry_entry, MODEL_REGISTRY


log = get_logger('local_models.downloader')


# ==================== 下载白名单 ====================

_COMMON_PATTERNS = [
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'added_tokens.json',
    'vocab.json',
    'special_tokens_map.json',
    'source.spm',
    'target.spm',
    '*.model',
]

# Opus-MT
QUANTIZED_PATTERNS = _COMMON_PATTERNS + [
    'onnx/encoder_model_quantized.onnx',
    'onnx/decoder_model_quantized.onnx',
    'onnx/decoder_with_past_model_quantized.onnx',
]

FP32_PATTERNS = _COMMON_PATTERNS + [
    'onnx/encoder_model.onnx',
    'onnx/decoder_model.onnx',
    'onnx/decoder_with_past_model.onnx',
]

# Gemma3：下载整个 onnx 目录
GEMMA3_Q4_PATTERNS = _COMMON_PATTERNS + [
    'onnx/model_q4.onnx',
    'onnx/model_q4.onnx_data',
    'onnx/model_q4.onnx_data_0',
    'onnx/model_q4.onnx_data_1',
    'onnx/model_q4.onnx_data_2',
    'onnx/model_q4.onnx_data_3',
    'onnx/model_quantized.onnx',
]

GEMMA3_FP32_PATTERNS = _COMMON_PATTERNS + [
    'onnx/model.onnx',
    'onnx/model.onnx_data',
    'onnx/model.onnx_data_0',
    'onnx/model.onnx_data_1',
    'onnx/model.onnx_data_2',
    'onnx/model.onnx_data_3',
    'onnx/model_fp16.onnx',
]


# ==================== 全局下载锁 ====================

_download_lock = threading.Lock()
_downloading_model: Optional[str] = None


def is_downloading() -> bool:
    with _download_lock:
        return _downloading_model is not None


def get_downloading_model() -> Optional[str]:
    with _download_lock:
        return _downloading_model


def _set_downloading(model_id: Optional[str]):
    global _downloading_model
    with _download_lock:
        _downloading_model = model_id


# ==================== 路径 ====================

def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent.parent


def get_app_models_dir() -> Path:
    """软件目录下的 models/"""
    return get_app_root() / 'models'


def get_user_models_dir() -> Path:
    """用户目录下的 models/（软件目录不可写时的回退位置）"""
    return Path(os.path.expanduser('~/.screen_translator/models'))


def get_models_dir() -> Path:
    """首选写入目录（软件目录优先，不可写则回退用户目录）"""
    base = get_app_models_dir()
    try:
        base.mkdir(parents=True, exist_ok=True)
        test_file = base / '.write_test'
        test_file.write_text('ok')
        test_file.unlink()
        return base
    except Exception as e:
        log.warning(f"[下载] 软件目录无写权限：{e}，回退用户目录")
        fallback = get_user_models_dir()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def iter_models_dirs():
    """所有候选模型目录（首选在前，去重，仅返回已存在的）

    ⭐ 关键：下载位置与读取位置可能不同（软件目录 / 用户回退目录），
    只认一个目录会导致「明明下载了却找不到模型」。
    """
    seen = set()
    for d in (get_models_dir(), get_user_models_dir(),
              get_app_models_dir()):
        try:
            key = str(d.resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if d.is_dir():
                yield d
        except Exception:
            continue


def has_model_weights(path) -> bool:
    """目录里是否存在真实模型权重（ONNX / CT2）

    只看 config.json 会把「只下了一半」的目录误判为已下载。
    """
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        for f in p.rglob('*.onnx'):
            try:
                if f.stat().st_size > 0:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    try:
        if (p / 'model.bin').exists():
            return True
    except Exception:
        pass
    return False


def find_model_dir(model_id: str):
    """在所有候选目录中查找含真实权重的模型目录"""
    if not model_id:
        return None
    for base in iter_models_dirs():
        p = base / model_id
        try:
            if p.is_dir() and has_model_weights(p):
                return p
        except Exception:
            continue
    return None


def dir_size_bytes(path) -> int:
    """目录总字节数（含下载中的临时分片）"""
    total = 0
    try:
        for f in Path(path).rglob('*'):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except Exception:
                continue
    except Exception:
        pass
    return total


def get_model_path(model_id: str) -> Optional[Path]:
    meta = get_registry_entry(model_id)
    if not meta:
        return None
    return find_model_dir(model_id)


def is_downloaded(model_id: str) -> bool:
    return get_model_path(model_id) is not None


def is_quantized_downloaded(model_id: str) -> bool:
    for base in iter_models_dirs():
        p = base / model_id
        if not p.is_dir():
            continue
        # Opus-MT int8
        if (p / 'onnx' / 'encoder_model_quantized.onnx').exists():
            return True
        # Gemma3 q4
        if (p / 'onnx' / 'model_q4.onnx').exists():
            return True
        if (p / 'onnx' / 'model_quantized.onnx').exists():
            return True
    return False


def is_fp32_downloaded(model_id: str) -> bool:
    for base in iter_models_dirs():
        p = base / model_id
        if not p.is_dir():
            continue
        # Opus-MT FP32
        if (p / 'onnx' / 'encoder_model.onnx').exists():
            return True
        # Gemma3 FP32
        if (p / 'onnx' / 'model.onnx').exists():
            return True
    return False


def list_downloaded_ids() -> set:
    """扫描全部候选目录，返回已下载且含权重的内置模型 id 集合"""
    found = set()
    for base in iter_models_dirs():
        try:
            for d in base.iterdir():
                if not d.is_dir() or d.name.startswith('.'):
                    continue
                if d.name not in MODEL_REGISTRY:
                    continue
                if has_model_weights(d):
                    found.add(d.name)
        except Exception:
            continue
    return found


def open_models_dir() -> bool:
    try:
        # 优先打开真正存放模型的目录，否则打开首选写入目录
        target = get_models_dir()
        for base in iter_models_dirs():
            try:
                has_any = any(
                    d.is_dir() and not d.name.startswith('.')
                    and has_model_weights(d)
                    for d in base.iterdir()
                )
            except Exception:
                has_any = False
            if has_any:
                target = base
                break

        path = str(target)
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
        return True
    except Exception as e:
        log.error(f"[下载] 打开目录失败：{e}")
        return False


# ==================== 下载器 ====================

class ModelDownloader:

    def __init__(self):
        self._cancel = threading.Event()
        self._model_id: Optional[str] = None
        self._variant: str = 'quantized'

    def cancel(self):
        self._cancel.set()

    def is_cancelled(self):
        return self._cancel.is_set()

    def download(self, model_id: str,
                 variant: str = 'quantized',
                 progress_cb: Optional[Callable] = None,
                 status_cb: Optional[Callable] = None
                 ) -> Optional[Path]:

        meta = get_registry_entry(model_id)
        if not meta:
            if status_cb:
                status_cb(f'未知模型：{model_id}')
            return None

        if is_downloading():
            other = get_downloading_model()
            if status_cb:
                status_cb(f'已有下载任务：{other}')
            return None

        if variant not in ('quantized', 'fp32'):
            variant = 'quantized'

        self._cancel.clear()
        self._model_id = model_id
        self._variant = variant
        _set_downloading(f"{model_id} ({variant})")

        try:
            return self._do_download(
                model_id, meta, variant, progress_cb, status_cb)
        finally:
            _set_downloading(None)

    def _do_download(self, model_id: str, meta: dict,
                     variant: str,
                     progress_cb: Optional[Callable],
                     status_cb: Optional[Callable]) -> Optional[Path]:

        base = get_models_dir()
        dest = base / model_id
        dest.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            if status_cb:
                status_cb('缺少 huggingface_hub，请 pip install '
                          'huggingface_hub')
            return None

        os.environ.setdefault('HF_ENDPOINT',
                              'https://hf-mirror.com')

        loader = meta.get('loader', 'hf_onnx')

        # ⭐ 根据 loader + variant 选择 patterns + 预期体积
        if loader == 'gemma3':
            if variant == 'fp32':
                patterns = GEMMA3_FP32_PATTERNS
                expected_mb = meta.get('fp32_mb', 1124)
                variant_label = 'GPU 版（FP32）'
            else:
                patterns = GEMMA3_Q4_PATTERNS
                expected_mb = meta.get('download_mb', 802)
                variant_label = '标准版（q4）'
        else:
            if variant == 'fp32':
                patterns = FP32_PATTERNS
                expected_mb = meta.get('fp32_mb', 647)
                variant_label = 'GPU 版（FP32）'
            else:
                patterns = QUANTIZED_PATTERNS
                expected_mb = meta.get(
                    'download_mb', meta.get('size_mb', 0)) or 171
                variant_label = '标准版（int8）'

        # ⭐ 进度轮询：snapshot_download 不提供进度回调，
        #    靠定时统计目标目录体积来驱动进度条，避免界面看起来卡死
        stop_evt = threading.Event()

        def _monitor():
            while not stop_evt.wait(0.6):
                got_mb = dir_size_bytes(dest) / (1024 * 1024)
                if progress_cb and expected_mb > 0:
                    try:
                        progress_cb(min(0.95, got_mb / expected_mb))
                    except Exception:
                        pass
                if status_cb:
                    try:
                        if expected_mb > 0:
                            status_cb(f'下载中… {got_mb:.0f} / '
                                      f'~{expected_mb:.0f} MB')
                        else:
                            status_cb(f'下载中… {got_mb:.0f} MB')
                    except Exception:
                        pass

        try:
            if status_cb:
                status_cb(
                    f"下载 {meta['name']} · {variant_label}"
                    f"（约 {expected_mb:.0f} MB）...")
            log.info(f"[下载] {model_id} ({variant}) ← {meta['hf_repo']}"
                     f" | 预期 {expected_mb:.0f} MB")

            mon = threading.Thread(target=_monitor, daemon=True)
            mon.start()
            try:
                snapshot_download(
                    repo_id=meta['hf_repo'],
                    local_dir=str(dest),
                    allow_patterns=patterns,
                    max_workers=4,
                )
            finally:
                stop_evt.set()

            if self._cancel.is_set():
                self._cleanup(dest, status_cb, '已取消')
                return None

            # ⭐ 用真实权重校验：只看 config.json 会把半成品当成功
            if not has_model_weights(dest):
                log.error("[下载] 缺少 onnx 权重，判定下载失败")
                if status_cb:
                    status_cb('❌ 模型文件不完整（缺少 onnx 权重）')
                self._cleanup(dest, status_cb)
                return None

            got_mb = dir_size_bytes(dest) / (1024 * 1024)
            if progress_cb:
                progress_cb(1.0)
            if status_cb:
                status_cb(f'✅ 下载完成（{variant_label}，'
                          f'{got_mb:.0f} MB）')
            log.info(f"[下载] 完成：{model_id} ({variant}) "
                     f"{got_mb:.0f} MB")
            return dest

        except Exception as e:
            stop_evt.set()
            log.error(f"[下载] 失败：{e}")
            if status_cb:
                status_cb(f'❌ 下载失败：{e}')
            self._cleanup(dest, status_cb)
            return None

    def _cleanup(self, path: Path, status_cb=None, msg: str = None):
        try:
            shutil.rmtree(path, ignore_errors=True)
            log.info(f"[下载] 已清理：{path}")
        except Exception as e:
            log.warning(f"[下载] 清理失败：{e}")

        if status_cb and msg:
            status_cb(msg)