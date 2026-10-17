"""模型下载器（ONNX 版，只下载，不转换，含完整校验和实时速度监控）"""

import fnmatch
import os
import sys
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, List

from logger import get_logger
from .registry import get_registry_entry

# PyInstaller 的窗口模式（console=False）没有 stdout/stderr。
# huggingface_hub 默认会向 stderr 画 tqdm 进度条，直接写 None 会抛
# "'NoneType' object has no attribute 'write'"，所以必须在导入前禁用。
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

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


def _try_begin_download(label: str) -> bool:
    """原子占位，避免两个窗口同时通过 is_downloading() 检查。"""
    global _downloading_model
    with _download_lock:
        if _downloading_model is not None:
            return False
        _downloading_model = label
        return True


def _end_download():
    global _downloading_model
    with _download_lock:
        _downloading_model = None


# ==================== 路径管理 ====================
def get_app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent.parent

def _user_models_dir() -> Path:
    return Path(os.path.expanduser('~/.screen_translator/models'))


def get_user_models_dir() -> Path:
    fallback = _user_models_dir()
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return fallback


def _try_writable_dir(path: Path) -> bool:
    test_file = path / f'.write_test_{os.getpid()}_{threading.get_ident()}'
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file.write_text('ok')
        return True
    except Exception:
        return False
    finally:
        try:
            test_file.unlink()
        except Exception:
            pass


_models_dir_cache: Optional[Path] = None
_models_dir_lock = threading.Lock()


def get_models_dir() -> Path:
    """下载/保存模型的首选目录（软件目录可写时优先）。

    目录只解析一次并缓存，避免下载线程与 UI 线程并发写权限探测时
    得到不同结果，造成「下载到了 A 目录、界面却去 B 目录找」。
    """
    global _models_dir_cache
    with _models_dir_lock:
        if _models_dir_cache is not None:
            return _models_dir_cache
        models_dir = get_app_root() / 'models'
        if _try_writable_dir(models_dir):
            _models_dir_cache = models_dir
        else:
            _models_dir_cache = get_user_models_dir()
        return _models_dir_cache


def get_app_models_dir() -> Path:
    return get_models_dir()


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except Exception:
        return str(path).casefold()


def get_model_roots() -> List[Path]:
    """所有可能存放模型的根目录（按优先级排序、去重）。

    打包后 exe 位于 ``dist/LST-.../`` 时，会同时检查 exe 目录、
    当前工作目录以及最多三级父目录下的 ``models``。这样源码目录、
    旧版便携包或用户目录里已有的模型都能被界面检测到。
    """
    raw: List[Path] = [
        get_app_root() / 'models',
        _user_models_dir(),
    ]

    try:
        raw.append(Path.cwd() / 'models')
    except Exception:
        pass

    if getattr(sys, 'frozen', False):
        try:
            cur = Path(sys.executable).resolve().parent
            for _ in range(3):
                cur = cur.parent
                raw.append(cur / 'models')
        except Exception:
            pass

        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            raw.append(Path(meipass) / 'models')

    result: List[Path] = []
    seen = set()
    for path in raw:
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _looks_like_model_dir(path: Path) -> bool:
    try:
        return bool(
            (path / 'config.json').is_file()
            or (path / 'model.bin').is_file()
            or (path / 'onnx').is_dir()
            or any(path.glob('*.onnx'))
        )
    except Exception:
        return False


def iter_models_dirs() -> List[Path]:
    """扫描全部候选目录下的模型文件夹（不含缓存/隐藏目录）。"""
    result: List[Path] = []
    seen = set()
    for root in get_model_roots():
        try:
            children = list(root.iterdir())
        except Exception:
            continue
        for d in children:
            if not d.is_dir() or d.name.startswith('.'):
                continue
            if not _looks_like_model_dir(d):
                continue
            key = _path_key(d)
            if key in seen:
                continue
            seen.add(key)
            result.append(d)
    return result


_CPU_WEIGHT_FILES = (
    'onnx/encoder_model_quantized.onnx',
    'onnx/model_q4.onnx',
)
_GPU_WEIGHT_FILES = (
    'onnx/encoder_model.onnx',
    'onnx/model.onnx',
)
_MIN_WEIGHT_BYTES = 1024 * 1024


def _variant_weight_file(path: Path, variant: str) -> Optional[Path]:
    files = _CPU_WEIGHT_FILES if variant == 'cpu' else _GPU_WEIGHT_FILES
    for rel in files:
        f = path / rel
        try:
            if f.is_file():
                return f
        except OSError:
            continue
    return None

def _iter_model_dir_candidates(model_id: str):
    """生成模型目录候选路径。

    兼容标准 ``models/<id>``、手动复制成 HF 仓库名的目录、
    ``Xenova/opus-mt-en-zh`` 多一层目录，以及 HF 缓存快照目录。
    """
    mid = (model_id or '').strip().replace('\\', '/').strip('/')
    if not mid or '..' in mid.split('/'):
        return

    names = [mid]
    owner = repo_name = ''
    try:
        from .registry import get_registry_entry
        meta = get_registry_entry(mid) or {}
        repo = str(meta.get('hf_repo') or '').strip()
        if '/' in repo:
            owner, repo_name = repo.split('/', 1)
            for name in (repo_name,
                         f'{owner}__{repo_name}',
                         f'models--{owner}--{repo_name}'):
                if name and name not in names:
                    names.append(name)
    except Exception:
        pass

    seen = set()

    def _emit(path: Path):
        key = _path_key(path)
        if key in seen:
            return None
        seen.add(key)
        return path

    for root in get_model_roots():
        for name in names:
            p = _emit(root / name)
            if p is not None:
                yield p
            if owner and name == repo_name:
                p = _emit(root / owner / repo_name)
                if p is not None:
                    yield p

        if not owner:
            continue
        cache_dir = root / f'models--{owner}--{repo_name}'
        snapshots = cache_dir / 'snapshots'
        if not snapshots.is_dir():
            continue
        try:
            snap_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            snap_dirs.sort(
                key=lambda d: d.stat().st_mtime, reverse=True)
            for snap in snap_dirs[:5]:
                p = _emit(snap)
                if p is not None:
                    yield p
        except Exception:
            pass


def _model_dir_score(path: Path):
    """多个目录都含同一模型时，优先选同时有 CPU/GPU 权重且较新的。"""
    has_cpu = bool(_variant_weight_file(path, 'cpu'))
    has_gpu = bool(_variant_weight_file(path, 'gpu'))
    try:
        mtime = max(
            (f.stat().st_mtime for f in (path / 'onnx').glob('*.onnx')),
            default=path.stat().st_mtime,
        )
    except Exception:
        mtime = 0.0
    return (int(has_cpu and has_gpu), int(has_cpu) + int(has_gpu), mtime)


def find_model_dir(model_id: str,
                   variant: Optional[str] = None) -> Optional[Path]:
    """在全部候选目录中查找可用模型目录。

    variant 为 ``cpu`` / ``gpu`` 时要求对应权重存在；为 None 时要求
    目录内至少有一个真实 ONNX 权重。
    """
    if variant not in (None, 'cpu', 'gpu'):
        variant = 'cpu'

    best: Optional[Path] = None
    best_score = (-1, -1, 0.0)
    for path in _iter_model_dir_candidates(model_id):
        try:
            if not path.is_dir() or not (path / 'config.json').is_file():
                continue

            if variant:
                weight = _variant_weight_file(path, variant)
                if weight is None or weight.stat().st_size < _MIN_WEIGHT_BYTES:
                    continue
                return path

            if not has_model_weights(path):
                continue
            score = _model_dir_score(path)
            if score > best_score:
                best, best_score = path, score
        except OSError:
            continue
    return best


def get_model_path(model_id: str,
                   variant: Optional[str] = None) -> Optional[Path]:
    return find_model_dir(model_id, variant)



# ==================== 状态检测 ====================
def is_downloaded(model_id: str) -> bool:
    return get_model_path(model_id) is not None


def list_downloaded_ids() -> List[str]:
    """已下载且包含真实 ONNX 权重的内置模型 id（供 API 配置自动同步）。"""
    from .registry import MODEL_REGISTRY
    result = []
    for mid in MODEL_REGISTRY:
        try:
            if get_model_path(mid) is not None:
                result.append(mid)
        except Exception:
            continue
    return result

def is_cpu_variant_downloaded(model_id: str) -> bool:
    return find_model_dir(model_id, 'cpu') is not None


def is_gpu_variant_downloaded(model_id: str) -> bool:
    return find_model_dir(model_id, 'gpu') is not None


def is_quantized_downloaded(model_id: str) -> bool:
    return is_cpu_variant_downloaded(model_id)


def is_fp32_downloaded(model_id: str) -> bool:
    return is_gpu_variant_downloaded(model_id)

def open_models_dir(model_id: Optional[str] = None) -> bool:
    try:
        path = find_model_dir(model_id) if model_id else None
        path = str(path or get_models_dir())
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
        return True
    except Exception:
        return False


# ==================== 完整性校验 ====================
def validate_model_files(model_id: str, variant: str) -> tuple[bool, str]:
    variant = 'gpu' if variant in ('gpu', 'fp32', 'fp16') else 'cpu'
    label = 'CPU' if variant == 'cpu' else 'GPU'

    found_dir = False
    found_config = False
    found_weight = False
    broken = False

    for path in _iter_model_dir_candidates(model_id):
        try:
            if not path.is_dir():
                continue
            found_dir = True
            if not (path / 'config.json').is_file():
                continue
            found_config = True

            weight = _variant_weight_file(path, variant)
            if weight is None:
                continue
            found_weight = True
            if weight.stat().st_size < _MIN_WEIGHT_BYTES:
                broken = True
                continue
            return True, "完整"
        except OSError:
            broken = True
            continue

    if not found_dir:
        return False, f"目录不存在（已扫描 {len(get_model_roots())} 个位置）"
    if not found_config:
        return False, "缺少配置文件 (config.json)"
    if found_weight and broken:
        return False, "文件损坏"
    return False, f"未找到 {label} 模型权重"


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


# ==================== 下载进度辅助 ====================
def _matches_patterns(name: str, patterns: List[str]) -> bool:
    """按 HuggingFace 的 allow_patterns 规则匹配仓库内相对路径。"""
    name = name.replace('\\', '/')
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _get_downloaded_bytes(dest: Path, patterns: List[str],
                          incomplete_since: Optional[float] = None) -> int:
    """统计当前变体已落盘的字节数。

    包括：
    - 已下载完成、且匹配 allow_patterns 的正式文件
    - ``.cache/huggingface/download`` 下的 ``*.incomplete`` 临时文件

    ``incomplete_since`` 用于忽略上次失败遗留的旧临时文件，避免进度条
    一开始就被旧文件撑到 99%。当前任务写入临时文件后 mtime 会更新，
    因此续传文件仍能正常计入。

    注意：不能直接统计整个模型目录，否则同时存在 CPU/GPU 两个版本时
    会把另一个变体的体积也算进来。
    """
    total = 0
    if dest.exists():
        for f in dest.rglob('*'):
            try:
                if not f.is_file():
                    continue
                rel = f.relative_to(dest).as_posix()
                if rel.startswith('.cache/'):
                    continue
                if _matches_patterns(rel, patterns):
                    total += f.stat().st_size
            except (OSError, ValueError):
                continue

        cache_dir = dest / '.cache' / 'huggingface' / 'download'
        if cache_dir.exists():
            for f in cache_dir.rglob('*.incomplete'):
                try:
                    if (incomplete_since is not None
                            and f.stat().st_mtime < incomplete_since):
                        continue
                    total += f.stat().st_size
                except OSError:
                    pass
    return total


def _resolve_total_bytes(repo_id: str, patterns: List[str],
                         fallback_bytes: int) -> int:
    """用 HF 元数据计算本次需要下载的精确总字节数。

    取不到元数据时（离线、镜像异常等）退回注册表里的估算值。
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        sizes = [
            sibling.size for sibling in (info.siblings or [])
            if sibling.size
            and _matches_patterns(sibling.rfilename, patterns)
        ]
        total = sum(sizes)
        if total > 0:
            log.info(f"[下载] {repo_id} 预计下载 {total / 1024 / 1024:.1f} MB"
                     f"（{len(sizes)} 个文件）")
            return total
    except Exception as e:
        log.warning(f"[下载] 获取文件体积失败，使用估算值：{e}")
    return max(int(fallback_bytes), 0)


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} B"


def _disable_hf_progress_bars():
    """关闭 huggingface_hub 自带进度条（窗口模式没有 stderr）。"""
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


# ==================== 缓存管理 ====================
class ModelCacheManager:
    def __init__(self, max_size_gb: float = 5.0):
        self.cache_dir = get_models_dir()
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)

    def get_cache_size(self) -> int:
        total = 0
        for d in iter_models_dirs():
            total += _get_dir_size(d)
        if total == 0:
            total = _get_dir_size(self.cache_dir)
        return total

    def get_cache_size_mb(self) -> float:
        return self.get_cache_size() / (1024 * 1024)

    def list_models(self) -> list:
        result = []
        seen = set()
        for d in iter_models_dirs():
            if d.name in seen:
                continue
            if not has_model_weights(d):
                continue
            try:
                size = sum(
                    f.stat().st_size for f in d.rglob('*')
                    if f.is_file())
                result.append({
                    'id': d.name, 'path': str(d),
                    'base': str(d.parent),
                    'size_mb': size / (1024 * 1024),
                    'mtime': d.stat().st_mtime,
                })
                seen.add(d.name)
            except Exception:
                continue
        return result

    def delete_model(self, model_id: str) -> bool:
        removed = False
        seen = set()
        for path in _iter_model_dir_candidates(model_id):
            try:
                if not path.exists():
                    continue
                key = _path_key(path)
                if key in seen:
                    continue
                seen.add(key)
                shutil.rmtree(path, ignore_errors=True)
                removed = True
                log.info(f'[缓存] 已删除：{path}')
            except Exception as e:
                log.error(f'[缓存] 删除失败 {path}：{e}')
        return removed


# ==================== 下载器 ====================
class ModelDownloader:
    def __init__(self):
        self._cancel = threading.Event()
        self._model_id: Optional[str] = None

    def cancel(self): self._cancel.set()
    def is_cancelled(self): return self._cancel.is_set()

    def download(self, model_id: str, variant: str = 'cpu',
                 progress_cb: Optional[Callable[[float, float, int, int], None]] = None,
                 status_cb: Optional[Callable[[str], None]] = None) -> Optional[Path]:
        """
        progress_cb(progress, speed_mb, downloaded_bytes, total_bytes) -> None

        progress 为 0.0~1.0；speed_mb 为最近约 2 秒的平均下载速度（MB/s）。
        """
        meta = get_registry_entry(model_id)
        if not meta:
            if status_cb: status_cb(f'未知模型：{model_id}')
            return None
        # UI 使用 quantized/fp32，下载器统一使用 cpu/gpu。
        if variant in ('gpu', 'fp32', 'fp16'):
            variant = 'gpu'
        else:
            variant = 'cpu'

        self._cancel.clear()
        self._model_id = model_id
        label = f"{model_id} ({variant})"
        if not _try_begin_download(label):
            if status_cb:
                status_cb(f'已有任务进行中：{get_downloading_model()}')
            return None

        try:
            if self._cancel.is_set():
                if status_cb: status_cb('已取消')
                return None
            return self._do_download(
                model_id, meta, variant, progress_cb, status_cb)
        finally:
            _end_download()

    def _do_download(self, model_id, meta, variant, progress_cb, status_cb):
        dest = get_models_dir() / model_id
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if status_cb:
                status_cb(f'❌ 无法创建模型目录：{e}')
            log.error(f'[下载] 无法创建目录 {dest}：{e}')
            return None

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            if status_cb: status_cb('缺少 huggingface_hub，无法下载')
            log.error('[下载] 缺少 huggingface_hub')
            return None

        # 窗口模式没有 stderr，必须关闭 HF 内置 tqdm（见文件头说明）。
        _disable_hf_progress_bars()
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

        def emit_progress(progress, speed_mb, done_bytes, total_bytes):
            if not progress_cb:
                return
            try:
                progress_cb(progress, speed_mb, done_bytes, total_bytes)
            except TypeError:
                # 兼容旧的二参数回调
                try:
                    progress_cb(progress, speed_mb)
                except Exception:
                    pass
            except Exception:
                pass

        if status_cb: status_cb("正在获取文件信息...")
        total_bytes = _resolve_total_bytes(
            meta['hf_repo'], patterns, int(size_mb * 1024 * 1024))

        if self._cancel.is_set():
            if status_cb: status_cb('已取消')
            return None

        # ⭐ 启动监控线程
        stop_monitor = threading.Event()
        incomplete_since = time.time() - 5
        
        def monitor_task():
            nonlocal total_bytes
            initial_done = min(_get_downloaded_bytes(
                dest, patterns, incomplete_since),
                total_bytes) if total_bytes > 0 else 0
            samples = deque([(time.monotonic(), initial_done)])
            initial_progress = initial_done / total_bytes if total_bytes > 0 else 0.0
            emit_progress(min(initial_progress, 0.99), 0.0,
                          initial_done, total_bytes)

            while not stop_monitor.wait(0.5):
                current = _get_downloaded_bytes(dest, patterns, incomplete_since)
                now = time.monotonic()
                
                # 元数据体积异常偏小时动态扩展，避免百分比长期卡在 99%
                if current > total_bytes:
                    total_bytes = current

                samples.append((now, current))
                # 使用最近约 2 秒的窗口计算平均速度，数字更稳定
                while len(samples) > 2 and now - samples[0][0] > 2.0:
                    samples.popleft()

                base_time, base_bytes = samples[0]
                dt = now - base_time
                if dt > 0:
                    speed_mb = max(0.0, current - base_bytes) / dt / (1024 * 1024)
                else:
                    speed_mb = 0.0

                progress = current / total_bytes if total_bytes > 0 else 0.0
                # 预留 1% 给最后的文件完整性校验
                progress = min(max(progress, 0.0), 0.99)
                emit_progress(progress, speed_mb, current, total_bytes)
                    
        mon_thread = threading.Thread(target=monitor_task, daemon=True)
        mon_thread.start()

        try:
            if status_cb: status_cb(f"下载 {meta['name']} · {v_label}...")
            log.info(f"[下载] {model_id} ({variant}) ← {meta['hf_repo']}"
                     f" · {_format_bytes(total_bytes)}")
            
            snapshot_download(
                repo_id=meta['hf_repo'],
                local_dir=str(dest),
                allow_patterns=patterns,
                resume_download=True,
                max_workers=4,
            )
            
            stop_monitor.set()
            mon_thread.join(timeout=1.5)

            if self._cancel.is_set():
                if status_cb: status_cb('已取消')
                return None

            actual_bytes = (_get_downloaded_bytes(
                dest, patterns, incomplete_since) or total_bytes)
            emit_progress(1.0, 0.0, actual_bytes, actual_bytes)
            if status_cb: status_cb("正在校验文件完整性...")
            
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
            log.exception(f"[下载] 网络或IO异常：{e}")
            text = str(e).strip()
            low = text.lower()
            if 'nonetype' in low and 'write' in low:
                text = '下载组件无法输出进度（打包模式），请查看日志'
            elif 'connection' in low or 'timeout' in low:
                text = '网络连接失败，请检查网络或代理设置'
            elif 'no space' in low or 'disk full' in low:
                text = '磁盘空间不足'
            if status_cb: status_cb(f'❌ 下载失败：{text[:120]}')
            return None
        finally:
            stop_monitor.set()
            if mon_thread.is_alive():
                mon_thread.join(timeout=1.0)
