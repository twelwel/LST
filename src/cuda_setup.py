"""GPU 加速环境安装
支持三种加速方式：
- cpu_multi：多线程 CPU（0 额外下载）
- directml：DirectML（~50MB，支持所有显卡）
- cuda：NVIDIA CUDA（~1.6GB，仅 NVIDIA）

注意：打包后的 exe 无法自动 pip 安装（sys.executable 指向 exe 本身），
      会通过 _check_not_frozen() 拒绝安装，避免无限弹窗。
"""

import os
import sys
import site
import subprocess
import threading

from logger import get_logger


log = get_logger('cuda')


PIP_INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple'

# CUDA 12 依赖（和 onnxruntime-gpu 1.20.0 对齐）
CUDA_PACKAGES = [
    'nvidia-cuda-runtime-cu12==12.6.77',
    'nvidia-cudnn-cu12==9.4.0.58',
    'nvidia-cublas-cu12==12.6.4.1',
    'nvidia-cufft-cu12==11.3.0.4',
    'nvidia-curand-cu12==10.3.7.77',
]

# onnxruntime-gpu 稳定版本（Python 3.13 兼容）
ORT_GPU_VERSION = '1.20.0'

DIRECTML_PACKAGE = 'onnxruntime-directml'
ORT_GPU_PACKAGE = f'onnxruntime-gpu=={ORT_GPU_VERSION}'
ORT_CPU_PACKAGE = 'onnxruntime'


# ==================== 打包环境检测 ====================

def is_frozen():
    """是否运行在打包后的 exe 中"""
    return getattr(sys, 'frozen', False)


def _check_not_frozen():
    """打包环境中禁止 pip 安装"""
    if is_frozen():
        raise RuntimeError(
            "当前是打包版本（exe），无法自动安装加速环境。\n\n"
            "如需 GPU 加速，请下载对应版本的 exe：\n"
            "  • LST-CPU.exe      （多线程 CPU）\n"
            "  • LST-DirectML.exe （支持所有显卡，推荐）\n"
            "  • LST-CUDA.exe     （仅 NVIDIA，最快）\n\n"
            "或使用源码运行（python main.py）后点此按钮。"
        )


def _no_window_flag():
    if sys.platform == 'win32':
        return 0x08000000
    return 0


def get_site_packages():
    try:
        for p in site.getsitepackages():
            if 'site-packages' in p:
                return p
    except Exception:
        pass
    return os.path.join(sys.prefix, 'Lib', 'site-packages')


# ==================== 检查安装状态 ====================

def _pkg_installed(pkg):
    try:
        import importlib.metadata as meta
        meta.version(pkg)
        return True
    except Exception:
        return False


def check_onnxruntime_variant():
    """
    返回当前安装的 onnxruntime 变体：
    - 'cpu' / 'gpu' / 'directml' / None（未安装）
    """
    try:
        import importlib.metadata as meta
        installed = [d.metadata['Name'].lower()
                     for d in meta.distributions()
                     if d.metadata['Name']]
    except Exception:
        return None

    for name in installed:
        n = name.lower()
        if n == 'onnxruntime-directml':
            return 'directml'
        if n == 'onnxruntime-gpu':
            return 'gpu'
        if n == 'onnxruntime':
            return 'cpu'
    return None


def check_cuda_ready():
    """CUDA 环境就绪？"""
    if check_onnxruntime_variant() != 'gpu':
        return False, "未安装 onnxruntime-gpu"

    load_cuda_dlls()

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            return False, f"CUDA EP 不可用（当前：{providers}）"
    except Exception as e:
        return False, f"onnxruntime 加载失败：{e}"

    return True, "CUDA 就绪"


def check_directml_ready():
    """DirectML 环境就绪？"""
    if check_onnxruntime_variant() != 'directml':
        return False, "未安装 onnxruntime-directml"

    try:
        import onnxruntime as ort
        if 'DmlExecutionProvider' not in ort.get_available_providers():
            return False, "DmlExecutionProvider 不可用"
    except Exception as e:
        return False, f"onnxruntime 加载失败：{e}"

    return True, "DirectML 就绪"


def check_nvidia_packages():
    """返回缺失的 NVIDIA 包列表"""
    result = []
    for pkg in CUDA_PACKAGES:
        # pkg 可能带版本号，取包名
        name = pkg.split('==')[0]
        if not _pkg_installed(name):
            result.append(pkg)
    return result


def load_cuda_dlls():
    """
    把 NVIDIA DLL 目录加到 PATH + add_dll_directory + 全量预加载
    - 遍历所有 DLL（不只关键几个），确保 cuDNN 完整加载
    - 按文件名排序，避免依赖顺序错误
    """
    sp = get_site_packages()
    nvidia_root = os.path.join(sp, 'nvidia')
    if not os.path.isdir(nvidia_root):
        log.warning("[CUDA] 未找到 nvidia 包目录")
        return False

    dirs = []
    try:
        for pkg in os.listdir(nvidia_root):
            pkg_path = os.path.join(nvidia_root, pkg)
            if not os.path.isdir(pkg_path):
                continue
            for sub in ('bin', 'lib', 'lib/x64', 'bin/x64'):
                p = os.path.join(pkg_path, sub)
                if os.path.isdir(p):
                    dirs.append(p)
    except Exception as e:
        log.warning(f"[CUDA] 扫描目录失败：{e}")
        return False

    if not dirs:
        log.warning("[CUDA] 未找到任何 DLL 目录")
        return False

    # 1. 加到 PATH
    cur = os.environ.get('PATH', '')
    added = [d for d in dirs if d not in cur]
    if added:
        os.environ['PATH'] = ';'.join(added) + ';' + cur
        log.info(f"[CUDA] PATH 加入 {len(added)} 个目录")

    # 2. add_dll_directory
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass

    # 3. 全量预加载所有 DLL
    try:
        import ctypes
    except Exception:
        return True

    # 收集所有 DLL
    all_dlls = []
    for d in dirs:
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith('.dll'):
                    all_dlls.append(os.path.join(d, fn))
        except Exception:
            continue

    # 优先级排序：cudart/cublas/cudnn 主 DLL 先
    priority_keywords = [
        'cudart',
        'cublasLt', 'cublas',
        'cudnn_graph', 'cudnn_heuristic',
        'cudnn_adv', 'cudnn_cnn', 'cudnn_ops',
        'cudnn_engines',
        'cudnn64',
        'cufft', 'curand',
    ]

    def priority(path):
        name = os.path.basename(path).lower()
        for i, kw in enumerate(priority_keywords):
            if kw.lower() in name:
                return (i, name)
        return (len(priority_keywords), name)

    all_dlls.sort(key=priority)

    loaded_names = []
    for path in all_dlls:
        name = os.path.basename(path)
        try:
            ctypes.WinDLL(path)
            loaded_names.append(name)
        except Exception as e:
            # 某些 DLL 可能被依赖加载失败，忽略
            pass

    log.info(f"[CUDA] 预加载 {len(loaded_names)} 个 DLL")

    # 打印关键 DLL 是否加载成功
    key_dlls = [
        'cudart64_12.dll',
        'cublas64_12.dll',
        'cublasLt64_12.dll',
        'cudnn64_9.dll',
        'cudnn_graph64_9.dll',
        'cudnn_engines_precompiled.dll',
        'cudnn_engines_runtime_compiled.dll',
        'cudnn_ops64_9.dll',
        'cudnn_cnn64_9.dll',
        'cudnn_adv64_9.dll',
        'cudnn_heuristic64_9.dll',
        'cufft64_11.dll',
        'curand64_10.dll',
    ]
    for kd in key_dlls:
        if kd in loaded_names:
            log.info(f"[CUDA]   ✅ {kd}")
        else:
            log.warning(f"[CUDA]   ❌ {kd}（未加载）")

    return True


def get_current_engine_text():
    """返回给 UI 显示的当前引擎名称"""
    v = check_onnxruntime_variant()
    return {
        'cpu': 'CPU 多线程',
        'directml': 'DirectML GPU',
        'gpu': 'NVIDIA CUDA',
    }.get(v, '未安装')


# ==================== 安装流程 ====================

def _pip_install(packages, uninstall_first=None, progress_cb=None):
    """通用 pip 安装"""
    if uninstall_first:
        for pkg in uninstall_first:
            name = pkg.split('==')[0]
            if progress_cb:
                progress_cb(f"卸载 {name}...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=_no_window_flag(),
            )

    if not packages:
        return

    if progress_cb:
        names = ' '.join(p.split('==')[0] for p in packages)
        progress_cb(f"安装 {names}...")

    cmd = [sys.executable, '-m', 'pip', 'install',
           '--upgrade', '-i', PIP_INDEX] + packages
    ret = subprocess.run(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=_no_window_flag(),
    )
    if ret.returncode != 0:
        out = ret.stdout.decode('utf-8', errors='ignore')
        raise RuntimeError(f"pip 安装失败：\n{out[-500:]}")


def install_directml(progress_cb=None, done_cb=None):
    """安装 DirectML 环境（~50MB）"""
    def _worker():
        try:
            _check_not_frozen()

            if progress_cb:
                progress_cb("步骤 1/2：卸载旧版 onnxruntime...")
            _pip_install(
                packages=[DIRECTML_PACKAGE],
                uninstall_first=['onnxruntime', 'onnxruntime-gpu',
                                 'onnxruntime-directml'],
                progress_cb=progress_cb,
            )

            if progress_cb:
                progress_cb("步骤 2/2：验证 DirectML...")

            ok, msg = check_directml_ready()
            if ok:
                done_cb(True, "✅ DirectML 安装完成！请重启程序")
            else:
                done_cb(False, f"安装完成但验证失败：{msg}")
        except Exception as e:
            log.exception("[DirectML] 安装失败")
            if done_cb:
                done_cb(False, str(e))

    threading.Thread(target=_worker, daemon=True,
                     name='DirectMLInstall').start()


def install_cuda(progress_cb=None, done_cb=None):
    """安装 CUDA 环境（~1.6GB）"""
    def _worker():
        try:
            _check_not_frozen()

            if progress_cb:
                progress_cb("步骤 1/4：卸载旧版 onnxruntime...")
            _pip_install(
                packages=[],
                uninstall_first=['onnxruntime', 'onnxruntime-directml',
                                 'onnxruntime-gpu'],
                progress_cb=progress_cb,
            )

            if progress_cb:
                progress_cb("步骤 2/4：安装 NVIDIA CUDA 运行时（~1.3GB）...")
            _pip_install(packages=CUDA_PACKAGES, progress_cb=progress_cb)

            if progress_cb:
                progress_cb(f"步骤 3/4：安装 onnxruntime-gpu...")
            _pip_install(packages=[ORT_GPU_PACKAGE], progress_cb=progress_cb)

            if progress_cb:
                progress_cb("步骤 4/4：加载 DLL 并验证...")
            load_cuda_dlls()

            ok, msg = check_cuda_ready()
            if ok:
                done_cb(True, "✅ CUDA 安装完成！请重启程序")
            else:
                done_cb(False, f"安装完成但验证失败：{msg}")
        except Exception as e:
            log.exception("[CUDA] 安装失败")
            if done_cb:
                done_cb(False, str(e))

    threading.Thread(target=_worker, daemon=True,
                     name='CUDAInstall').start()


def install_cpu_onnxruntime(progress_cb=None, done_cb=None):
    """恢复 CPU 版 onnxruntime"""
    def _worker():
        try:
            _check_not_frozen()

            if progress_cb:
                progress_cb("正在恢复 CPU 版 onnxruntime...")
            _pip_install(
                packages=[ORT_CPU_PACKAGE],
                uninstall_first=['onnxruntime-gpu', 'onnxruntime-directml'],
                progress_cb=progress_cb,
            )
            done_cb(True, "✅ 已恢复 CPU 版 onnxruntime，请重启程序")
        except Exception as e:
            log.exception("[CPU] 恢复失败")
            if done_cb:
                done_cb(False, str(e))

    threading.Thread(target=_worker, daemon=True).start()


# ==================== GPU 设备枚举 ====================

_gpu_cache = None


def list_gpu_devices(force=False):
    """
    枚举系统中的 GPU，返回 [(index, name, vram_mb), ...]
    """
    global _gpu_cache
    if _gpu_cache is not None and not force:
        return _gpu_cache

    devices = []

    try:
        import json as _json
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        ]
        out = subprocess.check_output(
            cmd, text=True, timeout=8,
            creationflags=_no_window_flag() if sys.platform == 'win32' else 0,
        ).strip()
        if out:
            data = _json.loads(out)
            if isinstance(data, dict):
                data = [data]
            for i, gpu in enumerate(data):
                name = (gpu.get("Name") or "Unknown GPU").strip()
                vram = gpu.get("AdapterRAM")
                try:
                    vram_mb = int(vram) // (1024 * 1024) if vram else 0
                except Exception:
                    vram_mb = 0
                devices.append((i, name, vram_mb))
    except Exception as e:
        log.warning(f"[GPU] PowerShell 枚举失败：{e}")

    if not devices:
        try:
            cmd = [
                "wmic", "path", "win32_VideoController",
                "get", "name", "/format:list"
            ]
            out = subprocess.check_output(
                cmd, text=True, timeout=8,
                creationflags=_no_window_flag() if sys.platform == 'win32' else 0,
            )
            for line in out.splitlines():
                line = line.strip()
                if line.lower().startswith("name="):
                    name = line.split("=", 1)[1].strip()
                    if name:
                        devices.append((len(devices), name, 0))
        except Exception:
            pass

    _gpu_cache = devices
    log.info(f"[GPU] 检测到 {len(devices)} 个设备：{devices}")
    return devices


def clear_gpu_cache():
    global _gpu_cache
    _gpu_cache = None


# ==================== 显存检测 ====================

def query_gpu_vram():
    """
    查询所有 GPU 的显存情况
    返回: [(device_index, total_mb, free_mb), ...]
    """
    result = []

    # 方式 1：NVIDIA nvidia-smi
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, text=True, timeout=5,
            creationflags=_no_window_flag() if sys.platform == 'win32' else 0,
        ).strip()

        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    total_mb = int(parts[0])
                    free_mb = int(parts[1])
                    result.append((i, total_mb, free_mb))
                    log.info(f"[显存] NVIDIA GPU {i}: "
                             f"总 {total_mb}MB, 剩 {free_mb}MB")
                except Exception:
                    pass
    except FileNotFoundError:
        log.info("[显存] nvidia-smi 不存在（非 NVIDIA 显卡）")
    except Exception as e:
        log.warning(f"[显存] nvidia-smi 查询失败：{e}")

    # 方式 2：WMI 查询非 NVIDIA 显卡
    try:
        import json as _json
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        ]
        out = subprocess.check_output(
            cmd, text=True, timeout=8,
            creationflags=_no_window_flag() if sys.platform == 'win32' else 0,
        ).strip()
        if out:
            data = _json.loads(out)
            if isinstance(data, dict):
                data = [data]
            for i, gpu in enumerate(data):
                if i < len(result):
                    continue
                vram = gpu.get("AdapterRAM")
                try:
                    total_mb = int(vram) // (1024 * 1024) if vram else 0
                except Exception:
                    total_mb = 0
                if total_mb <= 0:
                    total_mb = 2048
                free_mb = int(total_mb * 0.75)
                result.append((i, total_mb, free_mb))
                log.info(f"[显存] GPU {i} ({gpu.get('Name', '?')}): "
                         f"总 {total_mb}MB, 估计可用 {free_mb}MB")
    except Exception as e:
        log.warning(f"[显存] WMI 查询失败：{e}")

    return result


def estimate_max_engines_for_vram(free_mb, per_engine=350, reserve=800):
    """
    根据剩余显存估算单 GPU 能创建几个引擎
    """
    usable = free_mb - reserve
    if usable <= 0:
        return 0
    return max(0, usable // per_engine)