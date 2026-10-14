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

CUDA_PACKAGES = [
    'nvidia-cuda-runtime-cu12',
    'nvidia-cudnn-cu12',
    'nvidia-cublas-cu12',
    'nvidia-cufft-cu12',
    'nvidia-curand-cu12',
]

DIRECTML_PACKAGE = 'onnxruntime-directml'
ORT_GPU_PACKAGE = 'onnxruntime-gpu'
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

    missing = [p for p in CUDA_PACKAGES if not _pkg_installed(p)]
    if missing:
        return False, f"缺少 CUDA 包：{', '.join(missing)}"

    load_cuda_dlls()

    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            return False, "CUDA EP 不可用"
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
    return [p for p in CUDA_PACKAGES if not _pkg_installed(p)]


def load_cuda_dlls():
    """把 NVIDIA DLL 目录加到 PATH"""
    sp = get_site_packages()
    nvidia_root = os.path.join(sp, 'nvidia')
    if not os.path.isdir(nvidia_root):
        return False

    dirs = []
    for pkg in os.listdir(nvidia_root):
        pkg_path = os.path.join(nvidia_root, pkg)
        if not os.path.isdir(pkg_path):
            continue
        for sub in ('bin', 'lib', 'lib/x64'):
            p = os.path.join(pkg_path, sub)
            if os.path.isdir(p):
                dirs.append(p)

    if not dirs:
        return False

    cur = os.environ.get('PATH', '')
    added = [d for d in dirs if d not in cur]
    if added:
        os.environ['PATH'] = ';'.join(added) + ';' + cur
        log.info(f"[CUDA] PATH 加入 {len(added)} 个目录")

    for d in dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
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
            if progress_cb:
                progress_cb(f"卸载 {pkg}...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', pkg],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=_no_window_flag(),
            )

    if not packages:
        return

    if progress_cb:
        progress_cb(f"安装 {' '.join(packages)}...")

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
                progress_cb("步骤 3/4：安装 onnxruntime-gpu...")
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
    顺序对应 DirectML 的 device_id（通常：0=核显, 1=独显）。
    结果会缓存，force=True 时重新查询。
    """
    global _gpu_cache
    if _gpu_cache is not None and not force:
        return _gpu_cache

    devices = []

    # 方式 1：PowerShell + CIM（不依赖第三方库）
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

    # 方式 2：兜底 wmic
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