# -*- mode: python ; coding: utf-8 -*-

import os
import re
from PyInstaller.utils.hooks import collect_all


ROOT = os.path.abspath(os.getcwd())
SRC = os.path.join(ROOT, 'src')


def _read_app_version():
    """从 src/version.py 读取版本，保证打包版本与源码一致。"""
    version_file = os.path.join(SRC, 'version.py')
    with open(version_file, encoding='utf-8') as f:
        match = re.search(
            r"""^__version__\s*=\s*["']([^"']+)["']""",
            f.read(),
            re.M,
        )
    if not match:
        raise RuntimeError(f'无法从 {version_file} 读取 __version__')
    return match.group(1)


APP_VERSION = _read_app_version()
APP_NAME = 'LST-DirectML'
OUTPUT_NAME = f'{APP_NAME}-v{APP_VERSION}'
VERSION_INFO = os.path.join(ROOT, 'version_info.txt')
if not os.path.isfile(VERSION_INFO):
    raise FileNotFoundError(f'缺少 Windows 版本信息文件：{VERSION_INFO}')


datas = []
binaries = []
hiddenimports = []


def _merge(items):
    d, b, h = items
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


# ==================== 必须收集的包 ====================

for _pkg in ('customtkinter', 'rapidocr', 'onnxruntime', 'tkinterdnd2'):
    try:
        _merge(collect_all(_pkg))
    except Exception:
        pass


# ==================== 本地模型包（可选） ====================

# ctranslate2（CPU 推理引擎，体积小）
for _pkg in ('ctranslate2',):
    try:
        _merge(collect_all(_pkg))
    except Exception:
        pass

# torch：optimum 的 ORTModelForSeq2SeqLM / ORTModelForCausalLM
# 内部直接 import torch，不能放进 excludes，否则打包版会出现
# “Could not import module 'ORTModelForSeq2SeqLM'”。
try:
    _merge(collect_all('torch'))
except Exception:
    pass

# optimum：命名空间包，PyInstaller 静态分析无法自动发现其延迟导入，
# 必须显式收集 onnxruntime 子包，否则打包版缺少 optimum 目录。
try:
    _merge(collect_all('optimum.onnxruntime'))
except Exception:
    pass
hiddenimports += [
    'optimum',
    'optimum.onnxruntime',
    'optimum.onnxruntime.modeling_seq2seq',
    'optimum.onnxruntime.modeling_decoder',
    'optimum.exporters',
    'optimum.exporters.onnx',
    'optimum.exporters.tasks',
]

# huggingface_hub（下载器）
try:
    _merge(collect_all('huggingface_hub'))
except Exception:
    pass

# sentencepiece（分词器，有二进制）
try:
    _merge(collect_all('sentencepiece'))
except Exception:
    pass

# tokenizers（transformers 的分词依赖，Rust 编译）
try:
    _merge(collect_all('tokenizers'))
except Exception:
    pass

# safetensors：transformers 在模块级硬 import safetensors.torch
# （modeling_utils / pytorch_utils / model_debugging_utils / trainer），
# 缺了会让打包版本地模型加载时报 “No module named 'safetensors.torch'”。
# 注意：safetensors 带 Rust 扩展 _safetensors_rust，必须连二进制一起收。
try:
    _merge(collect_all('safetensors'))
except Exception:
    pass

# transformers：只收核心，不拉 torch
try:
    _merge(collect_all(
        'transformers',
        include_py_files=True,
    ))
except Exception:
    pass


# ==================== hiddenimports ====================

hiddenimports += [
    # 基础
    'keyboard',
    'mss',
    'cv2',
    'numpy',
    'PIL',
    'PIL._tkinter_finder',
    'requests',
    'urllib3',
    'uuid',
    'queue',
    'threading',
    'concurrent.futures',

    # torch / numpy / transformers 在运行期会 import unittest / doctest。
    # PyInstaller 默认可能把它们当测试模块裁掉，导致打包版本地模型加载时报
    # “No module named 'unittest'”。这里必须显式保留。
    'unittest',
    'unittest.case',
    'unittest.mock',
    'unittest.util',
    'unittest.loader',
    'unittest.suite',
    'unittest.result',
    'unittest.runner',
    'unittest.signals',
    'doctest',

    # safetensors：transformers 模块级硬依赖，绝不能进 excludes（见上方说明）
    'safetensors',
    'safetensors.torch',

    # 项目模块
    'emit_debouncer',
    'logger',
    'config',
    'config_manager',
    'cache',
    'layout',
    'inpaint',
    'pipeline',
    'ocr_engine',
    'dml_gate',
    'dml_device',
    'ocr_worker',
    'overlay',
    'region_manager',
    'scroll_detector',
    'translator',
    'hotkeys',
    'cuda_setup',
    'phrase_cache',
    'local_dict',
    'version',

    # UI 子包
    'ui',
    'ui.main_window',
    'ui.download_task',
    'ui.panels',
    'ui.widgets',
    'ui.constants',
    'ui.latency_dialog',
    'ui.model_manager_dialog',
    'ui.model_picker_dialog',

    # 翻译后端
    'translation',
    'translation.manager',
    'translation.api_config',
    'translation.providers',
    'translation.strategies',
    'translation.base',
    'translation.throttle',

    # 本地模型子包
    'translation.local_models',
    'translation.local_models.base',
    'translation.local_models.factory',
    'translation.local_models.hf_onnx',
    'translation.local_models.gemma3_onnx',
    'translation.local_models.downloader',
    'translation.local_models.registry',

    # 对话框
    'about_dialog',
    'api_setup_dialog',

    # transformers 动态导入
    'transformers.models.marian',
    'transformers.models.marian.tokenization_marian',
    'transformers.tokenization_utils',
    'transformers.tokenization_utils_base',
    'transformers.tokenization_utils_fast',
    'transformers.utils',
    'transformers.utils.hub',
]


# ==================== Analysis ====================

a = Analysis(
    [os.path.join(SRC, 'main.py')],
    pathex=[SRC, ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # ⚠️ 关键排除：防止 transformers 拖入大包
    excludes=[
        # 深度学习框架
        # 注意：torch 不能排除！optimum 的 ORTModel* 内部依赖 torch，
        # 排除后打包版会加载不了本地模型。
        'torchvision',
        'torchaudio',
        'tensorflow',
        'tensorboard',
        'keras',
        'jax',
        'flax',

        # 数据处理
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
        'skimage',
        'seaborn',

        # Notebook
        'IPython',
        'jupyter',
        'jupyter_core',
        'jupyter_client',
        'notebook',
        'nbformat',
        'nbconvert',
        'ipykernel',
        'ipywidgets',

        # 测试
        # ⚠️ 不能排除 unittest / doctest：torch 和 transformers 在
        # 正常模型加载路径中会 import unittest，排除后打包版会报
        # “No module named 'unittest'”（本地模型无法调用）。
        'pytest',

        # 打包工具
        'setuptools',
        'pip',
        'wheel',

        # ML 训练相关（transformers 会拉）
        'accelerate',
        'datasets',
        'evaluate',
        # ⚠️ 不要加 'safetensors.torch'：transformers 模块级硬 import 它，
        #    排除后打包版本地模型直接报 No module named 'safetensors.torch'（v1.40.7 实锤）。
        'tensorboardX',
        'wandb',
        'mlflow',

        # 不需要的 transformers 后端
        'tensorflow_text',
        'tf2onnx',
        'onnxruntime_training',
    ],

    noarchive=False,
    optimize=0,
)


# ==================== 过滤无用文件 ====================

def _is_junk(path):
    low = path.lower()
    markers = [
        '\\tests\\', '/tests/',
        '\\test\\', '/test/',
        '\\examples\\', '/examples/',
        '\\docs\\', '/docs/',
        '\\.git\\', '/.git/',
        '__pycache__',
        '\\benchmarks\\', '/benchmarks/',
        '\\scripts\\', '/scripts/',
    ]
    return any(m in low for m in markers)


a.datas = [d for d in a.datas if not _is_junk(d[0])]
a.binaries = [b for b in a.binaries if not _is_junk(b[0])]


# ==================== PYZ ====================

pyz = PYZ(a.pure)


# ==================== EXE ====================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    version=VERSION_INFO,           # Windows 文件属性版本（与 src/version.py 同步）
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # 关闭 UPX，避免杀软误报
    console=False,                  # 无控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets\\icon.ico',      # 有图标时取消注释
)


# ==================== COLLECT ====================

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=OUTPUT_NAME,
)