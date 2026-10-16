# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all


ROOT = os.path.abspath(os.getcwd())
SRC = os.path.join(ROOT, 'src')


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
    'ui.panels',
    'ui.widgets',
    'ui.constants',
    'ui.latency_dialog',
    'ui.model_manager_dialog',

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
    'translation.local_models.opus_mt',
    'translation.local_models.downloader',
    'translation.local_models.cache',
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
        # 深度学习框架（不需要）
        'torch',
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
        'pytest',
        'unittest',
        'doctest',

        # 打包工具
        'setuptools',
        'pip',
        'wheel',

        # ML 训练相关（transformers 会拉）
        'accelerate',
        'datasets',
        'evaluate',
        'safetensors.torch',
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
    name='LST-DirectML',
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
    name='LST-DirectML',
)