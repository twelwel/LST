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


for _pkg in ('customtkinter', 'rapidocr', 'onnxruntime', 'tkinterdnd2'):
    try:
        _merge(collect_all(_pkg))
    except Exception:
        pass


hiddenimports += [
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
    'ui',
    'ui.main_window',
    'ui.panels',
    'ui.widgets',
    'ui.constants',
    'translation',
    'translation.manager',
    'translation.api_config',
    'translation.providers',
    'translation.strategies',
    'translation.base',
    'about_dialog',
    'api_setup_dialog',
]


a = Analysis(
    [os.path.join(SRC, 'main.py')],
    pathex=[SRC, ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'pip',
        'wheel',
    ],
    noarchive=False,
    optimize=0,
)


def _is_junk(path):
    low = path.lower()
    markers = [
        '\\tests\\', '/tests/',
        '\\test\\', '/test/',
        '\\examples\\', '/examples/',
        '\\docs\\', '/docs/',
        '\\.git\\', '/.git/',
        '__pycache__',
    ]
    return any(m in low for m in markers)


a.datas = [d for d in a.datas if not _is_junk(d[0])]
a.binaries = [b for b in a.binaries if not _is_junk(b[0])]


pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LST-DirectML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LST-DirectML',
)