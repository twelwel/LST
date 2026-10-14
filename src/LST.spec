# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

for pkg in ['rapidocr', 'onnxruntime', 'cv2', 'customtkinter']:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    'PIL._tkinter_finder',
    'keyboard',
    'numpy',
    'mss',
    'requests',
    'hashlib',
    'collections',
    'json',
    'os',
    'api_config',
    'api_setup_dialog',
    'logger',
    'version',
    'about_dialog',
    'config_manager',
    'scroll_detector',
    'ui',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'scipy',
        'IPython',
        'notebook',
        'jupyter',
        'pytest',
        'setuptools',
        'pydoc',
        'unittest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LST',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',    # 👈 嵌入版本信息
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LST',
)
hiddenimports += [
    # ... 原有
    'cuda_setup',
]