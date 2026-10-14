# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.getcwd())
SRC = os.path.join(ROOT, 'src')

datas = []
binaries = []
hiddenimports = []

for pkg in ['rapidocr', 'onnxruntime', 'customtkinter', 'keyboard']:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += [
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',
    'multiprocessing',
    'multiprocessing.spawn',
    'multiprocessing.pool',
    'translation',
    'translation.base',
    'translation.providers',
    'translation.strategies',
    'translation.manager',
    'translation.api_config',
    'ocr_worker',
]

a = Analysis(
    [os.path.join(SRC, 'main.py')],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'tensorflow', 'matplotlib',
        'scipy', 'pandas', 'IPython', 'jupyter',
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
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='LST',
)