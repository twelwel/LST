# 第三方开源许可

LST 使用了以下开源软件，特此致谢。所有依赖的许可证均与本项目的 MIT 协议兼容。

## 核心依赖

| 库 | 版本 | 许可 | 用途 |
|----|------|------|------|
| PyInstaller | 6.x | GPL-2.0 + Exception | 打包工具 |
| customtkinter | 5.2+ | MIT | UI 框架 |
| rapidocr | 3.9+ | Apache-2.0 | OCR 引擎 |
| onnxruntime | 1.20+ | MIT | 推理运行时 |
| PaddleOCR (PP-OCRv6) | - | Apache-2.0 | OCR 模型 |
| OpenCV | 4.8+ | Apache-2.0 | 图像处理 |
| Pillow | 10.0+ | HPND | 图像处理 |
| numpy | 1.24+ | BSD-3-Clause | 数值计算 |
| mss | 9.0+ | MIT | 屏幕截图 |
| keyboard | 0.13+ | MIT | 全局热键 |
| requests | 2.31+ | Apache-2.0 | HTTP 请求 |

## 可选依赖

| 库 | 许可 | 用途 |
|----|------|------|
| onnxruntime-directml | MIT | DirectML 加速 |
| onnxruntime-gpu | MIT | CUDA 加速 |
| nvidia-cuda-runtime-cu12 | NVIDIA 许可 | CUDA 运行时 |
| nvidia-cudnn-cu12 | NVIDIA 许可 | cuDNN |

## 完整许可文本

各库的完整许可文本可在其源码仓库或 PyPI 页面查看：

- PyInstaller: https://github.com/pyinstaller/pyinstaller
- customtkinter: https://github.com/TomSchimansky/CustomTkinter
- rapidocr: https://github.com/RapidAI/RapidOCR
- onnxruntime: https://github.com/microsoft/onnxruntime
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- OpenCV: https://github.com/opencv/opencv
- Pillow: https://github.com/python-pillow/Pillow
- numpy: https://github.com/numpy/numpy
- mss: https://github.com/BoboTiG/python-mss
- keyboard: https://github.com/boppreh/keyboard
- requests: https://github.com/psf/requests

## 免责说明

本项目仅作为集成工具，不对第三方库的功能、安全性或合法性做任何担保。
使用者需自行遵守各第三方库的许可条款。