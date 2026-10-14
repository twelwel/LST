# LST - 实时屏幕翻译

一款轻量的实时屏幕翻译工具，截图 → OCR → 翻译 → 覆盖显示

## ✨ 功能特性

- 🔍 **实时 OCR** — 基于 RapidOCR，本地识别，无需联网
- 🌐 **三种翻译模式** — 本地 / 混合 / 联网
- 📖 **本地词典** — 支持文件夹导入，自动识别格式和编码
- 💾 **短语缓存** — 常见短语立即显示，节省 API
- 🎨 **覆盖显示** — 译文直接覆盖原文位置（类原生文字）
- 🚀 **多种加速** — CPU 多线程 / DirectML GPU / NVIDIA CUDA
- 📌 **三种译文模式** — 默认 / 跟随 / 钉住
- ⚙️ **完整设置面板** — 所有参数可调

## 🚀 快速开始

### 方式一：下载 exe（推荐普通用户）

1. 前往 [Releases](../../releases) 下载 `LST-DirectML.zip`
2. 解压到英文路径
3. **右键 `LST.exe` → 以管理员身份运行**

### 方式二：从源码运行

```bash
git clone https://github.com/twelwel/LST.git
cd LST/src
python -m venv .venv
.venv\Scripts\activate
pip install -r ../requirements.txt
python main.py