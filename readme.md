# LST — 实时屏幕翻译

> 截屏 → OCR → 翻译 → 覆盖显示。专为**视频字幕**、**游戏界面**、**外文文档**设计。

[![Version](https://img.shields.io/badge/version-1.40.0-blue.svg)](https://github.com/twelwel/LST/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

---

## ✨ 特性

| 类别 | 亮点 |
|------|------|
| **OCR** | PP-OCRv6（server / tiny 可选）；多引擎并行；高精度双引擎投票 |
| **翻译** | 6 种策略：离线 / 单 API / 多 API 轮询 / API+AI 混合 / 单 AI / 多 AI 投票 |
| **多语言** | 源语言 9 种（含自适应），目标语言 8 种，默认自适应 → 中文 |
| **加速** | 多线程 CPU / DirectML（所有显卡）/ CUDA（NVIDIA 最快） |
| **显示** | 覆盖 / 双语 / 小窗口三种模式；译文颜色全模式通用 |
| **过滤** | 通用（数字/单位/图像）+ 英语特化 + 标题/Logo + 二次 OCR 验证 |
| **防抖** | 位置匹配复用 key；全帧去重；滚动场景不堆叠 |
| **快速响应** | 本地词典 + 短语缓存；单词/词组不联网 |
| **测试** | 内置延迟测试工具；分阶段 + 全链路测量 |

---

## 📸 效果预览

| 覆盖模式 | 双语模式 | 小窗口模式 |
|---------|---------|-----------|
| ![overlay](docs/screenshots/overlay.png) | ![bilingual](docs/screenshots/bilingual.png) | ![window](docs/screenshots/window.png) |

---

## 🚀 快速开始

### 方式一：下载 exe（推荐）

1. 打开 [Releases](https://github.com/twelwel/LST/releases)
2. 下载 `LST-DirectML-v1.40.0.zip`（支持所有显卡，推荐）或 `LST-CPU-v1.40.0.zip`
3. 解压后双击 `LST-DirectML.exe`
4. 首次启动会**自动下载 PP-OCRv6 模型**（约 30s）

### 方式二：源码运行

```bash
git clone https://github.com/twelwel/LST.git
cd LST
python -m venv .venv
.venv\Scripts\activate

# 基础依赖
pip install -r requirements.txt

# 选择加速方式（三选一）
pip install onnxruntime-directml    # DirectML（推荐）
# pip install onnxruntime-gpu        # NVIDIA CUDA
# pip install onnxruntime            # 纯 CPU

python src/main.py
```

---

## 🎯 使用流程

### 1. 选择截图区域

启动后，屏幕上会出现**绿色边框**。拖动边框四角可调整范围。

也可以点击 **修改区域** 打开配置对话框，手动输入坐标或选择预设。

### 2. 选择语言

顶栏：

```
源语言： [自适应 ▼]    目标语言： [中文 ▼]
```

| 源语言 | 目标语言 |
|--------|---------|
| 自适应 / 英语 / 中文 / 日语 / 韩语 / 法语 / 德语 / 西班牙语 / 俄语 | 中文 / 英语 / 日语 / 韩语 / 法语 / 德语 / 西班牙语 / 俄语 |

### 3. 选择翻译策略

| 策略 | 适用场景 |
|------|---------|
| **离线** | 无网络；只用本地词典 |
| **单翻译 API** | 最常用；主 API + 兜底 |
| **多翻译 API 轮询** | 多 API 均衡负载 |
| **翻译 API + AI 混合** | 翻译 API 失败时自动切 AI |
| **单 AI 翻译** | GPT/DeepSeek/Qwen 等 |
| **多 AI 高精度投票** | 多 AI 并发 + 相似度投票 |

### 4. 点击"开始"

```
▶ 开始   清空译文   修改区域   隐藏边框
```

---

## 🔥 核心功能

### 多语言支持

- **源语言**：`auto` 时让 OCR 自动识别；指定语言则只识别该语言
- **目标语言**：决定翻译输出语言；已是目标语言的段落自动跳过
- **冲突处理**：
  - 目标 = 英语时，**英语特化自动禁用**（避免自相矛盾）
  - 目标 = 中文时，正常启用所有过滤

### 内容过滤（可开关）

| 开关 | 作用 |
|------|------|
| **通用过滤** | 只过滤数字 / 单位 / 图像（不过滤短文本） |
| **英语特化** | 只保留字母 ≥ 50%、中文 ≤ 20% 的段落 |
| **标题/Logo 过滤** | 全大写 或 超大字号 → 跳过 |
| **二次 OCR 验证** | emit 前局部复核，防一闪而过的原文残留 |
| **快速响应层** | 本地词典 + 短语缓存，不联网查单词 |

### 显示模式

| 模式 | 特点 |
|------|------|
| **覆盖** | 译文直接盖在原文位置（可图像修复背景） |
| **双语** | 原文下方显示小字译文（视频字幕推荐） |
| **小窗口** | 独立浮窗累积译文，可拖动、可复制 |

**双语距离**：`-30 ~ +100` 可调，控制译文与原文的间距。

### 输出防抖

| 机制 | 作用 |
|------|------|
| 位置匹配复用 key | OCR 位置微抖动不换窗口 |
| EmitDebouncer | 多帧投票 + 显示节流（0.12s） |
| Generation 保护 | 切换区域/模式时丢弃旧消息 |
| 全帧去重 | 同帧内相同文本只留一份 |

---

## ⚡ 性能优化

### 帧差检测

- 缩略图差分 → 连通域过滤（忽略小噪声）
- 变化区动态 padding（`max(60, 变化高度 × 0.3)`）
- 静止画面完全跳过 OCR

### 分块 OCR

- 均匀分块（每块高度一致，不切字）
- 重叠 40px（覆盖一行小字）
- IoU > 0.5 去重

### 快速响应

```
单词/词组 → 本地词典 → 短语缓存 → 命中则跳过网络
```

超过 8 词或 50 字符的文本直接走策略。

---

## 🧪 延迟测试

顶栏 **延迟测试** 按钮，或命令行：

```bash
python src/test_latency.py --api --e2e
```

**输出示例**：

```
阶段                              次数        平均        P50        P95
1.帧差检测                         20      2.15ms     2.10ms     2.85ms
2.裁剪+缩放                        20      0.85ms     0.80ms     1.20ms
3.OCR 识别                         20    127.45ms   125.30ms   145.20ms
4.过滤                             20      0.42ms     0.40ms     0.68ms
5.段落合并                         20      0.95ms     0.92ms     1.15ms
API 翻译                            5    485.20ms   462.10ms   610.30ms
★全链路（截图→译文）               5    628.75ms   615.30ms   720.50ms
```

---

## ⌨️ 默认热键

| 热键 | 功能 |
|------|------|
| `Ctrl+Alt+T` | 开始 / 暂停 |
| `F2` | 修改截图区域 |
| `F3` | 隐藏 / 显示区域框 |
| `Ctrl+Alt+L` | 切换译文模式（默认 / 跟随 / 钉住） |
| `Ctrl+Esc` | 退出程序 |

所有热键都可在 UI 里自定义。

---

## 🔧 配置文件

配置保存在：

```
~/.screen_translator/
├── user_config.json           # UI 配置
├── api_providers.json         # API 提供商配置
├── translation_strategy.json  # 翻译策略
├── phrase_cache.json          # 用户短语
└── logs/                      # 日志
```

**首次启动自动生成**，用户不需要手动创建。

---

## 🖥️ 加速方式

| 模式 | 安装 | 引擎数 | 适用 |
|------|------|--------|------|
| **多线程 CPU** | 无 | 1-8 可调 | 兼容性最好 |
| **DirectML** | `pip install onnxruntime-directml` | 固定 1 | 所有显卡 |
| **NVIDIA CUDA** | `pip install onnxruntime-gpu` + 5 个 CUDA 依赖 | 1-4 可调 | NVIDIA 最快 |

CUDA 依赖：

```bash
pip install nvidia-cuda-runtime-cu12==12.6.77
pip install nvidia-cudnn-cu12==9.4.0.58
pip install nvidia-cublas-cu12==12.6.4.1
pip install nvidia-cufft-cu12==11.3.0.4
pip install nvidia-curand-cu12==10.3.7.77
```

---

## 📦 打包

```bash
# DirectML 版（推荐）
pyinstaller --noconfirm --clean LST-DirectML.spec

# 输出：dist/LST-DirectML/
```

---

## 🛠️ 技术架构

```
┌─────────────────────────────────────────────┐
│  UI 层 (customtkinter)                       │
│  ui/ 包：主窗口 + 面板 Mixin + 折叠控件       │
├─────────────────────────────────────────────┤
│  调度层                                       │
│  main_window._tick → 截图节流 → pipeline     │
├─────────────────────────────────────────────┤
│  处理管道 (pipeline.py)                      │
│  帧差 → OCR → 过滤 → 合并 → 翻译 → 输出      │
├─────────────────────────────────────────────┤
│  防抖层 (emit_debouncer.py)                  │
│  位置即身份 + 多帧投票 + 显示节流             │
├─────────────────────────────────────────────┤
│  算法层                                       │
│  layout.py（段落合并 / 字号估算 / 过滤）      │
│  inpaint.py（图像修复）                       │
├─────────────────────────────────────────────┤
│  基础层                                       │
│  OCR 引擎池 / 翻译策略 / 缓存 / 词典          │
└─────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
LST/
├── src/
│   ├── main.py                  # 入口
│   ├── pipeline.py              # 处理管道
│   ├── ocr_engine.py            # OCR 引擎池
│   ├── layout.py                # 排版 / 过滤 / 字号
│   ├── inpaint.py               # 图像修复
│   ├── emit_debouncer.py        # 输出防抖
│   ├── overlay.py               # 覆盖显示
│   ├── region_manager.py        # 区域管理
│   ├── scroll_detector.py       # 滚动检测
│   ├── hotkeys.py               # 热键管理
│   ├── logger.py                # 日志
│   ├── config.py                # 常量
│   ├── config_manager.py        # 配置持久化
│   ├── cache.py                 # 段落缓存
│   ├── local_dict.py            # 本地词典
│   ├── phrase_cache.py          # 短语缓存
│   ├── translator.py            # 翻译门面
│   ├── test_latency.py          # 延迟测试
│   ├── ui/                      # UI 包
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── panels.py
│   │   ├── widgets.py
│   │   ├── constants.py
│   │   └── latency_dialog.py
│   ├── translation/             # 翻译后端
│   │   ├── manager.py
│   │   ├── providers.py
│   │   ├── strategies.py
│   │   └── api_config.py
│   └── ...
├── LST-DirectML.spec
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🐛 常见问题

### Q：OCR 不识别 / 识别错

1. **换 server 模型**：关闭「轻量型 OCR」
2. **开启高精度 OCR**：双引擎投票
3. **提高扫描间隔**：设置面板 → 刷新间隔调到 0.3~0.5s
4. **缩小区域**：只框住字幕带

### Q：翻译慢

1. **用 tiny 模型**
2. **切 GPU**：DirectML 或 CUDA
3. **开快速响应层**：本地词典命中不联网
4. **用延迟测试**：找出瓶颈

### Q：译文有残留 / 堆叠

1. **开启二次 OCR 验证**：防一闪而过
2. **调整防抖参数**：`emit_debouncer.py` 里 `MISS_TOLERANCE`（默认 2）
3. **切换目标语言** 会立即清空旧译文

### Q：视频字幕识别不准

1. **开启英语特化**：只处理字母主导的段落
2. **开启标题/Logo 过滤**：过滤水印
3. **双语模式 + 距离 -10**：字幕贴在原文下方

### Q：打包后启动闪退

1. 临时改 spec 里 `console=True` 打包，看报错
2. 检查 `_internal/customtkinter/assets/` 是否存在
3. 检查 `~/.rapidocr/models/` 是否有模型文件

---

## 📋 系统要求

| 项 | 要求 |
|----|------|
| 系统 | Windows 10/11 |
| Python | 3.10+ |
| 内存 | 4GB+（CUDA 版 8GB+） |
| 磁盘 | 1GB+（含模型） |
| GPU | 可选（DirectML 需 Win10+；CUDA 需 NVIDIA 显卡） |

---

## 🤝 贡献

欢迎 Issue 和 PR！

- 提交 bug：请附日志（`~/.screen_translator/logs/`）
- 提交功能：先开 Issue 讨论

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR 引擎
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) — UI
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv6 模型

---

## 📮 联系

- 作者：twelwel
- 邮箱：2033697674@qq.com
- 仓库：https://github.com/twelwel/LST