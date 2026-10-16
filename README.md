# 隐私声明

## 概述

LST（实时屏幕翻译）尊重用户隐私。本声明说明软件如何收集、使用和保护你的信息。

**核心承诺：LST 不收集、不上传、不存储任何用户数据到我们的服务器。我们没有服务器。**

## 本地数据处理

以下所有数据仅在本机处理，不会离开你的设备：

- **屏幕截图**：仅在本机内存中处理，不写入磁盘，不上传
- **OCR 识别**：完全本地运行（基于 RapidOCR + ONNX Runtime）
- **翻译缓存**：保存在 `~/.screen_translator/`，仅本机可读
- **日志**：保存在 `~/.screen_translator/logs/`，仅本机可读

## 第三方服务

翻译功能需要调用第三方翻译 API（如百度翻译、DeepSeek、OpenAI 等）。使用过程中：

- **发送内容**：仅识别出的文本（不含截图、不含屏幕位置信息、不含用户身份）
- **接收方**：你在「API 设置」中配置的翻译服务商
- **API 密钥**：保存在本机，不会发送给我们

请自行阅读你所使用的翻译服务商的隐私政策。

## 密钥存储

- API 密钥以明文形式存储在 `~/.screen_translator/api_providers.json`
- 请不要在公用电脑上使用本软件
- 请定期检查你的 API 服务商账户用量，防止泄露

## 数据删除

如需删除所有本地数据，只需删除目录：~/.screen_translator/

即 `C:\Users\<你的用户名>\.screen_translator\`。

## 儿童隐私

本软件不面向 14 岁以下儿童，也不会有意收集儿童信息。

## 声明变更

本隐私声明如有变更，将在 GitHub 仓库更新，请留意。

## 联系方式

如有隐私相关问题，请通过 GitHub Issue 联系：

https://github.com/twelwel/LST/issues

最后更新：2026-10-16


## 法律与合规

- 本项目采用 MIT 协议，详见 [LICENSE](LICENSE)
- 隐私声明：[PRIVACY.md](PRIVACY.md)
- 第三方许可：[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)

### 免责声明

- 本软件仅供个人学习研究使用
- 用户需自行遵守当地法律法规及第三方服务的服务条款
- 因使用本软件产生的任何后果，作者不承担责任
- 请勿将本软件用于任何商业用途或违反服务条款的场景

### 致谢

本项目基于以下开源项目构建，特此致谢：

- [RapidOCR](https://github.com/RapidAI/RapidOCR) - OCR 引擎
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - PP-OCRv6 模型
- [onnxruntime](https://github.com/microsoft/onnxruntime) - 推理运行时
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) - UI 框架
- [OpenCV](https://github.com/opencv/opencv) - 图像处理
- [Pillow](https://github.com/python-pillow/Pillow) - 图像处理
- [mss](https://github.com/BoboTiG/python-mss) - 屏幕截图
- [keyboard](https://github.com/boppreh/keyboard) - 全局热键