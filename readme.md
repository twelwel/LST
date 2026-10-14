# LST · 实时屏幕翻译

> Windows 平台实时屏幕 OCR 翻译工具
> 截图 → OCR 识别 → 翻译 → 类原生文字覆盖显示

[![Version](https://img.shields.io/badge/version-1.30.0-blue)](https://github.com/twelwel/LST/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()

---

## ✨ 特性

- **实时翻译**：截取屏幕任意区域，自动识别并翻译
- **多显示模式**：覆盖 / 双语 / 小窗口，可随时切换
- **GPU 加速**：DirectML 多引擎 + 多 GPU 并行（Intel / AMD / NVIDIA / 核显全支持）
- **三种翻译策略源**：翻译 API / 大语言模型 / 本地词典，可混合使用
- **5 种翻译策略**：离线 / 单翻译 API / 多 API 轮询 / 翻译+AI 混合 / 单 AI / 多 AI 投票
- **快速响应层**：本地词典 + 短语缓存，单词/短语秒出
- **类原生文字**：背景色采样 + 精细笔画修复，视觉接近原文
- **鼠标穿透 + 截图不可见**：不干扰操作，也不会被自己识别
- **完全离线可用**：本地词典 + 短语缓存，无需联网

---

## 📋 目录

- [快速开始](#快速开始)
- [首次运行提示](#-首次运行提示)
- [功能说明](#功能说明)
- [翻译策略](#翻译策略)
- [显示模式](#显示模式)
- [GPU 加速](#gpu-加速)
- [快捷键](#快捷键)
- [配置文件](#配置文件)
- [从源码运行](#从源码运行)
- [构建打包](#构建打包)
- [常见问题](#常见问题)
- [目录结构](#目录结构)
- [更新日志](#更新日志)
- [许可](#许可)

---

## 快速开始

### 1. 下载

从 [Releases](https://github.com/twelwel/LST/releases) 下载最新版：
