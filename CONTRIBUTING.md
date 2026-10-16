# 贡献指南

欢迎提交 Issue 和 Pull Request。

## Bug 报告

请通过 GitHub Issue 提交，并提供以下信息：

1. LST 版本号（程序顶栏可见）
2. 操作系统版本（Win10 / Win11）
3. 加速方式（多线程 CPU / DirectML / NVIDIA CUDA）
4. 完整日志（程序里点「打开日志」，把最新的 .log 文件内容贴出来）
5. 复现步骤
6. 相关截图（如有）

**请勿只写"不能用了"，无法排查。**

## 功能建议

请先开 Issue 讨论，确认方向后再实现，避免白做。

## Pull Request

1. Fork 本仓库
2. 创建你的分支（`git checkout -b feature/xxx`）
3. 提交改动（`git commit -m 'feat: xxx'`）
4. 推送（`git push origin feature/xxx`）
5. 提交 PR

### 代码风格

- 遵循 PEP 8
- 关键函数加中文注释
- 不引入新的重型依赖
- 改动尽量聚焦

## 开发环境

```bash
git clone https://github.com/twelwel/LST.git
cd LST
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py