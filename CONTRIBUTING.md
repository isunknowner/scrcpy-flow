# 贡献指南

感谢参与 scrcpy-flow。这个项目希望保持“可读、可迁移、可验证”。

## 开发环境

```bash
git clone <your-fork>
cd scrcpy-flow
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

Windows：

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

macOS 与 Linux 相同；设备工具使用：

```bash
brew install android-platform-tools scrcpy
```

## 运行

```bash
./run.sh
# 或
python -m scflow
```

## 测试

单元测试不需要手机与图形界面：

```bash
python -m unittest discover -s tests -v
# 或
pytest
```

代码风格：

```bash
ruff check scflow tests
```

GUI 冒烟测试（需要显示器，Linux 可用 xvfb）：

```bash
xvfb-run -a python -m scflow
```

真机测试步骤见仓库外的验收文档，避免在低配开发机上长时间跑截图循环。

## 提交 PR

1. 从 `main` 拉分支：`feat/xxx`、`fix/xxx`、`docs/xxx`。
2. 保持改动聚焦；新增行为要有测试或可复现步骤。
3. 运行 `ruff check` 与单元测试。
4. 在 PR 描述中说明：
   - 动机与行为变化；
   - 测试方式；
   - 是否涉及 Windows / macOS / scrcpy / ADB 差异；
   - 是否改变流程文件格式（需要同步 `docs/flow-format.md`）。
5. 不要提交 `data/`、`tools/` 中的二进制、真实账号截图或任何隐私数据。

## 适合贡献的方向

- Windows / macOS 下 scrcpy 窗口定位与输入的兼容性反馈。
- 新的图片向量方案（CLIP、局部特征、OCR 辅助）。
- 流程编辑器交互改进（缩放、框选多个区域、撤销）。
- 新的自动化配方（外卖、签到、消息回复、日常任务）。
- 文档与翻译。

## 行为准则

- 尊重他人，讨论问题而不是攻击人。
- 不提交用于绕过平台风控、验证码或侵犯他人隐私的流程。
- 示例截图请打码或使用虚构数据。
