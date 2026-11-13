# tools

Linux / Windows 交付包可以内置两套平台工具，复制整个项目文件夹即可带走；macOS 建议使用 Homebrew 安装系统工具。

```
tools/
├── linux/                          # Linux x86_64
│   ├── adb                         # Android Debug Bridge 37.0.1
│   ├── scrcpy                      # scrcpy 5.0.1
│   ├── scrcpy-server
│   └── ...
├── windows/
│   └── scrcpy-win64-v5.0.1/        # Windows x64
│       ├── adb.exe
│       ├── scrcpy.exe
│       ├── scrcpy-server
│       ├── *.dll
│       └── ...
└── VERSIONS.txt
```

程序启动时会自动在 `tools/` 下递归寻找当前平台对应的 `adb` / `scrcpy`，优先使用离程序最近的版本，无需在界面里选择路径。

- Linux 使用 `tools/linux/`。
- Windows 使用 `tools/windows/scrcpy-win64-v5.0.1/`。
- macOS 使用 Homebrew：`brew install android-platform-tools scrcpy`。
- 如果某个平台工具缺失，会再尝试系统 PATH、Homebrew、MacPorts 和 Android SDK 常见路径。
- 也可以用环境变量临时覆盖：

```bash
export SCFLOW_ADB=/path/to/adb
export SCFLOW_SCRCPY=/path/to/scrcpy
```

## 从 Git 仓库获取时

`tools/` 下的二进制被 `.gitignore` 排除，避免仓库体积过大。可以：

1. 直接复制已经带有 tools 的交付文件夹；
2. 或自行下载：
   - scrcpy 5.0.1 Linux / Windows：<https://github.com/Genymobile/scrcpy/releases/tag/v5.0.1>
   - Windows 包校验文件：`tools/windows/SHA256SUMS.txt`
3. 解压到 `tools/windows/` 或 `tools/linux/` 即可。

## 兼容性说明

- Windows 包是 x64 版本；ARM64 Windows 请换成 `scrcpy-winarm64`。
- Linux 包是 x86_64 版本，依赖常见的 glibc / libudev；极老或非 glibc 发行版若无法运行，可安装系统 scrcpy 后设置 `SCFLOW_SCRCPY`。
- 工具版本：scrcpy 5.0.1，adb 37.0.1。
- macOS：推荐 `brew install android-platform-tools scrcpy`；scrcpy 窗口输入可选 `brew install cliclick`。首次使用窗口截图/点击时，需要在系统设置中授予屏幕录制和辅助功能权限。
