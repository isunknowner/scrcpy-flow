#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" != "Darwin" ]; then
    echo "此脚本需要在 macOS 上运行。" >&2
    exit 1
fi
PY="${PYTHON:-python3}"
if [ -x ".venv2/bin/python" ]; then PY=".venv2/bin/python"; fi
"$PY" -m pip install "pyinstaller>=6.0"
"$PY" -m PyInstaller --noconfirm --clean --windowed --name scrcpy-flow-1.0 main.py
echo "已生成 dist/scrcpy-flow-1.0.app/"
echo "如需 adb / scrcpy，请用 Homebrew 安装，或复制 tools/ 到打包目录。"
