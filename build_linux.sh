#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
"$PY" -m pip install "pyinstaller>=6.0"
"$PY" -m PyInstaller --noconfirm --clean --windowed --name scrcpy-flow main.py
echo
echo "已生成 dist/scrcpy-flow/"
echo "把 tools/ 复制到 dist/scrcpy-flow/tools/ 即可随包携带 adb 与 scrcpy。"
