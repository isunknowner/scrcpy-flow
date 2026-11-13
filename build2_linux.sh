#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ -x ".venv2/bin/python" ]; then PY=".venv2/bin/python"; fi
"$PY" -m pip install "pyinstaller>=6.0"
"$PY" -m PyInstaller --noconfirm --clean --windowed --name scrcpy-flow-1.0 main.py
echo "已生成 dist/scrcpy-flow-1.0/"
echo "把 tools/ 复制到 dist/scrcpy-flow-1.0/tools/ 即可便携运行。"
