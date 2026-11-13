#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -m venv --clear .venv 2>/dev/null; then
    "$PYTHON" -m venv --clear .venv
fi
if [ ! -x ".venv/bin/pip" ]; then
    echo "当前 Python 缺少 venv/pip，无法创建可安装依赖的虚拟环境。" >&2
    echo "请安装：" >&2
    echo "  sudo apt install python3-venv python3-pip" >&2
    echo "如果系统 Python 已有 numpy/Pillow/tkinter，也可以直接运行 ./run.sh。" >&2
    exit 1
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "安装完成。运行 ./run.sh 启动。"
echo "如果 scrcpy 窗口输入提示缺少 xdotool，请安装：sudo apt install xdotool"
