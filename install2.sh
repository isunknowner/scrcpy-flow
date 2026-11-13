#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv --clear .venv2 2>/dev/null || true
if [ -x ".venv2/bin/pip" ]; then
    .venv2/bin/python -m pip install --upgrade pip
    .venv2/bin/python -m pip install -r requirements-ui2.txt
else
    echo "虚拟环境缺少 pip，尝试安装到当前用户环境（--user）。"
    "$PYTHON" -m pip install --user -r requirements-ui2.txt || \
        "$PYTHON" -m pip install --user --break-system-packages -r requirements-ui2.txt
fi
echo "安装完成。运行 ./run2.sh 启动 1.0 工作区。"
