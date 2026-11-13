#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=""
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import numpy, PIL, tkinter" >/dev/null 2>&1; then
    PY=".venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
if ! "$PY" -c "import numpy, PIL, tkinter" >/dev/null 2>&1; then
    echo "缺少运行依赖：numpy / Pillow / tkinter" >&2
    echo "请先运行 ./install.sh，或执行：" >&2
    echo "  python3 -m pip install -r requirements.txt" >&2
    exit 1
fi
exec "$PY" -m scflow "$@"
