#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=""
if [ -x ".venv2/bin/python" ] && .venv2/bin/python -c "import PySide6" >/dev/null 2>&1; then
    PY=".venv2/bin/python"
else
    PY="${PYTHON:-python3}"
fi
if ! "$PY" -c "import PySide6" >/dev/null 2>&1; then
    echo "缺少 PySide6。请先运行 ./install2.sh" >&2
    exit 1
fi
exec "$PY" -m scflow "$@"
