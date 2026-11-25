#!/usr/bin/env bash
# SO-101: robotd держит порт робота и камеры, команды по HTTP 127.0.0.1:8765 (robotctl.py, MCP, окно).
# Двойной клик в Finder или ./robotd.command; из Terminal, чтобы у камеры был доступ. Ctrl+C = выход, момент не трогается.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Нет .venv, см. README.md"
    exit 1
fi
exec .venv/bin/python robotd.py "$@"
