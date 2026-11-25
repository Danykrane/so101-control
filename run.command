#!/usr/bin/env bash
# SO-101: прототип ручного управления. Двойной клик в Finder или ./run.command
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Нет .venv. Создайте его (см. README.md):"
    echo "  /opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python -m venv --system-site-packages .venv"
    echo "  .venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi
exec .venv/bin/python app.py "$@"
