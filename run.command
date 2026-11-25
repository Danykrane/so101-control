#!/usr/bin/env bash
# SO-101 manual control GUI. Double-click in Finder or run ./run.command
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "No .venv. Create it (see README.md)."
    exit 1
fi
exec .venv/bin/python app.py "$@"
