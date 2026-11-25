#!/usr/bin/env bash
# SO-101: robotd owns the robot port and cameras, commands over HTTP 127.0.0.1:8765 (robotctl.py, MCP, GUI).
# Double-click in Finder or run ./robotd.command from Terminal so it gets camera access. Ctrl+C = exit, torque untouched.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "No .venv, see README.md"
    exit 1
fi
exec .venv/bin/python robotd.py "$@"
