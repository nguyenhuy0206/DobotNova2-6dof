#!/usr/bin/env bash
# Start the manual control dashboard at http://127.0.0.1:8765
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [[ -x $PY ]] || PY=python3
exec "$PY" ui/app.py "$@"
