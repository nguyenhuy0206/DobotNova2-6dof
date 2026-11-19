#!/usr/bin/env bash
# Every calibration of the cell (scripts/calibrate.py) with the project venv. Needs scripts/run_vision.sh.
#   scripts/calibrate.sh cameras --check | cameras [--dry] | survey contact A B C | table | colors
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
exec .venv/bin/python scripts/calibrate.py "$@"
