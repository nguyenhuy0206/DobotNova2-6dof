#!/usr/bin/env bash
# Robot calibrates the D455 by itself at 10 % speed. Needs scripts/run_vision.sh running.
#   scripts/run_calibration.sh  [--center x,y,z] [--span 80]
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
exec .venv/bin/python scripts/auto_calibrate.py "$@"
