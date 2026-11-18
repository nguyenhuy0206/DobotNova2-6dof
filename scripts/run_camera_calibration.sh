#!/usr/bin/env bash
# Both cameras from the ChArUco board (scripts/calibrate_cameras.py) with the project venv.
# Needs scripts/run_vision.sh running.
#   scripts/run_camera_calibration.sh --check      what each camera sees (no motion)
#   scripts/run_camera_calibration.sh --dry        full run, nothing saved
#   scripts/run_camera_calibration.sh              full run, saves what passes
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
exec .venv/bin/python scripts/calibrate_cameras.py "$@"
