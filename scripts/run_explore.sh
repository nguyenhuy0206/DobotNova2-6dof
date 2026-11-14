#!/usr/bin/env bash
# Astra explore mode. Start the cameras first in another terminal:  scripts/run_vision.sh
#
#   scripts/run_explore.sh                  task from config/explore.yaml
#   scripts/run_explore.sh --recalibrate    the camera was moved since the last calibration
#   scripts/run_explore.sh --mode confirm   y/N before every motion
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
exec .venv/bin/python scripts/run_explore.py "$@"
