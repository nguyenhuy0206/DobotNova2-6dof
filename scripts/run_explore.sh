#!/usr/bin/env bash
# Astra explore mode. Start the cameras first in another terminal:  scripts/run_vision.sh
#
#   scripts/run_explore.sh                  task from config/explore.yaml
#   scripts/run_explore.sh --mode confirm   y/N before every motion
#   scripts/run_explore.sh --episodes 5     cap autonomous attempts (threading config retries by default)
#   scripts/run_explore.sh --no-until-success --episodes 1   manual label/reset workflow
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
exec .venv/bin/python scripts/run_explore.py "$@"
