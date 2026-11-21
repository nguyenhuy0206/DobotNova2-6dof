#!/usr/bin/env bash
# Kept for old habits: the dashboard is started by scripts/run_vision.sh (cameras + depth helper + UI).
#   scripts/run_ui.sh          -> scripts/run_vision.sh --rgb   (no depth, no sudo)
#   scripts/run_ui.sh --depth  -> scripts/run_vision.sh         (D455 depth helper, asks for sudo)
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--depth" ]]; then
    shift
    echo ">> run_ui.sh --depth is now scripts/run_vision.sh"
    exec scripts/run_vision.sh "$@"
fi
echo ">> run_ui.sh is now scripts/run_vision.sh --rgb"
exec scripts/run_vision.sh --rgb "$@"
