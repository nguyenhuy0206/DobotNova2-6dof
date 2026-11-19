#!/usr/bin/env bash
# RealSense D455 depth helper. macOS only lets root open the depth sensor, so this
# small process (and nothing else) runs with sudo. Keep it open in its own terminal,
# then run scripts/run_ui.sh as usual and pick "RealSense (colour + depth)".
#
#   sudo scripts/run_depth.sh            (or without sudo: it asks for your password)
#   sudo scripts/run_depth.sh --width 640 --height 480 --fps 15
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(pwd)/.venv/bin/python"
[[ -d third_party/librealsense/lib ]] || { echo "Build first: scripts/dev/build_pyrealsense_mac.sh"; exit 1; }

echo ">> Releasing the D455 from macOS camera services (UI windows on its colour device will stop)"
sudo killall VDCAssistant UVCAssistant 2>/dev/null || true
sleep 1
# PYTHONDONTWRITEBYTECODE: root must not leave root-owned __pycache__ in the repo.
exec sudo PYTHONPATH="$(pwd)/src" PYTHONDONTWRITEBYTECODE=1 "$PY" -m drivers.realsense_helper "$@"
