#!/usr/bin/env bash
# Open one window with two views from the RealSense D455: COLOUR | DEPTH.
#
#   ./run_camera.sh              colour + depth via librealsense (asks for admin password),
#                                falls back to colour-only (OpenCV) if librealsense crashes
#   ./run_camera.sh rgb          skip librealsense, colour-only via OpenCV (no password)
#   ./run_camera.sh --save       extra options are passed to camera.py
#
# Press q in the window to quit; the data report is printed afterwards.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
SECONDS_LIMIT=3600   # run up to 1 hour, quit earlier with q

MODE=auto
if [[ "${1:-}" == "rgb" ]]; then MODE=rgb; shift; fi

if [[ "$MODE" == auto ]]; then
    echo ">> Starting colour + depth (librealsense). macOS needs your admin password."
    # macOS camera services hold the D455; release them so librealsense can open it.
    sudo killall VDCAssistant UVCAssistant 2>/dev/null
    sudo "$PY" camera.py --no-imu --show --seconds "$SECONDS_LIMIT" "$@"
    rc=$?
    if [[ $rc -eq 0 ]]; then exit 0; fi
    echo
    echo ">> librealsense failed (exit code $rc). Falling back to colour only (OpenCV)."
    sleep 2   # give macOS camera services time to come back
fi

exec "$PY" camera.py --opencv 0 --color 1280x800 --fps 30 --show --seconds "$SECONDS_LIMIT" "$@"
