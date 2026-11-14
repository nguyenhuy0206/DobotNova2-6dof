#!/usr/bin/env bash
# Start the control dashboard at http://127.0.0.1:8765
#
#   scripts/run_ui.sh            UI only (D455 colour via its webcam device, no sudo)
#   scripts/run_ui.sh --depth    also start the RealSense depth helper: asks your Mac
#                                password once; only the helper runs as root, the UI runs
#                                as you; Ctrl+C stops both
set -euo pipefail
cd "$(dirname "$0")/.."
# Python 3.13 ignores .pth files with the macOS "hidden" flag (Finder/iCloud can set it),
# which would silently break the editable install. Clear it every launch.
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
PY=.venv/bin/python; [[ -x $PY ]] || PY=python3

if [[ "${1:-}" == "--depth" ]]; then
    shift
    echo ">> Depth mode: the RealSense helper needs your Mac password (sudo)."
    sudo -v
    # $$ stays the UI's pid after the exec below; the helper exits when the UI does.
    scripts/run_depth.sh --watch-pid $$ &
    echo ">> UI: open http://127.0.0.1:8765/#debug and pick \"RealSense (colour + depth)\" in Environment"
fi
exec "$PY" ui/app.py "$@"
