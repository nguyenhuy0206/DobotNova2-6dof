#!/usr/bin/env bash
# One command for the whole vision system: D455 depth helper + wrist webcam + dashboard.
#
#   scripts/run_vision.sh               check cameras, (re)start depth helper, start UI, open browser
#   scripts/run_vision.sh --rgb         no depth / no sudo (D455 colour only through its webcam device)
#   scripts/run_vision.sh --no-browser  don't open the browser
#
# Safe to re-run any time: leftovers from a previous run (helper on :8766, UI on :8765)
# are stopped first. Ctrl+C stops everything. Helper log: data/logs/depth_helper.log
set -euo pipefail
cd "$(dirname "$0")/.."
chflags nohidden .venv/lib/python3*/site-packages/*.pth 2>/dev/null || true
PY=.venv/bin/python; [[ -x $PY ]] || PY=python3
UI_URL="http://127.0.0.1:8765/#debug"
HELPER="http://127.0.0.1:8766"
LOG=data/logs/depth_helper.log

DEPTH=1; BROWSER=1; UI_ARGS=()
for a in "$@"; do
    case "$a" in
        --rgb) DEPTH=0 ;;
        --no-browser) BROWSER=0 ;;
        *) UI_ARGS+=("$a") ;;
    esac
done

ok()   { echo "   ✓ $*"; }
warn() { echo "   ! $*"; }

# ------------------------------------------------------------------ 1. cameras plugged in?
echo ">> 1/4 Cameras on USB"
USB="$(system_profiler SPUSBDataType 2>/dev/null || true)"
if grep -q "RealSense" <<<"$USB"; then ok "Intel RealSense D455"
else warn "D455 NOT found — replug its USB-C cable (use a USB 3 port/cable if you can)"; fi
if grep -q "USB 2.0 Camera" <<<"$USB"; then ok "Wrist webcam (USB 2.0 Camera)"
else warn "Wrist webcam NOT found — replug it"; fi

# ------------------------------------------------------------------ 2. stop leftovers
echo ">> 2/4 Stopping leftovers from a previous run"
[[ $DEPTH == 1 ]] && { echo "   (sudo: Mac password for the depth helper — not asked if you typed it in the last few minutes)"; sudo -v; }
for port in 8765 8766; do
    pids="$( (sudo -n lsof -tiTCP:$port -sTCP:LISTEN 2>/dev/null || lsof -tiTCP:$port -sTCP:LISTEN 2>/dev/null) || true)"
    if [[ -n "$pids" ]]; then
        sudo -n kill $pids 2>/dev/null || kill $pids 2>/dev/null || true
        ok "stopped old process on :$port ($pids)"
    fi
done
sleep 1

# ------------------------------------------------------------------ 3. depth helper
if [[ $DEPTH == 1 ]]; then
    echo ">> 3/4 Depth helper (releases the D455 from macOS camera services)"
    mkdir -p data/logs
    # $$ stays the UI's pid after the exec below -> the helper exits when the UI does.
    scripts/run_depth.sh --watch-pid $$ >"$LOG" 2>&1 &
    ready=0
    last_status="no answer"
    printf "   opening the D455 (takes ~10–20 s, don't press Ctrl+C) "
    for _ in $(seq 1 80); do
        st="$(curl -s --max-time 1 "$HELPER/status" || true)"
        [[ -n "$st" ]] && last_status="$st"
        if grep -q '"has_frame": true' <<<"$st"; then ready=1; break; fi
        printf "."
        sleep 0.5
    done
    echo
    if [[ $ready == 1 ]]; then
        ok "depth live: $st"
    else
        warn "depth not ready after polling (last status: $last_status). Last log lines:"
        tail -n 8 "$LOG" | sed 's/^/      /'
        warn "Fix: unplug + replug the D455, then run this script again. UI starts anyway."
    fi
else
    echo ">> 3/4 Depth skipped (--rgb)"
fi

# ------------------------------------------------------------------ 4. UI
echo ">> 4/4 Dashboard: $UI_URL   (Ctrl+C stops everything)"
echo "   Wrong picture in a window? Pick another device in that window's dropdown (remembered)."
[[ $BROWSER == 1 ]] && (sleep 2; open "$UI_URL") &
exec "$PY" ui/app.py ${UI_ARGS[@]+"${UI_ARGS[@]}"}
