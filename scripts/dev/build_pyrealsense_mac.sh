#!/usr/bin/env bash
# Build librealsense + pyrealsense2 (v2.59.1, Python 3.13) into this repo:
#
#   third_party/librealsense/      (git-ignored)
#     src/     source checkout
#     build/   cmake build tree
#     lib/     pyrealsense2*.so + librealsense2*.dylib   <- what the code imports
#
#   scripts/dev/build_pyrealsense_mac.sh                    clone + build (~10–20 min on Apple Silicon)
#   scripts/dev/build_pyrealsense_mac.sh --reuse <Release>  install an existing build's Release/ folder
#                                                       (e.g. ~/src/librealsense-v2.59.1/build/Release)
#
# Why from source: the pip wheel (pyrealsense2-macosx 2.56.5) segfaults on macOS 15.
# src/drivers/camera.py imports from third_party/librealsense/lib directly —
# no .pth link in .venv (macOS keeps hiding those and Python 3.13 skips hidden .pth files).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
VERSION=v2.59.1
PY="${PY:-/opt/homebrew/bin/python3.13}"
TP="$ROOT/third_party/librealsense"
SRC="$TP/src"
BUILD="$TP/build"
LIB="$TP/lib"

[[ -x "$PY" ]] || { echo "Need Python 3.13: brew install python@3.13"; exit 1; }
brew list libusb >/dev/null 2>&1 || { echo "Need libusb: brew install libusb"; exit 1; }

if [[ "${1:-}" == "--reuse" ]]; then
    RELEASE="${2:?usage: $0 --reuse <path to build/Release>}"
    echo ">> Reusing existing build: $RELEASE"
else
    echo ">> 1/3 Source $VERSION -> $SRC"
    if [[ ! -d "$SRC" ]]; then
        mkdir -p "$TP"
        git clone --depth 1 --branch "$VERSION" https://github.com/realsenseai/librealsense.git "$SRC"
    fi
    echo ">> 2/3 Configure + build (library + Python bindings only)"
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DENABLE_CCACHE=OFF -DBUILD_WITH_OPENMP=OFF -DCHECK_FOR_UPDATES=OFF \
        -DENABLE_AI_ASSISTANT=OFF -DENABLE_STATS=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY"
    cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu)"
    RELEASE="$BUILD/Release"
fi

ls "$RELEASE"/pyrealsense2*.so >/dev/null 2>&1 || { echo "No pyrealsense2*.so in $RELEASE"; exit 1; }

echo ">> 3/3 Install into $LIB (self-contained: libraries found next to each other)"
rm -rf "$LIB"
mkdir -p "$LIB"
cp -P "$RELEASE"/pyrealsense2*.so "$RELEASE"/pyrsutils*.so "$RELEASE"/librealsense2*.dylib "$LIB"/ 2>/dev/null || true
for f in "$LIB"/*.so "$LIB"/*.dylib; do
    [[ -L "$f" ]] && continue                      # symlinks point at the real files
    # Replace absolute build-tree rpaths with "same folder as this file".
    otool -l "$f" | awk '/LC_RPATH/{getline; getline; print $2}' | while read -r rp; do
        install_name_tool -delete_rpath "$rp" "$f" 2>/dev/null || true
    done
    install_name_tool -add_rpath @loader_path "$f"
    codesign --force --sign - "$f" >/dev/null 2>&1   # editing invalidates the arm64 signature
done

echo ">> Python requirements into the existing .venv (not recreated)"
[[ -x "$ROOT/.venv/bin/python" ]] || "$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt" -e "$ROOT"

# The old installer linked ~/src/... into .venv with a .pth file; not needed any more.
rm -f "$ROOT"/.venv/lib/python3*/site-packages/pyrealsense2_local.pth

"$ROOT/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$ROOT/src')
from drivers.camera import import_pyrealsense2
rs = import_pyrealsense2()
print('pyrealsense2', rs.__version__, 'from', rs.__file__)"
echo
echo "Done. Depth on macOS needs root:  sudo scripts/run_depth.sh   (then scripts/run_ui.sh as usual)"
