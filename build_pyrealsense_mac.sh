#!/usr/bin/env bash
# Build pyrealsense2 2.59.1 (same version as Homebrew's working `realsense-viewer`)
# and put it into ./.venv (Python 3.13), replacing the crashing pip wheel.
#
#   ./build_pyrealsense_mac.sh        (~10-20 min on Apple M1)
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(pwd)"
VERSION=v2.59.1
PY=/opt/homebrew/bin/python3.13
SRC="$HOME/src/librealsense-$VERSION"

[[ -x "$PY" ]] || { echo "Need Python 3.13: brew install python@3.13"; exit 1; }

echo ">> 1/4 Download librealsense $VERSION"
if [[ ! -d "$SRC" ]]; then
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 --branch "$VERSION" https://github.com/realsenseai/librealsense.git "$SRC"
fi

echo ">> 2/4 Configure (library + Python bindings only)"
cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DENABLE_CCACHE=OFF -DBUILD_WITH_OPENMP=OFF -DCHECK_FOR_UPDATES=OFF \
    -DENABLE_AI_ASSISTANT=OFF -DENABLE_STATS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$PY" -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY"

echo ">> 3/4 Build (takes a while)"
cmake --build "$SRC/build" -j"$(sysctl -n hw.ncpu)"

SO="$(find "$SRC/build" -name 'pyrealsense2*.so' | head -1)"
[[ -n "$SO" ]] || { echo "Build finished but pyrealsense2*.so not found"; exit 1; }
echo "   built: $SO"

echo ">> 4/4 Recreate .venv with Python 3.13"
for _ in 1 2 3; do rm -rf "$PROJECT/.venv" 2>/dev/null && break; sleep 1; done  # Finder may recreate .DS_Store
"$PY" -m venv "$PROJECT/.venv"
"$PROJECT/.venv/bin/pip" install -q --upgrade pip
"$PROJECT/.venv/bin/pip" install -q numpy opencv-python
SITE="$("$PROJECT/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$(dirname "$SO")" > "$SITE/pyrealsense2_local.pth"

"$PROJECT/.venv/bin/python" -c "import pyrealsense2 as rs; print('pyrealsense2 OK, version', rs.__version__)"
echo
echo "Done. Now run:  ./run_camera.sh"
