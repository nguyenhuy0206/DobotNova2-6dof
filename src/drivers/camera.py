"""Camera drivers with a common ``read()`` interface.

* ``RealSenseCamera`` — Intel RealSense D455, aligned colour + depth.
* ``OpenCVCamera``    — any UVC webcam (or the D455's RGB on macOS when
                        librealsense cannot claim the device). Colour only.

For a full device/bandwidth benchmark use ``scripts/camera_benchmark.py``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.errors import RobotError


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Frame:
    color: np.ndarray                 # HxWx3 BGR uint8
    depth_m: np.ndarray | None = None  # HxW float32 metres, aligned to colour
    intrinsics: Intrinsics | None = None
    timestamp_ms: float = 0.0


class OpenCVCamera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        import cv2
        self._cv2 = cv2
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RobotError(f"Cannot open OpenCV camera index {index}")

    def grab(self) -> bool:
        """Pull the next frame off the driver without decoding (keeps latency low)."""
        return self.cap.grab()

    def retrieve(self) -> Frame:
        ok, image = self.cap.retrieve()
        if not ok or image is None:
            raise RobotError("OpenCV camera returned no frame")
        return Frame(color=image, timestamp_ms=self.cap.get(self._cv2.CAP_PROP_POS_MSEC))

    def read(self) -> Frame:
        ok, image = self.cap.read()
        if not ok:
            raise RobotError("OpenCV camera returned no frame")
        return Frame(color=image, timestamp_ms=self.cap.get(self._cv2.CAP_PROP_POS_MSEC))

    def close(self) -> None:
        self.cap.release()


def import_pyrealsense2():
    """Import pyrealsense2 from the repo's own build (third_party/librealsense/lib).

    Built/installed by scripts/build_pyrealsense_mac.sh. ``PYREALSENSE2_DIR``
    overrides the location; an already-importable pyrealsense2 (Linux pip
    wheel) is used as-is.
    """
    import os

    from common.paths import REALSENSE_LIB_DIR

    for directory in (os.environ.get("PYREALSENSE2_DIR"), str(REALSENSE_LIB_DIR)):
        if directory and Path(directory).is_dir() and directory not in sys.path:
            sys.path.insert(0, directory)
    try:
        import pyrealsense2
        return pyrealsense2
    except ImportError as exc:
        raise RobotError("pyrealsense2 not found in third_party/librealsense/lib — "
                         "run scripts/build_pyrealsense_mac.sh") from exc


class RealSenseCamera:
    # Tried in order. USB 2 (480 Mb/s) cannot carry colour+depth above ~640x480@15.
    FALLBACK_PROFILES = [(848, 480, 30), (640, 480, 15), (640, 480, 6), (424, 240, 15)]

    def __init__(self, width: int = 848, height: int = 480, fps: int = 30, serial: str | None = None):
        rs = import_pyrealsense2()
        self._rs = rs
        errors = []
        for w, h, f in [(width, height, fps)] + [p for p in self.FALLBACK_PROFILES if p != (width, height, fps)]:
            pipeline, config = rs.pipeline(), rs.config()
            if serial:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
            try:
                profile = pipeline.start(config)
            except RuntimeError as exc:
                errors.append(f"{w}x{h}@{f}: {exc}")
                if "no device" in str(exc).lower():
                    break                      # nothing plugged in; other profiles won't help
                continue
            self.pipeline, self.profile_used = pipeline, (w, h, f)
            break
        else:
            raise RobotError("RealSense: no colour+depth profile worked — " + " | ".join(errors[-2:]))
        if not hasattr(self, "pipeline"):
            raise RobotError("RealSense: " + errors[-1])
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy)

    def read(self) -> Frame:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RobotError("RealSense returned an incomplete frameset")
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return Frame(color=np.asanyarray(color.get_data()).copy(), depth_m=depth_m,
                     intrinsics=self.intrinsics, timestamp_ms=color.get_timestamp())

    def close(self) -> None:
        self.pipeline.stop()


def open_camera(prefer_realsense: bool = True, opencv_index: int = 0):
    """Try RealSense first, fall back to colour-only OpenCV."""
    if prefer_realsense:
        try:
            return RealSenseCamera()
        except Exception:  # librealsense often fails on macOS without sudo
            pass
    return OpenCVCamera(opencv_index)


class RealSenseRemoteCamera:
    """D455 colour + depth served by the root-only helper (drivers/realsense_helper.py).

    macOS only lets root open the depth sensor, so ``sudo scripts/run_depth.sh``
    owns the camera and this client — running in the normal-user UI — pulls the
    latest frame over localhost.
    """

    def __init__(self, url: str = "http://127.0.0.1:8766", timeout: float = 2.0):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.read()   # fail fast with a clear message if the helper isn't running

    def read(self) -> Frame:
        import io
        import urllib.error
        import urllib.request

        import cv2

        try:
            with urllib.request.urlopen(f"{self.url}/frame", timeout=self.timeout) as r:
                data = np.load(io.BytesIO(r.read()))
        except urllib.error.HTTPError as exc:
            raise RobotError(f"depth helper: {exc.read().decode(errors='replace')[:200]}") from exc
        except OSError as exc:
            raise RobotError(f"depth helper not running at {self.url} — start it with: "
                             f"sudo scripts/run_depth.sh") from exc
        color = cv2.imdecode(data["color_jpeg"], cv2.IMREAD_COLOR)
        k = data["intrinsics"]
        intr = Intrinsics(int(k[0]), int(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        return Frame(color=color, depth_m=data["depth_mm"].astype(np.float32) / 1000.0,
                     intrinsics=intr, timestamp_ms=float(data["timestamp_ms"]))

    def close(self) -> None:
        pass
