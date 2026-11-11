"""Camera drivers with a common ``read()`` interface.

* ``RealSenseCamera`` — Intel RealSense D455, aligned colour + depth.
* ``OpenCVCamera``    — any UVC webcam (or the D455's RGB on macOS when
                        librealsense cannot claim the device). Colour only.

For a full device/bandwidth benchmark use ``scripts/camera_benchmark.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RobotError(f"Cannot open OpenCV camera index {index}")

    def read(self) -> Frame:
        ok, image = self.cap.read()
        if not ok:
            raise RobotError("OpenCV camera returned no frame")
        return Frame(color=image, timestamp_ms=self.cap.get(self._cv2.CAP_PROP_POS_MSEC))

    def close(self) -> None:
        self.cap.release()


class RealSenseCamera:
    def __init__(self, width: int = 848, height: int = 480, fps: int = 30, serial: str | None = None):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RobotError("pyrealsense2 not installed (macOS: scripts/build_pyrealsense_mac.sh)") from exc
        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
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
