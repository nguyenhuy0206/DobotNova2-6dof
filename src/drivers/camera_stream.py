"""Background camera capture → latest JPEG, for live viewing (Debug tab MJPEG).

    hub = CameraHub(cfg.cameras)          # from config/robot.yaml `cameras:`
    jpeg = hub["wrist"].latest_jpeg()     # starts the camera on first use

Each ``CameraStream`` owns one device in its own thread:
  * opens lazily when someone watches, releases it after ``idle_timeout`` s
    without viewers (so other apps / scripts can use the camera),
  * keeps draining the driver buffer (low latency) but only encodes ``fps``
    frames per second,
  * reconnects automatically if the device is unplugged or fails to open.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from common.config import CameraConfig
from common.log import get_logger
from common.paths import MEMORY_DIR

log = get_logger(__name__)
OVERRIDES_FILE = MEMORY_DIR / "cameras.json"
SOURCES = ("opencv", "realsense")


def open_device(cfg: CameraConfig):
    """Default device factory: returns an object with read() -> Frame and close()."""
    from drivers.camera import OpenCVCamera, RealSenseRemoteCamera

    if cfg.source == "realsense":
        # Depth needs root on macOS; the root-only helper owns the D455 (scripts/run_depth.sh).
        return RealSenseRemoteCamera(cfg.helper_url)
    return OpenCVCamera(cfg.index, cfg.width, cfg.height, 30)


def depth_colormap(depth_m: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    import cv2

    scaled = np.clip(depth_m / max_m * 255.0, 0, 255).astype(np.uint8)
    view = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_JET)
    view[depth_m <= 0] = 0
    return view


def frame_to_image(frame) -> np.ndarray:
    """Colour only, or colour | depth side by side for RealSense frames."""
    if frame.depth_m is None:
        return frame.color
    depth = depth_colormap(frame.depth_m)
    if depth.shape[:2] != frame.color.shape[:2]:
        import cv2
        depth = cv2.resize(depth, (frame.color.shape[1], frame.color.shape[0]))
    return np.hstack([frame.color, depth])


class CameraStream:
    def __init__(self, name: str, cfg: CameraConfig,
                 opener: Callable[[CameraConfig], object] = open_device,
                 idle_timeout: float = 15.0, retry_s: float = 3.0, max_width: int = 960):
        self.name, self.cfg, self.opener = name, cfg, opener
        self.idle_timeout, self.retry_s, self.max_width = idle_timeout, retry_s, max_width
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._image: np.ndarray | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_access = 0.0
        self.error: str | None = None
        self.resolution: tuple[int, int] | None = None
        self.measured_fps = 0.0

    # -- public ----------------------------------------------------------
    def touch(self) -> None:
        """Mark that someone is watching; start the capture thread if needed."""
        self._last_access = time.monotonic()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
            self._thread.start()

    def latest_jpeg(self) -> bytes | None:
        self.touch()
        return self._jpeg

    def latest_image(self) -> np.ndarray | None:
        self.touch()
        return None if self._image is None else self._image.copy()

    def wait_jpeg(self, after_seq: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """Block until a frame newer than ``after_seq`` exists (or timeout)."""
        self.touch()
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq and self._jpeg is not None, timeout=timeout)
            return self._seq, self._jpeg

    def reconfigure(self, **changes) -> None:
        self.stop()
        self.cfg = replace(self.cfg, **changes)
        with self._cond:
            self._jpeg, self._image, self.error, self.resolution = None, None, None, None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def status(self) -> dict:
        running = self._thread is not None and self._thread.is_alive()
        return {"name": self.name, "label": self.cfg.label, "source": self.cfg.source, "index": self.cfg.index,
                "running": running, "has_frame": self._jpeg is not None, "error": self.error,
                "resolution": self.resolution, "fps": round(self.measured_fps, 1), "target_fps": self.cfg.fps}

    def _hint(self) -> str:
        """Actionable advice for the common macOS RealSense failure (not running as root)."""
        import os
        import sys
        if self.cfg.source == "realsense" and sys.platform == "darwin" and os.geteuid() != 0:
            return " — on macOS depth needs root: restart with  scripts/run_ui.sh --depth"
        return ""

    # -- capture thread --------------------------------------------------
    def _idle(self) -> bool:
        return time.monotonic() - self._last_access > self.idle_timeout

    def _run(self) -> None:
        while not self._stop.is_set() and not self._idle():
            try:
                device = self.opener(self.cfg)
            except Exception as exc:  # noqa: BLE001 — any driver failure: report, retry
                error = f"{self.cfg.source} #{self.cfg.index}: {exc}{self._hint()}"
                if error != self.error:          # log each distinct problem once
                    log.warning("camera %s: %s", self.name, error)
                self.error = error
                self._stop.wait(self.retry_s)
                continue
            log.info("camera %s opened (%s #%s)", self.name, self.cfg.source, self.cfg.index)
            try:
                self._capture(device)
            except Exception as exc:  # noqa: BLE001
                self.error = f"capture failed: {exc}"
                log.warning("camera %s: %s", self.name, exc)
                self._stop.wait(self.retry_s)
            finally:
                try:
                    device.close()
                except Exception:  # noqa: BLE001
                    pass
        log.info("camera %s released", self.name)

    def _capture(self, device) -> None:
        import cv2

        period = 1.0 / max(self.cfg.fps, 0.5)
        next_encode, window_start, frames, failures = 0.0, time.monotonic(), 0, 0
        grab = getattr(device, "grab", None)
        while not self._stop.is_set() and not self._idle():
            now = time.monotonic()
            if grab is not None and now < next_encode:
                if grab():                  # drain the driver buffer, don't decode
                    failures = 0
                    continue
                failures += 1
                if failures > 30:
                    raise RuntimeError("device stopped delivering frames (unplugged?)")
                time.sleep(0.05)
                continue
            frame = device.retrieve() if grab is not None else device.read()
            if grab is None and now < next_encode:
                continue
            next_encode = max(next_encode + period, now)   # fixed cadence, no drift
            image = frame_to_image(frame)
            if image.shape[1] > self.max_width:
                scale = self.max_width / image.shape[1]
                image = cv2.resize(image, (self.max_width, int(image.shape[0] * scale)))
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if not ok:
                continue
            frames += 1
            if now - window_start >= 2.0:
                self.measured_fps, window_start, frames = frames / (now - window_start), now, 0
            with self._cond:
                self._jpeg, self._image = buf.tobytes(), frame.color
                self.resolution = (frame.color.shape[1], frame.color.shape[0])
                self.error = None
                self._seq += 1
                self._cond.notify_all()


class CameraHub:
    """All configured cameras by name, with UI-chosen devices persisted to disk."""

    def __init__(self, configs: dict[str, CameraConfig], opener=open_device,
                 overrides_file: Path = OVERRIDES_FILE, **stream_kwargs):
        self.overrides_file = overrides_file
        overrides = self._load_overrides()
        self.streams = {name: CameraStream(name, replace(cfg, **overrides.get(name, {})), opener, **stream_kwargs)
                        for name, cfg in configs.items()}

    def __getitem__(self, name: str) -> CameraStream:
        return self.streams[name]

    def __contains__(self, name: str) -> bool:
        return name in self.streams

    def status(self) -> list[dict]:
        return [s.status() for s in self.streams.values()]

    def select_device(self, name: str, source: str, index: int) -> dict:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not 0 <= index <= 9:
            raise ValueError("index must be 0–9")
        stream = self.streams[name]
        stream.reconfigure(source=source, index=index)
        overrides = self._load_overrides()
        overrides[name] = {"source": source, "index": index}
        self.overrides_file.parent.mkdir(parents=True, exist_ok=True)
        self.overrides_file.write_text(json.dumps(overrides, indent=2) + "\n", encoding="utf-8")
        return stream.status()

    def stop_all(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    def _load_overrides(self) -> dict:
        try:
            value = json.loads(self.overrides_file.read_text(encoding="utf-8"))
            return {k: {f: v for f, v in d.items() if f in ("source", "index")} for k, d in value.items()}
        except (OSError, ValueError, AttributeError):
            return {}
