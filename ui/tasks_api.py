"""Manual D455 locate and pick/place operations, independent of HTTP transport."""
from __future__ import annotations

import math
from threading import Lock

import cv2

from actions.executor import ActionExecutor
from actions.validation import require_confirm
from calibration.workcell import load_workcell
from common.errors import RobotError, ValidationError
from perception.scene import Scene


class TasksAPI:
    def __init__(self, executor: ActionExecutor, workspace, *, scene_factory=None, workcell_loader=None):
        # Share the Debug executor, including its live mode and stopped-motion state.
        self.executor = executor
        self._scene_factory = scene_factory or (
            lambda: Scene.from_config(cameras=("env",), workspace=workspace))
        self._workcell_loader = workcell_loader or load_workcell
        self._scene = None
        self._has_frame = False
        self._frame_lock = Lock()

    def frame_jpeg(self) -> bytes:
        """Capture env once and encode its untouched, full-resolution colour frame."""
        with self._frame_lock:
            # A failed recapture must not leave a previous picture available for locating.
            self._has_frame = False
            try:
                if self._scene is None:
                    self._scene = self._scene_factory()
                self._scene.look("env")
                frame, _ = self._scene.last_frames["env"]
                if frame.depth_m is None or frame.intrinsics is None:
                    raise ValueError("env frame has no aligned depth or intrinsics")
                # Scene.look's JPEG can be downsized for a model. Use the cached raw
                # colour instead so browser pixels are always full-image coordinates.
                ok, jpeg = cv2.imencode(".jpg", frame.color)
                if not ok:
                    raise ValueError("could not encode the env colour frame")
            except (OSError, RobotError, ValueError, cv2.error) as exc:
                raise RobotError(
                    f"D455 depth helper unavailable: {exc}. Start scripts/run_vision.sh "
                    "with depth enabled, then Capture env again."
                ) from exc
            self._has_frame = True
            return jpeg.tobytes()

    def locate(self, data: dict) -> dict:
        with self._frame_lock:
            if not self._has_frame:
                raise ValidationError("Capture env successfully before locating a pixel.")
            pixel = data.get("pixel")
            if (not isinstance(pixel, list) or len(pixel) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                        not math.isfinite(v) for v in pixel)):
                raise ValidationError("pixel must be a finite full-image [u, v] pair.")
            return self._scene.locate_object(pixel, table_z=self._workcell_loader().get("table_z_mm"))

    def _motion(self, name: str, data: dict) -> dict:
        require_confirm(data.get("confirm"), "Turn on 'Workspace clear' before running motion.")
        return self.executor.execute(name, {key: value for key, value in data.items() if key != "confirm"})

    def pick(self, data: dict) -> dict:
        return self._motion("pick_object", data)

    def place(self, data: dict) -> dict:
        return self._motion("place_object", data)

    def insert(self, data: dict) -> dict:
        return self._motion("insert", data)
