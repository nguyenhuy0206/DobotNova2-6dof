"""The env camera watching the robot itself, through a coloured marker on the gripper.

    marker       a band of coloured tape wrapped around the gripper body
                 (config/explore.yaml view_guard.marker_color, default red — a
                 colour no task object uses)

    RobotViewGuard     "is the robot still in the D455 picture?"  -> used after every
                       move in explore mode; the move is undone when it says no
"""
from __future__ import annotations

import time

from perception.detection import detect_colored_objects


def find_marker(scene, frame, color: str, min_area_px: float):
    """Largest blob of the marker colour, or None."""
    dets = detect_colored_objects(frame.color, [color], min_area_px, scene.ranges)
    return dets[0] if dets else None


class RobotViewGuard:
    def __init__(self, scene, color: str = "red", min_area_px: float = 40, camera: str = "env",
                 attempts: int = 3, retry_s: float = 0.2):
        self.scene, self.color, self.min_area, self.camera = scene, color, min_area_px, camera
        self.attempts, self.retry_s = attempts, retry_s

    def __call__(self) -> dict:
        for attempt in range(self.attempts):
            frame = self.scene.observe(self.camera)
            det = find_marker(self.scene, frame, self.color, self.min_area)
            if det is not None:
                return {"visible": True, "marker_pixel": list(det.center), "marker_area_px": int(det.area)}
            if attempt + 1 < self.attempts:
                time.sleep(self.retry_s)
        return {"visible": False,
                "detail": f"{self.color} marker on the gripper not found in the {self.camera} image"}
