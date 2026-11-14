"""The env camera watching the robot itself, through a coloured marker on the gripper.

    marker       a band of coloured tape wrapped around the gripper body
                 (config/explore.yaml view_guard.marker_color, default red — a
                 colour no task object uses)

    RobotViewGuard     "is the robot still in the D455 picture?"  -> used after every
                       move in explore mode; the move is undone when it says no
    SelfCalibration    Astra calibrates the env camera itself: move the arm to varied
                       poses, record_calibration_point at each (marker 3D from depth +
                       robot flange pose), then solve_calibration. Solved with the
                       marker offset unknown (perception.calibration.solve_with_marker_offset),
                       so the tape can be anywhere on the gripper.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.errors import RobotError, ValidationError
from common.paths import MEMORY_DIR
from perception.detection import detect_colored_objects
from perception.transforms import HandEye, pose_to_matrix, rotation_angle_deg

SESSION_FILE = MEMORY_DIR / "calibration_session.json"


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


class SelfCalibration:
    """Env camera -> robot base, collected and solved by the agent at runtime."""

    MIN_POINTS = 6

    def __init__(self, scene, primitives, color: str = "red", min_area_px: float = 40,
                 session_file: Path = SESSION_FILE):
        self.scene, self.p = scene, primitives
        self.color, self.min_area = color, min_area_px
        self.session_file = session_file
        self.points: list[dict] = []
        if session_file.exists():
            self.points = json.loads(session_file.read_text(encoding="utf-8")).get("points", [])

    def _save_session(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(json.dumps({"points": self.points}, indent=1) + "\n", encoding="utf-8")

    def reset(self) -> dict:
        """Forget all recorded points (e.g. the camera was moved)."""
        self.points = []
        self._save_session()
        return {"points": 0}

    def _rotation_spread_deg(self) -> float:
        Rs = [pose_to_matrix(p["flange_pose"])[:3, :3] for p in self.points]
        return max((rotation_angle_deg(a, b) for i, a in enumerate(Rs) for b in Rs[i + 1:]), default=0.0)

    def _measure(self, samples: int) -> tuple[list[float], list[int], float]:
        """Marker 3D point (camera frame, m), median over a few frames to beat depth noise."""
        cams, pixels = [], []
        for i in range(samples):
            frame = self.scene.observe("env")
            det = find_marker(self.scene, frame, self.color, self.min_area)
            if det is not None:
                _, cam, _ = self.scene.locate(frame, *det.center)
                if cam is not None:
                    cams.append(cam)
                    pixels.append(list(det.center))
            if i + 1 < samples:
                time.sleep(0.1)
        if not pixels:
            raise ValidationError(f"{self.color} marker (with depth) not visible in the env image — "
                                  "move so the camera sees it.")
        cam = np.median(np.array(cams), axis=0)
        spread_mm = float(np.max(np.linalg.norm(np.array(cams) - cam, axis=1)) * 1000) if len(cams) > 1 else 0.0
        return [float(c) for c in cam], pixels[len(pixels) // 2], spread_mm

    def record(self, samples: int = 5) -> dict:
        """Robot must be standing still with the marker visible to the env camera."""
        state = self.p.get_state()
        if state["moving"]:
            raise ValidationError("robot is moving — record only when it stands still")
        cam, pixel, noise_mm = self._measure(samples)
        if noise_mm > 15:
            raise ValidationError(f"marker depth unstable ({noise_mm:.0f} mm between frames) — "
                                  "marker at an edge or too oblique; try another pose")
        point = {"camera_m": [round(c, 5) for c in cam], "flange_pose": state["flange_pose"],
                 "pixel": pixel, "depth_m": round(cam[2], 4), "noise_mm": round(noise_mm, 1), "t": time.time()}
        if any(np.linalg.norm(np.subtract(point["flange_pose"][:3], q["flange_pose"][:3])) < 20 and
               rotation_angle_deg(pose_to_matrix(point["flange_pose"])[:3, :3],
                                  pose_to_matrix(q["flange_pose"])[:3, :3]) < 5 for q in self.points):
            raise ValidationError("almost the same pose as an earlier point — move ≥ 20 mm or rotate ≥ 5° first")
        self.points.append(point)
        self._save_session()
        spread = self._rotation_spread_deg()
        return {"points": len(self.points), "marker_pixel": point["pixel"], "depth_m": point["depth_m"],
                "noise_mm": point["noise_mm"],
                "rotation_spread_deg": round(spread, 1),
                "advice": self._advice(spread)}

    @staticmethod
    def leave_one_out(cam: np.ndarray, poses: list) -> np.ndarray:
        """Predict each marker point from a calibration solved without it (mm)."""
        from perception.calibration import solve_with_marker_offset

        errs = []
        for i in range(len(poses)):
            keep = [j for j in range(len(poses)) if j != i]
            try:
                T, o, _, _ = solve_with_marker_offset(cam[keep], [poses[j] for j in keep])
            except (ValueError, np.linalg.LinAlgError):
                errs.append(np.inf)
                continue
            P = pose_to_matrix(poses[i])
            predicted = T[:3, :3] @ cam[i] + T[:3, 3]
            actual = P[:3, 3] + P[:3, :3] @ (o / 1000.0)
            errs.append(float(np.linalg.norm(predicted - actual) * 1000))
        return np.array(errs)

    def _advice(self, spread: float) -> str:
        need = max(0, self.MIN_POINTS - len(self.points))
        tips = []
        if need:
            tips.append(f"record {need} more point(s)")
        if spread < 30:
            tips.append("vary the gripper orientation more (J4/J5/J6 by 15–20°) between points")
        tips.append("spread points over the area you will work in (around the box and the free table)")
        return "; ".join(tips)

    def solve(self) -> dict:
        from perception.calibration import solve_with_marker_offset

        if len(self.points) < self.MIN_POINTS:
            raise ValidationError(f"need ≥ {self.MIN_POINTS} points, have {len(self.points)}")
        cam = np.array([p["camera_m"] for p in self.points])
        poses = [p["flange_pose"] for p in self.points]
        # Drop bad points one at a time: the worst leave-one-out error, while it is far
        # above the rest (a marker seen at its edge, a depth glitch) and ≥ 8 points remain.
        keep = list(range(len(poses)))
        dropped = []
        loo = self.leave_one_out(cam, poses)
        while len(keep) > 8:
            worst = int(np.argmax(loo))
            rest = np.delete(loo, worst)
            if not (loo[worst] > max(3 * float(np.median(rest)), 25.0)):
                break
            dropped.append({"point": keep[worst], "leave_one_out_mm": round(float(loo[worst]), 1)})
            del keep[worst]
            loo = self.leave_one_out(cam[keep], [poses[i] for i in keep])
        cam, poses = cam[keep], [poses[i] for i in keep]
        try:
            T, offset_mm, rms, errors = solve_with_marker_offset(cam, poses)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RobotError(f"calibration solve failed: {exc}") from None
        spread = self._rotation_spread_deg()
        calib = HandEye(T=T, eye_in_hand=False)
        calib.save(self.scene.cfg.calibration_path)
        self.scene.env_calibration = calib
        worst = int(np.argmax(errors))
        return {"saved": str(self.scene.cfg.calibration_path), "points": len(keep),
                "recorded": len(self.points), "dropped_outliers": dropped,
                "rms_mm": round(float(rms), 1), "worst_point": worst, "worst_error_mm": round(float(errors[worst]), 1),
                "marker_offset_in_flange_mm": [round(float(o), 1) for o in offset_mm],
                "rotation_spread_deg": round(spread, 1),
                "leave_one_out_mm": {"median": round(float(np.median(loo)), 1), "max": round(float(np.max(loo)), 1)},
                "note": ("leave_one_out = error on a point when solved WITHOUT it: the honest accuracy. "
                         "find_objects / look now report base_xyz_mm. RMS on the points themselves is "
                         "optimistic — verify by hovering over a block and checking with the wrist camera."
                         + (" Rotation spread < 30°: the result may be unreliable." if spread < 30 else ""))}
