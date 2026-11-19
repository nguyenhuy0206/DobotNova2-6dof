"""LEGACY env-camera calibration from a coloured marker (red tape) on the gripper.

Superseded on 18 Nov 2026 by the ChArUco calibration (calibration.cameras), which is more accurate
and also calibrates the wrist camera. Kept because explore mode still builds a SelfCalibration.

    SelfCalibration  record (marker 3D from depth + flange pose) at varied poses, then solve with
                     the marker offset unknown (calibration.hand_eye.solve_with_marker_offset)
    auto_calibrate   the robot visits the poses by itself

Scripted marker calibration: the robot visits poses by itself (no model in the loop).

Uses the normal primitives, so every guard applies (speed cap, table keep-out along
the path, IK reachability, robot must stay in the D455 view — a pose that leaves
the view is undone and skipped).

Candidates around a CENTRE (fingertip): a ±span grid in X/Y at several heights, each
with a different gripper orientation (tilts up to 20°, twists up to 20°) so the
marker offset can be solved (perception.calibration.solve_with_marker_offset).
Unreachable candidates are rejected by the controller's IK *before* moving; the run
stops once `target_points` good points are recorded.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from calibration.files import MARKER_SESSION_FILE
from common.errors import RobotError, SafetyError, ValidationError
from common.log import get_logger
from perception.robot_view import find_marker
from perception.transforms import HandEye, matrix_to_rpy_deg, pose_to_matrix, rotation_angle_deg

log = get_logger(__name__)

# X/Y offsets as fractions of span (spread first, fill in later)
XY = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1),
      (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5),
      (1, 0.5), (-1, -0.5), (0.5, -1), (-0.5, 1), (0, 0), (1, 0), (0, 1)]
HEIGHTS = [0, 80, 40, 120, 20, 100]                       # mm above the centre
# rotation of the gripper about its own X, Y, Z (deg), applied to "pointing down"
ORIENT = [(0, 0, 0), (20, 0, 0), (0, 20, 0), (-20, 0, 0), (0, -20, 0), (15, 15, 0), (-15, -15, 0),
          (0, 0, 20), (0, 0, -20), (15, -15, 10), (-15, 15, -10), (10, 0, 15)]


def candidates(center: list[float], rz: float, span_mm: float) -> list[tuple[str, list[float]]]:
    """Fingertip poses [x, y, z, rx, ry, rz] with a short label each."""
    cx, cy, cz = center
    down = pose_to_matrix([0, 0, 0, 180, 0, rz])[:3, :3]
    out = []
    for i, (fx, fy) in enumerate(XY):
        tilt = ORIENT[i % len(ORIENT)]
        R = down @ pose_to_matrix([0, 0, 0, *tilt])[:3, :3]
        pos = [cx + fx * span_mm, cy + fy * span_mm, cz + HEIGHTS[i % len(HEIGHTS)]]
        out.append((f"xyz {[round(c) for c in pos]} tilt {tilt}", [*pos, *matrix_to_rpy_deg(R)]))
    return out


def auto_calibrate(prims, calibration, center: list[float] | None = None, span_mm: float = 100,
                   center_height_mm: float = 150, speed: int = 10, settle_s: float = 0.6,
                   target_points: int = 12, log: Callable[[str], None] = print) -> dict:
    """Record points until `target_points` (or candidates run out), then solve.
    Returns the solve report plus skipped poses and why. Ends at the centre."""
    tip = prims.get_state()["pose"]
    rz = tip[5]
    if center is None:
        z_min = (prims.table_z_mm + center_height_mm) if prims.table_z_mm is not None else tip[2]
        center = [tip[0], tip[1], max(tip[2], z_min)]
    cx, cy, cz = center

    # 1. straight up first (allowed even from below the safe floor), then gripper down at the centre
    while prims.get_state()["pose"][2] < cz - 1:
        lift = min(prims.m.max_relative_mm, cz - prims.get_state()["pose"][2])
        log(f"lift {lift:.0f} mm")
        prims.move_relative(dz=lift, speed=speed)
    log(f"centre {[round(c) for c in center]} (gripper down)")
    prims.move_to_pose(cx, cy, cz, 180, 0, rz, linear=True, speed=speed)

    skipped, recorded = [], 0
    for i, (label, pose) in enumerate(candidates(center, rz, span_mm), 1):
        if recorded >= target_points:
            break
        label = f"{i:2d} {label}"
        try:
            prims._ik(prims.flange_from_tip(pose))        # unreachable: skip without moving
            prims.move_to_pose(*pose, linear=True, speed=speed)
            time.sleep(settle_s)
            rec = calibration.record()
            recorded += 1
            log(f"{label}: ✓ point {rec['points']} (marker px {rec['marker_pixel']}, "
                f"depth {rec['depth_m']} m, noise {rec['noise_mm']} mm)")
        except ValidationError as exc:
            why = "unreachable" if "reachable" in str(exc) else str(exc)
            skipped.append({"pose": label, "why": why})
            log(f"{label}: skipped — {why}")
        except SafetyError as exc:
            if "camera view" not in str(exc):
                raise                                    # alarm / not ready: stop everything
            skipped.append({"pose": label, "why": "left the camera view (undone)"})
            log(f"{label}: skipped — left the camera view (move undone)")
        except RobotError as exc:
            if prims.get_state()["error"]:
                raise
            skipped.append({"pose": label, "why": str(exc)})
            log(f"{label}: skipped — {exc}")

    log("back to centre")
    prims.move_to_pose(cx, cy, cz, 180, 0, rz, linear=True, speed=speed)
    report = calibration.solve()
    report["skipped"] = skipped
    report["center"] = [round(c, 1) for c in center]
    report["rotation_spread_ok"] = report.get("rotation_spread_deg", 0) >= 30
    return report


def suggest_center(prims, max_radius_mm: float = 430.0) -> list[float] | None:
    """If the fingertip is near the edge of the reach, a centre pulled toward the base."""
    x, y = prims.get_state()["pose"][:2]
    r = float(np.hypot(x, y))
    if r <= max_radius_mm:
        return None
    s = max_radius_mm / r
    return [x * s, y * s]


class SelfCalibration:
    """Env camera -> robot base, collected and solved by the agent at runtime."""

    MIN_POINTS = 6
    # quality gate: a solve worse than this is NOT saved or used (camera moved, old points mixed in)
    MAX_RMS_MM = 10.0
    MAX_LOO_MEDIAN_MM = 15.0
    MIN_Z_SPREAD_MM = 60.0

    def __init__(self, scene, primitives, color: str = "red", min_area_px: float = 40,
                 session_file: Path = MARKER_SESSION_FILE):
        self.scene, self.p = scene, primitives
        self.color, self.min_area = color, min_area_px
        self.session_file = session_file
        self.points: list[dict] = []
        self.solution: dict | None = None     # accepted solve: marker offset + quality (for verify)
        if session_file.exists():
            data = json.loads(session_file.read_text(encoding="utf-8"))
            self.points, self.solution = data.get("points", []), data.get("solution")

    def _save_session(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(json.dumps({"points": self.points, "solution": self.solution}, indent=1) + "\n",
                                     encoding="utf-8")

    def reset(self) -> dict:
        """Forget all recorded points (e.g. the camera was moved). The active calibration
        stays until a new one is accepted or verify_calibration rejects it."""
        self.points = []
        self._save_session()
        return {"points": 0, "calibrated": self.scene.env_calibration is not None}

    def _z_spread_mm(self) -> float:
        zs = [p["flange_pose"][2] for p in self.points]
        return (max(zs) - min(zs)) if zs else 0.0

    def deactivate(self, reason: str) -> None:
        """Stop using the current calibration (file kept as *.rejected-<time>.json)."""
        self.scene.env_calibration = None
        self.solution = None
        self._save_session()
        path = self.scene.cfg.calibration_path
        if path.exists():
            path.rename(path.with_name(f"{path.stem}.rejected-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        log.warning("env calibration deactivated: %s", reason[:200])

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
        from calibration.hand_eye import solve_with_marker_offset

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
        if len(self.points) >= 3 and self._z_spread_mm() < self.MIN_Z_SPREAD_MM:
            tips.append(f"vary the HEIGHT more (z spread {self._z_spread_mm():.0f} mm, want ≥ {self.MIN_Z_SPREAD_MM:.0f}): "
                        "include low points just above the approach zone near the box, else cube heights come out wrong")
        tips.append("spread points over the area you will work in (around the box and the free table)")
        return "; ".join(tips)

    def solve(self) -> dict:
        from calibration.hand_eye import solve_with_marker_offset

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
        loo_median = float(np.median(loo))
        if rms > self.MAX_RMS_MM or loo_median > self.MAX_LOO_MEDIAN_MM:
            active = "the previous calibration stays active" if self.scene.env_calibration is not None \
                else "nothing is calibrated"
            raise ValidationError(
                f"calibration REJECTED (not saved, {active}): rms {rms:.1f} mm, leave-one-out median "
                f"{loo_median:.1f} mm (limits {self.MAX_RMS_MM:.0f} / {self.MAX_LOO_MEDIAN_MM:.0f}) over {len(keep)} points. "
                "Points from before a camera move are probably mixed in, or the poses are too similar: "
                "reset_calibration, record 8–12 NEW points spread in XY, height (≥ 60 mm) and orientation, "
                "solve again, then verify_calibration at a pose near the box.")
        calib = HandEye(T=T, eye_in_hand=False)
        calib.save(self.scene.cfg.calibration_path)
        self.scene.env_calibration = calib
        self.solution = {"marker_offset_mm": [round(float(v), 2) for v in offset_mm], "rms_mm": round(float(rms), 2),
                         "loo_median_mm": round(loo_median, 2), "points": len(keep), "t": time.time()}
        self._save_session()
        worst = int(np.argmax(errors))
        return {"saved": str(self.scene.cfg.calibration_path), "points": len(keep),
                "recorded": len(self.points), "dropped_outliers": dropped,
                "rms_mm": round(float(rms), 1), "worst_point": worst, "worst_error_mm": round(float(errors[worst]), 1),
                "marker_offset_in_flange_mm": [round(float(o), 1) for o in offset_mm],
                "rotation_spread_deg": round(spread, 1),
                "leave_one_out_mm": {"median": round(float(np.median(loo)), 1), "max": round(float(np.max(loo)), 1)},
                "z_spread_mm": round(self._z_spread_mm(), 1),
                "note": ("leave_one_out = error on a point when solved WITHOUT it: the honest accuracy. "
                         "find_objects / look now report base_xyz_mm. RMS on the points themselves is "
                         "optimistic — check it with verify_calibration at a pose that was not recorded (near the box)."
                         + (" Rotation spread < 30°: the result may be unreliable." if spread < 30 else ""))}

    def verify(self, samples: int = 5, max_error_mm: float | None = None) -> dict:
        """Check the ACTIVE calibration at the current pose: where the env camera sees the
        marker vs where the robot says it is. No motion. A large error means the camera
        moved or the solve was bad -> the calibration is deactivated (look then says
        calibrated=false) so no one targets with wrong coordinates."""
        calib = self.scene.env_calibration
        if calib is None:
            return {"calibration_valid": False, "calibrated": False,
                    "advice": "no active calibration: record 8–12 varied points, then solve_calibration"}
        if not self.solution:
            return {"calibration_valid": None, "calibrated": True,
                    "advice": "calibration has no stored marker offset (solved before verify existed): "
                              "reset_calibration and calibrate again to be able to verify it"}
        state = self.p.get_state()
        if state["moving"]:
            raise ValidationError("robot is moving — verify only when it stands still")
        cam, pixel, noise_mm = self._measure(samples)
        P = pose_to_matrix(state["flange_pose"])
        actual = P[:3, 3] + P[:3, :3] @ (np.array(self.solution["marker_offset_mm"]) / 1000.0)
        T = calib.T
        predicted = T[:3, :3] @ np.array(cam) + T[:3, 3]
        err = (predicted - actual) * 1000.0
        limit = float(max_error_mm or max(15.0, 3 * float(self.solution.get("loo_median_mm", 5.0))))
        dist = float(np.linalg.norm(err))
        out = {"error_mm": round(dist, 1), "error_xyz_mm": [round(float(e), 1) for e in err], "limit_mm": round(limit, 1),
               "marker_pixel": pixel, "noise_mm": round(noise_mm, 1)}
        if dist <= limit:
            return {"calibration_valid": True, "calibrated": True, **out,
                    "advice": "calibration agrees with the robot here: camera coordinates are usable (±a few cm)"}
        self.deactivate(f"verify error {dist:.0f} mm > {limit:.0f} mm")
        return {"calibration_valid": False, "calibrated": False, "deactivated": True, **out,
                "advice": "the camera disagrees with the robot by more than the limit: the camera probably moved. "
                          "The calibration was switched off. reset_calibration, record 8–12 new varied points, "
                          "solve_calibration, then verify_calibration again before picking."}
