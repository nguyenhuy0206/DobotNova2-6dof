"""Scripted env-camera calibration: the robot visits poses by itself (no model in the loop).

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

import time
from typing import Callable

import numpy as np

from common.errors import RobotError, SafetyError, ValidationError
from perception.transforms import matrix_to_rpy_deg, pose_to_matrix

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
