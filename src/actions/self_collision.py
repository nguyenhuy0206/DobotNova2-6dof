"""Self-collision guard: the gripper must stay clear of the robot's own elbow, upper arm and forearm,
and the wrist must not fold onto the upper arm.

The controller's IK only says whether a pose is reachable; it happily plans a path that folds
the arm until the gripper hits the elbow (17 Nov: auto_calibrate, tilted pose above the box ->
controller alarm, gripper against the elbow). This module checks that BEFORE a move is sent.

Kinematics: drivers.nova2_kinematics (DH fitted to this controller, 0.04 mm).

Clearance = shortest distance between the gripper axis (flange -> fingertip) and the axis of
the shoulder->elbow and elbow->wrist links. It is a coarse measure (the real gripper is wider on
the side of its fingers and wrist camera), so the limit is set from what actually happened:
  * 564 poses from the episodes, all without alarm: >= 157 mm
  * auto_calibrate 17 Nov (replayed with this model; the replay passes through the measured
    alarm joints within 1°): poses 2 and 4 ran fine at 79 / 80 mm; pose 6 (the target of the
    move that hit the elbow) is at 53 mm — the gripper axis 13 mm from the elbow joint axis,
    i.e. inside the elbow. Contact happened on the way there (at 96 mm, 67 % of the path).
motion.self_clearance_mm (default 65, between 53 and 79) is the limit.

Wrist clearance = shortest distance between the wrist links (J4 -> J5 -> J6 -> flange axes)
and the upper arm (shoulder -> elbow). It shrinks when the elbow folds (J3 far negative):
  * place_block_on_table 17 Nov: a straight move toward (-200, -250) near the base folded the
    elbow; the arm stopped with an alarm at 183.5 mm (J3 -120.5°), the target needed 166 mm.
  * 38 poses from the episodes with the elbow folded as deep (J3 <= -100°), all without alarm:
    >= 196.8 mm (J3 never below -119.0°)
  * with the elbow open the measure means little: a tilted calibration pose (J3 -88°) ran fine
    at 180 mm, the wrist only reaches the upper arm when the elbow is folded.
So it is checked only while J3 <= motion.wrist_fold_deg (default -100°), with the limit
motion.wrist_clearance_mm (default 190, between 183.5 and 196.8).

A move is refused when its path or target goes below a limit AND below the clearance it starts
with, so a move that only increases the clearance is always allowed (the arm can always get out).
"""
from __future__ import annotations

import math

import numpy as np

from common.errors import ValidationError
from drivers.nova2_kinematics import flange_pose, frames, ik_near  # noqa: F401 (flange_pose re-exported)
from common.poses import matrix_to_rpy_deg, pose_to_matrix

def _seg_dist(p1, q1, p2, q2) -> float:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > 1e-9 else 0.0
    t = min(1.0, max(0.0, (b * s + f) / e)) if e > 1e-9 else 0.0
    s = min(1.0, max(0.0, (b * t - c) / a)) if a > 1e-9 else 0.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))

def clearance(joints_deg, tool_length_mm: float) -> tuple[float, str]:
    """(mm, link): closest approach of the gripper axis to the upper arm / forearm."""
    F, p = frames(joints_deg)
    flange = F[:3, 3]
    tip = flange + F[:3, 2] * tool_length_mm
    upper = _seg_dist(flange, tip, p[0], p[1])
    fore = _seg_dist(flange, tip, p[1], p[2])
    return (upper, "upper arm / elbow") if upper <= fore else (fore, "forearm / elbow")

def wrist_clearance(joints_deg) -> float:
    """mm between the wrist links and the upper arm (shrinks as the elbow folds)."""
    F, p = frames(joints_deg)
    wrist = ((p[2], p[3]), (p[3], p[4]), (p[4], F[:3, 3]))
    return min(_seg_dist(a, b, p[0], p[1]) for a, b in wrist)

WRIST_FOLD_DEG = -100.0          # J3 at or below this = elbow folded deep enough for the wrist check

def _measure(q, tool_length_mm: float, fold_deg: float = WRIST_FOLD_DEG) -> dict:
    mm, link = clearance(q, tool_length_mm)
    w = wrist_clearance(q) if q[2] <= fold_deg else math.inf
    return {"gripper": (mm, f"the gripper would come within {mm:.0f} mm of the robot's {link}"),
            "wrist": (w, f"the elbow folds to J3 {q[2]:.0f}° and the wrist comes within {w:.0f} mm of the "
                         "upper arm")}

def _floors(q, tool_length_mm: float, limits: dict) -> dict:
    now = _measure(q, tool_length_mm)
    return {k: min(lim, now[k][0] - 1) for k, lim in limits.items() if lim > 0}

def check_joint_path(start, end, tool_length_mm: float, min_mm: float, step_deg: float = 2.0,
                     wrist_min_mm: float = 0.0) -> None:
    """Joint-interpolated motion (JointMovJ / MovJ)."""
    limits = {"gripper": min_mm, "wrist": wrist_min_mm}
    floors = _floors(start, tool_length_mm, limits)
    span = max(abs(e - s) for s, e in zip(start, end))
    n = max(1, math.ceil(span / step_deg))
    for k in range(1, n + 1):
        q = [s + (e - s) * k / n for s, e in zip(start, end)]
        _check(q, tool_length_mm, floors, limits, "target" if k == n else f"joint path {k * 100 // n}%")

def check_linear_path(start_joints, start_flange: list[float], target_flange: list[float],
                      tool_length_mm: float, min_mm: float, step_mm: float = 10.0, step_deg: float = 3.0,
                      wrist_min_mm: float = 0.0) -> None:
    """Straight-line motion (MovL): follow the line with IK from the current joints."""
    from common.poses import interpolate_rotation
    p0, p1 = np.array(start_flange[:3]), np.array(target_flange[:3])
    R0, R1 = pose_to_matrix(start_flange)[:3, :3], pose_to_matrix(target_flange)[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R1) - 1) / 2))))
    n = max(1, math.ceil(max(np.linalg.norm(p1 - p0) / step_mm, ang / step_deg)))
    q = list(start_joints)
    limits = {"gripper": min_mm, "wrist": wrist_min_mm}
    floors = _floors(q, tool_length_mm, limits)
    for k in range(1, n + 1):
        s = k / n
        pose = [*map(float, p0 + (p1 - p0) * s), *matrix_to_rpy_deg(interpolate_rotation(R0, R1, s))]
        nxt = ik_near(pose, q)
        if nxt is None:
            return            # the controller's own IK / reachability check reports this case
        q = nxt
        _check(q, tool_length_mm, floors, limits, "target" if k == n else f"path {k * 100 // n}%")

def _check(q, tool_length_mm: float, floors: dict, limits: dict, where: str) -> None:
    now = _measure(q, tool_length_mm)
    for key, floor in floors.items():
        mm, text = now[key]
        if mm < floor:
            raise ValidationError(f"self-collision ({where}): {text} (limit {limits[key]:g} mm) — the arm would "
                                  "fold onto itself there. Not moved.")
