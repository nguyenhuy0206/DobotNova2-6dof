"""Nova 2 kinematics: joints -> flange pose (FK) and back near a given configuration (IK).

Standard DH, fitted (17 Nov 2026) to 564 (joints, flange pose) pairs measured by this controller:
flange position error median 0.04 mm / max 0.09 mm, orientation median 0.004° / max 0.017°. The
lengths come out at the nominal Nova 2 values (d1 223.4, a2 280, a3 225, d4 117.5, d5 120, d6 88 mm).
The controller's own PositiveSolution / InverseSolution stay authoritative for motion; this model
serves the checks that need many poses fast (self-collision along a path) and offline geometry
(the touch survey). 18 Nov: still matches the controller to 0.05 mm on new poses.
"""
from __future__ import annotations

import math

import numpy as np

from common.poses import matrix_to_rpy_deg, pose_to_matrix

# fitted DH (see module doc): d1, a2, a3, d4, d5, d6 (mm), joint offsets (rad), joint signs,
# and the constant rotation from the DH end frame to the controller's flange frame
_D1, _A2, _A3, _D4, _D5, _D6 = 223.41, 280.01, 225.01, -117.53, 119.98, -88.02
_OFFSET = np.radians([180.0, 90.0, 0.0, 90.0, 0.0, 168.96])
_SIGN = np.array([1, -1, -1, -1, 1, -1], float)
_ALPHA = np.radians([90, 0, 0, 90, -90, 0])
_A = [0, _A2, _A3, 0, 0, 0]
_D = [_D1, 0, 0, _D4, _D5, _D6]
_FLANGE = np.array([[0.9815, 0.1915, 0.0], [0.1915, -0.9815, 0.0], [0.0, 0.0, -1.0]])


def frames(joints_deg) -> tuple[np.ndarray, list[np.ndarray]]:
    """Flange pose (4x4, controller convention) and the origins of the 6 DH frames."""
    T = np.eye(4)
    pts = []
    for i, q in enumerate(np.radians(np.asarray(joints_deg, float))):
        th = _SIGN[i] * q + _OFFSET[i]
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(_ALPHA[i]), math.sin(_ALPHA[i])
        T = T @ np.array([[ct, -st * ca, st * sa, _A[i] * ct],
                          [st, ct * ca, -ct * sa, _A[i] * st],
                          [0, sa, ca, _D[i]],
                          [0, 0, 0, 1]])
        pts.append(T[:3, 3].copy())
    F = T.copy()
    F[:3, :3] = T[:3, :3] @ _FLANGE
    return F, pts

def flange_pose(joints_deg) -> list[float]:
    F, _ = frames(joints_deg)
    return [*map(float, F[:3, 3]), *matrix_to_rpy_deg(F[:3, :3])]

def ik_near(target: list[float], joints_deg, iters: int = 40) -> list[float] | None:
    """Joints for a flange pose, found from `joints_deg` (the branch a straight-line move
    stays on). Damped least squares on position (mm) + orientation (rad·100)."""
    q = np.asarray(joints_deg, float).copy()
    Tt = np.eye(4)
    Tt[:3, :3] = pose_to_matrix(target)[:3, :3]        # rotation only: pose_to_matrix works in metres
    Tt[:3, 3] = target[:3]
    for _ in range(iters):
        F, _ = frames(q)
        e = _pose_error(F, Tt)
        if np.linalg.norm(e[:3]) < 0.05 and np.linalg.norm(e[3:]) < 0.005:
            return list(map(float, q))
        Jm = np.empty((6, 6))
        for k in range(6):
            dq = np.zeros(6)
            dq[k] = 1e-3
            Jm[:, k] = (_pose_error(frames(q + dq)[0], Tt) - e) / 1e-3
        q = q - np.linalg.solve(Jm.T @ Jm + 1e-4 * np.eye(6), Jm.T @ e)
    F, _ = frames(q)
    e = _pose_error(F, Tt)
    return list(map(float, q)) if np.linalg.norm(e[:3]) < 0.5 and np.linalg.norm(e[3:]) < 0.05 else None

def _pose_error(F: np.ndarray, Tt: np.ndarray) -> np.ndarray:
    Rerr = Tt[:3, :3].T @ F[:3, :3]
    w = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
    return np.r_[F[:3, 3] - Tt[:3, 3], 100 * w]
