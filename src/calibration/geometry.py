"""Rigid-transform helpers shared by every calibration method.

All 4x4 transforms are in METRES (as perception.transforms.pose_to_matrix), named T_a_b = pose
of frame b in frame a (maps a point in b to a). Point sets given in mm say so in their names.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

def transform(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).ravel()
    return T

def to_rvec_t(T: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]

def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti

def rot_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))

def nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

def average_poses(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean of nearby rigid transforms (chordal rotation mean)."""
    R = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(R)
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out

def solve_rigid_transform(camera_pts_m: np.ndarray, base_pts_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (T 4x4 camera->base in metres, RMS error in mm)."""
    a = np.asarray(camera_pts_m, float)
    b = np.asarray(base_pts_mm, float) / 1000.0
    if a.shape != b.shape or a.shape[0] < 4 or a.shape[1] != 3:
        raise ValueError("need >= 4 matching 3D point pairs")
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    H = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))          # guard against a reflection
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    residual = (a @ R.T + t) - b
    rms_mm = float(np.sqrt((residual ** 2).sum(axis=1).mean()) * 1000.0)
    return T, rms_mm

def outliers(errors_mm: np.ndarray, factor: float = 2.5, floor_mm: float = 10.0) -> list[int]:
    """Indices whose error is far above the rest (and above an absolute floor)."""
    med = float(np.median(errors_mm))
    return [i for i, e in enumerate(errors_mm) if e > max(factor * med, floor_mm)]

def solve_points(camera_pts_m, base_pts_mm, drop_outliers: bool = True, min_points: int = 5):
    """Plain point-pair calibration (marker position known, e.g. touch calibration).

    Returns (T, rms_mm, per_point_error_mm with NaN for dropped points, dropped indices).
    """
    cam = np.asarray(camera_pts_m, float)
    base = np.asarray(base_pts_mm, float)
    T, rms = solve_rigid_transform(cam, base)
    errors = np.linalg.norm((cam @ T[:3, :3].T + T[:3, 3]) * 1000.0 - base, axis=1)
    dropped = outliers(errors) if drop_outliers else []
    if dropped and len(cam) - len(dropped) >= min_points:
        keep = [i for i in range(len(cam)) if i not in dropped]
        T, rms = solve_rigid_transform(cam[keep], base[keep])
        errors = np.full(len(cam), np.nan)
        errors[keep] = np.linalg.norm((cam[keep] @ T[:3, :3].T + T[:3, 3]) * 1000.0 - base[keep], axis=1)
    else:
        dropped = []
    return T, rms, errors, dropped

def leave_one_out_errors(camera_pts_m, base_pts_mm) -> np.ndarray:
    """Error (mm, xyz) on each point when the calibration is solved WITHOUT it —
    an honest estimate of accuracy on spots the calibration has never seen."""
    cam = np.asarray(camera_pts_m, float)
    base = np.asarray(base_pts_mm, float)
    if len(cam) < 5:
        raise ValueError("leave-one-out needs >= 5 points")
    errs = []
    for i in range(len(cam)):
        keep = [j for j in range(len(cam)) if j != i]
        T, _ = solve_rigid_transform(cam[keep], base[keep])
        errs.append((cam[i] @ T[:3, :3].T + T[:3, 3]) * 1000.0 - base[i])
    return np.array(errs)
