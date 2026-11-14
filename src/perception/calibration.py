"""Camera -> robot-base calibration from matched 3D points (eye-to-hand).

Record N >= 4 pairs: where the camera sees a marker (camera frame, metres) and
where the robot says the marker is (base frame, mm — the TCP plus the gripper
offset). ``solve_rigid_transform`` finds the rotation + translation that maps
one onto the other (Kabsch / SVD), so every later detection can be turned into
a robot coordinate. scripts/calibrate_env_camera.py collects the pairs.
"""
from __future__ import annotations

import numpy as np


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


def solve_with_marker_offset(camera_pts_m, tcp_poses, initial_offset_mm=None,
                             iterations: int = 20) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Calibrate when the marker's exact place on the gripper is unknown.

    Model, for every recorded point i (metres):
        R @ c_i + t  =  p_i + R_i @ o
    c_i = marker seen by the camera, (R_i, p_i) = robot TCP pose, o = marker in
    the tool frame, (R, t) = camera -> base. Solved directly: one linear least
    squares with R relaxed to any 3x3 matrix, R snapped to the nearest rotation,
    then (t, o) re-solved and refined. Needs >= 5 points with varied tool
    rotations, otherwise ``o`` and ``t`` cannot be told apart.
    ``initial_offset_mm`` is accepted for API compatibility and ignored.

    Returns (T, offset_mm, rms_mm, per_point_error_mm).
    """
    from perception.transforms import pose_to_matrix

    cam = np.asarray(camera_pts_m, float)
    if len(cam) < 5:
        raise ValueError("need >= 5 points")
    poses = [pose_to_matrix(p) for p in tcp_poses]
    Rs = [P[:3, :3] for P in poses]
    ps = np.array([P[:3, 3] for P in poses])
    I3 = np.eye(3)

    # 1) linear: unknowns vec(M) (9, column-major), t (3), o (3)
    A = np.vstack([np.hstack([np.kron(c, I3), I3, -R]) for c, R in zip(cam, Rs)])
    x = np.linalg.lstsq(A, ps.reshape(-1), rcond=None)[0]
    M = x[:9].reshape(3, 3, order="F")
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    def solve_t_o(R):
        B = np.vstack([np.hstack([I3, -Ri]) for Ri in Rs])
        rhs = (ps - cam @ R.T).reshape(-1)
        y = np.linalg.lstsq(B, rhs, rcond=None)[0]
        return y[:3], y[3:]

    t, o = solve_t_o(R)
    # 2) refine: Kabsch for (R, t) given o, then (t, o) given R
    for _ in range(iterations):
        base = ps + np.array([Ri @ o for Ri in Rs])
        T, _ = solve_rigid_transform(cam, base * 1000.0)
        R = T[:3, :3]
        t_new, o_new = solve_t_o(R)
        done = np.linalg.norm(o_new - o) < 1e-7
        t, o = t_new, o_new
        if done:
            break

    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    base = ps + np.array([Ri @ o for Ri in Rs])
    errors = np.linalg.norm((cam @ R.T + t) - base, axis=1) * 1000.0
    rms = float(np.sqrt((errors ** 2).mean()))
    return T, o * 1000.0, rms, errors


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
