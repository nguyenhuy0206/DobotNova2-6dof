"""Hand-eye calibration: where a camera is relative to the robot.

    hand_eye                 camera on the flange, a board fixed in the cell: T_flange_cam and
                             T_base_board from many (flange pose, board corners) views
    env_from_board           a fixed camera that sees the same board: T_base_env through the board
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calibration.board import BoardSpec, Detection, board_pose, camera_matrix, object_points
from calibration.geometry import inv, nearest_rotation, to_rvec_t, transform

@dataclass
class HandEyeResult:
    T_flange_cam: np.ndarray
    T_base_board: np.ndarray
    reproj_px: float                      # RMS over all corners of all views (after refinement)
    per_view_px: list[float] = field(default_factory=list)
    views: int = 0

def _project_all(params, T_base_flange, dets, spec, K, dist):
    T_flange_cam, T_base_board = transform(params[:3], params[3:6]), transform(params[6:9], params[9:12])
    T_cam_flange = inv(T_flange_cam)
    out = []
    for Tbf, d in zip(T_base_flange, dets):
        T_cam_board = T_cam_flange @ inv(Tbf) @ T_base_board
        r, t = to_rvec_t(T_cam_board)[:3], T_cam_board[:3, 3]
        proj, _ = cv2.projectPoints(object_points(spec, d.ids), r, t, K, dist)
        out.append(proj.reshape(-1, 2) - d.corners)
    return out

def hand_eye_linear(T_base_flange: list[np.ndarray], T_cam_board: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed form for  T_base_flange_i · X · T_cam_board_i = Z  (X = T_flange_cam, Z = T_base_board),
    after Li et al. 2010: rotations from the null space of a Kronecker system, then translations by
    linear least squares. Needs >= 3 views with rotations about different axes."""
    rows = [np.hstack([np.kron(Tc[:3, :3].T, Tb[:3, :3]), -np.eye(9)]) for Tb, Tc in zip(T_base_flange, T_cam_board)]
    v = np.linalg.svd(np.vstack(rows))[2][-1]
    RX, RZ = v[:9].reshape(3, 3, order="F"), v[9:].reshape(3, 3, order="F")
    s = np.cbrt(np.linalg.det(RX))                 # null vector has arbitrary scale and sign
    RX, RZ = nearest_rotation(RX / s), nearest_rotation(RZ / s)
    A = np.vstack([np.hstack([Tb[:3, :3], -np.eye(3)]) for Tb in T_base_flange])
    b = np.concatenate([-Tb[:3, 3] - Tb[:3, :3] @ RX @ Tc[:3, 3] for Tb, Tc in zip(T_base_flange, T_cam_board)])
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    X, Z = np.eye(4), np.eye(4)
    X[:3, :3], X[:3, 3], Z[:3, :3], Z[:3, 3] = RX, t[:3], RZ, t[3:]
    return X, Z

def hand_eye(T_base_flange: list[np.ndarray], dets: list[Detection], spec: BoardSpec, K, dist,
             iterations: int = 30) -> HandEyeResult:
    """Board fixed in the cell, camera on the flange. Closed form (hand_eye_linear) from the
    per-view board poses, then Levenberg–Marquardt on the reprojection error of every corner of
    every view — the PnP poses alone are noisy in rotation for a small board."""
    if len(dets) < 3:
        raise ValueError("need >= 3 views")
    K, dist = camera_matrix(K), np.asarray(dist, np.float64).ravel()
    T_cam_board = [board_pose(d, spec, K, dist)[0] for d in dets]
    X, Z = hand_eye_linear(T_base_flange, T_cam_board)

    x = np.r_[to_rvec_t(X), to_rvec_t(Z)]

    def residual(p):
        return np.concatenate([r.ravel() for r in _project_all(p, T_base_flange, dets, spec, K, dist)])

    r, lam = residual(x), 1e-3
    for _ in range(iterations):
        J = np.empty((r.size, 12))
        for k in range(12):
            dp = np.zeros(12)
            dp[k] = 1e-6
            J[:, k] = (residual(x + dp) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = residual(x + step)
            if r_new @ r_new < r @ r:
                x, r, lam = x + step, r_new, max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e8:
                break
        if lam > 1e8 or np.linalg.norm(step) < 1e-10:
            break
    per = [float(np.sqrt((e ** 2).sum(1).mean())) for e in _project_all(x, T_base_flange, dets, spec, K, dist)]
    return HandEyeResult(transform(x[:3], x[3:6]), transform(x[6:9], x[9:12]), float(np.sqrt((r ** 2).sum() / (r.size / 2))),
                         per, len(dets))

def env_from_board(T_base_board: np.ndarray, T_env_board: np.ndarray) -> np.ndarray:
    """T_base_env."""
    return T_base_board @ inv(T_env_board)

def depth_check_mm(det: Detection, spec: BoardSpec, T_cam_board: np.ndarray, depth_m: np.ndarray) -> float | None:
    """Median (depth camera z − board-pose z) over the detected corners, mm. Tells whether the
    board pose (from the image) and the depth image agree on the distance."""
    if depth_m is None:
        return None
    pts = object_points(spec, det.ids) @ T_cam_board[:3, :3].T + T_cam_board[:3, 3]
    diffs = []
    for (u, v), p in zip(det.corners, pts):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < depth_m.shape[0] and 0 <= ui < depth_m.shape[1] and depth_m[vi, ui] > 0:
            diffs.append(depth_m[vi, ui] - p[2])
    return round(float(np.median(diffs)) * 1000, 1) if diffs else None
