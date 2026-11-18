"""ChArUco board: detection, board pose, intrinsics, and the two camera calibrations built on it.

One board lying flat on the table, seen by both cameras:

    wrist (eye-in-hand)   the robot looks at the board from many poses
                          -> wrist intrinsics, flange->camera, base->board        (hand_eye)
    env   (eye-to-hand)   the D455 sees the same board without moving
                          -> env->board; base->env = base->board · board->env    (env_from_board)

Every transform here is a 4x4 in METRES (as perception.transforms.pose_to_matrix), named
T_a_b = pose of frame b in frame a (maps a point in b to a).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_CORNERS = 6          # fewer and a planar board pose is unreliable


@dataclass
class BoardSpec:
    squares: tuple[int, int] = (7, 5)        # (x, y) squares
    square_mm: float = 20.85
    marker_mm: float = 15.0
    dictionary: str = "DICT_5X5_100"

    def board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(tuple(self.squares), self.square_mm / 1000, self.marker_mm / 1000, d)

    @property
    def center_m(self) -> np.ndarray:
        """Board centre in the board frame (origin = a corner of the first square)."""
        return np.array([self.squares[0] * self.square_mm / 2000, self.squares[1] * self.square_mm / 2000, 0.0])

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSpec":
        return cls(tuple(d.get("squares", (7, 5))), float(d.get("square_mm", 20.85)),
                   float(d.get("marker_mm", 15.0)), str(d.get("dictionary", "DICT_5X5_100")))


@dataclass
class Detection:
    corners: np.ndarray          # (N, 2) float32 pixels
    ids: np.ndarray              # (N,) int, ChArUco corner ids
    markers: int                 # ArUco markers found
    image_size: tuple[int, int]  # (w, h)

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict:
        return {"corners": self.corners.round(3).tolist(), "ids": self.ids.tolist(), "markers": self.markers,
                "image_size": list(self.image_size)}

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(np.asarray(d["corners"], np.float32), np.asarray(d["ids"], int), int(d["markers"]),
                   tuple(d["image_size"]))


def _params(tuned: bool):
    p = cv2.aruco.DetectorParameters()
    if tuned:           # small / oblique boards (the D455 sees the table at a grazing angle)
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 53, 4
        p.minMarkerPerimeterRate = 0.01
        p.polygonalApproxAccuracyRate = 0.05
        p.perspectiveRemovePixelPerCell = 8
        p.perspectiveRemoveIgnoredMarginPerCell = 0.2
        p.errorCorrectionRate = 0.8
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def detect(image: np.ndarray, spec: BoardSpec, scales=(1, 2)) -> Detection | None:
    """Best ChArUco detection over plain/tuned parameters and upscaling; None if < MIN_CORNERS."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    board, best = spec.board(), None
    for scale in scales:
        g = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for tuned in (False, True):
            cc, ci, _, mi = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), _params(tuned)).detectBoard(g)
            n = 0 if ci is None else len(ci)
            if n and (best is None or n > len(best)):
                best = Detection(cc.reshape(-1, 2).astype(np.float32) / scale, ci.ravel().astype(int),
                                 0 if mi is None else len(mi), (w, h))
        if best is not None and len(best) >= 0.8 * (spec.squares[0] - 1) * (spec.squares[1] - 1):
            break                                         # nearly all corners: no need to upscale
    return best if best is not None and len(best) >= MIN_CORNERS else None


def object_points(spec: BoardSpec, ids) -> np.ndarray:
    return spec.board().getChessboardCorners()[np.asarray(ids, int)].astype(np.float64)


def _K(K) -> np.ndarray:
    return np.asarray(K, np.float64).reshape(3, 3)


def board_pose(det: Detection, spec: BoardSpec, K, dist=None) -> tuple[np.ndarray, float]:
    """T_cam_board and RMS reprojection error (px)."""
    obj, img = object_points(spec, det.ids), det.corners.astype(np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
    ok, rv, tv = cv2.solvePnP(obj, img, _K(K), dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
    rv, tv = cv2.solvePnPRefineLM(obj, img, _K(K), dist, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, _K(K), dist)
    err = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    return _T(rv, tv), err


def calibrate_intrinsics(dets: list[Detection], spec: BoardSpec) -> tuple[np.ndarray, np.ndarray, float]:
    """(K, dist, RMS px) from >= 4 views. Distortion: k1, k2, p1, p2 (k3 fixed: too few views for it)."""
    views = [d for d in dets if len(d) >= MIN_CORNERS]
    if len(views) < 4:
        raise ValueError(f"need >= 4 views with >= {MIN_CORNERS} corners (have {len(views)})")
    obj = [object_points(spec, d.ids).astype(np.float32) for d in views]
    img = [d.corners.reshape(-1, 1, 2) for d in views]
    w, h = views[0].image_size
    K0 = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (w, h), K0, np.zeros(5), flags=flags)
    return K, dist.ravel(), float(rms)


# --- rigid transform helpers ----------------------------------------------------------------

def _T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).ravel()
    return T


def _vec(T: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


# --- wrist: eye-in-hand ------------------------------------------------------------------------

@dataclass
class HandEyeResult:
    T_flange_cam: np.ndarray
    T_base_board: np.ndarray
    reproj_px: float                      # RMS over all corners of all views (after refinement)
    per_view_px: list[float] = field(default_factory=list)
    views: int = 0


def _project_all(params, T_base_flange, dets, spec, K, dist):
    T_flange_cam, T_base_board = _T(params[:3], params[3:6]), _T(params[6:9], params[9:12])
    T_cam_flange = inv(T_flange_cam)
    out = []
    for Tbf, d in zip(T_base_flange, dets):
        T_cam_board = T_cam_flange @ inv(Tbf) @ T_base_board
        r, t = _vec(T_cam_board)[:3], T_cam_board[:3, 3]
        proj, _ = cv2.projectPoints(object_points(spec, d.ids), r, t, K, dist)
        out.append(proj.reshape(-1, 2) - d.corners)
    return out


def _so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def hand_eye_linear(T_base_flange: list[np.ndarray], T_cam_board: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Closed form for  T_base_flange_i · X · T_cam_board_i = Z  (X = T_flange_cam, Z = T_base_board),
    after Li et al. 2010: rotations from the null space of a Kronecker system, then translations by
    linear least squares. Needs >= 3 views with rotations about different axes."""
    rows = [np.hstack([np.kron(Tc[:3, :3].T, Tb[:3, :3]), -np.eye(9)]) for Tb, Tc in zip(T_base_flange, T_cam_board)]
    v = np.linalg.svd(np.vstack(rows))[2][-1]
    RX, RZ = v[:9].reshape(3, 3, order="F"), v[9:].reshape(3, 3, order="F")
    s = np.cbrt(np.linalg.det(RX))                 # null vector has arbitrary scale and sign
    RX, RZ = _so3(RX / s), _so3(RZ / s)
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
    K, dist = _K(K), np.asarray(dist, np.float64).ravel()
    T_cam_board = [board_pose(d, spec, K, dist)[0] for d in dets]
    X, Z = hand_eye_linear(T_base_flange, T_cam_board)

    x = np.r_[_vec(X), _vec(Z)]

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
    return HandEyeResult(_T(x[:3], x[3:6]), _T(x[6:9], x[9:12]), float(np.sqrt((r ** 2).sum() / (r.size / 2))),
                         per, len(dets))


# --- env: eye-to-hand through the board ----------------------------------------------------------

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


def average_poses(Ts: list[np.ndarray]) -> np.ndarray:
    """Mean of nearby rigid transforms (chordal rotation mean)."""
    R = sum(T[:3, :3] for T in Ts)
    U, _, Vt = np.linalg.svd(R)
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return out
