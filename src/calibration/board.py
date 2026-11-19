"""ChArUco board: detection, board pose and camera intrinsics.

The board in the cell (config/perception.yaml calibration.board): 7x5 squares, DICT_5X5_100,
printed smaller than the PDF, so the measured sizes (square 20.85 mm, marker 15 mm) are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from calibration.geometry import transform

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

def camera_matrix(K) -> np.ndarray:
    return np.asarray(K, np.float64).reshape(3, 3)

def board_pose(det: Detection, spec: BoardSpec, K, dist=None) -> tuple[np.ndarray, float]:
    """T_cam_board and RMS reprojection error (px)."""
    obj, img = object_points(spec, det.ids), det.corners.astype(np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
    ok, rv, tv = cv2.solvePnP(obj, img, camera_matrix(K), dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
    rv, tv = cv2.solvePnPRefineLM(obj, img, camera_matrix(K), dist, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, camera_matrix(K), dist)
    err = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    return transform(rv, tv), err

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
