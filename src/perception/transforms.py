"""Pixel → camera → robot-base transforms.

Conventions: camera frame in metres (RealSense: +X right, +Y down, +Z forward);
robot base frame in millimetres, as reported by Nova 2 feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.paths import MEMORY_DIR

HAND_EYE_FILE = MEMORY_DIR / "hand_eye.json"


def deproject_pixel(u: float, v: float, depth_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole deprojection; returns [x, y, z] in metres in the camera frame."""
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m])


def pose_to_matrix(pose: list[float]) -> np.ndarray:
    """Dobot pose [X, Y, Z mm, Rx, Ry, Rz deg] (XYZ-fixed / Rz·Ry·Rx) → 4x4 (metres)."""
    x, y, z, rx, ry, rz = pose
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


def matrix_to_rpy_deg(R: np.ndarray) -> list[float]:
    """Inverse of pose_to_matrix's rotation (R = Rz·Ry·Rx) -> [Rx, Ry, Rz] degrees."""
    ry = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return [float(v) for v in np.degrees([rx, ry, rz])]


def rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    """Angle of the rotation taking R0 to R1."""
    c = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def interpolate_rotation(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Shortest-path rotation from R0 (s=0) to R1 (s=1), via axis–angle (Rodrigues)."""
    Rd = R0.T @ R1
    angle = np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return R0.copy()
    if np.pi - angle < 1e-6:     # 180°: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(Rd) + 1.0) / 2.0, 0.0, None))
    else:
        axis = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / (2 * np.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = angle * s
    return R0 @ (np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k)


@dataclass
class HandEye:
    """Camera extrinsics from a hand-eye calibration.

    eye_in_hand=True:  T is tool→camera (camera mounted on the flange)
    eye_in_hand=False: T is base→camera (camera fixed in the cell)
    """
    T: np.ndarray = field(default_factory=lambda: np.eye(4))
    eye_in_hand: bool = True

    def camera_to_base(self, point_cam_m: np.ndarray, tcp_pose: list[float] | None = None) -> np.ndarray:
        """Return the point in robot base frame, millimetres."""
        p = np.append(point_cam_m, 1.0)
        if self.eye_in_hand:
            if tcp_pose is None:
                raise ValueError("eye-in-hand needs the current TCP pose")
            p = pose_to_matrix(tcp_pose) @ self.T @ p
        else:
            p = self.T @ p
        return p[:3] * 1000.0

    def save(self, path: Path = HAND_EYE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"T": self.T.tolist(), "eye_in_hand": self.eye_in_hand}, indent=2))

    @classmethod
    def load(cls, path: Path = HAND_EYE_FILE) -> "HandEye":
        if not path.exists():
            return cls()   # identity until calibrated
        data = json.loads(path.read_text())
        return cls(np.array(data["T"]), bool(data["eye_in_hand"]))
