"""Where the fingertips are in the wrist picture (infrastructure, not detection).

The wrist camera does not look along the gripper: after the 20 Nov calibration it sits 89 mm off
the tool axis and looks 69° away from it, almost forward. What appears "between the fingers" in the
picture is therefore not what is under the fingertips (episode 164552: a green cube showed between
the fingers while the fingertips were right above the blue one). This projects the fingertip points
— computed from the robot's own pose, the gripper model and the wrist calibration — into the
picture, so the image shows where the gripper actually is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.poses import pose_to_matrix


class WristOverlay:
    def __init__(self, calibration_file: Path, state_fn, tool_length_mm: float, finger_half_gap_mm: float,
                 open_position: float = 1000.0):
        cal = json.loads(Path(calibration_file).read_text())
        self.K = np.asarray(cal["wrist_intrinsics"]["K"], float)
        self.dist = np.asarray(cal["wrist_intrinsics"]["dist"], float)
        X = np.asarray(cal["T_flange_wrist"], float).copy()
        if np.abs(X[:3, 3]).max() < 5:            # stored in metres
            X[:3, 3] *= 1000.0
        self.T_flange_cam = X
        self.state_fn, self.tool, self.half_gap, self.open = state_fn, tool_length_mm, finger_half_gap_mm, open_position

    def points_flange(self, gripper_position: float | None) -> dict:
        """Fingertip points in the flange frame (mm): the centre between them, and each fingertip at the
        current opening (linear in the gripper position; fully open when unknown)."""
        frac = 1.0 if gripper_position is None else max(0.0, min(1.0, float(gripper_position) / self.open))
        g = self.half_gap * frac
        return {"centre": np.array([0.0, 0.0, self.tool]), "finger+": np.array([0.0, g, self.tool]),
                "finger-": np.array([0.0, -g, self.tool])}

    def project(self, flange_pose: list, gripper_position: float | None = None) -> dict:
        import cv2
        T_bf = pose_to_matrix(flange_pose)
        T_bf[:3, 3] = flange_pose[:3]
        T_cam_flange = np.linalg.inv(self.T_flange_cam)
        out = {}
        for name, p in self.points_flange(gripper_position).items():
            pc = T_cam_flange @ np.r_[p, 1.0]
            if pc[2] <= 1.0:
                continue
            uv, _ = cv2.projectPoints(pc[:3].reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
            out[name] = [round(float(v), 1) for v in uv.ravel()]
        return out

    def __call__(self) -> dict:
        state = self.state_fn()
        grip = (state.get("gripper") or {}).get("position")
        return {"fingertips_px": self.project(state["flange_pose"], grip)}

    @staticmethod
    def draw(img, fingertips_px: dict) -> None:
        import cv2
        h, w = img.shape[:2]
        for name, (u, v) in fingertips_px.items():
            if not (0 <= u < w and 0 <= v < h):
                continue
            centre = name == "centre"
            cv2.drawMarker(img, (int(u), int(v)), (255, 0, 255), cv2.MARKER_CROSS if centre else cv2.MARKER_TILTED_CROSS,
                           16 if centre else 10, 2 if centre else 1)
            if centre:
                cv2.putText(img, "tip", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
