"""Work-cell survey by hand-guided touches (drag mode + Save position in the Debug tab).

The robot is the ruler: a saved position's joints give the flange pose exactly (FK matches the
controller to 0.05 mm), so what remains is WHICH point of the gripper touched.

    fit_contact      three touches of ONE dot: gripper vertical with the dot between the fingertips,
                     then tilted twice, each time one open fingertip on the dot
                     -> centre between the fingertips (tool length) + open-fingertip offset
    contact_points   centre and both open fingertips for any saved pose
    lowest_contact   the point that touches a surface below (the gripper may be tilted)
    table_height     z of the table from touches on it
    enclosing_cylinder  a keep-out zone around touches of an obstacle (all contact candidates)

18 Nov 2026: tool 127.1 mm (was 118 from a tape measure), fingertips ±31.6 mm, table z 0.3.
Flange poses here are (position mm, rotation 3x3).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from calibration.files import SURVEY_FILE
from common.poses import pose_to_matrix

Flange = tuple[np.ndarray, np.ndarray]          # (position mm, rotation 3x3)


def flange_from_joints(joints_deg) -> Flange:
    from drivers.nova2_kinematics import flange_pose
    f = flange_pose(joints_deg)
    return np.array(f[:3], float), pose_to_matrix(f)[:3, :3]


@dataclass
class ContactModel:
    centre_mm: np.ndarray                 # centre between the fingertips, flange frame
    finger_mm: np.ndarray                 # open fingertip = centre ± this (flange frame, across the axis)
    residual_mm: list[float] = field(default_factory=list)
    dot_mm: np.ndarray | None = None      # the touched dot, base frame

    @property
    def tool_length_mm(self) -> float:
        return float(self.centre_mm[2])

    @property
    def finger_half_gap_mm(self) -> float:
        return float(np.linalg.norm(self.finger_mm))

    def to_dict(self) -> dict:
        return {"centre_in_flange_mm": self.centre_mm.round(1).tolist(),
                "open_finger_half_gap_mm": round(self.finger_half_gap_mm, 1),
                "finger_dir_in_flange": (self.finger_mm / self.finger_half_gap_mm).round(3).tolist(),
                "fit_residual_mm": [round(r, 1) for r in self.residual_mm]}

    @classmethod
    def from_dict(cls, d: dict) -> "ContactModel":
        return cls(np.array(d["centre_in_flange_mm"], float),
                   np.array(d["finger_dir_in_flange"], float) * float(d["open_finger_half_gap_mm"]),
                   list(d.get("fit_residual_mm", [])))


def fit_contact(vertical: Flange, tilt_a: Flange, tilt_b: Flange) -> ContactModel:
    """The same dot touched three times: by the centre (vertical), by one open fingertip (tilt_a)
    and by the other (tilt_b). Linear least squares in (centre, finger offset ⟂ axis, dot):
        p_v + R_v c = q,   p_a + R_a (c + e) = q,   p_b + R_b (c − e) = q,   e_z = 0.
    The two tilts must lean in different directions; the residual tells whether one dot was hit."""
    rows, rhs = [], []
    for (p, R), s in ((vertical, 0), (tilt_a, 1), (tilt_b, -1)):
        rows.append(np.hstack([R, s * R[:, :2], -np.eye(3)]))
        rhs.append(-p)
    A, b = np.vstack(rows), np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = np.linalg.norm((A @ x - b).reshape(3, 3), axis=1)
    return ContactModel(x[:3], np.array([x[3], x[4], 0.0]), list(map(float, res)), x[5:])


def contact_points(flange: Flange, model: ContactModel) -> dict[str, np.ndarray]:
    p, R = flange
    return {"centre": p + R @ model.centre_mm, "finger+": p + R @ (model.centre_mm + model.finger_mm),
            "finger-": p + R @ (model.centre_mm - model.finger_mm)}


def lowest_contact(flange: Flange, model: ContactModel) -> np.ndarray:
    return min(contact_points(flange, model).values(), key=lambda q: q[2])


def table_height(z_mm: list[float]) -> dict:
    z = np.asarray(z_mm, float)
    return {"mean": round(float(z.mean()), 1), "min": round(float(z.min()), 1), "max": round(float(z.max()), 1),
            "n": int(z.size)}


def enclosing_cylinder(points_mm: list[np.ndarray], margin_mm: float = 40.0) -> dict:
    """Vertical cylinder around every point (+ margin): centre of the bounding box in x/y."""
    P = np.asarray(points_mm, float)
    c = (P[:, :2].max(0) + P[:, :2].min(0)) / 2
    r = float(np.linalg.norm(P[:, :2] - c, axis=1).max())
    return {"xy": c.round(0).tolist(), "radius_mm": round(r + margin_mm), "enclosing_mm": round(r, 1),
            "z_touched": [round(float(P[:, 2].min()), 1), round(float(P[:, 2].max()), 1)]}


def load() -> dict:
    try:
        return json.loads(SURVEY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(update: dict) -> dict:
    """Merge into data/memory/workcell_survey.json."""
    data = load()
    data.update(update)
    data["surveyed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    SURVEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SURVEY_FILE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def saved_model() -> ContactModel | None:
    g = load().get("gripper")
    return ContactModel.from_dict(g) if g else None
