"""Trusted read-only acquisition with an operator-validated evidence profile."""
from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from judge.predicates import polygon_relation
from judge.records import Evidence, Observation
from judge.spec import TaskSpec, digest
from perception.detection import detect_colored_objects

PROFILE_FILE = Path(__file__).parent / "config" / "evidence_profile.json"


class EvidenceCapture:
    def __init__(self, scene, state_fn, root: Path, spec: TaskSpec, profile=None):
        self.scene, self.state_fn, self.root, self.spec = scene, state_fn, root, spec
        self.profile = json.loads(PROFILE_FILE.read_text()) if profile is None else json.loads(json.dumps(profile))
        self.profile_hash = digest(self.profile)
        self.stream_id, self.sequence = uuid.uuid4().hex, 0
        self.state = {}
        self.opened_since_ns = self.retracted_since_ns = None

    def _save(self, data: bytes, suffix: str) -> tuple[str, str]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{uuid.uuid4().hex}.{suffix}"
        with path.open("xb") as f:
            f.write(data)
        return str(path), hashlib.sha256(data).hexdigest()

    def _applicable(self):
        p = self.profile
        return (p.get("validated") is True and self.spec.calibration_hash == self.profile_hash
                and p.get("object_id") == self.spec.object_id and p.get("object_color") == self.spec.object_color
                and p.get("single_object_scope_confirmed") is True)

    def robot(self) -> Evidence:
        self.state = state = self.state_fn()
        captured_ns = time.monotonic_ns()
        self.sequence += 1
        file, sha = self._save(json.dumps(state, sort_keys=True, allow_nan=False).encode(), "json")
        pose, grip, p = state.get("pose"), state.get("gripper") or {}, self.profile
        status = str(grip.get("grip_status", ""))
        holding = True if "caught" in status else False if "no object" in status or "dropped" in status else None
        opened, retracted = None, None
        if "open_position" in p and grip.get("position") is not None:
            opened = abs(grip["position"] - p["open_position"]) <= p["open_position_tolerance"]
        if pose and "retracted_z_min_mm" in p:
            retracted = pose[2] - p.get("robot_uncertainty_mm", float("inf")) > p["retracted_z_min_mm"]
        self.opened_since_ns = (self.opened_since_ns or captured_ns) if opened else None
        self.retracted_since_ns = (self.retracted_since_ns or captured_ns) if retracted else None
        obs = Observation(z_mm=pose[2] if pose and len(pose) > 2 else None,
                          uncertainty_mm=p.get("robot_uncertainty_mm"), holding=holding,
                          opened=opened, retracted=retracted, opened_since_ns=self.opened_since_ns,
                          retracted_since_ns=self.retracted_since_ns,
                          dropped=True if "dropped" in status else None)
        return Evidence(uuid.uuid4().hex, "robot", captured_ns, self.sequence, self.stream_id,
                        file, sha, self.spec.object_id, True, self.profile_hash,
                        self._applicable(), observation=obs)

    def _camera_applicable(self, name, frame):
        p, pose = self.profile, self.state.get("pose")
        if (not self._applicable() or not frame.stream_id
                or p.get("stream_ids", {}).get(name) != frame.stream_id
                or p.get("image_sizes", {}).get(name) != [frame.color.shape[1], frame.color.shape[0]]):
            return False
        if name == "wrist":
            limits = p.get("wrist_pose_bounds")
            if not pose or not limits or len(limits) != len(pose):
                return False
            return all(lo <= value <= hi for value, (lo, hi) in zip(pose, limits))
        return name == "env"

    def _support(self, frame, bbox):
        """Bound support using raw depth, an operator transform and known cube height."""
        p, k = self.profile, frame.intrinsics
        needed = {"camera_to_base_mm", "cube_height_mm", "table_z_mm", "support_uncertainty_mm"}
        if frame.depth_m is None or k is None or not needed <= p.keys():
            return None, None
        x, y, w, h = bbox
        depth = frame.depth_m[y:y + h, x:x + w]
        if depth.size == 0 or not np.all(np.isfinite(depth) & (depth > 0)):
            return None, None
        vv, uu = np.mgrid[y:y + h, x:x + w]
        points = np.stack(((uu - k.cx) * depth / k.fx * 1000,
                           (vv - k.cy) * depth / k.fy * 1000, depth * 1000, np.ones(depth.shape)), axis=-1)
        transform = np.asarray(p["camera_to_base_mm"], dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return None, None
        z = (points @ transform.T)[..., 2]
        lo, hi = float(z.min()), float(z.max())
        return ((lo + hi) / 2 - p["cube_height_mm"] - p["table_z_mm"],
                (hi - lo) / 2 + p["support_uncertainty_mm"])

    def _visibility(self, name, detection):
        """A blob away from the image edge alone does not establish full visibility."""
        if detection.cut_by_edge:
            return "partial"
        bounds = self.profile.get("bbox_size_bounds", {}).get(name)
        fill = self.profile.get("minimum_fill_ratio")
        if not bounds or fill is None:
            return "unknown"
        w, h = detection.bbox[2:]
        complete = all(lo <= value <= hi for value, (lo, hi) in zip((w, h), bounds))
        return "full" if complete and detection.area / (w * h) >= fill else "partial"

    def camera(self, name: str) -> Evidence:
        frame = self.scene.observe(name)
        # Preserve raw depth and intrinsics with the pixels; processed tool outputs are not evidence.
        buf = io.BytesIO()
        np.savez_compressed(buf, color=frame.color, depth_m=frame.depth_m if frame.depth_m is not None else [],
                            intrinsics=json.dumps(asdict(frame.intrinsics)) if frame.intrinsics else "null")
        file, sha = self._save(buf.getvalue(), "npz")
        color = self.spec.object_color
        found = detect_colored_objects(frame.color, [color]) if color else []
        applicable = self._camera_applicable(name, frame)
        obs = Observation()
        if len(found) == 1:
            det, p = found[0], self.profile
            obs = Observation(pixel=tuple(det.center), bbox=tuple(det.bbox), edge_deg=det.angle_deg,
                              visibility=self._visibility(name, det),
                              uncertainty_px=p.get("pixel_uncertainty_px"),
                              edge_uncertainty_deg=p.get("edge_uncertainty_deg"))
            if name == "wrist" and p.get("finger_polygon"):
                probe = Evidence("probe", name, None, None, None, None, None, None, observation=obs)
                relation = polygon_relation(probe, replace(self.spec, destination=tuple(map(tuple, p["finger_polygon"]))))
                obs = replace(obs, between_fingers={"true": True, "false": False, "unknown": None}[relation])
            if name == "env":
                gap, uncertainty = self._support(frame, det.bbox)
                obs = replace(obs, support_gap_mm=gap, uncertainty_mm=uncertainty)
        return Evidence(uuid.uuid4().hex, name, frame.captured_ns, frame.sequence_id, frame.stream_id,
                        file, sha, self.spec.object_id if len(found) == 1 else None,
                        applicable and len(found) == 1, self.profile_hash, applicable,
                        legacy=frame.captured_ns is None, observation=obs)

    def acquire(self) -> tuple[Evidence, ...]:
        """Camera evidence only when the profile can make it applicable: an unconfigured judge would
        otherwise save colour + raw depth on every action (a few MB each) that can never earn credit."""
        evidence = []
        cameras = (lambda: self.camera("wrist"), lambda: self.camera("env")) if self._applicable() else ()
        for fn in (self.robot, *cameras):
            try:
                evidence.append(fn())
            except Exception:  # noqa: BLE001 — unavailable sensors mean missing evidence
                continue
        return tuple(evidence)
