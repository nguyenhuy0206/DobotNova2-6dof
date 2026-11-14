"""Selective vision: which camera images actually reach Astra.

Routine perception runs locally (colour detection + depth + calibration) and goes to
the model as structured data (objects, pixels, base_xyz_mm, angle). An image is
attached only when it adds information:

  * the first image of each camera in an episode (initial visual understanding)
  * Astra asked for it explicitly: look(..., image="always")
  * the event router flagged that camera (failed grasp/skill, contact, ambiguous
    check, object not seen, scene changed)
  * detection is ambiguous: none of the requested colours, or one colour seen twice
and never when it is (nearly) identical to the last image sent from that camera.
policy "all" = baseline: every image goes, as before.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field


@dataclass
class ImagePolicy:
    mode: str = "selective"          # selective | all
    env_detail: str = "low"          # input_image detail per camera (low | high | auto)
    wrist_detail: str = "high"
    same_threshold: float = 4.0      # mean abs grey difference (0–255) below which two images are "the same"
    _last_sent: dict = field(default_factory=dict)      # camera -> 16x16 thumbnail
    _seen_camera: set = field(default_factory=set)

    def detail(self, camera: str) -> str:
        return self.env_detail if camera == "env" else self.wrist_detail

    def decide(self, args: dict, result: dict, flagged: bool) -> tuple[bool, str]:
        """(send?, reason) for a look result that carries an image."""
        camera = result.get("camera", args.get("camera", "env"))
        request = args.get("image", "auto")
        if request == "never":
            return False, "you asked image=never"
        if self.mode == "all":
            return True, "policy all"
        thumb = _thumb(result.get("image_jpeg_b64"))
        if thumb is not None and camera in self._last_sent and request != "always":
            if _diff(thumb, self._last_sent[camera]) < self.same_threshold:
                return False, "same as the last image already sent"
        reasons = []
        if camera not in self._seen_camera:
            reasons.append("first image from this camera")
        if request == "always":
            reasons.append("requested")
        if flagged:
            reasons.append("after an unexpected event")
        objs = [o.get("label") for o in result.get("objects", [])]
        wanted = [c for c in (args.get("colors") or []) if c != "white"]
        if wanted and not any(c in objs for c in wanted):
            reasons.append("requested colours not detected")
        if _ambiguous_duplicates(result.get("objects", [])):
            reasons.append("a colour detected more than once")
        if not reasons:
            return False, "detections are clear; structured data only (look with image='always' to see it)"
        return True, ", ".join(reasons)

    def sent(self, camera: str, image_b64: str | None) -> None:
        self._seen_camera.add(camera)
        thumb = _thumb(image_b64)
        if thumb is not None:
            self._last_sent[camera] = thumb


def _ambiguous_duplicates(objects: list[dict], ratio: float = 0.3) -> bool:
    """Same colour twice with comparable size (specks < 30 % of the largest blob and the
    white box are noise, not ambiguity)."""
    biggest: dict[str, float] = {}
    for o in objects:
        if o.get("label") != "white":
            biggest[o["label"]] = max(biggest.get(o["label"], 0.0), float(o.get("area_px") or 0))
    seen: dict[str, int] = {}
    for o in objects:
        lab = o.get("label")
        if lab != "white" and float(o.get("area_px") or 0) >= ratio * biggest.get(lab, 0):
            seen[lab] = seen.get(lab, 0) + 1
    return any(n > 1 for n in seen.values())


def _thumb(image_b64: str | None):
    if not image_b64:
        return None
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_GRAYSCALE)
        return None if img is None else cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA).astype("float32")
    except Exception:  # noqa: BLE001 — a broken image is simply "not comparable"
        return None


def _diff(a, b) -> float:
    return float(abs(a - b).mean())
