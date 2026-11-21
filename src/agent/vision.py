"""Image delivery and detail per camera.

20 Nov: look returns pictures only, so every requested picture reaches Astra.
The legacy image="never" option still suppresses delivery; mode also selects
whether the harness sends explicit per-camera detail.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImagePolicy:
    mode: str = "selective"          # selective = explicit detail per camera | all = API default detail
    env_detail: str = "low"          # input_image detail per camera (low | high | auto)
    wrist_detail: str = "high"

    def detail(self, camera: str) -> str:
        return self.env_detail if camera == "env" else self.wrist_detail

    def decide(self, args: dict) -> tuple[bool, str]:
        """(send?, reason) for a result that carries an image."""
        if args.get("image") == "never":
            return False, "you asked image=never"
        if self.mode == "all":
            return True, "policy all"
        return True, "the picture is the result of look"
