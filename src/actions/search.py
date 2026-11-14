"""search_object: find an object that is hidden (usually under the arm) or out of view.

Looks with the env camera (3D) and the wrist camera; if neither sees it, rotates
J1 back and forth in steps (+step … +max, then -step … -max) and looks again
after each step. Every rotation goes through the ActionExecutor, so dry_run /
confirm / speed cap / safety checks apply. If nothing is found the arm returns
to where it started.
"""
from __future__ import annotations

from common.errors import ValidationError


class ObjectSearch:
    def __init__(self, scene, joint: int = 1):
        self.scene = scene
        self.joint = joint
        self.executor = None          # set by agent.toolkit.build_toolkit

    def _look(self, color: str) -> dict | None:
        objs = self.scene.find(color, "env")
        if objs and objs[0].base_xyz_mm is not None:
            return {"camera": "env", "object": objs[0].summary()}
        if "wrist" in self.scene.sources:
            try:
                wrist = self.scene.find(color, "wrist")
            except Exception:  # noqa: BLE001 — wrist camera optional
                wrist = []
            if wrist:
                return {"camera": "wrist", "object": wrist[0].summary(),
                        "note": "seen by the wrist camera only (pixels, no 3D)"}
        return None

    MAX_JOG_DEG = 20.0   # jog_joint's per-command limit (config joint_step.max_deg)

    def _jog(self, degrees: float) -> None:
        """Rotate the joint, split into steps the jog limit allows."""
        remaining = degrees
        while abs(remaining) > 1e-6:
            chunk = max(-self.MAX_JOG_DEG, min(self.MAX_JOG_DEG, remaining))
            result = self.executor.execute("jog_joint", {"joint": self.joint, "degrees": chunk})
            if not result["ok"]:
                raise ValidationError(f"search stopped: {result.get('error')}")
            remaining -= chunk

    def search(self, color: str, max_sweep_deg: float = 20, step_deg: float = 10) -> dict:
        if self.executor is None:
            raise RuntimeError("ObjectSearch.executor not set")
        if not 1 <= step_deg <= 20 or not step_deg <= max_sweep_deg <= 40:
            raise ValidationError("step_deg 1–20, max_sweep_deg step–40")
        found = self._look(color)
        if found and found["camera"] == "env":
            return {"found": True, "rotated_deg": 0, **found}
        n = int(max_sweep_deg // step_deg)
        offsets = [step_deg * i for i in range(1, n + 1)] + [-step_deg * i for i in range(1, n + 1)]
        current = 0.0
        wrist_hit = found
        for target in offsets:
            self._jog(target - current)
            current = target
            found = self._look(color)
            if found and found["camera"] == "env":
                return {"found": True, "rotated_deg": current, "joint": self.joint, **found}
            wrist_hit = wrist_hit or found
        if current:
            self._jog(-current)        # back to where we started
        return {"found": wrist_hit is not None, "rotated_deg": 0, "returned_to_start": True,
                **(wrist_hit or {}), "last_seen": self.scene.last_seen_info(color)}
