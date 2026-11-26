"""Stop an episode that is going round in circles.

The same ACTION in the same STATE (same tool, same arguments — numbers rounded to 0.1 — with
the robot at the same pose and gripper position afterwards) or the same HYPOTHESIS (diagnose /
propose_hypothesis landing on the same knowledge hypothesis) three times in a row ends the
episode: repeating it unchanged cannot give a new result. Seen on 17 Nov: diagnose sent with the
same wrong id 6×, open_gripper on open fingers again and again. A move that changes the pose
each time is progress, not a repeat (lifting in 25 mm steps: move_relative dz +25 three times).

Calls that only observe (look, get_state, list_*, query_observations …)
neither count nor break a run: looking again changes nothing in the world.
"""
from __future__ import annotations

import json

OBSERVE_ONLY = {"get_state", "look", "depth_at", "find_objects", "compare_reference", "locate_object", "list_skills", "list_routines", "read_routine",
                "list_concepts", "query_observations", "read_history", "check_robot_in_view"}


def _round(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 1)                       # 30 == 30.0 == 30.04
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def _where(state: dict | None) -> str:
    """Pose (1 mm / 1°) and gripper position: what an unchanged repeat leaves unchanged."""
    if not state:
        return ""
    pose = state.get("pose")
    grip = (state.get("gripper") or {}).get("position")
    p = ",".join(f"{v:.0f}" for v in pose) if isinstance(pose, (list, tuple)) else ""
    return f" @[{p}] grip {grip}"


def signature(name: str, args: dict, result: dict) -> str | None:
    if name in OBSERVE_ONLY:
        return None
    if isinstance(result, dict) and not result.get("ok", True) and result.get("error"):
        return f"{name} refused: {str(result['error'])[:120]}"      # same refusal, whatever the wording
    res = result.get("result") if isinstance(result, dict) and isinstance(result.get("result"), dict) else {}
    if name == "propose_hypothesis" and res.get("id"):
        return f"hypothesis {res['id']}"
    if name == "diagnose" and res.get("hypotheses"):
        ids = sorted(str((h.get("knowledge") or {}).get("id") or h.get("cause", ""))[:80] for h in res["hypotheses"])
        return "hypothesis " + ", ".join(ids)
    return f"{name} {json.dumps(_round(args or {}), sort_keys=True, ensure_ascii=False, default=str)}"


class RepeatGuard:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self.last: str | None = None
        self.count = 0

    def add(self, name: str, args: dict, result: dict, state: dict | None = None) -> str | None:
        """Record a call (state = measured right after it); returns the stop reason when the same
        thing ran `limit` times in a row."""
        if self.limit <= 0:
            return None
        sig = signature(name, args, result)
        if sig is None:
            return None
        if not sig.startswith("hypothesis "):
            sig += _where(state)
        self.count = self.count + 1 if sig == self.last else 1
        self.last = sig
        if self.count >= self.limit:
            return f"the same {'hypothesis' if sig.startswith('hypothesis ') else 'action'} {self.count}× in a row: {sig[:160]}"
        return None
