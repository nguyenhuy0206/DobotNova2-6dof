"""Bounded context: what Astra sees on each call, independent of episode length.

Baseline (V0) chains requests with ``previous_response_id`` — the API then bills the
WHOLE history (every image included) again on every call, so cost grows ~quadratically.
The bounded policy sends a fresh, self-contained request each time (no
previous_response_id) whose size has a fixed upper bound:

    instructions + tools                      stable  -> served from the prompt cache
    TASK                                      stable  -> cached prefix
    STATE (measured by the harness, not by the model): pose, gripper, zone, alarms,
          verified progress, unresolved error, active skill, operator approval mode
    EPISODE SO FAR: deterministic one-line digests, ≤ summary_max_chars, oldest dropped
          (milestones — grasps, placements, errors — are kept longest)
    LAST TURNS: the last ``recent_turns`` model turns with compact results
    LATEST RESULTS: the results of the previous turn in full (images removed)
    image(s): only those the image policy lets through, this turn only

The full history stays in data/episodes/<id>.jsonl for audit and learning.
No LLM is used to summarise: every digest line comes from agent.events.digest_line.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field

MILESTONE_TAGS = ("grasp_caught", "grasp_missed", "released", "skill_ok", "skill_failed", "contact",
                  "verify_ambiguous", "scene_changed", "safety_block", "ERROR")


@dataclass
class ContextPolicy:
    mode: str = "bounded"            # bounded | chain (previous_response_id, baseline)
    recent_turns: int = 4
    summary_max_chars: int = 4000    # ≈ 1000 tokens
    latest_result_chars: int = 6000
    turn_result_chars: int = 300


def compact_result(result: dict, limit: int) -> str:
    """Tool result for the model without images and bulky fields."""
    res = result.get("result")
    if isinstance(res, dict):
        res = {k: v for k, v in res.items() if k not in ("image_jpeg_b64", "image_file")}
        if "objects" in res:
            res["objects"] = [{k: o.get(k) for k in ("label", "pixel", "bbox", "area_px", "base_xyz_mm", "angle_deg")
                               if o.get(k) is not None} for o in res["objects"]]
    out = {"ok": result.get("ok"), **({"error": result["error"]} if "error" in result else {}),
           **({"result": res} if res is not None else {})}
    text = json.dumps(out, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class WorkingMemory:
    task: str
    policy: ContextPolicy = field(default_factory=ContextPolicy)
    digest: list = field(default_factory=list)                 # (is_milestone, line)
    turns: deque = field(default_factory=deque)
    latest: list = field(default_factory=list)                 # [(name, args, result)] of the last turn
    notes_for_model: list = field(default_factory=list)        # harness messages (deferred calls, budget…)
    progress: list = field(default_factory=list)               # verified facts ("green placed at …")
    unresolved_error: str | None = None
    active_skill: str | None = None

    def record_turn(self, step: int, text: str, executed: list, deferred: list, reason: str,
                    digests: list[str]) -> None:
        for line in digests:
            self.digest.append((any(t in line for t in MILESTONE_TAGS), line))
        self._trim_digest()
        self.turns.append({"step": step, "said": (text or "")[:300],
                           "calls": [f"{n}({json.dumps(a, ensure_ascii=False, default=str)[:160]}) -> "
                                     f"{compact_result(r, self.policy.turn_result_chars)}" for n, a, r in executed]})
        while len(self.turns) > self.policy.recent_turns:
            self.turns.popleft()
        self.latest = executed
        self.notes_for_model = []
        if deferred:
            self.notes_for_model.append(
                f"NOT executed ({reason}): " + ", ".join(f"{c.name}({json.dumps(getattr(c, 'parsed_args', {}), default=str)[:100]})"
                                                          for c in deferred) + ". Re-issue them if still wanted.")
        failed = [r for _, _, r in executed if not r.get("ok")]
        self.unresolved_error = str(failed[-1].get("error"))[:200] if failed else (
            None if executed else self.unresolved_error)

    def _trim_digest(self) -> None:
        limit = self.policy.summary_max_chars
        size = sum(len(line) + 1 for _, line in self.digest)
        i = 0
        while size > limit and i < len(self.digest):         # drop routine lines first, oldest first
            if not self.digest[i][0]:
                size -= len(self.digest[i][1]) + 1
                del self.digest[i]
            else:
                i += 1
        while size > limit and self.digest:                  # then the oldest milestones
            size -= len(self.digest[0][1]) + 1
            del self.digest[0]

    def build(self, state: dict | None, extra: str = "") -> str:
        parts = [f"TASK:\n{self.task.strip()}"]
        st = {k: v for k, v in (state or {}).items() if v is not None}
        if self.progress:
            st["verified_progress"] = self.progress
        if self.unresolved_error:
            st["unresolved_error"] = self.unresolved_error
        if self.active_skill:
            st["active_skill"] = self.active_skill
        if st:
            parts.append("STATE (measured by the harness now):\n" + json.dumps(st, ensure_ascii=False, default=str))
        if self.digest:
            parts.append("EPISODE SO FAR (one line per action, oldest routine lines dropped):\n"
                         + "\n".join(line for _, line in self.digest))
        older = list(self.turns)[:-1]
        if older:
            parts.append("EARLIER TURNS:\n" + "\n".join(
                f"[{t['step']}] you: {t['said']}\n  " + "\n  ".join(t["calls"]) for t in older))
        if self.latest:
            budget = self.policy.latest_result_chars
            lines = []
            for n, a, r in self.latest:
                line = f"{n}({json.dumps(a, ensure_ascii=False, default=str)[:300]}) -> {compact_result(r, budget)}"
                budget = max(400, budget - len(line))
                lines.append(line)
            parts.append("RESULTS OF YOUR LAST TOOL CALLS:\n" + "\n".join(lines))
        if self.notes_for_model:
            parts.append("HARNESS: " + " ".join(self.notes_for_model))
        if extra:
            parts.append(extra)
        parts.append("Decide the next tool call(s), or answer with the final report (no tool call) when done.")
        return "\n\n".join(parts)
