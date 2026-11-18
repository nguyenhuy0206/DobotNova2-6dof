"""Skill library: reusable, named sequences of primitive steps the agent can recall.

A skill is data, not code, so the agent can write new ones (``save_skill``)::

    {"name": "pick_from_above", "params": {"x": null, "y": null, "z": null, "hover_mm": 100},
     "steps": [{"tool": "open_gripper"},
               {"tool": "move_to_pose", "args": {"x": "{x}", "z": "={z} + hover_mm", ...}}]}

Step args may reference params: "{x}" (the value) or "=<expression>" (+ - * / on
params and numbers). actions/skill_runner.py fills them in and runs the steps.
Retrieval is keyword overlap for now; swap ``search`` for embeddings later.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.paths import MEMORY_DIR

SKILLS_FILE = MEMORY_DIR / "skills.json"


@dataclass
class Skill:
    name: str
    description: str
    steps: list[dict] = field(default_factory=list)   # [{"tool": ..., "args": {...}}]
    params: dict = field(default_factory=dict)         # name -> default (None = required)
    version: int = 1
    tags: list[str] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    history: list[dict] = field(default_factory=list)  # earlier versions: steps + their counts
    # --- validated-skill interface (local executor, actions/skill_runner.py) ---------
    # steps may also be {"check": name, "args"} (local checkpoint, no LLM) or
    # {"routine": name, "args"} (built-in local loop: descend_to, lift_to, wrist_center).
    preconditions: list[dict] = field(default_factory=list)       # checks before the first step
    success_conditions: list[dict] = field(default_factory=list)  # checks after the last step
    failure_conditions: dict = field(default_factory=dict)        # failure code -> what it means / next try
    recovery: list[dict] = field(default_factory=list)            # safe steps after a definite failure
    calibration: str = ""   # coordinate definition the numbers use (memory.knowledge.CALIBRATION); "" = current
    validation_status: str = "candidate"   # candidate (needs a supervised run) | validated | deprecated
    evidence: list[dict] = field(default_factory=list)            # {t, ok, supervised, failure, episode}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SkillLibrary:
    def __init__(self, path: Path = SKILLS_FILE):
        self.path = path

    def all(self) -> list[Skill]:
        if not self.path.exists():
            return []
        return [Skill(**s) for s in json.loads(self.path.read_text(encoding="utf-8"))]

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.all() if s.name == name), None)

    def record_outcome(self, name: str, success: bool) -> None:
        skill = self.get(name)
        if skill:
            if success:
                skill.successes += 1
            else:
                skill.failures += 1
            self.save(skill)

    def ensure_seeds(self) -> None:
        """Add built-in starter skills; upgrade stored copies of an older seed version
        (keeping their success/failure counts). Seed names are reserved."""
        existing = {s.name: s for s in self.all()}
        for seed in SEED_SKILLS:
            old = existing.get(seed.name)
            if old is None:
                self.save(seed)
            elif old.version < seed.version:
                self.save(Skill(**{**asdict(seed), "successes": old.successes, "failures": old.failures}))

    def revise(self, skill: Skill, keep: int = 5) -> Skill:
        """Save a new version of a skill (or the first one). The previous version's steps,
        params and success/failure counts go into ``history``; counts restart."""
        old = self.get(skill.name)
        if old is not None:
            skill.version = old.version + 1
            skill.history = (old.history + [{"version": old.version, "steps": old.steps, "params": old.params,
                                             "successes": old.successes, "failures": old.failures,
                                             "validation_status": old.validation_status,
                                             "recovery": old.recovery, "preconditions": old.preconditions,
                                             "success_conditions": old.success_conditions}])[-keep:]
            skill.successes = skill.failures = 0
            skill.evidence = []
        # A new or revised procedure is never trusted until it is run under supervision.
        skill.validation_status = "candidate"
        self.save(skill)
        return skill

    def add_evidence(self, name: str, ok: bool, supervised: bool, failure: str | None = None,
                     episode: str | None = None, keep: int = 20, step: str | None = None,
                     measured: dict | None = None) -> None:
        import time
        skill = self.get(name)
        if skill is None:
            return
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": bool(ok), "supervised": bool(supervised),
               "failure": failure, "episode": episode}
        if step:
            rec["step"] = step
        if measured:
            rec["measured"] = measured
        skill.evidence = (skill.evidence + [rec])[-keep:]
        self.save(skill)

    def quarantined(self, name: str, after: int = 3) -> dict | None:
        """The CURRENT version failed its last `after` runs in a row (and never succeeded):
        it must be revised (save_skill), rolled back or retired before it runs again."""
        skill = self.get(name)
        if skill is None or after <= 0 or skill.successes:
            return None
        last = skill.evidence[-after:]
        if len(last) < after or any(e.get("ok") for e in last):
            return None
        return {"skill": name, "version": skill.version, "consecutive_failures": after,
                "stats": self.failure_stats(name)}

    def failure_stats(self, name: str) -> dict:
        """Where and how the current version fails, with the measured numbers (no LLM)."""
        skill = self.get(name)
        if skill is None:
            return {}
        by_step: dict = {}
        for e in skill.evidence:
            if e.get("ok"):
                continue
            key = f"{e.get('step') or '?'}:{e.get('failure') or '?'}"
            by_step[key] = by_step.get(key, 0) + 1
        nums: dict = {}
        for e in skill.evidence:
            for k, v in (e.get("measured") or {}).items():
                nums.setdefault(k, []).append(v)
        return {"runs": len(skill.evidence), "ok": sum(1 for e in skill.evidence if e.get("ok")),
                "failed_at": dict(sorted(by_step.items(), key=lambda kv: -kv[1])),
                "measured": {k: v[-8:] for k, v in nums.items()}}

    def promote(self, name: str, min_supervised_successes: int = 2) -> dict:
        """Operator action only (never called by Astra): candidate -> validated, when the
        CURRENT version has enough supervised, camera-verified successes."""
        skill = self.get(name)
        if skill is None:
            return {"promoted": False, "reason": f"no skill {name!r}"}
        ok = sum(1 for e in skill.evidence if e.get("ok") and e.get("supervised"))
        if ok < min_supervised_successes:
            return {"promoted": False, "reason": f"{name} v{skill.version}: {ok} supervised successes, "
                                                 f"need {min_supervised_successes}"}
        skill.validation_status = "validated"
        self.save(skill)
        return {"promoted": True, "name": name, "version": skill.version, "supervised_successes": ok}

    def auto_promote(self, name: str, after: int) -> dict | None:
        """Autonomous mode: a candidate whose CURRENT version succeeded (camera-verified
        success conditions) in its last ``after`` runs, with no failure among them, becomes
        validated. Counts restart with every new version, so a revision must earn it again."""
        skill = self.get(name)
        if skill is None or after <= 0 or skill.validation_status != "candidate":
            return None
        last = skill.evidence[-after:]
        if len(last) < after or not all(e.get("ok") for e in last):
            return None
        skill.validation_status = "validated"
        self.save(skill)
        return {"promoted": True, "name": name, "version": skill.version, "after_successes": after}

    def retire(self, name: str, reason: str = "") -> dict:
        """Mark a skill deprecated (kept in the file with its history; run_skill refuses it)."""
        skill = self.get(name)
        if skill is None:
            return {"retired": None, "known": [s.name for s in self.all()]}
        if "builtin" in skill.tags:
            raise ValueError(f"{name} is a built-in skill: save a new version or rollback_skill instead of retiring it")
        skill.validation_status = "deprecated"
        if reason:
            skill.description = f"[retired: {reason[:120]}] " + skill.description
        self.save(skill)
        return {"retired": name, "version": skill.version}

    def rollback(self, name: str, version: int) -> dict:
        """Restore an earlier version's procedure as the newest version (history kept)."""
        skill = self.get(name)
        old = next((h for h in (skill.history if skill else []) if h["version"] == version), None)
        if old is None:
            return {"rolled_back": False, "reason": f"{name} has no stored version {version}"}
        restored = Skill(name=name, description=skill.description, steps=old["steps"], params=old["params"],
                         tags=skill.tags, preconditions=old.get("preconditions", []),
                         success_conditions=old.get("success_conditions", []),
                         recovery=old.get("recovery", []), failure_conditions=skill.failure_conditions)
        status = old.get("validation_status", "candidate")
        self.revise(restored)
        restored.validation_status = status          # a rollback restores the trust it had
        self.save(restored)
        return {"rolled_back": True, "name": name, "from_version": version, "now_version": restored.version,
                "validation_status": status}

    def delete(self, name: str) -> dict:
        skills = self.all()
        keep = [s for s in skills if s.name != name]
        if len(keep) == len(skills):
            return {"deleted": None, "known": [s.name for s in skills]}
        self.path.write_text(json.dumps([asdict(s) for s in keep], ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        return {"deleted": name}

    def save(self, skill: Skill) -> None:
        skills = [s for s in self.all() if s.name != skill.name] + [skill]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(s) for s in skills], ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    def index(self) -> list[dict]:
        """Verified facts about the usable skills (no descriptions): what exists and how its
        CURRENT version has done. Deprecated skills are left out."""
        out = []
        for s in self.all():
            if s.validation_status == "deprecated":
                continue
            out.append({"name": s.name, "version": s.version, "status": s.validation_status,
                        "ok": s.successes, "fail": s.failures,
                        "params": [k for k in s.params]})
        return out

    def search(self, query: str, k: int = 3) -> list[Skill]:
        q = _tokens(query)
        scored = [(len(q & _tokens(" ".join([s.name, s.description, *s.tags]))), s) for s in self.all()]
        return [s for score, s in sorted(scored, key=lambda x: -x[0]) if score > 0][:k]


# ---------------------------------------------------------------- starter skills
# Poses are fingertip positions (primitives add the gripper length). Tool pointing
# straight down = Rx 180, Ry 0. Heights are relative to the object's top surface z
# (from find_objects). Tune grasp_depth_mm to grip below the top surface.
_DOWN = {"rx": "{rx}", "ry": "{ry}", "rz": "{rz}"}
_ORIENT = {"rx": 180, "ry": 0, "rz": 0}

SEED_SKILLS = [
    Skill(
        name="pick_from_above",
        description="Pick an object whose top surface is at (x, y, z) mm in the base frame: open, hover "
                    "above, descend straight down, close, lift. Result of close_gripper tells if it holds.",
        params={"x": None, "y": None, "z": None, "hover_mm": 100, "grasp_depth_mm": 10, "speed": 20, **_ORIENT},
        steps=[
            {"tool": "open_gripper"},
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} + hover_mm",
                                              **_DOWN, "speed": "{speed}"}},
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} - grasp_depth_mm",
                                              **_DOWN, "linear": True, "speed": "=speed / 2"}},
            {"tool": "close_gripper"},
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} + hover_mm",
                                              **_DOWN, "linear": True, "speed": "{speed}"}},
        ],
        tags=["pick", "grasp", "block", "top-down", "seed"], version=3,
    ),
    Skill(
        name="place_from_above",
        description="Place the held object onto a surface at (x, y, z) mm: hover above, descend to "
                    "release_height_mm above the surface, open, lift.",
        params={"x": None, "y": None, "z": None, "hover_mm": 100, "release_height_mm": 10, "speed": 20, **_ORIENT},
        steps=[
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} + hover_mm",
                                              **_DOWN, "speed": "{speed}"}},
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} + release_height_mm",
                                              **_DOWN, "linear": True, "speed": "=speed / 2"}},
            {"tool": "open_gripper"},
            {"tool": "move_to_pose", "args": {"x": "{x}", "y": "{y}", "z": "={z} + hover_mm",
                                              **_DOWN, "linear": True, "speed": "{speed}"}},
        ],
        tags=["place", "release", "drop", "frame", "target", "top-down", "seed"], version=3,
    ),
]
