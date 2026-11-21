"""Run composite skills (memory/skills.py) through the ActionExecutor.

Every step goes through the executor, so dry_run / confirm / live and all
validation + safety checks apply to each primitive exactly as if the agent had
called it directly.
"""
from __future__ import annotations

import ast
import json
import operator
import re

from common.errors import ValidationError
from memory.skills import Skill, SkillLibrary

_PLACEHOLDER = re.compile(r"^\{(\w+)\}$")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(expr: str, params: dict) -> float:
    """Arithmetic on params only: '{z} + hover_mm - 5'. No calls, no attributes."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in params or params[node.id] is None:
                raise ValidationError(f"skill expression uses unknown param '{node.id}'")
            return params[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValidationError(f"unsupported skill expression: {expr!r}")
    return ev(ast.parse(expr.replace("{", "").replace("}", ""), mode="eval"))


def fill(value, params: dict):
    """Resolve "{name}" and "=expr" strings inside step args."""
    if isinstance(value, dict):
        return {k: fill(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, params) for v in value]
    if isinstance(value, str):
        m = _PLACEHOLDER.match(value)
        if m:
            if params.get(m.group(1)) is None:
                raise ValidationError(f"missing skill param '{m.group(1)}'")
            return params[m.group(1)]
        if value.startswith("="):
            result = _eval(value[1:], params)
            return int(result) if float(result).is_integer() and "speed" in value else round(result, 3)
    return value


# tools a stored skill may call (no memory/calibration/skill tools: a skill cannot
# change what is safe, what is remembered, or other skills)
SKILL_TOOLS = {"get_state", "move_to_pose", "move_to_joints", "move_relative", "jog_joint", "go_to_position",
               "open_gripper", "close_gripper", "set_gripper", "stop", "look", "find_objects",
               "check_robot_in_view", "depth_at"}     # depth_at: 20 Nov — the only perception look still offers


class SkillRunner:
    """Runs a skill locally: preconditions -> steps (tools, checks, routines) -> success
    conditions. No model call happens inside a run. On the first failing step it runs
    the skill's recovery steps (only open_gripper / lift_to — never further down) for a
    definite failure, and returns control with a failure code; an ambiguous check
    returns control without recovery (e.g. the block may still be held).

    require_validation: a skill whose validation_status is not "validated" only runs
    live after the operator approves this run (``supervisor(name, params) -> bool``);
    the outcome is stored as supervised evidence. Promotion is an operator command.
    Autonomous mode runs candidates without asking. Execution checks remain operational
    diagnostics; promotion requires an operator citing the independent judge ledger.
    """

    def __init__(self, library: SkillLibrary, constants: dict | None = None, local=None,
                 require_validation: bool = False, supervisor=None):
        self.library = library
        self.constants = constants or {}   # extra named values usable in expressions
        self.executor = None               # set by agent.toolkit.build_toolkit
        self.local = local                 # actions.local_steps.LocalSteps (checks + routines)
        self.require_validation = require_validation
        self.supervisor = supervisor
        self.episode: str | None = None    # set by the harness, stored with evidence
        self.auto_promote_after = 0        # compatibility only; automatic promotion is disabled
        self.current_dependency_hash = None
        self.quarantine_after = 0          # refuse a version that failed this many runs in a row (0 = off)

    def list(self, name: str | None = None) -> list[dict]:
        """Compact catalogue (no steps: they cost tokens on every call). name=… gives one
        skill in full, with steps, checks, recovery and earlier versions."""
        if name:
            s = self.library.get(name)
            if s is None:
                raise ValidationError(f"no skill '{name}'")
            return [{"name": s.name, "version": s.version, "validation_status": s.validation_status,
                     "description": s.description, "params": s.params, "preconditions": s.preconditions,
                     "steps": s.steps, "success_conditions": s.success_conditions, "recovery": s.recovery,
                     "failure_conditions": s.failure_conditions, "successes": s.successes, "failures": s.failures,
                     "previous_versions": [{"version": h["version"], "successes": h["successes"],
                                            "failures": h["failures"]} for h in s.history]}]
        return [{"name": s.name, "version": s.version, "status": s.validation_status, "successes": s.successes,
                 "failures": s.failures, "params": s.params, "description": s.description[:160]}
                for s in self.library.all() if s.validation_status != "deprecated"]

    # ------------------------------------------------------------------ run
    def _step(self, st: dict, values: dict, memo: dict) -> dict:
        args = fill(st.get("args", {}), values)
        if "check" in st:
            if self.local is None:
                raise ValidationError("checks need the local step executor")
            out = self.local.check(st["check"], memo, **args)
            return {"ok": bool(out.get("ok")), "kind": "check", "name": st["check"], **out}
        if "routine" in st:
            if self.local is None:
                raise ValidationError("routines need the local step executor")
            out = self.local.routine(st["routine"], memo, **args)
            if not out.get("ok") and "error_result" in out:
                out["error"] = out["error_result"].get("error")
            return {"kind": "routine", "name": st["routine"], **out}
        r = self.executor.execute(st["tool"], args)
        if r.get("ok") and isinstance(r.get("result"), dict):
            memo[st["tool"]] = r["result"]
        return {"kind": "tool", "name": st["tool"], **r}

    def run(self, name: str, params: dict | None = None) -> dict:
        if self.executor is None:
            raise RuntimeError("SkillRunner.executor not set")
        skill = self.library.get(name)
        if skill is None:
            raise ValidationError(f"no skill '{name}'. Known: {[s.name for s in self.library.all()]}")
        if skill.validation_status == "deprecated":
            raise ValidationError(f"skill '{name}' is deprecated")
        values = {**self.constants, **skill.params, **(params or {})}
        missing = [k for k, v in skill.params.items() if v is None and values.get(k) is None]
        if missing:
            raise ValidationError(f"{name}: missing params {missing}")
        live = self.executor.mode == "live"
        q = self.library.quarantined(name, self.quarantine_after) if live and self.quarantine_after else None
        if q:
            raise ValidationError(
                f"skill '{name}' v{skill.version} is QUARANTINED: it failed its last {q['consecutive_failures']} runs "
                f"in a row. Failure pattern: {json.dumps(q['stats'], ensure_ascii=False)[:600]}. Fix the procedure "
                "before running it again: save_skill a new version that changes the step that keeps failing (or its "
                "defaults), or rollback_skill to a version that worked. Repeating it unchanged is not allowed.")
        supervised = False
        trusted = (skill.validation_status == "validated" and skill.validated_dependency_hash is not None
                   and skill.validated_dependency_hash == self.current_dependency_hash)
        if live and self.require_validation and not trusted:
            if self.supervisor is None or not self.supervisor(name, params or {}):
                raise ValidationError(f"skill '{name}' v{skill.version} is a {skill.validation_status}: it needs "
                                      "operator approval to run (not given). Use primitives, or ask for a supervised run.")
            supervised = True

        memo: dict = {}
        plan = ([{**c, "_phase": "pre"} for c in skill.preconditions] + list(skill.steps)
                + [{**c, "_phase": "post"} for c in skill.success_conditions])
        trace, failed, grasps, last = [], None, 0, None
        for i, st in enumerate(plan):
            if not isinstance(st, dict) or not any(k in st for k in ("tool", "check", "routine")):
                raise ValidationError(f"{name}: step {i} needs tool, check or routine")
            out = self._step(st, values, memo)
            values.update(memo.get("vars", {}))          # e.g. measured_top from routine probe_top
            grasps += out.get("name") == "close_gripper" and out.get("kind") == "tool"
            trace.append(_brief(i, out))
            if out.get("kind") == "tool":
                last = out.get("result")
            if not out.get("ok"):
                failed = (i, st, out)
                break

        result = {"skill": name, "version": skill.version, "ok": failed is None,
                  "steps_run": len(trace), "steps_total": len(plan), "trace": trace, "grasp_attempts": grasps,
                  "last_result": last, "failed_step": None}
        if failed:
            i, st, out = failed
            code = _failure_code(st, out)
            result.update(failure=code, failed_index=i, failed_step={k: v for k, v in out.items() if k != "error_result"},
                          detail=out.get("detail") or out.get("error"),
                          ambiguous=bool(out.get("ambiguous")), camera=out.get("camera"))
            if code in skill.failure_conditions:
                result["meaning"] = skill.failure_conditions[code]
            if not out.get("ambiguous") and skill.recovery:
                rec = []
                for st2 in skill.recovery:
                    r2 = self._step(st2, values, memo)
                    rec.append(_brief(-1, r2))
                    if not r2.get("ok"):
                        break
                result["recovery"] = rec
        if live:
            fs = result.get("failed_step") or {}
            self.library.add_evidence(name, result["ok"], supervised, result.get("failure"), self.episode,
                                      step=f"{fs.get('kind')}:{fs.get('name')}" if fs else None,
                                      measured=_measured(trace))
        if self.executor.mode == "dry_run":
            result["note"] = "dry_run: motions validated only, checks not evaluated"
        return result

    def save(self, name: str, description: str, steps: list, params: dict | None = None,
             tags: list | None = None, recovery: list | None = None, preconditions: list | None = None,
             success_conditions: list | None = None, failure_conditions: dict | None = None) -> dict:
        from actions.local_steps import CHECKS, RECOVERY_ROUTINES, RECOVERY_TOOLS, ROUTINES
        if self.local is not None:                       # + Astra's own (src/agent/astra_skills)
            CHECKS, ROUTINES = self.local.known("check"), self.local.known("routine")  # noqa: N806
        known = set(self.executor.specs) if self.executor else set()

        def ok_step(st, where):
            if not isinstance(st, dict) or not any(k in st for k in ("tool", "check", "routine")):
                raise ValidationError(f"{where}: each step needs 'tool', 'check' or 'routine'")
            if "tool" in st:
                if st["tool"] not in SKILL_TOOLS or (known and st["tool"] not in known):
                    raise ValidationError(f"{where}: tool '{st['tool']}' is not allowed in a skill "
                                          f"(allowed: {sorted(SKILL_TOOLS & known if known else SKILL_TOOLS)})")
            elif "check" in st and st["check"] not in CHECKS:
                raise ValidationError(f"{where}: unknown check '{st['check']}' (known: {sorted(CHECKS)})")
            elif "routine" in st and st["routine"] not in ROUTINES:
                raise ValidationError(f"{where}: unknown routine '{st['routine']}' (known: {sorted(ROUTINES)})")

        for st in steps:
            ok_step(st, "step")
        for st in (preconditions or []) + (success_conditions or []):
            ok_step(st, "condition")
            if "check" not in st:
                raise ValidationError("preconditions / success_conditions may only be checks")
        for st in recovery or []:
            ok_step(st, "recovery")
            if st.get("tool") not in RECOVERY_TOOLS and st.get("routine") not in RECOVERY_ROUTINES:
                raise ValidationError("recovery may only open the gripper or lift (open_gripper, routine lift_to)")
        saved = self.library.revise(Skill(name=name, description=description, steps=steps,
                                          params=params or {}, tags=[t for t in (tags or []) if t != "builtin"], recovery=recovery or [],
                                          preconditions=preconditions or [],
                                          success_conditions=success_conditions or [],
                                          failure_conditions=failure_conditions or {}))
        if self.require_validation:
            note = "new versions are candidates: they run only under operator supervision until promoted"
        else:
            note = "saved as a candidate: you may run it now; validation requires operator ledger review"
        return {"saved": name, "version": saved.version, "steps": len(steps),
                "validation_status": saved.validation_status, "note": note}


def _measured(trace: list[str]) -> dict:
    """Key numbers of a run, from its trace (for failure patterns across runs)."""
    out = {}
    for line in trace:
        for key in ("measured_top", "depth", "offset", "error_px", "grip"):
            m = re.search(rf"\b{key}=(\[[^\]]*\]|-?[\d.]+)", line)
            if m:
                out[key] = m.group(1)
    return out


def _brief(i: int, out: dict) -> str:
    res = out.get("result")
    extra = ""
    if isinstance(res, dict):
        if "pose" in res and isinstance(res["pose"], list):
            extra = " z=%.0f" % res["pose"][2]
        if "position" in res:
            extra += f" grip={res['position']} holding={res.get('holding')}"
    for k in ("z", "iterations", "error_px", "offset", "measured_top", "depth", "skipped"):
        if k in out:
            extra += f" {k}={out[k]}"
    status = "ok" if out.get("ok") else ("AMBIGUOUS" if out.get("ambiguous") else "FAIL")
    why = "" if out.get("ok") else " " + str(out.get("detail") or out.get("error") or "")[:120]
    return f"{i}:{out.get('kind')}:{out.get('name')} {status}{extra}{why}"


def _failure_code(st: dict, out: dict) -> str:
    text = str(out.get("detail") or out.get("error") or "").lower()
    if st.get("routine") == "probe_top" and "no contact" in text:
        return "top_not_found"
    if st.get("routine") == "grasp_ladder" and "contact" not in text:
        return "grasp_missed"
    if "contact" in text and "no contact" not in text:
        return "contact"
    if "check" in st:
        return {"holding": "grasp_missed", "lifted": "not_lifted", "placed": "not_placed",
                "wrist_sees": "block_not_in_wrist_view", "wrist_clear": "still_held"}.get(st["check"], st["check"])
    if st.get("routine") == "wrist_center":
        return "not_centred"
    if st.get("routine") == "wrist_search":
        return "block_not_in_wrist_view"
    if st.get("routine") == "probe_top":
        return "top_not_found"
    if "table" in text or "approach zone" in text or "camera view" in text or "not reachable" in text:
        return "safety_block"
    return "step_failed"
