"""Tool-style action execution with a single execution-mode switch.

An ``ActionSpec`` describes one callable action (name, JSON-schema params,
whether it moves hardware). ``ActionExecutor`` runs specs by name and enforces
the execution mode:

  dry_run  validate arguments only; nothing is sent to the robot
  confirm  ask a human (callback) before any action with ``moves_robot``
  live     run directly (validation + safety checks still apply)

The Astra harness and any future planner call ``execute(name, args)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from common.errors import RobotError, ValidationError
from common.log import get_logger
from common.types import ActionResult

log = get_logger(__name__)

ConfirmFn = Callable[[str, dict], bool]


@dataclass
class ActionSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    moves_robot: bool = False   # motion or gripper actuation
    read_only: bool = False     # safe to run even in dry_run (e.g. get_state)
    delegates: bool = False     # runs other tools through this executor (run_skill): those
                                # inner calls get dry_run/confirm, so don't gate the outer one
    reports_status: bool = False  # handler's structured ok also determines the tool envelope's ok

    def validate_args(self, args: dict) -> dict:
        """Light JSON-schema check: required keys, no unknown keys, basic types."""
        props = self.parameters.get("properties", {})
        missing = [k for k in self.parameters.get("required", []) if k not in args]
        if missing:
            raise ValidationError(f"{self.name}: thiếu tham số {missing}")
        unknown = [k for k in args if k not in props]
        if unknown:
            raise ValidationError(f"{self.name}: tham số không hợp lệ {unknown}")
        casts = {"integer": int, "number": float, "string": str, "boolean": bool}
        out = {}
        for key, value in args.items():
            kind = props[key].get("type")
            if kind in casts and kind != "boolean" and not isinstance(value, (list, dict)):
                value = casts[kind](value)
            if "enum" in props[key] and value not in props[key]["enum"]:
                raise ValidationError(f"{self.name}.{key} phải thuộc {props[key]['enum']}")
            out[key] = value
        return out


def _short(value, limit: int = 400) -> str:
    """Log-friendly repr: images and long payloads are cut."""
    if isinstance(value, dict) and "image_jpeg_b64" in value:
        value = {**value, "image_jpeg_b64": f"<{len(value['image_jpeg_b64'])} b64 chars>"}
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _deny_all(name: str, args: dict) -> bool:
    return False


class ActionExecutor:
    def __init__(self, specs: list[ActionSpec], mode: str = "dry_run",
                 confirm: ConfirmFn = _deny_all, allowed: list[str] | None = None):
        # all_specs: what Astra's programs/skills may call (internal=True; still validated and safety-checked);
        # specs: the tools the MODEL is offered. 23 Nov: program mode offers no motion tools, but programs move —
        # filtering the executor itself made every program fail with "Unknown action open_gripper".
        self.all_specs = {s.name: s for s in specs}
        for spec in specs:
            owner = getattr(spec.handler, "__self__", None)
            bind = getattr(owner, "bind_executor", None)
            if bind is not None:
                bind(self)
        self.specs = {k: s for k, s in self.all_specs.items() if not allowed or k in allowed}
        self.mode = mode
        self.confirm = confirm
        self.lifecycle = None       # private harness hook; never an Astra tool
        # set when a motion was stopped by the safety layer (contact / alarm / timeout -> Pause()). 23 Nov audit:
        # a program could catch the error and keep sending moves, which queue behind the Pause and would run on
        # Continue. While set, every motion is refused; the harness ends the episode on it. Cleared per episode.
        self.stopped: str | None = None
        # tools refused for the whole session, for model AND program calls: name -> reason (23 Nov threading
        # task: the gripper is set up holding the threading object and must not move)
        self.locked: dict[str, str] = {}
        # 24 Nov (operator): one picture per motion — after a picture the model's next picture is refused until
        # a motion has run (the agent kept looking and reasoning instead of moving). None = off.
        # 24 Nov: 1 -> 2 (the agent made fake 0.2 mm retractions to get a second view): look_limit pictures
        self.look_gate: set[str] | None = None
        self.look_limit = 1
        self.looks_since_motion = 0
        # per-episode facts session wrappers keep (e.g. which axes were calibrated); the harness clears it
        self.episode_state: dict = {}

    def list(self) -> list[ActionSpec]:
        return list(self.specs.values())

    def execute(self, name: str, args: dict | None = None, internal: bool = False) -> ActionResult:
        """internal=True: a call from a running program/skill (any registered tool); else a model call
        (only the offered tools)."""
        args = args or {}
        spec = (self.all_specs if internal else self.specs).get(name)
        if spec is None:
            return {"ok": False, "action": name, "executed": False, "error": f"Unknown action {name}"}
        if self.look_gate and not internal and name in self.look_gate and \
                self.looks_since_motion >= self.look_limit:
            n = self.look_limit
            return {"ok": False, "action": name, "executed": False,
                    "error": f"no new picture yet: {n} picture{'s' if n > 1 else ''} per motion — run a motion "
                             "(move_relative, insert or go_to_position) first; the motion and the next picture may "
                             "go in the same turn"}
        if name in self.locked:
            return {"ok": False, "action": name, "executed": False, "error": f"{name} is locked: {self.locked[name]}"}
        if self.stopped and spec.moves_robot and not spec.delegates:
            return {"ok": False, "action": name, "executed": False, "error_type": "MotionStopped",
                    "error": f"not sent: the robot was stopped earlier in this episode ({self.stopped[:160]})"}
        started = False
        try:
            clean = spec.validate_args(args)
            if self.mode == "dry_run" and not (spec.read_only or spec.delegates):
                log.info("[dry_run] %s %s", name, clean)
                return {"ok": True, "action": name, "executed": False,
                        "result": f"DRY RUN: {name}({clean}) validated, not sent."}
            if self.mode == "confirm" and spec.moves_robot and not spec.delegates and not self.confirm(name, clean):
                return {"ok": False, "action": name, "executed": False, "error": "Operator rejected the action."}
            watched = self.lifecycle is not None and ((spec.moves_robot and not spec.delegates) or name == "look")
            feedback = self.lifecycle.before(name, clean) if watched else {}
            started = True
            try:
                result = spec.handler(**clean)
            finally:
                if watched:
                    feedback = self.lifecycle.after(name, clean) or feedback
            if name == "stop":
                self.stopped = "stop() paused the robot"   # a move sent now would queue behind the Pause
            if self.look_gate and not internal:
                if name in self.look_gate:
                    self.looks_since_motion += 1
                elif spec.moves_robot and not (isinstance(result, dict) and result.get("ok") is False):
                    self.looks_since_motion = 0
            log.info("%s %s -> %s", name, clean, _short(result))
            if spec.reports_status and isinstance(result, dict) and result.get("ok") is False:
                return {"ok": False, "action": name, "executed": True, "result": result,
                        "error": result.get("error") or result.get("detail") or "action failed"}
            return {"ok": True, "action": name, "executed": True, "result": result,
                    **({"grasp_measurements": feedback} if feedback else {})}
        except (RobotError, OSError, ValueError, TypeError) as exc:
            log.warning("%s %s failed: %s", name, args, exc)
            kind = type(exc).__name__
            if kind in ("ContactError", "MotionStopped"):
                self.stopped = str(exc)
            # executed: True = the move started and was stopped on the way (contact/alarm/timeout: the arm is
            # somewhere between start and target); False = refused before anything was sent
            return {"ok": False, "action": name, "executed": started and kind in ("ContactError", "MotionStopped"),
                    "error": str(exc), "error_type": kind}

    def run_sequence(self, steps: list[dict], stop_on_error: bool = True) -> list[ActionResult]:
        """Run [{"tool": name, "args": {...}}, …] — a composite skill built from primitives."""
        results = []
        for step in steps:
            result = self.execute(step["tool"], step.get("args", {}))
            results.append(result)
            if stop_on_error and not result["ok"]:
                break
        return results
