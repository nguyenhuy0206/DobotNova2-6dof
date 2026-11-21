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
        self.specs = {s.name: s for s in specs}
        if allowed:
            self.specs = {k: s for k, s in self.specs.items() if k in allowed}
        self.mode = mode
        self.confirm = confirm
        self.lifecycle = None       # private harness hook; never an Astra tool

    def list(self) -> list[ActionSpec]:
        return list(self.specs.values())

    def execute(self, name: str, args: dict | None = None) -> ActionResult:
        args = args or {}
        spec = self.specs.get(name)
        if spec is None:
            return {"ok": False, "action": name, "executed": False, "error": f"Unknown action {name}"}
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
            try:
                result = spec.handler(**clean)
            finally:
                if watched:
                    feedback = self.lifecycle.after(name, clean) or feedback
            log.info("%s %s -> %s", name, clean, _short(result))
            return {"ok": True, "action": name, "executed": True, "result": result,
                    **({"grasp_measurements": feedback} if feedback else {})}
        except (RobotError, OSError, ValueError, TypeError) as exc:
            log.warning("%s %s failed: %s", name, args, exc)
            return {"ok": False, "action": name, "executed": False, "error": str(exc), "error_type": type(exc).__name__}

    def run_sequence(self, steps: list[dict], stop_on_error: bool = True) -> list[ActionResult]:
        """Run [{"tool": name, "args": {...}}, …] — a composite skill built from primitives."""
        results = []
        for step in steps:
            result = self.execute(step["tool"], step.get("args", {}))
            results.append(result)
            if stop_on_error and not result["ok"]:
                break
        return results
