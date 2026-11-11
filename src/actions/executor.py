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

    def list(self) -> list[ActionSpec]:
        return list(self.specs.values())

    def execute(self, name: str, args: dict | None = None) -> ActionResult:
        args = args or {}
        spec = self.specs.get(name)
        if spec is None:
            return {"ok": False, "action": name, "executed": False, "error": f"Unknown action {name}"}
        try:
            clean = spec.validate_args(args)
            if self.mode == "dry_run" and not spec.read_only:
                log.info("[dry_run] %s %s", name, clean)
                return {"ok": True, "action": name, "executed": False,
                        "result": f"DRY RUN: {name}({clean}) validated, not sent."}
            if self.mode == "confirm" and spec.moves_robot and not self.confirm(name, clean):
                return {"ok": False, "action": name, "executed": False, "error": "Operator rejected the action."}
            result = spec.handler(**clean)
            log.info("%s %s -> %s", name, clean, result)
            return {"ok": True, "action": name, "executed": True, "result": result}
        except (RobotError, OSError, ValueError, TypeError) as exc:
            log.warning("%s %s failed: %s", name, args, exc)
            return {"ok": False, "action": name, "executed": False, "error": str(exc)}
