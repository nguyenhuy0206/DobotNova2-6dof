"""HTTP-facing glue for the primitive skills (used by the Debug tab).

Kept separate from app.py so the web server stays a thin router:

    GET  /api/skills          -> catalogue (name, description, JSON-schema params, flags)
    GET  /api/skills/state    -> live robot + gripper state (never blocks on a running move)
    POST /api/skills/run      -> {"name": ..., "args": {...}, "confirm": true}
    POST /api/gripper/driver  -> {"op": status | init | go_to, …}: the gripper driver of THIS process,
                                 for other processes (gripper.via_ui) — the app is the bus's only master
"""
from __future__ import annotations

from actions.executor import ActionExecutor
from actions.primitives import PrimitiveSkills
from actions.registry import build_registry
from actions.validation import require_confirm
from common.errors import ValidationError


class SkillsAPI:
    def __init__(self, primitives: PrimitiveSkills):
        self.primitives = primitives
        # The operator pressing a button is the confirmation, so the executor
        # runs live; motion still requires the request's explicit confirm flag.
        self.executor = ActionExecutor(build_registry(primitives), mode="live")

    def catalogue(self) -> dict:
        return {"skills": [{"name": s.name, "description": s.description, "parameters": s.parameters,
                            "moves_robot": s.moves_robot, "read_only": s.read_only}
                           for s in self.executor.list()],
                "defaults": {"speed": self.primitives.m.default_speed,
                             "gripper_force": self.primitives.cfg.gripper.force,
                             "gripper_speed": self.primitives.cfg.gripper.speed}}

    def state(self) -> dict:
        try:
            return self.primitives.get_state()
        except OSError as exc:   # timeout / refused / no route
            ip = self.primitives.cfg.ip
            raise OSError(f"Cannot reach controller {ip} ({exc}). Check the Mac's Ethernet "
                          f"adapter is 192.168.5.100/24 and the CCBOX is in TCP/IP mode.") from exc

    def system(self, data: dict) -> dict:
        """Enable / disable / clear alarm (dashboard commands, not motion skills)."""
        action = str(data.get("action", ""))
        if action in ("enable", "power_on"):
            require_confirm(data.get("confirm"), "Bật 'Workspace clear' trước khi Enable (nhả phanh).")
        if action == "drag_on":
            require_confirm(data.get("confirm"), "Bật 'Workspace clear' trước khi Drag (tay đỡ robot).")
        return {"ok": True, "action": action, "result": self.primitives.a.system(action)}

    def run(self, data: dict) -> dict:
        name = str(data.get("name", ""))
        spec = self.executor.specs.get(name)
        if spec is None:
            raise ValidationError(f"Unknown skill: {name}")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValidationError("args must be an object")
        if spec.moves_robot:
            require_confirm(data.get("confirm"), "Bật 'Workspace clear' trước khi chạy lệnh chuyển động.")
        return self.executor.execute(name, args)

    def gripper_driver(self, data: dict) -> dict:
        """Driver-level gripper calls from other processes (drivers.gripper.PGCGripperRemote)."""
        g = self.primitives.gripper                     # opened + initialized on first use, one lock
        op = str(data.get("op", ""))
        if op == "status":
            return g.status()
        if op == "init":
            return g.init(force_reinit=bool(data.get("force_reinit")), full=bool(data.get("full")))
        if op == "go_to":
            require_confirm(data.get("confirm"), "gripper motion needs confirm")
            return g.go_to(int(data["position"]), int(data.get("force", 30)), int(data.get("speed", 50)),
                           wait=bool(data.get("wait", True)))
        raise ValidationError(f"unknown gripper op {op!r} (status | init | go_to)")
