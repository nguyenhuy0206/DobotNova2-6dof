"""Fingertip pick/place sequences; every actuation delegates to the executor."""
from __future__ import annotations

import math

from actions.validation import check_joint_limits
from common.errors import RobotError, ValidationError


class PickPlace:
    def __init__(self, primitives):
        self.p = primitives
        self.executor = None

    def bind_executor(self, executor):
        self.executor = executor

    def _yaw(self, yaw, state, targets):
        current = state["pose"][5]
        centre = round((current - yaw) / 90)
        candidates = [yaw + 90 * k for k in range(centre - 4, centre + 5)]
        candidates = sorted((a for a in candidates if -360 <= a <= 360), key=lambda a: abs(a - current))
        for angle in candidates:
            try:
                for x, y, z in targets:
                    flange = self.p.flange_from_tip([x, y, z, 180, 0, angle])
                    joints = self.p._ik(flange, state["joints_deg"])
                    if joints is not None:
                        check_joint_limits(joints, self.p.m.joint_limits_deg)
            except (RobotError, ValueError):
                continue
            return angle
        raise ValidationError("no equivalent square grasp yaw within joint limits")

    @staticmethod
    def _numbers(*values):
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("pick/place coordinates and dimensions must be finite")

    def _step(self, steps, phase, tool, args):
        result = self.executor.execute(tool, args, internal=True)
        steps.append({"phase": phase, "tool": tool, "args": args, **result})
        if not result["ok"]:
            return {"ok": False, "failed_step": phase, "detail": result.get("error", "step failed"), "steps": steps}
        return None

    def _move(self, steps, phase, pose, start):
        zone = self.p.approach_top()
        distance = math.dist(start[:3], pose[:3])
        count = 1
        if zone is not None and min(start[2], pose[2]) < zone:
            count = max(1, math.ceil(distance / max(1, self.p.m.approach_max_step_mm)))
        for i in range(1, count + 1):
            xyz = [a + (b - a) * i / count for a, b in zip(start[:3], pose[:3])]
            args = dict(zip(("x", "y", "z", "rx", "ry", "rz"), [*xyz, *pose[3:]]))
            # review 23 Nov: the transit to "above" may start anywhere (home, the far box): a forced straight
            # line can cross the base or a singularity. Joint-space there; the primitive itself switches to a
            # straight line near the table. Descend / lift stay straight.
            args["linear"] = phase != "above"
            failed = self._step(steps, phase, "move_to_pose", args)
            if failed:
                return failed
        return None

    def _run(self, sequence, state):
        steps, pose = [], state["pose"]
        for phase, target in sequence:
            if isinstance(target, list):
                failed = self._move(steps, phase, target, pose)
                pose = target
            else:
                failed = self._step(steps, phase, target, {})
            if failed:
                return failed
        grip = next((s["result"] for s in steps if s["tool"] == "close_gripper" and
                     isinstance(s.get("result"), dict)), {})
        return {"ok": True, "steps": steps, "pose": pose,
                "holding": grip.get("holding"), "position": grip.get("position")}

    def pick_object(self, x, y, top_z, yaw_deg, height_mm=None):
        """yaw_deg is a base-frame footprint edge; square-equivalent grasps use the nearest feasible rz."""
        try:
            height = 26.0 if height_mm is None else float(height_mm)
            self._numbers(x, y, top_z, yaw_deg, height)
            if height <= 0:
                raise ValueError("height_mm must be positive")
            state = self.p.get_state()
            grasp = top_z - min(height / 2, 15)
            targets = [[x, y, z] for z in (top_z + 60, grasp, top_z + 80)]
            # Flange Y joins the fingers. At Rx=180 its XY direction is rz-90 degrees;
            # rz=yaw aligns closing with the footprint's short axis (or its square equivalent).
            rz = self._yaw(yaw_deg, state, targets)
            above, down, lift = [[*xyz, 180, 0, rz] for xyz in targets]
        except (RobotError, ValueError, TypeError) as exc:
            return {"ok": False, "failed_step": "prepare", "detail": str(exc), "steps": []}
        result = self._run([("open", "open_gripper"), ("above", above), ("descend", down),
                            ("close", "close_gripper"), ("lift", lift)], state)
        if result["ok"] and result.get("holding") is False:
            # review 23 Nov: an empty grasp was reported as ok=true
            return {**result, "ok": False, "failed_step": "grasp",
                    "detail": f"the fingers closed to position {result.get('position')} without holding anything"}
        return result

    def place_object(self, x, y, top_z, yaw_deg=None, held_height_mm=26):
        """Release with the held object's bottom 3 mm above the support; fingertips hold at its grasp height."""
        try:
            self._numbers(x, y, top_z, held_height_mm)
            if held_height_mm <= 0:
                raise ValueError("held_height_mm must be positive")
            state = self.p.get_state()
            above_z = top_z + held_height_mm + 60
            bottom_offset = held_height_mm - min(held_height_mm / 2, 15)
            down_z = top_z + bottom_offset + 3
            rz = state["pose"][5] if yaw_deg is None else self._yaw(yaw_deg, state, [[x, y, above_z], [x, y, down_z]])
            self._numbers(rz)
            above, down = [x, y, above_z, 180, 0, rz], [x, y, down_z, 180, 0, rz]
        except (RobotError, ValueError, TypeError) as exc:
            return {"ok": False, "failed_step": "prepare", "detail": str(exc), "steps": []}
        result = self._run([("above", above), ("descend", down), ("open", "open_gripper"), ("lift", above)], state)
        if result["ok"]:
            result["holding"] = False if self.executor.mode != "dry_run" else None
        return result
