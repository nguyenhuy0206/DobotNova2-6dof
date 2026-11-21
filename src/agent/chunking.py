"""Action chunking: Astra may propose several tool calls in one turn.

20 Nov: free_chunks skips zone horizons and verification boundaries, subject to the
per-turn call/action limits. Errors and unexpected events still abort the turn.
Strict policy (free_chunks off), horizon L (robot-moving actions per proposal):
  * contact-rich (gripper, descending inside the approach zone)   L = 1
  * uncertain alignment (sideways / turning inside the zone)      L ≤ 2
  * verified free-space motion (above the approach zone)          L ≤ max_horizon (4)
Read-only calls (get_state, look, notes…) do not count, but after a verification
boundary (close/open gripper, run_skill, calibration solve) no further MOTION runs:
the result has to be seen first. A gripper call that cannot change anything (open_gripper
when the fingers are already fully open, set_gripper to where they already are) is neither
a boundary nor a counted motion. Any error / unexpected event aborts the rest
(see agent.events.ABORTS_CHUNK). Actions run one after another, never in parallel,
and every motion is still validated and confirmed by robot feedback in primitives.
"""
from __future__ import annotations

from dataclasses import dataclass

BOUNDARY = {"close_gripper", "open_gripper", "set_gripper", "run_skill", "stop"}
MOTION = {"move_to_pose", "move_to_joints", "move_relative", "jog_joint", "go_to_position",
          "search_object"} | BOUNDARY


@dataclass
class ChunkPolicy:
    enabled: bool = True
    max_horizon: int = 4          # strict: 1 … 4; free: harness uses its larger action limit
    max_calls: int = 8            # read-only calls included
    # free (20 Nov, operator): Astra plans as many actions as it wants; no horizon by zone, no verification
    # boundary. Every action is still validated (target, path, joint/self collision, table) when it runs,
    # and the rest of the plan is dropped after any error, refusal or unexpected event (ABORTS_CHUNK).
    free: bool = False

    def horizon_for(self, name: str, args: dict, tip_z: float | None, zone_top: float | None) -> int:
        if self.free:
            return self.max_horizon
        if name in BOUNDARY:
            return 1
        if zone_top is None or tip_z is None:
            return min(2, self.max_horizon)               # no table taught / state unknown: be careful
        target_z = _target_z(name, args, tip_z)
        in_zone = tip_z < zone_top or (target_z is not None and target_z < zone_top)
        if not in_zone:
            return self.max_horizon
        if target_z is None or target_z < tip_z - 0.5:
            return 1                                      # going down near the block / table
        return min(2, self.max_horizon)                   # sideways / turning / lifting in the zone

    def split(self, calls: list, tip_z: float | None, zone_top: float | None,
              grip: float | None = None) -> tuple[list, list, str]:
        """(calls to run now, calls deferred, reason). calls: objects with .name and parsed .args.
        grip: measured gripper position (0 closed … 1000 open), to recognise no-op gripper calls."""
        if not self.enabled:
            return list(calls), [], ""
        run, limit, moves, z = [], self.max_horizon, 0, tip_z
        crossed_boundary = False
        for i, call in enumerate(calls):
            name, args = call.name, getattr(call, "parsed_args", {}) or {}
            if len(run) >= self.max_calls:
                return run, list(calls[i:]), f"at most {self.max_calls} calls per turn"
            if name not in MOTION or _gripper_noop(name, args, grip):
                run.append(call)
                continue
            if crossed_boundary and not self.free:
                return run, list(calls[i:]), "verification boundary: look at the result before moving again"
            limit = min(limit, self.horizon_for(name, args, z, zone_top))
            if moves + 1 > limit:
                return run, list(calls[i:]), f"safe horizon here is L={limit}"
            run.append(call)
            moves += 1
            z = _target_z(name, args, z) if z is not None else None
            if name in BOUNDARY:
                crossed_boundary = True
            if name in ("close_gripper", "open_gripper", "set_gripper"):
                grip = None                               # unknown until measured again
        return run, [], ""


def _gripper_noop(name: str, args: dict, grip: float | None) -> bool:
    if grip is None:
        return False
    if name == "open_gripper":
        return grip >= 990
    if name == "set_gripper" and isinstance(args.get("position"), (int, float)):
        return abs(float(args["position"]) - grip) <= 5
    return False


def _target_z(name: str, args: dict, tip_z: float | None) -> float | None:
    """Fingertip Z after the call, when it can be known without the robot (else None)."""
    if name == "move_to_pose" and "z" in args:
        return float(args["z"])
    if name == "move_relative" and tip_z is not None:
        if args.get("frame", "base") == "base":
            return tip_z + float(args.get("dz", 0) or 0)
        if not any(args.get(k) for k in ("dz", "drx", "dry")):
            return tip_z                                  # tool dx/dy/drz with the gripper down: sideways
        return None                                       # tool-frame Z / tilt: unknown, treat as descending
    if name in ("close_gripper", "open_gripper", "set_gripper", "stop"):
        return tip_z
    return None
