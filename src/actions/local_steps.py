"""Local checkpoints and routines that skills use without asking Astra.

Checks (skill step {"check": name, "args": {...}}) only READ the robot and cameras.
They return {"ok": bool, "ambiguous": bool, "detail": str, "camera": ...}:
ambiguous = the evidence could not decide (object occluded …) -> Astra decides,
with an image.

Routines (skill step {"routine": name, "args": {...}}) are tested loops in
code. Astra can add its own and replace these (actions/astra_code.py, files in
src/agent/astra_skills/); the primitives they call stay fixed. Every motion they make goes through the ActionExecutor, so dry_run / confirm
/ live, argument validation and all safety checks (table floor, approach zone,
contact sensing, view guard, IK, speed cap) apply exactly as for Astra's own calls.

The wrist-centring numbers come from the successful episode 20261112-181222:
at z_top + 30 the green cube was grasped and lifted when the bottom-centre of its
wrist-image box was at ≈ (355, 140) px; tool +dy 10 mm moved it ≈ +56 px in u,
tool +dx 10 mm ≈ −26 px in v.
"""
from __future__ import annotations

import math
import re

from common.errors import RobotError


class LocalSteps:
    def __init__(self, primitives, scene, executor=None, block_top_z: float | None = None,
                 settle_s: float = 0.4, frame_retries: int = 3, code=None):
        self.p, self.scene, self.executor = primitives, scene, executor
        self.block_top_z = block_top_z      # taught cube top (workcell.json), None = unknown
        # Astra's own routines/checks (src/agent/astra_skills): replace core ones of the same name
        from actions.astra_code import AstraCode
        self.code = code if code is not None else AstraCode.default()
        if self.code.core is None:
            self.code.core = type(self)
        # The wrist frame comes from the UI stream and can lag a motion by a few hundred
        # ms: wait before reading it, and re-read a few frames before saying "not seen".
        self.settle_s, self.frame_retries = settle_s, frame_retries

    # ------------------------------------------------------------------ helpers
    def _tip(self) -> list[float]:
        return self.p.get_state()["pose"]

    def _dry(self) -> bool:
        return self.executor is not None and self.executor.mode == "dry_run"

    def _move(self, **args) -> dict:
        return self.executor.execute("move_relative", {"frame": "base", **args})

    def _wrist(self, color: str, min_area: float = 0) -> list[dict]:
        import time
        objs: list[dict] = []
        for i in range(max(1, self.frame_retries)):
            if self.settle_s:
                time.sleep(self.settle_s)
            objs = [o.summary() for o in self.scene.find(color, "wrist")]
            if any(o["area_px"] >= min_area for o in objs):
                break
        return objs

    def _env(self, color: str) -> list[dict]:
        return [o.summary() for o in self.scene.find(color, "env")]

    @staticmethod
    def _anchor(obj: dict) -> tuple[float, float]:
        x, y, w, h = obj["bbox"]
        return x + w / 2, y + h                           # bottom-centre: not clipped at the image top

    # ------------------------------------------------------------------ checks
    def check(self, name: str, memo: dict, **args) -> dict:
        own = self.code.get("check", name) if self.code is not None else None
        fn = getattr(self, f"check_{name}", None)
        if own is None and fn is None:
            raise ValueError(f"unknown check '{name}' (known: {sorted(self.known('check'))})")
        if self._dry() and not (own and own["dry_run"]):
            return {"ok": True, "detail": f"dry_run: {name} not evaluated"}
        try:
            if own is not None:
                return self.code.call(self, "check", name, memo, args)
            return fn(memo, **args)
        except (RobotError, OSError) as exc:
            return {"ok": False, "ambiguous": True, "detail": f"{name}: {exc}"}

    def check_robot_ready(self, memo) -> dict:
        s = self.p.get_state()
        ok = s.get("mode") == 5 and not s.get("error")
        return {"ok": ok, "detail": "" if ok else f"robot not ready (mode {s.get('mode')}, alarm={s.get('error')})"}

    def check_z_top_plausible(self, memo, z_top: float, tol_mm: float = 8.0, min_above_table_mm: float = 15.0) -> dict:
        """z_top given to a pick must be above the table (a cube on the box is) and agree
        with the taught cube top (if one is taught)."""
        table = getattr(self.p, "table_z_mm", None) if self.p is not None else None
        if table is not None and float(z_top) < table + min_above_table_mm:
            return {"ok": False, "detail": f"z_top={float(z_top):.1f} is not above the table ({table:.1f} + "
                    f"{min_above_table_mm:.0f} mm): the env-camera heights are wrong — verify_calibration (and "
                    "recalibrate if it fails) or use a measured contact height"}
        if self.block_top_z is None:
            return {"ok": True, "detail": "no taught cube top to compare with"}
        diff = float(z_top) - self.block_top_z
        ok = abs(diff) <= tol_mm
        return {"ok": ok, "detail": "" if ok else
                f"z_top={float(z_top):.1f} differs from the taught cube top {self.block_top_z:.1f} by {diff:+.1f} mm "
                "(env-camera Z error?) — call again with z_top = STATE.block_top_z"}

    def check_above_zone(self, memo, tol_mm: float = 3.0) -> dict:
        top = self.p.approach_top()
        z = self._tip()[2]
        ok = top is None or z >= top - tol_mm
        return {"ok": ok, "detail": "" if ok else f"fingertip z={z:.0f} is inside the approach zone (top {top:.0f})"}

    def check_holding(self, memo, max_position: int = 850) -> dict:
        g = memo.get("close_gripper") or {}
        pos, holding = g.get("position"), g.get("holding")
        ok = bool(holding) and pos is not None and pos < max_position
        return {"ok": ok, "detail": "" if ok else f"grasp missed: holding={holding} position={pos}"
                + (" (edge contact, would slip)" if holding else " (fingers met nothing)"), "camera": "wrist"}

    def check_wrist_sees(self, memo, color: str, min_area: float = 1500) -> dict:
        objs = [o for o in self._wrist(color, min_area) if o["area_px"] >= min_area]
        ok = bool(objs)
        return {"ok": ok, "ambiguous": not ok, "camera": "wrist",
                "detail": "" if ok else f"wrist camera does not see {color}"}

    def check_wrist_clear(self, memo, color: str, max_area: float = 15000) -> dict:
        big = [o for o in self._wrist(color) if o["area_px"] > max_area]
        return {"ok": not big, "camera": "wrist",
                "detail": "" if not big else f"{color} still fills the wrist view (area {big[0]['area_px']:.0f}): still held?"}

    def check_env_ref(self, memo, color: str, x: float | None = None, y: float | None = None) -> dict:
        """Remember where the env camera sees the block now (pixel), for check 'lifted'."""
        objs = self._env(color)
        if objs and x is not None and y is not None and objs[0].get("base_xyz_mm"):
            objs.sort(key=lambda o: math.hypot(o["base_xyz_mm"][0] - x, o["base_xyz_mm"][1] - y)
                      if o.get("base_xyz_mm") else 1e9)
        memo["env_ref"] = objs[0]["pixel"] if objs else None
        return {"ok": True, "detail": "" if objs else f"env does not see {color} now (no reference)"}

    def check_lifted(self, memo, color: str, min_shift_px: float = 8) -> dict:
        ref = memo.get("env_ref")
        objs = self._env(color)
        if ref is None:
            return {"ok": False, "ambiguous": True, "camera": "env", "detail": "no env reference of the block"}
        if any(math.dist(o["pixel"], ref) < min_shift_px for o in objs):
            return {"ok": False, "camera": "env", "detail": f"{color} is still where it was on the box: not lifted"}
        # gone from its place is not enough: a dropped cube also left it (16 Nov 16:11)
        g = (self.p.get_state().get("gripper") or {}) if self.p is not None else {}
        status = str(g.get("grip_status", "")).lower()
        if g.get("holding") is False or "drop" in status or "no object" in status:
            return {"ok": False, "camera": "env", "detail": f"{color} left its place but the gripper is not holding it "
                    f"({g.get('grip_status')}, position {g.get('position')}): dropped or knocked off, not lifted"}
        if objs:
            return {"ok": True, "detail": f"{color} moved away from its place (env) and the gripper holds it"}
        return {"ok": False, "ambiguous": True, "camera": "env",
                "detail": f"env camera does not see {color} (occluded?): cannot tell if it was lifted"}

    def check_placed(self, memo, color: str, x: float, y: float, radius_mm: float = 40) -> dict:
        objs = [o for o in self._env(color) if o.get("base_xyz_mm")]
        near = [o for o in objs if math.hypot(o["base_xyz_mm"][0] - x, o["base_xyz_mm"][1] - y) <= radius_mm]
        if near:
            return {"ok": True, "detail": f"{color} seen at {near[0]['base_xyz_mm'][:2]}"}
        if objs:
            return {"ok": False, "camera": "env",
                    "detail": f"{color} seen at {objs[0]['base_xyz_mm'][:2]}, not within {radius_mm} mm of target"}
        return {"ok": False, "ambiguous": True, "camera": "env", "detail": f"env camera does not see {color}"}

    # ------------------------------------------------------------------ routines
    def routine(self, name: str, memo: dict, **args) -> dict:
        if self.code is not None and self.code.get("routine", name) is not None:
            return self.code.call(self, "routine", name, memo, args)      # Astra's version wins
        fn = getattr(self, f"routine_{name}", None)
        if fn is None:
            raise ValueError(f"unknown routine '{name}' (known: {sorted(self.known('routine'))})")
        return fn(memo, **args)

    def known(self, kind: str) -> set[str]:
        """Core + Astra's own routines / checks."""
        core = CHECKS if kind == "check" else ROUTINES
        return core | (self.code.names(kind) if self.code is not None else set())

    # lifting straight up can leave the arm's reach at the far edge of the box: then step
    # sideways TOWARDS the robot base (short, slow, contact-sensed) and keep lifting
    DETOUR_MM, MAX_DETOURS = 15.0, 3

    def _vertical(self, z: float, step_mm: float, speed: int | None, sign: int) -> dict:
        start = self._tip()[2]
        cur, moves, detours = start, 0, 0
        while (cur - z) * sign < -0.5 and moves < 40:
            dz = sign * min(step_mm, abs(z - cur))
            r = self._move(dz=round(dz, 2), **({"speed": speed} if speed else {}))
            moves += 1
            if not r["ok"] and sign > 0 and "not reachable" in str(r.get("error", "")) and detours < self.MAX_DETOURS:
                x, y = self._tip()[:2]
                d = math.hypot(x, y)
                if d > 1:
                    hop = self._move(dx=round(-self.DETOUR_MM * x / d, 2), dy=round(-self.DETOUR_MM * y / d, 2),
                                     speed=1)
                    detours += 1
                    if hop["ok"]:
                        continue                              # retry the lift from the detour
            if not r["ok"]:
                return {"ok": False, "detail": r.get("error", ""), "z": cur, "moves": moves, "error_result": r,
                        **({"detours": detours} if detours else {})}
            pose = (r.get("result") or {}).get("pose") if isinstance(r.get("result"), dict) else None
            cur = pose[2] if pose else cur + dz                # dry_run: planned z
        return {"ok": True, "z": round(cur, 1), "moves": moves,
                **({"detours": detours, "detail": f"moved {detours * self.DETOUR_MM:.0f} mm towards the base "
                    "to stay in reach while lifting"} if detours else {})}

    def routine_descend_to(self, memo, z: float, step_mm: float = 19, speed: int | None = None) -> dict:
        """Straight down to fingertip z in steps (contact sensing stops it on a surface)."""
        return self._vertical(float(z), float(step_mm), speed, -1)

    def routine_lift_to(self, memo, z: float, step_mm: float = 23, speed: int | None = None) -> dict:
        """Straight up to fingertip z (only upward moves)."""
        return self._vertical(float(z), float(step_mm), speed, +1)

    # tool-frame offsets (mm) tried when the cube is not in the wrist view. The camera looks
    # past the fingers, so a cube that is missed usually lies beyond the top of the image:
    # -dx (moves the image content down) is tried first. Each hop ≤ 25 mm (approach zone).
    SEARCH = [(-15, 0), (-30, 0), (-15, 15), (-15, -15), (0, 15), (0, -15), (-30, 15), (-30, -15), (15, 0)]

    def routine_wrist_search(self, memo, color: str, min_area: float = 1500, speed: int | None = 1,
                             pattern: list | None = None) -> dict:
        """Find the cube with the wrist camera by small sideways moves at the current height.
        Ends where it was seen (ok) or back at the start (not seen)."""
        if self._dry() or any(o["area_px"] >= min_area for o in self._wrist(color, min_area)):
            return {"ok": True, "offset": [0, 0], "moves": 0}
        cur = (0.0, 0.0)
        for i, (dx, dy) in enumerate(pattern or self.SEARCH):
            r = self._tool_hop(dx - cur[0], dy - cur[1], speed)
            if not r["ok"]:
                return {"ok": False, "detail": r.get("error", ""), "offset": list(cur), "moves": i + 1,
                        "error_result": r}
            cur = (dx, dy)
            if any(o["area_px"] >= min_area for o in self._wrist(color, min_area)):
                memo["search_offset"] = [dx, dy]
                return {"ok": True, "offset": [dx, dy], "moves": i + 1,
                        "detail": f"found {color} at tool offset dx={dx} dy={dy} mm from the given x,y"}
        self._tool_hop(-cur[0], -cur[1], speed)
        return {"ok": False, "ambiguous": True, "camera": "wrist", "moves": len(pattern or self.SEARCH) + 1,
                "detail": f"{color} not in the wrist view within ±30 mm of the given x,y at this height — "
                          "x,y (env camera) or z_top is off by more than that"}

    def routine_probe_top(self, memo, z_est: float, below_mm: float = 25, step_mm: float = 8,
                          speed: int | None = 1, grip: int = 500) -> dict:
        """Measure the cube top by touch: fingers half closed (narrower than the cube, so a
        fingertip lands ON it), straight down in small slow steps until the contact sensing
        of the approach zone stops the arm (it then backs off by itself). The measured Z is
        stored as variable measured_top for the next steps. No contact down to
        z_est - below_mm = the fingers went past the cube (not centred) -> fail."""
        if self._dry():
            memo.setdefault("vars", {})["measured_top"] = float(z_est)
            return {"ok": True, "measured_top": float(z_est), "detail": "dry_run: not probed"}
        specs = getattr(self.executor, "specs", None)
        r = (self.executor.execute("set_gripper", {"position": int(grip)}) if specs is None or "set_gripper" in specs
             else self.executor.execute("close_gripper", {}))     # explore: close_gripper stops at 50 %
        if not r["ok"]:
            return {"ok": False, "detail": r.get("error", ""), "error_result": r}
        z_min = float(z_est) - float(below_mm)
        cur = self._tip()[2]
        while cur > z_min + 0.5:
            r = self._move(dz=-round(min(step_mm, cur - z_min), 2), **({"speed": speed} if speed else {}))
            if not r["ok"]:
                m = re.search(r"CONTACT at fingertip Z=(-?[\d.]+)", str(r.get("error", "")))
                if not m:
                    return {"ok": False, "detail": r.get("error", ""), "error_result": r}
                top = float(m.group(1))
                memo.setdefault("vars", {})["measured_top"] = top
                return {"ok": True, "measured_top": top,
                        "detail": f"cube top measured by touch at z={top:.1f} (estimate was {float(z_est):.1f})"}
            pose = (r.get("result") or {}).get("pose")
            cur = pose[2] if pose else cur - step_mm
        return {"ok": False, "camera": "wrist", "detail": f"no contact down to z={z_min:.1f}: the fingers went past "
                "the cube (not centred) or the cube is lower than estimated"}

    def _tool_hop(self, dx: float, dy: float, speed: int | None, max_mm: float = 20.0) -> dict:
        """Sideways tool-frame move split into pieces ≤ max_mm (approach-zone limit is 25)."""
        n = max(1, math.ceil(math.hypot(dx, dy) / max_mm))
        r = {"ok": True}
        for _ in range(n):
            r = self.executor.execute("move_relative", {"dx": round(dx / n, 2), "dy": round(dy / n, 2), "frame": "tool",
                                                        **({"speed": speed} if speed else {})})
            if not r["ok"]:
                break
        return r

    def routine_wrist_center(self, memo, color: str, target_u: float = 355, target_v: float = 140,
                             u_per_dy: float = 5.6, v_per_dx: float = -2.6, tol_px: float = 12,
                             max_iter: int = 6, max_step_mm: float = 15, gain: float = 0.8,
                             speed: int | None = 1, max_travel_mm: float = 35) -> dict:
        """Centre the block between the fingers with tool-frame dx/dy from the wrist image.
        Never wanders more than max_travel_mm (≤ 50) from where it started: further away it is
        chasing another object or the px/mm mapping is wrong for this height."""
        errs = []
        travel_limit = min(float(max_travel_mm), 50.0)
        tx = ty = 0.0
        for i in range(int(max_iter)):
            objs = sorted(self._wrist(color, 500), key=lambda o: -o["area_px"])
            if not objs:
                return {"ok": False, "ambiguous": True, "camera": "wrist", "iterations": i,
                        "detail": f"wrist camera lost {color}"}
            u, v = self._anchor(objs[0])
            eu, ev = u - target_u, v - target_v
            errs.append(round(math.hypot(eu, ev), 1))
            if abs(eu) <= tol_px and abs(ev) <= tol_px:
                return {"ok": True, "iterations": i, "error_px": [round(eu), round(ev)], "anchor": [round(u), round(v)]}
            if len(errs) >= 3 and errs[-1] > 0.9 * errs[-3]:
                return {"ok": False, "camera": "wrist", "iterations": i, "errors_px": errs,
                        "detail": "centring does not converge (mapping wrong for this block/height?)"}
            dy = max(-max_step_mm, min(max_step_mm, -gain * eu / u_per_dy))
            dx = max(-max_step_mm, min(max_step_mm, -gain * ev / v_per_dx))
            if math.hypot(tx + dx, ty + dy) > travel_limit:
                return {"ok": False, "camera": "wrist", "iterations": i, "errors_px": errs,
                        "travel_mm": [round(tx, 1), round(ty, 1)],
                        "detail": f"centring would move {math.hypot(tx + dx, ty + dy):.0f} mm from its start "
                                  f"(limit {travel_limit:.0f}): it is chasing another object or the px/mm mapping "
                                  "is wrong at this height — look wrist and check which blob is the cube"}
            tx, ty = tx + dx, ty + dy
            r = self.executor.execute("move_relative", {"dx": round(dx, 1), "dy": round(dy, 1), "frame": "tool",
                                                        **({"speed": speed} if speed else {})})
            if not r["ok"]:
                return {"ok": False, "detail": r.get("error", ""), "iterations": i + 1, "error_result": r}
            if self._dry():
                return {"ok": True, "iterations": 1, "detail": "dry_run: one correction validated"}
        return {"ok": False, "camera": "wrist", "iterations": int(max_iter), "errors_px": errs,
                "detail": f"not centred after {max_iter} corrections"}


CHECKS = {n[len("check_"):] for n in dir(LocalSteps) if n.startswith("check_")}
ROUTINES = {n[len("routine_"):] for n in dir(LocalSteps) if n.startswith("routine_")}
# what recovery steps may do after a failure: only open the fingers or go up
RECOVERY_TOOLS = {"open_gripper"}
RECOVERY_ROUTINES = {"lift_to"}
