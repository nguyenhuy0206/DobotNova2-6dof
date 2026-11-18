"""Primitive skills: the smallest robot behaviours an agent composes into tasks.

    observe   get_state
    arm       move_to_pose · move_to_joints · move_relative · jog_joint
    memory    go_to_position · save_position
    gripper   open_gripper · close_gripper · set_gripper
    safety    stop

Cartesian poses are the FINGERTIP (real flange + gripper.tool_length_mm along tool Z):
what you pass to move_to_pose and what get_state reports.

Three frames, kept apart on purpose:
  controller pose  what feedback reports and Mov* commands take: the controller's
                   ACTIVE tool (on this robot Tool 1, ~125 mm off the flange!)
  flange           the real mounting face = PositiveSolution(joints, tool 0)
  fingertip        flange + tool_length_mm along the flange Z axis
The active-tool offset is measured from the controller (FK vs feedback) before
every motion, so a tool selected on the pendant can't shift the fingertip again.

Guards (all optional, set by create_primitives / the explore toolkit):
  table_z_mm          fingertip/flange keep-out above the table, checked along the
                      whole path (joint moves via controller FK, every ~2°)
  view_guard()        after a move: is the robot still visible to the env camera?
                      If not, the move is undone and the call fails
  grip_close_position close_gripper stops here (e.g. 500 = 50 %) instead of 0

Every motion primitive:  validate → check robot ready → send → wait until arrived
(or fail + stop on timeout/alarm) → return the resulting state. Each call blocks,
so a sequence of primitives is a sequence of finished motions — which is what
an agent or a learned composite skill needs.

Command syntax follows Dobot's TCP-IP-Python-V3 SDK (external/…/dobot_api.py).
"""
from __future__ import annotations

import dataclasses

import math
import threading
import time
from typing import Callable

import numpy as np

from actions import validation as v
from actions.robot_actions import RobotActions
from actions import self_collision
from common.errors import ContactError, RobotError, SafetyError, ValidationError
from common.log import get_logger
from common.types import Feedback
from perception.transforms import interpolate_rotation, matrix_to_rpy_deg, pose_to_matrix, rotation_angle_deg

log = get_logger(__name__)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class PrimitiveSkills:
    def __init__(self, actions: RobotActions, gripper_factory: Callable[[], object] | None = None):
        self.a = actions
        self.robot = actions.robot
        self.cfg = actions.config
        self.m = actions.config.motion
        self._gripper_factory = gripper_factory
        self._gripper = None
        self._gripper_lock = threading.Lock()
        self.tool_length_mm = float(actions.config.gripper.tool_length_mm)
        # open fingertips sit ± this far from the tool axis (along flange Y): when the gripper tilts,
        # the lower one is what reaches the table first, so the floor is checked on it
        self.finger_half_gap_mm = float(getattr(actions.config.gripper, "finger_half_gap_mm", 0.0))
        self.speed_cap: int | None = None   # set for the agent (agent.yaml max_speed)
        self.table = actions.config.table
        self.table_z_mm: float | None = None              # data/memory/workcell.json
        self.view_guard: Callable[[], dict] | None = None # explore mode
        self.grip_close_position: int | None = None       # None = close fully
        self.grip_open_position: int | None = None        # None = open fully (1000)
        self.grip_speed: int | None = None                # fixed finger speed % for open/close (None = per call)
        # controller active tool, translation in the flange frame (mm); measured per motion
        self.active_tool_offset = np.zeros(3)
        self.tool_offset_source = "not measured yet"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _speed(self, speed: int | None) -> int:
        value = self.m.default_speed if speed is None else int(speed)
        if self.speed_cap is not None and value > self.speed_cap:
            log.info("speed %d%% capped to %d%%", value, self.speed_cap)
            value = self.speed_cap
        return v.check_speed(value, self.cfg.speed)

    # -- frames: feedback pose (active tool) / flange / fingertip ---------------
    # Commands always name Tool=0, User=0 (the real flange) — never the implicit
    # active tool, which the controller may switch (it changed after ResetRobot).
    # Feedback still reports the active tool, so it is converted with the offset
    # measured from controller FK.
    @staticmethod
    def _rot(pose: list[float]) -> np.ndarray:
        return pose_to_matrix(pose)[:3, :3]

    def _tool_axis(self, pose: list[float]):
        return self._rot(pose)[:, 2]                # tool Z axis in the base frame

    def measure_tool_offset(self, fb: Feedback | None = None) -> np.ndarray:
        """Where the feedback's active tool sits relative to the real flange.

        PositiveSolution(joints, tool 0) = flange; feedback = active tool. A robot
        without controller FK (simulator) is assumed to report the flange; once FK
        has worked, losing it is an error (never silently assume 0 again).
        """
        fb = fb or self.robot.feedback()
        had_fk = self.tool_offset_source == "controller FK"
        last_exc = None
        for attempt in range(3 if had_fk else 1):
            try:
                flange = self.robot.positive_solution(fb["joints_deg"])
                break
            except (RobotError, OSError, AttributeError) as exc:
                last_exc = exc
                time.sleep(0.2 * (attempt + 1))
        else:
            if had_fk:
                raise RobotError(f"controller FK unavailable ({last_exc}) — refusing to guess the tool offset")
            if self.tool_offset_source != "assumed flange (no controller FK)":
                log.warning("no controller FK (%s) — assuming feedback is the flange", last_exc)
            self.tool_offset_source = "assumed flange (no controller FK)"
            self.active_tool_offset = np.zeros(3)
            return self.active_tool_offset
        Rf = self._rot(flange)
        if rotation_angle_deg(Rf, self._rot(fb["pose"])) > 1.0:
            raise SafetyError("The controller's active tool frame is rotated relative to the flange; "
                              "select Tool 0 (or a tool without rotation) on the controller.")
        self.active_tool_offset = Rf.T @ (np.array(fb["pose"][:3]) - np.array(flange[:3]))
        self.tool_offset_source = "controller FK"
        return self.active_tool_offset

    def flange_from_controller(self, pose: list[float]) -> list[float]:
        """Feedback pose (active tool) -> real flange pose."""
        p = np.array(pose[:3]) - self._rot(pose) @ self.active_tool_offset
        return [*map(float, p), *pose[3:]]

    def tip_from_flange(self, pose: list[float]) -> list[float]:
        p = np.array(pose[:3]) + self._tool_axis(pose) * self.tool_length_mm
        return [*map(float, p), *pose[3:]]

    def flange_from_tip(self, pose: list[float]) -> list[float]:
        p = np.array(pose[:3]) - self._tool_axis(pose) * self.tool_length_mm
        return [*map(float, p), *pose[3:]]

    def tip_from_controller(self, pose: list[float]) -> list[float]:
        return self.tip_from_flange(self.flange_from_controller(pose))

    def _lowest_tip(self, centre, R: np.ndarray) -> list[float]:
        """The lowest point of the fingertips: the centre, or an open fingertip when the gripper tilts."""
        c = np.asarray(centre[:3], float)
        if not self.finger_half_gap_mm:
            return list(map(float, c))
        side = R[:, 1] * self.finger_half_gap_mm
        return list(map(float, min((c, c + side, c - side), key=lambda q: q[2])))

    def _points(self, p: np.ndarray, R: np.ndarray) -> tuple[list[float], list[float]]:
        """(lowest fingertip xyz, flange xyz) for a flange position/rotation."""
        return self._lowest_tip(p + R[:, 2] * self.tool_length_mm, R), list(p)

    def _summary(self, fb: Feedback) -> dict:
        flange = self.flange_from_controller(fb["pose"])
        return {"pose": [round(x, 2) for x in self.tip_from_flange(flange)],
                "flange_pose": [round(x, 2) for x in flange],
                "joints_deg": [round(x, 2) for x in fb["joints_deg"]]}

    def _ready(self) -> Feedback:
        """Robot ready to move + fresh active-tool offset."""
        fb = self.a._motion_ready()
        self.measure_tool_offset(fb)
        return fb

    # -- table keep-out --------------------------------------------------------
    def floor_limits(self) -> tuple[float, float] | None:
        """(min fingertip Z, min flange Z) or None when the table isn't taught."""
        if self.table_z_mm is None:
            return None
        return (self.table_z_mm + self.table.tip_clearance_mm, self.table_z_mm + self.table.flange_clearance_mm)

    def _check_points(self, tip: list[float], flange: list[float], where: str = "target",
                      use_floor: bool = True) -> None:
        """Fingertip and flange inside the workspace and above the table. A 6-value tip pose is
        checked at its lowest open fingertip."""
        if len(tip) >= 6:
            tip = self._lowest_tip(tip, self._rot(tip))
        ws, floors = self.cfg.workspace, (self.floor_limits() if use_floor else None)
        for name, point, zmin in (("fingertip", tip, floors and floors[0]), ("flange", flange, floors and floors[1])):
            box = ws if zmin is None else dataclasses.replace(ws, z=(zmin, ws.z[1]))
            try:
                v.check_pose_in_workspace(point, box)
            except ValidationError as exc:
                extra = (f" (table Z={self.table_z_mm:g}; {name} must stay ≥ {zmin:g})"
                         if zmin is not None and point[2] < zmin else "")
                raise ValidationError(f"{where}: {name} {exc}{extra}") from None

    def _check_linear_path(self, start: list[float], end: list[float], samples: int = 10) -> None:
        """Straight-line controller motion: interpolate position + rotation, check each point."""
        p0, p1 = np.array(start[:3]), np.array(end[:3])
        R0, R1 = self._rot(start), self._rot(end)
        for k in range(1, samples + 1):
            s = k / samples
            tip, flange = self._points(p0 + (p1 - p0) * s, interpolate_rotation(R0, R1, s))
            self._check_points(tip, flange, "target" if k == samples else f"path {k * 100 // samples}%")

    def _fk_points(self, joints: list[float]) -> tuple[list[float], list[float]]:
        flange = self.robot.positive_solution(joints)
        R = self._rot(flange)
        return self._lowest_tip(np.array(flange[:3]) + R[:, 2] * self.tool_length_mm, R), flange[:3]

    def _check_joint_path(self, start: list[float], end: list[float]) -> None:
        """Joint-interpolated motion: FK every ~path_step_deg; only needed near a table."""
        if self.floor_limits() is None:
            return
        span = max(abs(e - s) for s, e in zip(start, end))
        n = min(90, max(1, math.ceil(span / self.table.path_step_deg)))
        for k in range(1, n + 1):
            joints = [s + (e - s) * k / n for s, e in zip(start, end)]
            try:
                tip, flange = self._fk_points(joints)
            except (RobotError, OSError) as exc:
                raise SafetyError(f"Cannot check the joint path against the table (no FK: {exc}).") from None
            self._check_points(tip, flange, "target" if k == n else f"joint path {k * 100 // n}%")

    # -- self-collision: gripper vs the robot's own elbow / arm (actions/self_collision.py) -----
    def _self_check_on(self, fb: Feedback) -> bool:
        """On when enabled AND the kinematic model reproduces the flange the controller reports
        now (within 5 mm): another robot, or a simulator, is never judged by the wrong model."""
        if (self.m.self_clearance_mm <= 0 and self.m.wrist_clearance_mm <= 0) \
                or self.tool_offset_source != "controller FK":
            return False
        model = self_collision.flange_pose(fb["joints_deg"])[:3]
        return float(np.linalg.norm(np.subtract(model, self.flange_from_controller(fb["pose"])[:3]))) <= 5.0

    def _self_check_joints(self, fb: Feedback, end: list[float]) -> None:
        if self._self_check_on(fb):
            self_collision.check_joint_path(fb["joints_deg"], end, self.tool_length_mm, self.m.self_clearance_mm,
                                            wrist_min_mm=self.m.wrist_clearance_mm)

    def _self_check_linear(self, fb: Feedback, start: list[float], target: list[float]) -> None:
        if self._self_check_on(fb):
            self_collision.check_linear_path(fb["joints_deg"], start, target, self.tool_length_mm,
                                             self.m.self_clearance_mm, wrist_min_mm=self.m.wrist_clearance_mm)

    def _ik(self, flange_pose: list[float]) -> list[float] | None:
        """Joints for a flange pose from the controller's IK. Raises ValidationError
        when unreachable (instead of letting the move alarm); None when the
        controller can't tell (simulator)."""
        if self.tool_offset_source != "controller FK":
            return None
        try:
            return self.robot.inverse_solution(flange_pose)
        except ValidationError:
            raise
        except (RobotError, OSError):
            return None

    # -- approach zone: short, slow moves near the table; stall = contact -----------
    def approach_top(self) -> float | None:
        return None if self.table_z_mm is None else self.table_z_mm + self.m.approach_height_mm

    def _approach(self, tip0: list[float], tip1: list[float], speed: int) -> tuple[int, bool]:
        """(speed to use, contact sensing on?) for a straight fingertip move tip0 -> tip1.
        Raises ValidationError when the part of the move inside the zone is too long."""
        top = self.approach_top()
        if top is None or (tip0[2] >= top and tip1[2] >= top):
            return speed, False
        a, b = np.array(tip0[:3], float), np.array(tip1[:3], float)
        if b[2] >= a[2] and np.hypot(*(b - a)[:2]) < 5:
            return speed, False                          # (nearly) straight up: leaving, fine
        entry = a if a[2] <= top else a + (b - a) * (a[2] - top) / (a[2] - b[2])
        inside = float(np.linalg.norm(b - entry))
        if inside > self.m.approach_max_step_mm + 0.5:
            raise ValidationError(
                f"approach zone (fingertip below Z={top:.0f}): at most {self.m.approach_max_step_mm:g} mm per "
                f"move there, this one is {inside:.0f} mm. Stop at Z≈{top:.0f}, then go down/sideways in "
                "small steps and look with the wrist camera in between.")
        return min(speed, self.m.approach_speed), True

    def _recover_from_alarm(self) -> bool:
        """ClearError, re-enable if the controller dropped to disabled, and wait until it is
        ready (mode 5, no alarm, enabled). False = it did not come back: stay stopped."""
        deadline = time.monotonic() + self.m.recovery_timeout_s
        enabled_once = False
        try:
            self.robot.clear_error()
            while time.monotonic() < deadline:
                time.sleep(max(self.m.poll_s, 0.2))
                fb = self.robot.feedback()
                if not fb["error"] and fb.get("mode") == 5 and fb.get("enabled", 1):
                    return True
                if not fb["error"] and not enabled_once and (fb.get("mode") == 4 or not fb.get("enabled", 1)):
                    self.robot.enable()
                    enabled_once = True
                elif fb["error"]:
                    self.robot.clear_error()
        except (RobotError, OSError):
            return False
        return False

    def _back_off(self) -> list[float]:
        """After a contact: straight up by contact_backoff_mm (pure lift, always allowed)."""
        fb = self.robot.feedback()
        flange = self.flange_from_controller(fb["pose"])
        up = [flange[0], flange[1], flange[2] + self.m.contact_backoff_mm, *flange[3:]]
        self.robot.movl(up, self.m.approach_speed)
        self._wait(self._pose_reached(up), "back off")
        return self.tip_from_flange(flange)

    # -- env camera must keep seeing the robot ------------------------------------
    def _after_motion(self, result: dict, undo: Callable[[], None]) -> dict:
        if self.view_guard is None:
            return result
        info = self.view_guard()
        if info.get("visible"):
            return {**result, "robot_in_view": info}
        undo()
        raise SafetyError("Robot left the env camera view — moved back to where it was. "
                          f"Try a smaller move or another direction. ({info.get('detail', '')})")

    def _undo_joints(self, joints: list[float], speed: int) -> Callable[[], None]:
        def undo():
            self.robot.joint_movj(joints, speed)
            self._wait(self._joints_reached(joints), "undo")
        return undo

    def _undo_linear(self, flange_pose: list[float], speed: int) -> Callable[[], None]:
        def undo():
            self.robot.movl(flange_pose, speed)
            self._wait(self._pose_reached(flange_pose), "undo")
        return undo

    def _pose_reached(self, target: list[float], check_rotation: bool = True) -> Callable[[Feedback], bool]:
        """Arrived when the FLANGE (feedback converted) matches the flange target."""
        def reached(fb: Feedback) -> bool:
            pose = self.flange_from_controller(fb["pose"])
            pos_ok = all(abs(c - t) <= self.m.arrive_tol_mm for c, t in zip(pose[:3], target[:3]))
            rot_ok = not check_rotation or all(
                _angle_diff(c, t) <= self.m.arrive_tol_deg for c, t in zip(pose[3:], target[3:]))
            return pos_ok and rot_ok
        return reached

    def _joints_reached(self, target: list[float]) -> Callable[[Feedback], bool]:
        return lambda fb: all(abs(c - t) <= self.m.arrive_tol_deg for c, t in zip(fb["joints_deg"], target))

    def _wait(self, reached: Callable[[Feedback], bool] | None, label: str, contact: bool = False) -> dict:
        """Block until the move finishes. Stops the robot on alarm or timeout.
        contact=True: a move that stops making progress is blocked by something —
        stop, back off upward and raise ContactError."""
        start = time.monotonic()
        last_p, last_t, moved = None, start, False
        while True:
            now = time.monotonic()
            elapsed = now - start
            fb = self.robot.feedback()
            if fb["error"]:
                self.robot.stop()
                if contact and self.m.collision_recovery:
                    tip = self._summary(fb)["pose"]
                    if self._recover_from_alarm():
                        self._back_off()
                        raise ContactError(
                            f"{label}: CONTACT at fingertip Z={tip[2]:.1f} (X={tip[0]:.1f}, Y={tip[1]:.1f}) — the "
                            "controller's collision detection stopped the arm (alarm cleared, robot enabled again) "
                            f"and it moved up {self.m.contact_backoff_mm:g} mm. That Z is where something blocked "
                            "the gripper.")
                raise SafetyError(f"{label}: robot alarm during motion — stopped.")
            idle = fb["running"] == 0
            if idle and (reached(fb) if reached else elapsed >= self.m.start_grace_s):
                return self._summary(fb)
            if contact:
                p = np.array(fb["pose"][:3])
                if last_p is None or np.linalg.norm(p - last_p) > self.m.stall_mm:
                    moved = moved or last_p is not None
                    last_p, last_t = p, now
                elif (moved or elapsed > 3.0) and now - last_t > self.m.stall_s:
                    self.robot.stop()
                    tip = self._summary(fb)["pose"]
                    self._back_off()
                    raise ContactError(
                        f"{label}: CONTACT at fingertip Z={tip[2]:.1f} (X={tip[0]:.1f}, Y={tip[1]:.1f}) — "
                        f"something blocked the gripper; stopped and moved up {self.m.contact_backoff_mm:g} mm. "
                        "That Z is a measured surface height: use it.")
            if elapsed > self.m.move_timeout_s:
                self.robot.stop()
                raise RobotError(f"{label}: not arrived after {self.m.move_timeout_s}s — stopped.")
            time.sleep(self.m.poll_s)

    @property
    def gripper(self):
        """Connected + initialized gripper, created on first use."""
        with self._gripper_lock:
            if self._gripper is None:
                if self._gripper_factory is None:
                    raise RobotError("No gripper configured.")
                g = self._gripper_factory()
                gc = self.cfg.gripper
                g.connect(gc.slave_id, gc.baud, gc.parity)
                g.init()
                self._gripper = g
            return self._gripper

    # ------------------------------------------------------------------
    # 1. observe
    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """Robot mode/alarm, joints, TCP pose, gripper (if connected), saved position names."""
        fb = self.robot.feedback()
        self.measure_tool_offset(fb)
        state = {"mode": fb["mode"], "enabled": bool(fb["enabled"]), "error": bool(fb["error"]),
                 "moving": bool(fb["running"]), "motion_locked": self.a.motion_locked,
                 **self._summary(fb),
                 "saved_positions": [p["name"] for p in self.a.positions.all()]}
        floors = self.floor_limits()
        if floors:
            state["table"] = {"table_z_mm": self.table_z_mm, "min_fingertip_z": floors[0],
                              "min_flange_z": floors[1]}
        if self._gripper is not None:
            state["gripper"] = self._gripper.status()
        return state

    # ------------------------------------------------------------------
    # 2–5. arm motion
    # ------------------------------------------------------------------
    def move_to_pose(self, x: float, y: float, z: float, rx: float, ry: float, rz: float,
                     linear: bool = False, speed: int | None = None) -> dict:
        """Move the FINGERTIP to an absolute base-frame pose.

        linear: straight line (MovL, Tool=0). Otherwise joint-space: the controller's
        IK gives the target joints and JointMovJ goes there (path checked against the
        table). Near the table the move is always a straight line.
        """
        tip = v.check_pose6([x, y, z, rx, ry, rz])
        speed = self._speed(speed)
        with self.a.lock:
            fb = self._ready()
            start = self.flange_from_controller(fb["pose"])
            target = self.flange_from_tip(tip)
            note = None
            self._check_points(tip, target[:3])     # floor / workspace first: the clearer error
            speed, contact = self._approach(self.tip_from_flange(start), tip, speed)
            if contact:
                linear = True                       # near the table: always straight, slow, sensed
            joints = self._ik(target)              # unreachable -> ValidationError, no alarm
            if not linear and self.floor_limits() is not None:
                low = self.table_z_mm + self.table.movj_min_height_mm
                try:
                    if min(self.tip_from_flange(start)[2], tip[2]) < low:
                        raise ValidationError("near the table")
                    if joints is None:
                        raise RobotError("no controller IK to check the joint path")
                    self._check_joint_path(fb["joints_deg"], joints)
                except (RobotError, OSError) as why:
                    linear, note = True, f"used a straight line (MovL): {why}"
            if linear:
                self._check_linear_path(start, target)
                self._self_check_linear(fb, start, target)
                self.robot.movl(target, speed)
                result = self._wait(self._pose_reached(target), "move_to_pose", contact=contact)
                undo = self._undo_linear(start, speed)
            else:
                self._check_points(tip, target[:3])
                if joints is not None:
                    self._self_check_joints(fb, joints)
                    self.robot.joint_movj(joints, speed)
                    result = self._wait(self._joints_reached(joints), "move_to_pose")
                else:
                    self.robot.movj(target, speed)
                    result = self._wait(self._pose_reached(target), "move_to_pose")
                undo = self._undo_joints(fb["joints_deg"], speed)
            if note:
                result["note"] = note
            return self._after_motion(result, undo)

    def move_to_joints(self, joints: list[float], speed: int | None = None) -> dict:
        """Move all six joints to absolute angles (JointMovJ)."""
        joints = v.check_joints(joints)
        v.check_joint_limits(joints, self.m.joint_limits_deg)
        speed = self._speed(speed)
        with self.a.lock:
            fb = self._ready()
            self._check_joint_path(fb["joints_deg"], joints)
            self._self_check_joints(fb, joints)
            self.robot.joint_movj(joints, speed)
            result = self._wait(self._joints_reached(joints), "move_to_joints")
            return self._after_motion(result, self._undo_joints(fb["joints_deg"], speed))

    def move_relative(self, dx: float = 0, dy: float = 0, dz: float = 0,
                      drx: float = 0, dry: float = 0, drz: float = 0,
                      frame: str = "base", speed: int | None = None) -> dict:
        """Straight-line offset of the FINGERTIP, in the base or tool frame.
        Rotations pivot around the fingertip. Sent as an absolute MovL (Tool=0)."""
        if frame not in ("base", "tool"):
            raise ValidationError("frame phải là 'base' hoặc 'tool'.")
        delta = v.check_pose6([dx, dy, dz, drx, dry, drz])
        v.check_relative(delta, self.m.max_relative_mm, self.m.max_relative_deg)
        speed = self._speed(speed)
        with self.a.lock:
            fb = self._ready()
            start = self.flange_from_controller(fb["pose"])
            R0 = self._rot(start)
            tip0 = np.array(self.tip_from_flange(start)[:3])
            D = self._rot([0, 0, 0, *delta[3:]])
            if frame == "base":
                tip1, R1 = tip0 + np.array(delta[:3]), D @ R0
            else:
                tip1, R1 = tip0 + R0 @ np.array(delta[:3]), R0 @ D
            flange1 = tip1 - R1[:, 2] * self.tool_length_mm
            target = [*map(float, flange1), *matrix_to_rpy_deg(R1)]
            # A pure lift only increases clearance, so it is allowed even when starting
            # below the safe floor (e.g. right after teaching the table).
            lift = frame == "base" and delta[0] == delta[1] == 0 and not any(delta[3:]) and delta[2] > 0
            if lift:
                self._check_points(list(tip1), list(flange1), use_floor=False)
            else:
                self._check_linear_path(start, target)
                self._self_check_linear(fb, start, target)
            speed, contact = self._approach(list(tip0), list(tip1), speed)
            self._ik(target)
            self.robot.movl(target, speed)
            result = self._wait(self._pose_reached(target), "move_relative", contact=contact)
            return self._after_motion(result, self._undo_linear(start, speed))

    def jog_joint(self, joint: int, degrees: float, speed: int | None = None) -> dict:
        """Rotate one joint by a small relative angle."""
        joint, degrees = int(joint), float(degrees)
        v.check_joint_step(joint, degrees, self.cfg.joint_step)
        speed = self._speed(speed)
        with self.a.lock:
            fb = self._ready()
            target = list(fb["joints_deg"])
            target[joint - 1] += degrees
            v.check_joint_limits(target, self.m.joint_limits_deg)
            self._check_joint_path(fb["joints_deg"], target)
            self._self_check_joints(fb, target)
            deltas = [0.0] * 6
            deltas[joint - 1] = degrees
            self.robot.rel_joint_movj(deltas, speed)
            result = self._wait(self._joints_reached(target), "jog_joint")
            return self._after_motion(result, self._undo_joints(fb["joints_deg"], speed))

    # ------------------------------------------------------------------
    # 6–7. named positions
    # ------------------------------------------------------------------
    def go_to_position(self, name: str, speed: int | None = None) -> dict:
        """Move to a saved position by name. 'home' falls back to config motion.home_joints."""
        entry = self.a.positions.find_by_name(name)
        if entry is not None:
            return self.move_to_joints(entry["joints_deg"], speed)
        if name == "home":
            return self.move_to_joints(list(self.m.home_joints), speed)
        raise ValidationError(f"Không có vị trí '{name}'. Có: {[p['name'] for p in self.a.positions.all()]}")

    def save_position(self, name: str) -> dict:
        """Remember the current joints + pose under a name (overwrites the same name)."""
        name = v.check_name(name)
        with self.a.lock:
            fb = self.robot.feedback()
            existing = self.a.positions.find_by_name(name)
            if existing:
                self.a.positions.delete(existing["id"])
            self.a.positions.add(name, fb["joints_deg"], fb["pose"])
        return {"saved": name, **self._summary(fb)}

    # ------------------------------------------------------------------
    # 8–10. gripper
    # ------------------------------------------------------------------
    def _g(self, force, speed):
        gc = self.cfg.gripper
        if self.grip_speed is not None:
            speed = self.grip_speed                          # fixed: whatever the caller asked
        return (gc.force if force is None else int(force)), (gc.speed if speed is None else int(speed))

    def open_gripper(self, force: int | None = None, speed: int | None = None) -> dict:
        """Open the fingers (fully, or to grip_open_position — wide enough for a block
        but not so wide that the fingers hit its neighbours)."""
        if self.grip_open_position is None:
            return self.gripper.open(*self._g(force, speed))
        return self.gripper.go_to(self.grip_open_position, *self._g(force, speed))

    def close_gripper(self, force: int | None = None, speed: int | None = None) -> dict:
        """Close until the fingers hit an object (or reach grip_close_position, default
        fully closed); reports 'holding'."""
        if self.grip_close_position is None:
            status = self.gripper.close(*self._g(force, speed))
        else:
            status = self.gripper.go_to(self.grip_close_position, *self._g(force, speed))
        return {**status, "holding": status.get("grip_state") == 2,
                "close_target": 0 if self.grip_close_position is None else self.grip_close_position}

    def set_gripper(self, position: int, force: int | None = None, speed: int | None = None) -> dict:
        """Move the fingers to 0 (closed) … 1000 (open)."""
        position = int(position)
        if self.grip_close_position is not None and position < self.grip_close_position:
            raise ValidationError(f"gripper may not close beyond {self.grip_close_position} (asked {position})")
        return self.gripper.go_to(position, *self._g(force, speed))

    # ------------------------------------------------------------------
    # safety
    # ------------------------------------------------------------------
    def stop(self) -> dict:
        """Stop immediately and clear the motion queue."""
        return {"result": self.robot.stop()}


def gripper_factory(actions: RobotActions, owner: bool = False) -> Callable[[], object]:
    """Pick the gripper transport from config (gripper.transport: direct | controller).
    gripper.via_ui: every process except the bus owner (ui/app.py, owner=True) goes through it."""
    from drivers.gripper import PGCGripper, PGCGripperDirect, PGCGripperRemote

    gc, cfg = actions.config.gripper, actions.config
    if gc.via_ui and not owner:
        return lambda: PGCGripperRemote(f"http://{cfg.ui_host}:{cfg.ui_port}")
    if gc.transport == "direct":
        return lambda: PGCGripperDirect(gc.port)
    return lambda: PGCGripper(actions.robot)


def create_primitives(robot_cfg=None, actions: RobotActions | None = None, owner: bool = False) -> PrimitiveSkills:
    """Wire up robot + gripper from config/robot.yaml. Gripper connects lazily.

    Pass an existing ``actions`` to share its lock, motion lock and position store
    (the UI does this so manual controls and skills stay consistent).
    """
    if actions is None:
        from common.config import load_robot_config
        from drivers.nova2 import Nova2

        cfg = robot_cfg or load_robot_config()
        actions = RobotActions(Nova2(cfg.ip, cfg.timeout_s), cfg)
    from memory.workcell import load_workcell

    prims = PrimitiveSkills(actions, gripper_factory(actions, owner=owner))
    prims.table_z_mm = load_workcell().get("table_z_mm")
    return prims
