"""High-level robot actions = validation + safety + driver call.

Both the manual UI and the Astra agent go through this class, so every safety
rule is enforced in exactly one place.
"""
from __future__ import annotations

import json
import threading
import time

from actions import validation as v
from common.config import RobotConfig
from common.errors import SafetyError, ValidationError
from common.log import get_logger
from common.paths import CAPTURES_DIR
from drivers.gripper import PGCGripper
from drivers.nova2 import Nova2
from memory.store import PositionStore

log = get_logger(__name__)


class RobotActions:
    def __init__(self, robot: Nova2, config: RobotConfig, positions: PositionStore | None = None):
        self.robot = robot
        self.config = config
        self.gripper = PGCGripper(robot)
        self.positions = positions or PositionStore()
        self.lock = threading.RLock()
        # Speed 0 in the UI = software motion lock (no command is sent).
        self.motion_locked = False

    # -- state -----------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            state = dict(self.robot.feedback())
            state["positions"] = self.positions.all()
            state["motion_locked"] = self.motion_locked
            state["gripper_connected"] = self.gripper.connected
            return state

    def _motion_ready(self):
        if self.motion_locked:
            raise SafetyError("Motion lock đang bật (speed = 0).")
        return v.require_motion_ready(self.robot.feedback())

    # -- dashboard -------------------------------------------------------
    _SYSTEM = {"enable": "enable", "disable": "disable", "clear": "clear_error", "stop": "stop"}

    def system(self, action: str) -> str:
        if action not in self._SYSTEM:
            raise ValidationError("Thao tác không hợp lệ.")
        with self.lock:
            log.info("system %s", action)
            return getattr(self.robot, self._SYSTEM[action])()

    def set_speed(self, speed: int) -> str:
        limits = self.config.speed
        if not 0 <= speed <= limits.max:
            raise ValidationError(f"Tốc độ phải trong 0–{limits.max}%.")
        with self.lock:
            if speed == 0:
                self.motion_locked = True
                return "Motion lock: app sẽ từ chối lệnh chuyển động khi speed = 0."
            self.motion_locked = False
            return self.robot.speed_factor(speed)

    # -- motion ----------------------------------------------------------
    def joint_step(self, joint: int, degrees: float, speed: int) -> str:
        v.check_joint_step(joint, degrees, self.config.joint_step)
        v.check_speed(speed, self.config.speed)
        with self.lock:
            self._motion_ready()
            deltas = [0.0] * 6
            deltas[joint - 1] = degrees
            log.info("joint_step J%d %+g° @%d%%", joint, degrees, speed)
            return self.robot.rel_joint_movj(deltas, speed)

    def move_joints(self, joints: list[float], speed: int) -> str:
        joints = v.check_joints(joints)
        v.check_speed(speed, self.config.speed)
        with self.lock:
            self._motion_ready()
            log.info("move_joints %s @%d%%", joints, speed)
            return self.robot.joint_movj(joints, speed)

    def move_to_saved(self, position_id: str, speed: int) -> str:
        entry = self.positions.get(position_id)
        if entry is None:
            raise ValidationError("Không tìm thấy vị trí lưu.")
        return self.move_joints(entry.get("joints_deg", []), speed)

    # -- saved positions -------------------------------------------------
    def capture_position(self, name: str) -> list:
        name = v.check_name(name)
        with self.lock:
            live = self.robot.feedback()
            return self.positions.add(name, live["joints_deg"], live["pose"])

    def delete_position(self, position_id: str) -> list:
        with self.lock:
            return self.positions.delete(position_id)

    # -- gripper ---------------------------------------------------------
    def gripper_connect(self, slave_id: int | None = None, baud: int | None = None, parity: str | None = None) -> dict:
        g = self.config.gripper
        with self.lock:
            return self.gripper.connect(slave_id or g.slave_id, baud or g.baud, (parity or g.parity).upper())

    def gripper_status(self) -> dict:
        with self.lock:
            return self.gripper.status()

    def gripper_initialize(self) -> str:
        with self.lock:
            return self.gripper.initialize()

    def gripper_move(self, position: int, force: int | None = None, speed: int | None = None) -> dict:
        g = self.config.gripper
        with self.lock:
            return self.gripper.move(position, force or g.force, speed or g.speed)

    def gripper_open(self) -> dict:
        return self.gripper_move(self.config.gripper.open_position)

    def gripper_close(self) -> dict:
        return self.gripper_move(self.config.gripper.closed_position)

    def gripper_disconnect(self) -> str:
        with self.lock:
            return self.gripper.disconnect()

    # -- inspection capture ----------------------------------------------
    def capture_inspection(self, name: str, jpeg: bytes) -> dict:
        name = str(name).strip() or "camera-check"
        if len(name) > 40:
            raise ValidationError("Tên operation tối đa 40 ký tự.")
        v.check_jpeg(jpeg)
        with self.lock:
            live = self.robot.feedback()
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            image_name = f"{stamp}-{int(time.time() * 1000) % 1000:03d}.jpg"
            (CAPTURES_DIR / image_name).write_bytes(jpeg)
            record = {"operation": name,
                      "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "image": f"data/captures/{image_name}",
                      "pose": live["pose"], "joints_deg": live["joints_deg"]}
            (CAPTURES_DIR / f"{image_name}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return record
