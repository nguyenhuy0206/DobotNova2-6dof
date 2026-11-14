"""Simulated Nova 2: subclasses the real driver so the real command strings are
built and parsed, but nothing touches the network."""
from __future__ import annotations

import re
import struct

import numpy as np

from drivers.nova2 import Nova2
from perception.transforms import pose_to_matrix


class SimNova2(Nova2):
    def __init__(self, mode: int = 5, enabled: int = 1, error: int = 0, running: int = 0):
        super().__init__("sim", 0.1)
        self.state = {"mode": mode, "enabled": enabled, "running": running, "error": error, "robot_type": 101,
                      "joints_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
                      "pose": [300.0, 0.0, 400.0, 180.0, 0.0, 0.0]}
        self.dashboard_log: list[str] = []
        self.motion_log: list[str] = []
        self.stall = False        # True: accept moves but never arrive
        self._next_master = 0
        self.fk = None            # joints -> flange pose; enables PositiveSolution (like the real controller)
        self.ik = None            # flange pose -> joints; enables InverseSolution
        self.active_tool = None   # [x,y,z] mm: feedback reports this tool, not the flange;
                                  # also enables FK (= feedback - R·tool)

    def dashboard(self, command: str) -> str:
        self.dashboard_log.append(command)
        if command.startswith(("ModbusRTUCreate", "ModbusCreate")):
            return f"0,{{{self._next_master}}},{command};"
        fk = self.fk
        if fk is None and self.active_tool is not None:
            fk = lambda j: self._shift(self.state["pose"], -1)
        for name, fn in (("PositiveSolution", fk), ("InverseSolution", self.ik)):
            if command.startswith(name) and fn is not None:
                nums = [float(x) for x in re.match(r"\w+\((.*)\)", command).group(1).split(",")[:6]]
                return "0,{" + ",".join(f"{v:.6f}" for v in fn(nums)) + "}," + command + ";"
        return f"0,{{}},{command};"

    def feedback(self):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.state.items()}

    def motion(self, command: str) -> str:
        self.motion_log.append(command)
        if self.stall:
            return "queued"
        name, args = re.match(r"(\w+)\((.*)\)", command).groups()
        nums = [float(x) for x in args.split(",") if "=" not in x]
        s = self.state
        if name == "RelJointMovJ":
            s["joints_deg"] = [j + d for j, d in zip(s["joints_deg"], nums[:6])]
        elif name == "JointMovJ":
            s["joints_deg"] = nums[:6]
        elif name in ("MovJ", "MovL"):
            s["pose"] = self._shift(nums[:6], +1) if "Tool=0" in command else nums[:6]
        elif name == "RelMovLUser":
            s["pose"] = [p + d for p, d in zip(s["pose"], nums[:6])]
        elif name == "RelMovLTool":
            R = pose_to_matrix(s["pose"])[:3, :3]
            xyz = np.array(s["pose"][:3]) + R @ np.array(nums[:3])
            s["pose"] = [*map(float, xyz), *s["pose"][3:]]
        return "Đã gửi vào motion queue."

    def _shift(self, pose, sign):
        """flange <-> active tool point (sign +1: flange -> tool)."""
        if self.active_tool is None:
            return list(pose)
        R = pose_to_matrix(pose)[:3, :3]
        xyz = np.array(pose[:3]) + sign * (R @ np.array(self.active_tool))
        return [*map(float, xyz), *pose[3:]]


class FakeGripper:
    """Same API as PGCGripperDirect; an object is 'present' if object_at is set."""

    def __init__(self, object_at: int | None = None):
        self.object_at, self.position, self.connected_with = object_at, 1000, None

    def connect(self, slave_id=1, baud=115200, parity="N"):
        self.connected_with = (slave_id, baud, parity)

    def init(self):
        return self.status()

    def status(self):
        caught = self.object_at is not None and self.position == self.object_at
        return {"initialized": 1, "grip_state": 2 if caught else 1, "position": self.position}

    def go_to(self, position, force=30, speed=50, wait=True):
        self.position = max(position, self.object_at) if self.object_at is not None else position
        return self.status()

    def open(self, force=30, speed=50):
        return self.go_to(1000, force, speed)

    def close(self, force=30, speed=50):
        return self.go_to(0, force, speed)


def make_feedback_packet(mode=5, enabled=1, running=0, error=0, robot_type=101,
                         joints=(1, 2, 3, 4, 5, 6), pose=(10, 20, 30, 40, 50, 60)) -> bytes:
    data = bytearray(1440)
    struct.pack_into("<q", data, 0, 1440)
    struct.pack_into("<Q", data, 24, mode)
    struct.pack_into("<Q", data, 48, 0x123456789ABCDEF)
    struct.pack_into("<6d", data, 432, *joints)
    struct.pack_into("<6d", data, 624, *pose)
    data[1026], data[1028], data[1029], data[1031] = enabled, running, error, robot_type
    return bytes(data)
