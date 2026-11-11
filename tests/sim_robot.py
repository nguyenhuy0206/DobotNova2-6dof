"""In-memory stand-in for ``drivers.Nova2``: records commands, no network."""
from __future__ import annotations

import re
import struct


class SimNova2:
    def __init__(self, mode: int = 5, enabled: int = 1, error: int = 0, running: int = 0):
        self.ip, self.timeout = "sim", 0.1
        self.state = {"mode": mode, "enabled": enabled, "running": running, "error": error, "robot_type": 101,
                      "joints_deg": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
                      "pose": [300.0, 0.0, 400.0, 180.0, 0.0, 0.0]}
        self.dashboard_log: list[str] = []
        self.motion_log: list[str] = []
        self._next_master = 0

    def dashboard(self, command: str) -> str:
        self.dashboard_log.append(command)
        if command.startswith(("ModbusRTUCreate", "ModbusCreate")):
            return f"0,{{{self._next_master}}},{command};"
        return f"0,{{}},{command};"

    def feedback(self):
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.state.items()}

    def motion(self, command: str) -> str:
        self.motion_log.append(command)
        m = re.match(r"RelJointMovJ\(([^S]+),SpeedJ", command)
        if m:
            for i, d in enumerate(float(x) for x in m.group(1).split(",")):
                self.state["joints_deg"][i] += d
        return "Đã gửi vào motion queue."

    enable = lambda self: self.dashboard("EnableRobot()")          # noqa: E731
    disable = lambda self: self.dashboard("DisableRobot()")        # noqa: E731
    clear_error = lambda self: self.dashboard("ClearError()")      # noqa: E731
    stop = lambda self: self.dashboard("ResetRobot()")             # noqa: E731

    def speed_factor(self, speed: int) -> str:
        return self.dashboard(f"SpeedFactor({speed})")

    def rel_joint_movj(self, deltas, speed):
        from drivers.nova2 import format_values
        return self.motion(f"RelJointMovJ({format_values(deltas)},SpeedJ={speed},AccJ={speed})")

    def joint_movj(self, joints, speed):
        from drivers.nova2 import format_values
        return self.motion(f"JointMovJ({format_values(joints, '.4f')},SpeedJ={speed},AccJ={speed})")


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
