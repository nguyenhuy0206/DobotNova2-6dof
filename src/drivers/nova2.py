"""Dobot Nova 2 (V3 controller) TCP/IP driver.

Ports:
  29999  dashboard  — Enable/Disable/ClearError/SpeedFactor/Modbus…
  30003  motion     — JointMovJ, RelJointMovJ … (queued)
  30004  feedback   — 1440-byte realtime packet, 8 ms

Originally designed from the small local architecture of
https://github.com/salmondavidp/DOBOT-Nova5-Control-UI, rewritten for Nova 2 V3.
"""
from __future__ import annotations

import socket
import struct
import time

from common.errors import RobotError
from common.types import Feedback

DASHBOARD_PORT = 29999
MOTION_PORT = 30003
FEEDBACK_PORT = 30004
FEEDBACK_SIZE = 1440
FEEDBACK_MAGIC = 0x123456789ABCDEF
NOVA2_ROBOT_TYPE = 101


def recv_frame(sock: socket.socket, timeout: float) -> str:
    """Read one ``;``-terminated dashboard reply."""
    data = bytearray()
    deadline = time.monotonic() + timeout
    while b";" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RobotError("Controller không trả phản hồi đầy đủ.")
        sock.settimeout(remaining)
        part = sock.recv(4096)
        if not part:
            raise RobotError("Controller đóng kết nối.")
        data.extend(part)
        if len(data) > 65536:
            raise RobotError("Phản hồi controller quá lớn.")
    return bytes(data).split(b";", 1)[0].decode("utf-8", "replace") + ";"


def dashboard_response_ok(raw: str, command: str) -> str:
    if not raw.lstrip().startswith("0,"):
        raise RobotError(f"Controller từ chối {command}: {raw}")
    return raw


def parse_feedback(data: bytes) -> Feedback:
    """Decode a V3 realtime feedback packet.

    Offsets follow Dobot's official V3 ``MyType`` layout. Padding after byte
    fields means they are not simply consecutive from an eyeballed layout.
    """
    if len(data) != FEEDBACK_SIZE or struct.unpack_from("<q", data, 0)[0] != FEEDBACK_SIZE:
        raise RobotError("Không nhận đúng gói feedback V3 (1440 byte).")
    if struct.unpack_from("<Q", data, 48)[0] != FEEDBACK_MAGIC:
        raise RobotError("Gói feedback không khớp giao thức V3.")
    robot_type = data[1031]
    if robot_type != NOVA2_ROBOT_TYPE:
        raise RobotError(f"robot_type={robot_type}; app này chỉ dành cho Nova 2 (101).")
    return {
        "mode": struct.unpack_from("<Q", data, 24)[0],
        "enabled": data[1026], "running": data[1028], "error": data[1029],
        "robot_type": robot_type,
        "joints_deg": list(struct.unpack_from("<6d", data, 432)),
        "pose": list(struct.unpack_from("<6d", data, 624)),
    }


def format_values(values: list[float], fmt: str = "g") -> str:
    return ",".join(f"{value:{fmt}}" for value in values)


class Nova2:
    """Thin, stateless wrapper: one short TCP connection per call."""

    def __init__(self, ip: str, timeout: float = 3):
        self.ip, self.timeout = ip, timeout

    # -- raw ports -----------------------------------------------------
    def dashboard(self, command: str) -> str:
        with socket.create_connection((self.ip, DASHBOARD_PORT), self.timeout) as sock:
            sock.sendall(command.encode("ascii"))
            return dashboard_response_ok(recv_frame(sock, self.timeout), command)

    def feedback(self) -> Feedback:
        with socket.create_connection((self.ip, FEEDBACK_PORT), self.timeout) as sock:
            packet = bytearray()
            deadline = time.monotonic() + self.timeout
            while len(packet) < FEEDBACK_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RobotError("Timeout feedback port 30004.")
                sock.settimeout(remaining)
                block = sock.recv(FEEDBACK_SIZE - len(packet))
                if not block:
                    raise RobotError("Feedback bị đóng trước khi đủ 1440 byte.")
                packet.extend(block)
        return parse_feedback(bytes(packet))

    def motion(self, command: str) -> str:
        """V3 motion server may acknowledge immediately or not at all."""
        with socket.create_connection((self.ip, MOTION_PORT), self.timeout) as sock:
            sock.sendall(command.encode("ascii"))
            sock.settimeout(0.6)
            try:
                raw = sock.recv(4096).decode("utf-8", "replace").strip()
            except socket.timeout:
                raw = ""
        if raw and not raw.lstrip().startswith("0,"):
            raise RobotError(f"Controller từ chối {command}: {raw}")
        return raw or "Đã gửi vào motion queue."

    # -- dashboard commands ---------------------------------------------
    def enable(self) -> str:
        return self.dashboard("EnableRobot()")

    def disable(self) -> str:
        return self.dashboard("DisableRobot()")

    def clear_error(self) -> str:
        return self.dashboard("ClearError()")

    def stop(self) -> str:
        return self.dashboard("ResetRobot()")

    def speed_factor(self, speed: int) -> str:
        return self.dashboard(f"SpeedFactor({speed})")

    # -- motion commands (no validation here; see actions.validation) --
    def rel_joint_movj(self, deltas: list[float], speed: int) -> str:
        return self.motion(f"RelJointMovJ({format_values(deltas)},SpeedJ={speed},AccJ={speed})")

    def joint_movj(self, joints: list[float], speed: int) -> str:
        return self.motion(f"JointMovJ({format_values(joints, '.4f')},SpeedJ={speed},AccJ={speed})")
