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

from common.log import get_logger
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


def _braced_numbers(raw: str, count: int, command: str) -> list[float]:
    """'0,{a,b,c},Cmd(...);' -> [a, b, c]."""
    start, end = raw.find("{"), raw.find("}")
    try:
        values = [float(x) for x in raw[start + 1:end].split(",")] if 0 <= start < end else []
    except ValueError:
        values = []
    if len(values) != count:
        raise RobotError(f"{command}: unexpected reply {raw!r}")
    return values


def format_values(values: list[float], fmt: str = "g") -> str:
    def one(value: float) -> str:
        text = f"{value:{fmt}}"
        return text.lstrip("-") if float(text) == 0 else text   # never "-0.0000"
    return ",".join(one(value) for value in values)


log = get_logger(__name__)


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
    def power_on(self) -> str:
        """Power the arm (mode 3 → 4). Takes ~10 s before EnableRobot() is accepted."""
        return self.dashboard("PowerOn()")

    def enable(self) -> str:
        return self.dashboard("EnableRobot()")

    def disable(self) -> str:
        return self.dashboard("DisableRobot()")

    def clear_error(self) -> str:
        return self.dashboard("ClearError()")

    # Stop commands tried in order (operator, 19 Nov): never ResetRobot() unless every one is refused.
    # ResetRobot changed the controller's active tool once and preceded controller shutdowns; Stop()
    # is not in this firmware (-10000).
    stop_commands: tuple[str, ...] = ("Pause()",)

    def stop(self) -> str:
        """Stop the motion. Falls back to ResetRobot() only if the controller refuses every stop
        command: a move that cannot be stopped is worse."""
        refused = []
        for cmd in self.stop_commands:
            try:
                return self.dashboard(cmd)
            except RobotError as exc:
                refused.append(f"{cmd}: {exc}")
        log.warning("every stop command refused (%s) — falling back to ResetRobot()", "; ".join(refused))
        return self.dashboard("ResetRobot()")

    def reset_robot(self) -> str:
        """ResetRobot(): only when the operator asks for it explicitly."""
        return self.dashboard("ResetRobot()")

    def start_drag(self) -> str:
        """Hand-guiding (mode 6): the arm can be pushed by hand. Needs the robot enabled and still."""
        return self.dashboard("StartDrag()")

    def stop_drag(self) -> str:
        return self.dashboard("StopDrag()")

    def speed_factor(self, speed: int) -> str:
        return self.dashboard(f"SpeedFactor({speed})")

    def set_collision_level(self, level: int) -> str:
        """Controller collision detection: 0 = off, 1–5 = more sensitive."""
        return self.dashboard(f"SetCollisionLevel({int(level)})")

    def positive_solution(self, joints: list[float], user: int = 0, tool: int = 0) -> list[float]:
        """Forward kinematics on the controller: joints (deg) -> [X,Y,Z,Rx,Ry,Rz].

        tool=0 gives the bare FLANGE, whatever tool is active. (The feedback pose
        and Cartesian Mov* commands use the controller's *active* tool instead.)
        """
        raw = self.dashboard(f"PositiveSolution({format_values(joints, '.4f')},{user},{tool})")
        return _braced_numbers(raw, 6, "PositiveSolution")

    def inverse_solution(self, pose: list[float], user: int = 0, tool: int = 0,
                         near: list[float] | None = None) -> list[float]:
        """Inverse kinematics on the controller: pose -> joints (deg).

        near: the joints to stay close to (the robot's current joints). Without it the controller
        solves on its own default branch, which can differ from the branch the arm is on: it then
        refuses poses a straight move from here reaches (19 Nov: a 20 mm lift refused 12× while
        the 20 mm descent next to it was accepted). With it: IsJointNear=1, JointNear={…}
        (TCP-IP V3 syntax); if that form is refused too, the plain form is asked as well, so the
        check is never stricter than before.
        """
        base = f"InverseSolution({format_values(pose, '.4f')},{user},{tool}"
        forms = ([f"{base},1,{{{format_values(near, '.4f')}}})"] if near is not None else []) + [f"{base})"]
        refused = []
        for cmd in forms:
            try:
                raw = self.dashboard(cmd)
            except RobotError as exc:                    # the controller said no
                refused.append(str(exc))
                continue
            return _braced_numbers(raw, 6, "InverseSolution")   # unreadable reply: RobotError (can't tell)
        from common.errors import ValidationError
        raise ValidationError("pose not reachable (no inverse-kinematics solution): " + " | ".join(refused))

    # -- motion commands (no validation here; see actions.validation) --
    def rel_joint_movj(self, deltas: list[float], speed: int) -> str:
        return self.motion(f"RelJointMovJ({format_values(deltas)},SpeedJ={speed},AccJ={speed})")

    def joint_movj(self, joints: list[float], speed: int) -> str:
        return self.motion(f"JointMovJ({format_values(joints, '.4f')},SpeedJ={speed},AccJ={speed})")

    # Cartesian commands (syntax per Dobot TCP-IP-Python-V3 SDK, user/tool frame 0).
    # pose = [X, Y, Z mm, Rx, Ry, Rz deg] in the base frame.
    # User/Tool are always named: the controller's implicit tool for motion commands
    # is not stable (it changed after ResetRobot on this robot). Tool 0 = flange.
    def movj(self, pose: list[float], speed: int, user: int = 0, tool: int = 0) -> str:
        """Point-to-point to a Cartesian pose (joint-interpolated path)."""
        return self.motion(f"MovJ({format_values(pose, '.4f')},User={user},Tool={tool},"
                           f"SpeedJ={speed},AccJ={speed})")

    def movl(self, pose: list[float], speed: int, user: int = 0, tool: int = 0) -> str:
        """Straight-line TCP motion to a Cartesian pose."""
        return self.motion(f"MovL({format_values(pose, '.4f')},User={user},Tool={tool},"
                           f"SpeedL={speed},AccL={speed})")

    def rel_movl_user(self, delta: list[float], speed: int, user: int = 0) -> str:
        """Straight-line motion by an offset in a user frame (user 0 = robot base)."""
        return self.motion(f"RelMovLUser({format_values(delta, '.4f')},{user},SpeedL={speed},AccL={speed})")

    def rel_movl_tool(self, delta: list[float], speed: int, tool: int = 0) -> str:
        """Straight-line motion by an offset expressed in the tool frame."""
        return self.motion(f"RelMovLTool({format_values(delta, '.4f')},{tool},SpeedL={speed},AccL={speed})")
