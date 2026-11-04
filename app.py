#!/usr/bin/env python3
"""Local-only control panel for the user's Dobot Nova 2 (six axes).

Designed from the small local TCP/IP architecture of
https://github.com/salmondavidp/DOBOT-Nova5-Control-UI, but rewritten for
Nova 2 V3, macOS and a deliberately narrow, safety-oriented command surface.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import socket
import struct
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
POSITIONS_FILE = APP_DIR / "positions.json"
CAPTURES_DIR = APP_DIR / "captures"
DEFAULT_IP = "192.168.5.1"
MAX_SPEED = 100
LOCK = threading.Lock()


class RobotError(RuntimeError):
    pass


def recv_frame(sock: socket.socket, timeout: float) -> str:
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


class Nova2:
    def __init__(self, ip: str, timeout: float = 3):
        self.ip, self.timeout = ip, timeout
        self.gripper_master: int | None = None
        self.gripper_transport: str | None = None

    def dashboard(self, command: str) -> str:
        with socket.create_connection((self.ip, 29999), self.timeout) as sock:
            sock.sendall(command.encode("ascii"))
            return dashboard_response_ok(recv_frame(sock, self.timeout), command)

    def feedback(self) -> dict:
        with socket.create_connection((self.ip, 30004), self.timeout) as sock:
            packet = bytearray()
            deadline = time.monotonic() + self.timeout
            while len(packet) < 1440:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RobotError("Timeout feedback port 30004.")
                sock.settimeout(remaining)
                block = sock.recv(1440 - len(packet))
                if not block:
                    raise RobotError("Feedback bị đóng trước khi đủ 1440 byte.")
                packet.extend(block)
        data = bytes(packet)
        if struct.unpack_from("<q", data, 0)[0] != 1440:
            raise RobotError("Không nhận đúng gói feedback V3 (1440 byte).")
        if struct.unpack_from("<Q", data, 48)[0] != 0x123456789ABCDEF:
            raise RobotError("Gói feedback không khớp giao thức V3.")
        # Offsets follow the official V3 feedback layout. Nova 2 robot_type=101.
        # Keep these aligned with Dobot's ``MyType`` structure; padding after
        # byte fields means they are not simply consecutive from an eyeballed
        # packet layout.
        robot_type = data[1031]
        if robot_type != 101:
            raise RobotError(f"robot_type={robot_type}; app này chỉ dành cho Nova 2 (101).")
        return {
            "mode": struct.unpack_from("<Q", data, 24)[0],
            "enabled": data[1026], "running": data[1028], "error": data[1029],
            "robot_type": robot_type,
            "joints_deg": list(struct.unpack_from("<6d", data, 432)),
            "pose": list(struct.unpack_from("<6d", data, 624)),
        }

    def motion(self, command: str) -> str:
        """V3 motion server may acknowledge immediately or not at all."""
        with socket.create_connection((self.ip, 30003), self.timeout) as sock:
            sock.sendall(command.encode("ascii"))
            sock.settimeout(0.6)
            try:
                raw = sock.recv(4096).decode("utf-8", "replace").strip()
            except socket.timeout:
                raw = ""
        if raw and not raw.lstrip().startswith("0,"):
            raise RobotError(f"Controller từ chối {command}: {raw}")
        return raw or "Đã gửi vào motion queue."

    def live_safe_for_motion(self) -> dict:
        feedback = self.feedback()
        if not (feedback["mode"] == 5 and feedback["enabled"] == 1 and feedback["error"] == 0 and feedback["running"] == 0):
            raise RobotError("Robot phải ở Mode 5, Enabled, không alarm và đang đứng yên trước khi chuyển động.")
        return feedback

    def joint_step(self, joint: int, degrees: float, speed: int) -> str:
        if not 1 <= joint <= 6 or not 0.1 <= abs(degrees) <= 20:
            raise RobotError("Chỉ hỗ trợ J1–J6 với bước 0.1–20 độ.")
        if not 1 <= speed <= MAX_SPEED:
            raise RobotError(f"Tốc độ phải trong 1–{MAX_SPEED}%.")
        self.live_safe_for_motion()
        values = [0.0] * 6
        values[joint - 1] = degrees
        payload = ",".join(f"{value:g}" for value in values)
        return self.motion(f"RelJointMovJ({payload},SpeedJ={speed},AccJ={speed})")

    def move_joints(self, joints: list[float], speed: int) -> str:
        if len(joints) != 6 or any(not isinstance(v, (int, float)) for v in joints):
            raise RobotError("Vị trí lưu phải gồm đúng 6 góc khớp.")
        if not 1 <= speed <= MAX_SPEED:
            raise RobotError(f"Tốc độ phải trong 1–{MAX_SPEED}%.")
        self.live_safe_for_motion()
        payload = ",".join(f"{value:.4f}" for value in joints)
        return self.motion(f"JointMovJ({payload},SpeedJ={speed},AccJ={speed})")

    def gripper_connect(self, slave_id: int, baud: int, parity: str) -> dict:
        if not 1 <= slave_id <= 247 or baud not in {4800, 9600, 19200, 38400, 57600, 115200}:
            raise RobotError("Cấu hình RS485 không hợp lệ.")
        if parity not in {"N", "O", "E"}:
            raise RobotError("Parity phải là N, O hoặc E.")
        if self.gripper_master is not None:
            self.dashboard(f"ModbusClose({self.gripper_master})")
            self.gripper_master = None
        try:
            raw = self.dashboard(f'ModbusRTUCreate({slave_id},{baud},"{parity}",8,1)')
            transport = "ModbusRTUCreate"
        except RobotError as exc:
            # Nova 2 V3 controller in this workspace returns -10000 for the
            # newer direct RS485 API.  Its supported route is the local V3
            # serial transparent port; baud/parity are configured in CCBOX.
            if "-10000" not in str(exc):
                raise
            raw = self.dashboard(f"ModbusCreate(127.0.0.1,60000,{slave_id},1)")
            transport = "V3 transparent port 60000"
        match = re.search(r"\{\s*(\d+)\s*\}", raw)
        if not match:
            raise RobotError(f"Không đọc được Modbus master ID: {raw}")
        self.gripper_master = int(match.group(1))
        self.gripper_transport = transport
        return {"master_id": self.gripper_master, "transport": transport, "response": raw}

    def _gripper_master(self) -> int:
        if self.gripper_master is None:
            raise RobotError("Chưa kết nối PGC qua RS485.")
        return self.gripper_master

    def gripper_status(self) -> dict:
        master = self._gripper_master()
        # 0x0200: initialized, 0x0201: state, 0x0202: actual position.
        return {"master_id": master, "response": self.dashboard(f"GetHoldRegs({master},512,3)")}

    def _set_gripper_register(self, address: int, value: int) -> str:
        master = self._gripper_master()
        return self.dashboard(f"SetHoldRegs({master},{address},1,{{{value}}})")

    def gripper_initialize(self) -> str:
        return self._set_gripper_register(0x0100, 0xA5)

    def gripper_move(self, position: int, force: int, speed: int) -> dict:
        if not 0 <= position <= 1000 or not 20 <= force <= 100 or not 1 <= speed <= 100:
            raise RobotError("PGC: position 0–1000, force 20–100, speed 1–100.")
        return {"force": self._set_gripper_register(0x0101, force),
                "speed": self._set_gripper_register(0x0104, speed),
                "position": self._set_gripper_register(0x0103, position)}

    def gripper_disconnect(self) -> str:
        master = self._gripper_master()
        try:
            return self.dashboard(f"ModbusClose({master})")
        finally:
            self.gripper_master = None
            self.gripper_transport = None


def read_positions() -> list[dict]:
    if not POSITIONS_FILE.exists():
        return []
    value = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def write_positions(value: list[dict]) -> None:
    temporary = POSITIONS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(POSITIONS_FILE)


class Handler(BaseHTTPRequestHandler):
    robot: Nova2

    def log_message(self, fmt, *args):
        print("[web] " + fmt % args)

    def respond(self, status: int, payload: dict):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 5_500_000:
            raise RobotError("Yêu cầu quá lớn.")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise RobotError("Dữ liệu không hợp lệ.")
        return value

    def do_GET(self):
        if self.path == "/":
            page = (APP_DIR / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/api/state":
            try:
                with LOCK:
                    state = self.robot.feedback()
                    state["positions"] = read_positions()
                self.respond(HTTPStatus.OK, state)
            except (OSError, RobotError, ValueError) as exc:
                print(f"[state] {exc}")
                self.respond(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})

    def do_POST(self):
        try:
            data = self.body()
            if self.path == "/api/action":
                action = data.get("action")
                commands = {"enable": "EnableRobot()", "disable": "DisableRobot()",
                            "clear": "ClearError()", "stop": "ResetRobot()"}
                if action not in commands:
                    raise RobotError("Thao tác không hợp lệ.")
                with LOCK:
                    result = self.robot.dashboard(commands[action])
                self.respond(HTTPStatus.OK, {"result": result})
                return
            if self.path == "/api/speed":
                speed = int(data.get("speed", 0))
                if not 0 <= speed <= MAX_SPEED:
                    raise RobotError(f"Tốc độ phải trong 0–{MAX_SPEED}%.")
                if speed == 0:
                    self.respond(HTTPStatus.OK, {"result": "Motion lock: app sẽ từ chối lệnh chuyển động khi speed = 0."})
                    return
                with LOCK:
                    result = self.robot.dashboard(f"SpeedFactor({speed})")
                self.respond(HTTPStatus.OK, {"result": result})
                return
            if self.path == "/api/joint-step":
                if data.get("confirm") is not True:
                    raise RobotError("Cần xác nhận vùng làm việc an toàn.")
                with LOCK:
                    result = self.robot.joint_step(int(data.get("joint")), float(data.get("degrees")), int(data.get("speed")))
                self.respond(HTTPStatus.OK, {"result": result})
                return
            if self.path == "/api/gripper":
                action = data.get("action")
                with LOCK:
                    if action == "connect":
                        result = self.robot.gripper_connect(int(data.get("slave_id", 1)),
                                                            int(data.get("baud", 115200)),
                                                            str(data.get("parity", "N")).upper())
                    elif action == "status":
                        result = self.robot.gripper_status()
                    elif action == "disconnect":
                        result = {"response": self.robot.gripper_disconnect()}
                    elif action == "initialize":
                        if data.get("confirm") is not True:
                            raise RobotError("Xác nhận vùng quanh gripper an toàn trước khi Initialize.")
                        result = {"response": self.robot.gripper_initialize()}
                    elif action == "move":
                        if data.get("confirm") is not True:
                            raise RobotError("Xác nhận vùng quanh gripper an toàn trước khi chuyển động.")
                        result = self.robot.gripper_move(int(data.get("position")), int(data.get("force")), int(data.get("speed")))
                    else:
                        raise RobotError("Thao tác gripper không hợp lệ.")
                self.respond(HTTPStatus.OK, result)
                return
            if self.path == "/api/positions":
                action = data.get("action")
                with LOCK:
                    positions = read_positions()
                    if action == "capture":
                        name = str(data.get("name", "")).strip()
                        if not 1 <= len(name) <= 40:
                            raise RobotError("Tên vị trí phải dài 1–40 ký tự.")
                        live = self.robot.feedback()
                        entry = {"id": str(int(time.time() * 1000)), "name": name,
                                 "joints_deg": live["joints_deg"], "pose": live["pose"]}
                        positions.append(entry); write_positions(positions)
                        self.respond(HTTPStatus.OK, {"positions": positions}); return
                    if action == "delete":
                        positions = [p for p in positions if p.get("id") != data.get("id")]
                        write_positions(positions); self.respond(HTTPStatus.OK, {"positions": positions}); return
                    if action == "move":
                        if data.get("confirm") is not True:
                            raise RobotError("Cần xác nhận vùng làm việc an toàn.")
                        entry = next((p for p in positions if p.get("id") == data.get("id")), None)
                        if entry is None:
                            raise RobotError("Không tìm thấy vị trí lưu.")
                        result = self.robot.move_joints(entry.get("joints_deg", []), int(data.get("speed")))
                        self.respond(HTTPStatus.OK, {"result": result}); return
                raise RobotError("Thao tác vị trí không hợp lệ.")
            if self.path == "/api/inspection":
                name = str(data.get("name", "")).strip() or "camera-check"
                if len(name) > 40:
                    raise RobotError("Tên operation tối đa 40 ký tự.")
                image = str(data.get("image", ""))
                prefix = "data:image/jpeg;base64,"
                if not image.startswith(prefix):
                    raise RobotError("Webcam phải gửi ảnh JPEG hợp lệ.")
                try:
                    binary = base64.b64decode(image[len(prefix):], validate=True)
                except ValueError as exc:
                    raise RobotError("Không giải mã được ảnh webcam.") from exc
                if not 512 <= len(binary) <= 4_000_000 or not binary.startswith(b"\xff\xd8"):
                    raise RobotError("Ảnh webcam không hợp lệ hoặc quá lớn.")
                with LOCK:
                    live = self.robot.feedback()
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    CAPTURES_DIR.mkdir(exist_ok=True)
                    image_name = f"{stamp}-{int(time.time() * 1000) % 1000:03d}.jpg"
                    (CAPTURES_DIR / image_name).write_bytes(binary)
                    record = {"operation": name, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "image": f"captures/{image_name}", "pose": live["pose"], "joints_deg": live["joints_deg"]}
                    (CAPTURES_DIR / f"{image_name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self.respond(HTTPStatus.OK, record)
                return
            self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})
        except (OSError, RobotError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.respond(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Local control for Dobot Nova 2")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.robot = Nova2(args.ip)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Nova 2 Control: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
