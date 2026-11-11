#!/usr/bin/env python3
"""Local-only manual control dashboard for the Dobot Nova 2.

HTTP layer only: parses requests, calls ``actions.RobotActions`` (where every
limit and safety rule lives) and serializes the result.

    python ui/app.py [--ip 192.168.5.1] [--port 8765]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without `pip install -e .`

from actions.robot_actions import RobotActions  # noqa: E402
from actions.validation import require_confirm  # noqa: E402
from common.config import load_robot_config  # noqa: E402
from common.errors import RobotError, ValidationError  # noqa: E402
from common.log import get_logger, setup_logging  # noqa: E402
from common.paths import UI_DIR  # noqa: E402
from drivers.nova2 import Nova2  # noqa: E402

log = get_logger("ui")
MAX_BODY = 5_500_000
JPEG_PREFIX = "data:image/jpeg;base64,"


class Handler(BaseHTTPRequestHandler):
    actions: RobotActions

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    # -- helpers ---------------------------------------------------------
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
        if length > MAX_BODY:
            raise ValidationError("Yêu cầu quá lớn.")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValidationError("Dữ liệu không hợp lệ.")
        return value

    # -- GET -------------------------------------------------------------
    def do_GET(self):
        if self.path == "/":
            page = (UI_DIR / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/api/state":
            try:
                self.respond(HTTPStatus.OK, self.actions.state())
            except (OSError, RobotError, ValueError) as exc:
                log.warning("state: %s", exc)
                self.respond(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})

    # -- POST ------------------------------------------------------------
    def do_POST(self):
        routes = {
            "/api/action": self.post_action,
            "/api/speed": self.post_speed,
            "/api/joint-step": self.post_joint_step,
            "/api/gripper": self.post_gripper,
            "/api/positions": self.post_positions,
            "/api/inspection": self.post_inspection,
        }
        route = routes.get(self.path)
        if route is None:
            self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})
            return
        try:
            self.respond(HTTPStatus.OK, route(self.body()))
        except (OSError, RobotError, ValueError, TypeError) as exc:
            self.respond(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def post_action(self, data: dict) -> dict:
        return {"result": self.actions.system(str(data.get("action")))}

    def post_speed(self, data: dict) -> dict:
        return {"result": self.actions.set_speed(int(data.get("speed", 0)))}

    def post_joint_step(self, data: dict) -> dict:
        require_confirm(data.get("confirm"))
        return {"result": self.actions.joint_step(int(data.get("joint")), float(data.get("degrees")),
                                                  int(data.get("speed")))}

    def post_gripper(self, data: dict) -> dict:
        a, action = self.actions, data.get("action")
        if action == "connect":
            return a.gripper_connect(int(data.get("slave_id", 1)), int(data.get("baud", 115200)),
                                     str(data.get("parity", "N")).upper())
        if action == "status":
            return a.gripper_status()
        if action == "disconnect":
            return {"response": a.gripper_disconnect()}
        if action == "initialize":
            require_confirm(data.get("confirm"), "Xác nhận vùng quanh gripper an toàn trước khi Initialize.")
            return {"response": a.gripper_initialize()}
        if action == "move":
            require_confirm(data.get("confirm"), "Xác nhận vùng quanh gripper an toàn trước khi chuyển động.")
            return a.gripper_move(int(data.get("position")), int(data.get("force")), int(data.get("speed")))
        raise ValidationError("Thao tác gripper không hợp lệ.")

    def post_positions(self, data: dict) -> dict:
        a, action = self.actions, data.get("action")
        if action == "capture":
            return {"positions": a.capture_position(str(data.get("name", "")))}
        if action == "delete":
            return {"positions": a.delete_position(data.get("id"))}
        if action == "move":
            require_confirm(data.get("confirm"))
            return {"result": a.move_to_saved(data.get("id"), int(data.get("speed")))}
        raise ValidationError("Thao tác vị trí không hợp lệ.")

    def post_inspection(self, data: dict) -> dict:
        image = str(data.get("image", ""))
        if not image.startswith(JPEG_PREFIX):
            raise ValidationError("Webcam phải gửi ảnh JPEG hợp lệ.")
        try:
            binary = base64.b64decode(image[len(JPEG_PREFIX):], validate=True)
        except ValueError as exc:
            raise ValidationError("Không giải mã được ảnh webcam.") from exc
        return self.actions.capture_inspection(str(data.get("name", "")), binary)


def main():
    cfg = load_robot_config()
    parser = argparse.ArgumentParser(description="Local control for Dobot Nova 2")
    parser.add_argument("--ip", default=cfg.ip)
    parser.add_argument("--port", type=int, default=cfg.ui_port)
    args = parser.parse_args()
    setup_logging()

    Handler.actions = RobotActions(Nova2(args.ip, cfg.timeout_s), cfg)
    server = ThreadingHTTPServer((cfg.ui_host, args.port), Handler)
    log.info("Nova 2 Control: http://%s:%d  (robot %s)", cfg.ui_host, args.port, args.ip)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
