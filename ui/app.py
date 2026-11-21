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
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]  # run without `pip install -e .`

from actions.primitives import create_primitives  # noqa: E402
from actions.robot_actions import RobotActions  # noqa: E402
from actions.validation import require_confirm  # noqa: E402
from common.config import load_robot_config  # noqa: E402
from common.errors import RobotError, ValidationError  # noqa: E402
from common.log import get_logger, setup_logging  # noqa: E402
from common.paths import UI_DIR  # noqa: E402
from drivers.nova2 import Nova2  # noqa: E402
from drivers.camera_stream import CameraHub  # noqa: E402
from ui.camera_api import BOUNDARY, CameraAPI  # noqa: E402
from ui.skills_api import SkillsAPI  # noqa: E402

log = get_logger("ui")
MAX_BODY = 5_500_000
JPEG_PREFIX = "data:image/jpeg;base64,"
# Static assets next to index.html that the browser may load.
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    actions: RobotActions
    skills: SkillsAPI
    cameras: CameraAPI

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

    def send_file(self, path: Path, content_type: str):
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- GET -------------------------------------------------------------
    def do_GET(self):
        if self.path == "/":
            self.send_file(UI_DIR / "index.html", "text/html; charset=utf-8")
            return
        static = self.static_path()
        if static is not None:
            self.send_file(static, STATIC_TYPES[static.suffix])
            return
        if self.path.startswith("/api/cameras/"):
            self.get_camera()
            return
        live = {"/api/state": self.actions.state, "/api/skills/state": self.skills.state,
                "/api/skills": self.skills.catalogue, "/api/cameras": self.cameras.status}.get(self.path)
        if live is not None:
            try:
                self.respond(HTTPStatus.OK, live())
            except (OSError, RobotError, ValueError) as exc:
                log.warning("%s: %s", self.path, exc)
                self.respond(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})

    # -- cameras -----------------------------------------------------------
    def camera_route(self) -> tuple[str, str]:
        """/api/cameras/<name>/<action> -> (name, action)."""
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) != 4:
            raise ValidationError("Không tìm thấy.")
        return parts[2], parts[3]

    def get_camera(self):
        try:
            name, action = self.camera_route()
            self.cameras.require(name)
            if action == "stream":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.cameras.stream_mjpeg(name, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass   # browser closed the window / switched tab
                return
            if action == "frame.jpg":
                jpeg, metadata = self.cameras.frame_packet(name)
                if jpeg is None:
                    self.respond(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Camera chưa có hình."})
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                for key, value in zip(("X-Capture-Monotonic-Ns", "X-Frame-Sequence", "X-Frame-Stream"), metadata):
                    if value is not None:
                        self.send_header(key, str(value))
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            self.respond(HTTPStatus.NOT_FOUND, {"error": "Không tìm thấy."})
        except (RobotError, ValueError) as exc:
            self.respond(HTTPStatus.NOT_FOUND, {"error": str(exc)})

    def post_camera(self, data: dict) -> dict:
        name, action = self.camera_route()
        if action == "capture":
            return self.cameras.capture(name, data)
        if action == "device":
            return self.cameras.select_device(name, data)
        raise ValidationError("Thao tác camera không hợp lệ.")

    def static_path(self) -> Path | None:
        """Plain file names in ui/ with an allowed extension; nothing else."""
        name = self.path.lstrip("/")
        if "/" in name or "\\" in name or name.startswith("."):
            return None
        path = UI_DIR / name
        return path if path.suffix in STATIC_TYPES and path.is_file() else None

    # -- POST ------------------------------------------------------------
    def do_POST(self):
        routes = {
            "/api/action": self.post_action,
            "/api/speed": self.post_speed,
            "/api/joint-step": self.post_joint_step,
            "/api/gripper": self.post_gripper,
            "/api/positions": self.post_positions,
            "/api/inspection": self.post_inspection,
            "/api/skills/run": self.skills.run,
            "/api/skills/system": self.skills.system,
            "/api/gripper/driver": self.skills.gripper_driver,
        }
        route = routes.get(self.path)
        if route is None and self.path.startswith("/api/cameras/"):
            route = self.post_camera
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
    # the app OWNS the gripper bus (gripper.via_ui): other processes call /api/gripper/driver
    Handler.skills = SkillsAPI(create_primitives(actions=Handler.actions, owner=True))
    hub = CameraHub(cfg.cameras)
    if os.environ.get("NOVA2_DEPTH") == "1" and "env" in hub:
        hub.select_device("env", "realsense", 0)     # scripts/run_vision.sh
    Handler.cameras = CameraAPI(hub, robot_state=Handler.skills.state)
    server = ThreadingHTTPServer((cfg.ui_host, args.port), Handler)
    log.info("Nova 2 Control: http://%s:%d  (robot %s)", cfg.ui_host, args.port, args.ip)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop_all()
        server.server_close()


if __name__ == "__main__":
    main()
