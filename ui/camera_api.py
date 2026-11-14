"""HTTP glue for the Debug tab camera windows.

    GET  /api/cameras                 -> status of every configured camera
    GET  /api/cameras/<name>/stream   -> MJPEG (multipart/x-mixed-replace), use as <img src>
    GET  /api/cameras/<name>/frame.jpg-> latest single frame
    POST /api/cameras/<name>/capture  -> save frame + robot pose to data/captures/
    POST /api/cameras/<name>/device   -> {"source": "opencv"|"realsense", "index": n}
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from common.errors import ValidationError
from common.log import get_logger
from common.paths import CAPTURES_DIR
from drivers.camera_stream import CameraHub

log = get_logger(__name__)
BOUNDARY = "nova2frame"


class CameraAPI:
    def __init__(self, hub: CameraHub, robot_state=None):
        self.hub = hub
        self.robot_state = robot_state   # callable -> dict (pose/joints), optional

    def require(self, name: str):
        """The camera stream by name; raises before any response is sent if unknown."""
        if name not in self.hub:
            raise ValidationError(f"Unknown camera: {name}")
        return self.hub[name]

    _stream = require

    def status(self) -> dict:
        return {"cameras": self.hub.status()}

    def frame(self, name: str) -> bytes | None:
        return self._stream(name).latest_jpeg()

    @staticmethod
    def placeholder(message: str) -> bytes:
        """Dark 'no signal' frame with the camera's error, so the window explains itself."""
        image = np.full((360, 640, 3), 30, np.uint8)
        cv2.putText(image, "NO SIGNAL", (225, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        for i, line in enumerate([message[j:j + 60] for j in range(0, len(message), 60)][:3]):
            cv2.putText(image, line, (20, 210 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 170, 255), 1)
        return cv2.imencode(".jpg", image)[1].tobytes()

    def stream_mjpeg(self, name: str, wfile) -> None:
        """Write frames until the browser disconnects. Runs in the request thread."""
        stream = self._stream(name)
        seq = 0
        while True:
            new_seq, jpeg = stream.wait_jpeg(seq, timeout=1.0)
            if new_seq == seq or jpeg is None:
                # No new frame within 1 s: show why (also detects closed browsers).
                jpeg = self.placeholder(stream.error or "starting camera…")
            seq = new_seq
            wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode())
            wfile.write(jpeg)
            wfile.write(b"\r\n")
            wfile.flush()

    def capture(self, name: str, data: dict) -> dict:
        image = self._stream(name).latest_image()
        if image is None:
            raise ValidationError(f"Camera {name} has no frame yet.")
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        image_name = f"{name}-{stamp}.jpg"
        cv2.imwrite(str(CAPTURES_DIR / image_name), image)
        record = {"camera": name, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "image": f"data/captures/{image_name}", "note": str(data.get("note", ""))[:200]}
        if self.robot_state is not None:
            try:
                state = self.robot_state()
                record.update(pose=state.get("pose"), joints_deg=state.get("joints_deg"))
            except Exception as exc:  # noqa: BLE001 — capture still useful without pose
                record["robot_state_error"] = str(exc)
        (CAPTURES_DIR / f"{image_name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log.info("captured %s", record["image"])
        return record

    def select_device(self, name: str, data: dict) -> dict:
        self._stream(name)
        return self.hub.select_device(name, str(data.get("source", "opencv")), int(data.get("index", 0)))
