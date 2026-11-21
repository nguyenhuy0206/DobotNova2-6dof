"""RealSense depth helper — the only process that runs as root.

macOS only lets root take the D455 away from its camera service, which
librealsense needs for the depth sensor. Rather than running the whole UI
(and robot control) as root, this tiny server owns the camera and hands the
latest frame to the normal-user UI over localhost:

    GET /frame    npz: color_jpeg (uint8), depth_mm (uint16), intrinsics, timestamp_ms
    GET /status   JSON: profile, fps, error

Start with:  sudo scripts/run_depth.sh      (client: drivers.camera.RealSenseRemoteCamera)

It writes nothing to disk (logs go to the terminal), so no root-owned files
end up in the repo.
"""
from __future__ import annotations

import argparse
import os
import io
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import numpy as np

log = logging.getLogger("realsense_helper")


def pack_frame(frame, jpeg_quality: int = 85) -> bytes:
    import cv2

    ok, jpeg = cv2.imencode(".jpg", frame.color, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    depth_mm = np.clip(frame.depth_m * 1000.0, 0, 65535).astype(np.uint16)
    i = frame.intrinsics
    intr = np.array([i.width, i.height, i.fx, i.fy, i.cx, i.cy] if i else [0, 0, 0, 0, 0, 0], np.float64)
    buf = io.BytesIO()
    np.savez(buf, color_jpeg=jpeg.reshape(-1), depth_mm=depth_mm, intrinsics=intr,
             timestamp_ms=np.float64(frame.timestamp_ms),
             captured_ns=np.int64(frame.captured_ns if frame.captured_ns is not None else -1),
             sequence_id=np.int64(frame.sequence_id if frame.sequence_id is not None else -1),
             stream_id=np.str_(frame.stream_id or ""))
    return buf.getvalue()


class FrameSource:
    """Keeps the camera open and the newest frame in memory; reopens on failure."""

    def __init__(self, open_camera: Callable[[], object], retry_s: float = 3.0):
        self.open_camera, self.retry_s = open_camera, retry_s
        self.frame = None
        self.error: str | None = "starting"
        self.profile = None
        self.fps = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="realsense", daemon=True)

    def start(self) -> "FrameSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def latest(self):
        with self._lock:
            return self.frame

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cam = self.open_camera()
            except Exception as exc:  # noqa: BLE001
                if str(exc) != self.error:
                    log.warning("open failed: %s", exc)
                self.error = str(exc)
                self._stop.wait(self.retry_s)
                continue
            self.profile = getattr(cam, "profile_used", None)
            self.preset = getattr(cam, "preset_used", None)
            if getattr(cam, "preset_error", None):
                log.warning("visual preset not set (%s): streaming with the camera's own setting", cam.preset_error)
            log.info("RealSense streaming %s, preset %s, USB %s", self.profile, self.preset or "camera's own",
                     getattr(cam, "usb", "?") or "?")
            count, t0 = 0, time.monotonic()
            try:
                while not self._stop.is_set():
                    frame = cam.read()
                    with self._lock:
                        self.frame, self.error = frame, None
                    count += 1
                    if time.monotonic() - t0 >= 2:
                        self.fps, count, t0 = count / (time.monotonic() - t0), 0, time.monotonic()
            except Exception as exc:  # noqa: BLE001 — unplugged / USB reset: reopen
                log.warning("stream failed: %s", exc)
                self.error = str(exc)
                self._stop.wait(self.retry_s)
            finally:
                try:
                    cam.close()
                except Exception:  # noqa: BLE001
                    pass


def make_handler(source: FrameSource):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, status, body: bytes, content_type: str):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Polling clients may time out while the camera is opening.
                # Keep their disconnects from hiding the actual camera error.
                self.close_connection = True

        def do_GET(self):
            if self.path == "/frame":
                frame = source.latest()
                if frame is None:
                    self._send(HTTPStatus.SERVICE_UNAVAILABLE, (source.error or "no frame yet").encode(), "text/plain")
                else:
                    self._send(HTTPStatus.OK, pack_frame(frame), "application/octet-stream")
            elif self.path == "/status":
                body = {"profile": source.profile, "preset": getattr(source, "preset", None),
                        "fps": round(source.fps, 1), "error": source.error,
                        "has_frame": source.latest() is not None}
                self._send(HTTPStatus.OK, json.dumps(body).encode(), "application/json")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    return Handler


def serve(source: FrameSource, host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(source))


def stop_when_process_exits(pid: int, server: ThreadingHTTPServer, poll_s: float = 1.0) -> threading.Thread:
    """Shut the helper down once ``pid`` (the UI started with run_vision.sh) is gone."""
    import os

    def watch():
        while True:
            time.sleep(poll_s)
            try:
                os.kill(pid, 0)              # signal 0 = existence check only
            except ProcessLookupError:
                log.info("UI (pid %d) exited — stopping depth helper", pid)
                server.shutdown()
                return
            except PermissionError:
                pass                         # exists, owned by someone else

    thread = threading.Thread(target=watch, name="watch-ui", daemon=True)
    thread.start()
    return thread


def main() -> None:
    from drivers.camera import RealSenseCamera

    p = argparse.ArgumentParser(description="RealSense D455 depth helper (run as root on macOS)")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=15)
    # Off by default: on USB 2 / macOS (20 Nov) setting it timed out (get_xu) and the stream then stalled
    # ("Frame didn't arrive") until the D455 was replugged. Try high_accuracy again on USB 3.
    p.add_argument("--preset", default=os.environ.get("D455_PRESET", ""),
                   help="D4xx visual preset on the depth sensor, e.g. high_accuracy ('' = leave the camera's setting)")
    p.add_argument("--watch-pid", type=int, help="exit when this process exits (used by run_vision.sh)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    source = FrameSource(lambda: RealSenseCamera(args.width, args.height, args.fps, preset=args.preset)).start()
    server = serve(source, "127.0.0.1", args.port)   # localhost only
    log.info("depth helper on http://127.0.0.1:%d  (Ctrl+C to stop)", args.port)
    if args.watch_pid:
        stop_when_process_exits(args.watch_pid, server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        source.stop()


if __name__ == "__main__":
    main()
