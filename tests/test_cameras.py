"""Camera streaming: fake devices, real threads, real HTTP server."""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import CameraConfig  # noqa: E402
from drivers.camera import Frame  # noqa: E402
from drivers.camera_stream import CameraHub, CameraStream, frame_to_image  # noqa: E402


class FakeDevice:
    opened = []

    def __init__(self, cfg, depth=False):
        self.cfg, self.depth, self.n = cfg, depth, 0
        FakeDevice.opened.append((cfg.source, cfg.index))

    def read(self):
        time.sleep(0.01)
        self.n += 1
        color = np.full((48, 64, 3), self.cfg.index * 40, np.uint8)
        return Frame(color=color, depth_m=np.ones((48, 64), np.float32) if self.depth else None)

    def close(self):
        pass


def fake_opener(cfg):
    if cfg.index == 9:
        raise RuntimeError("no such device")
    return FakeDevice(cfg, depth=cfg.source == "realsense")


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_stream_produces_jpeg_at_target_rate():
    s = CameraStream("w", CameraConfig(index=1, fps=20), fake_opener)
    s.touch()
    assert wait_for(lambda: s.latest_jpeg() is not None)
    assert s.latest_jpeg()[:2] == b"\xff\xd8"
    assert s.status()["resolution"] == (64, 48)
    s.stop()


def test_stream_reports_open_error_and_releases_when_idle():
    s = CameraStream("w", CameraConfig(index=9), fake_opener, idle_timeout=0.3, retry_s=0.05)
    s.touch()
    assert wait_for(lambda: s.error and "no such device" in s.error)
    assert wait_for(lambda: not s.status()["running"], timeout=2)


def test_realsense_frames_show_colour_and_depth_side_by_side():
    frame = Frame(color=np.zeros((48, 64, 3), np.uint8), depth_m=np.ones((48, 64), np.float32))
    assert frame_to_image(frame).shape == (48, 128, 3)


def test_hub_remembers_device_choice(tmp_path):
    overrides = tmp_path / "cams.json"
    hub = CameraHub({"wrist": CameraConfig(index=0)}, fake_opener, overrides)
    hub.select_device("wrist", "opencv", 3)
    assert json.loads(overrides.read_text())["wrist"] == {"source": "opencv", "index": 3}
    hub2 = CameraHub({"wrist": CameraConfig(index=0)}, fake_opener, overrides)
    assert hub2["wrist"].cfg.index == 3
    with pytest.raises(ValueError):
        hub.select_device("wrist", "webcam", 1)
    hub.stop_all()


# ---------------------------------------------------------------- HTTP
@pytest.fixture
def cam_server(actions, prims, tmp_path, monkeypatch):
    from ui import camera_api
    from ui.app import Handler
    from ui.camera_api import CameraAPI
    from ui.skills_api import SkillsAPI

    monkeypatch.setattr(camera_api, "CAPTURES_DIR", tmp_path / "captures")
    hub = CameraHub({"env": CameraConfig(label="Env", index=1), "wrist": CameraConfig(label="Wrist", index=2)},
                    fake_opener, tmp_path / "cams.json")
    Handler.actions, Handler.skills = actions, SkillsAPI(prims)
    Handler.cameras = CameraAPI(hub, robot_state=Handler.skills.state)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()
    httpd.server_close()
    hub.stop_all()


def frame_ready(url):
    try:
        return urllib.request.urlopen(url).status == 200
    except urllib.error.HTTPError:
        return False          # 503 until the camera delivers its first frame


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_camera_endpoints(cam_server):
    base, tmp = cam_server
    cams = json.loads(urllib.request.urlopen(base + "/api/cameras").read())["cameras"]
    assert [c["name"] for c in cams] == ["env", "wrist"]

    # MJPEG: read the first part of the multipart stream
    with urllib.request.urlopen(base + "/api/cameras/wrist/stream", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        head = r.read(200)
    assert b"--nova2frame" in head and b"image/jpeg" in head

    assert wait_for(lambda: frame_ready(base + "/api/cameras/wrist/frame.jpg"))

    rec = post(base + "/api/cameras/wrist/capture", {"note": "test"})
    assert rec["pose"][:3] == [300, 0, 400]            # robot pose saved with the image
    saved = tmp / "captures" / Path(rec["image"]).name
    assert saved.exists() and saved.with_name(saved.name + ".json").exists()

    status = post(base + "/api/cameras/env/device", {"source": "opencv", "index": 4})
    assert status["index"] == 4


def test_unknown_camera_404(cam_server):
    base, _ = cam_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/api/cameras/nope/stream")
    assert e.value.code == 404


def test_placeholder_when_camera_missing(cam_server):
    base, _ = cam_server
    post(base + "/api/cameras/env/device", {"source": "opencv", "index": 9})   # fails to open
    with urllib.request.urlopen(base + "/api/cameras/env/stream", timeout=5) as r:
        assert b"image/jpeg" in r.read(200)        # 'NO SIGNAL' frame, connection stays useful


def test_stream_after_device_switch_does_not_crash(cam_server):
    """Regression: switching device cleared the frame but kept the counter -> len(None)."""
    base, _ = cam_server
    assert wait_for(lambda: frame_ready(base + "/api/cameras/wrist/frame.jpg"))
    post(base + "/api/cameras/wrist/device", {"source": "opencv", "index": 9})   # clears frame, seq > 0
    with urllib.request.urlopen(base + "/api/cameras/wrist/stream", timeout=5) as r:
        assert b"image/jpeg" in r.read(200)
