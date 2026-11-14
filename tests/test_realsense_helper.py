"""Root-only depth helper <-> normal-user client, over real localhost HTTP (fake camera)."""
import threading
import time

import numpy as np
import pytest

from common.config import CameraConfig
from common.errors import RobotError
from drivers.camera import Frame, Intrinsics, RealSenseRemoteCamera
from drivers.camera_stream import CameraStream, open_device
from drivers.realsense_helper import FrameSource, serve


class FakeRealSense:
    profile_used = (640, 480, 15)

    def read(self):
        time.sleep(0.01)
        depth = np.full((48, 64), 1.234, np.float32)
        depth[0, 0] = 0.0                                    # invalid pixel stays 0
        return Frame(color=np.full((48, 64, 3), 128, np.uint8), depth_m=depth,
                     intrinsics=Intrinsics(64, 48, 50.0, 51.0, 32.0, 24.0), timestamp_ms=42.0)

    def close(self):
        pass


@pytest.fixture
def helper():
    source = FrameSource(FakeRealSense).start()
    server = serve(source, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    end = time.monotonic() + 3
    while source.latest() is None and time.monotonic() < end:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    source.stop()


def test_client_gets_colour_depth_and_intrinsics(helper):
    frame = RealSenseRemoteCamera(helper).read()
    assert frame.color.shape == (48, 64, 3)
    assert frame.depth_m[1, 1] == pytest.approx(1.234, abs=0.001)    # mm precision
    assert frame.depth_m[0, 0] == 0
    assert frame.intrinsics.fx == 50.0 and frame.intrinsics.cy == 24.0
    assert frame.timestamp_ms == 42.0


def test_camera_window_shows_colour_and_depth(helper):
    stream = CameraStream("env", CameraConfig(source="realsense", helper_url=helper, fps=20), open_device)
    stream.touch()
    end = time.monotonic() + 3
    while stream.status()["resolution"] is None and time.monotonic() < end:
        time.sleep(0.02)
    assert stream.status()["resolution"] == (64, 48) and stream.error is None
    stream.stop()


def test_clear_message_when_helper_not_running():
    with pytest.raises(RobotError, match="sudo scripts/run_depth.sh"):
        RealSenseRemoteCamera("http://127.0.0.1:9", timeout=0.5)


def test_helper_reports_camera_open_error():
    def broken():
        raise RuntimeError("No device connected")
    source = FrameSource(broken, retry_s=0.05).start()
    time.sleep(0.2)
    assert source.latest() is None and "No device connected" in source.error
    source.stop()


def test_helper_stops_when_ui_process_exits():
    import subprocess

    from drivers.realsense_helper import stop_when_process_exits
    ui = subprocess.Popen(["sleep", "0.3"])
    source = FrameSource(FakeRealSense).start()
    server = serve(source, port=0)
    stop_when_process_exits(ui.pid, server, poll_s=0.05)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    ui.wait()
    t.join(timeout=3)
    assert not t.is_alive()          # serve_forever returned -> helper shut down
    server.server_close()
    source.stop()
