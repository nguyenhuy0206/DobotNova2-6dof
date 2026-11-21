"""Scripts and agents get a frame captured after their request, never the one left from before the
idle stop (20 Nov episode 165941: the wrist picture was one move behind the robot)."""
import time

import numpy as np

from common.config import CameraConfig
from drivers.camera import Frame
from drivers.camera_stream import CameraStream


class Counter:
    """A camera whose picture is the number of frames delivered so far."""
    def __init__(self):
        self.n = 0

    def read(self):
        self.n += 1
        time.sleep(0.01)
        return Frame(color=np.full((8, 8, 3), self.n % 250, np.uint8), sequence_id=self.n)

    def close(self):
        pass


def test_fresh_packet_waits_for_a_new_frame_after_the_idle_stop():
    s = CameraStream("w", CameraConfig(index=0, fps=20), lambda cfg: Counter(), idle_timeout=0.2, retry_s=0.05)
    s.touch()
    deadline = time.monotonic() + 3
    while s.latest_jpeg() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    old_meta = s.latest_packet()[1]
    time.sleep(0.6)                                   # idle: the capture thread stops, the old frame stays
    stale_jpeg, stale_meta = s._jpeg, s._frame_metadata
    assert stale_jpeg is not None and stale_meta == s._frame_metadata
    jpeg, meta = s.fresh_packet(timeout=3)
    assert jpeg is not None and meta[1] is not None
    assert meta != stale_meta or meta[1] != old_meta[1]
    seq_before = s._seq
    s.fresh_packet(timeout=3)
    assert s._seq > seq_before                          # always a frame newer than the call
    s.stop()
