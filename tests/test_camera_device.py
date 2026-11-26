"""Cameras found by USB id, not by an index that moves when another camera is plugged in (24 Nov)."""
import json
from types import SimpleNamespace as NS


def test_device_index_follows_the_usb_id(monkeypatch):
    import drivers.camera_stream as cs
    listing = {"SPCameraDataType": [{"spcamera_unique-id": "EAB7A68F-EC2B"},
                                    {"spcamera_unique-id": "0x1100000c45636b"},
                                    {"spcamera_unique-id": "0x12000032e49422"}]}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: NS(stdout=json.dumps(listing)))
    assert cs.device_index("0c45:636b") == 1 and cs.device_index("32E4:9422") == 2
    assert cs.device_index("1234:5678") is None
