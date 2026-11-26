"""Height-only geometry on synthetic aligned RGB-D frames."""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.config import PerceptionConfig, WorkspaceLimits
from drivers.camera import Frame, Intrinsics
from perception.scene import Scene
from perception.transforms import HandEye


def synthetic(surfaces=(), camera_yaw=23):
    k = Intrinsics(640, 480, 900, 900, 320, 240)
    a = np.radians(camera_yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(a), np.sin(a), 0],
                          [np.sin(a), -np.cos(a), 0], [0, 0, -1]])
    T[:3, 3] = [0.4, 0, 1]
    v, u = np.indices((480, 640))
    depth = np.ones((480, 640), np.float32)
    for x, y, z, long, short, yaw in surfaces:
        d = (1000 - z) / 1000
        cam = np.stack([(u - k.cx) * d / k.fx, (v - k.cy) * d / k.fy,
                        np.full(u.shape, d)], axis=-1)
        base = (cam @ T[:3, :3].T + T[:3, 3]) * 1000
        a = np.radians(yaw)
        dx, dy = base[..., 0] - x, base[..., 1] - y
        inside = (abs(dx * np.cos(a) + dy * np.sin(a)) <= long / 2) & (
            abs(-dx * np.sin(a) + dy * np.cos(a)) <= short / 2)
        depth[inside] = np.minimum(depth[inside], d)
    frame = Frame(np.full((480, 640, 3), 90, np.uint8), depth, k, sequence_id=7)
    return frame, HandEye(T, eye_in_hand=False)


def locate(frame, calibration, pixel=(320, 240), **kwargs):
    from perception.locate import locate_object
    return locate_object(frame, calibration, pixel, table_z=0, **kwargs)


@pytest.mark.parametrize("yaw,long,short", [(0, 26, 26), (31, 26, 26), (-63, 54, 26), (88, 54, 26)])
def test_cube_top_and_base_yaw(yaw, long, short):
    frame, cal = synthetic([(400, 0, 26, long, short, yaw)])
    before = frame.color.copy(), frame.depth_m.copy()
    out = locate(frame, cal)
    assert out["ok"] and out["kind"] == "object"
    assert out["top_center_mm"] == pytest.approx([400, 0, 26], abs=2)
    assert out["support_z_mm"] == pytest.approx(0, abs=2)
    assert out["height_mm"] == pytest.approx(26, abs=2)
    assert out["footprint_mm"] == pytest.approx([long, short], abs=2)
    period = 90 if long == short else 180
    assert abs((out["yaw_deg"] - yaw + period / 2) % period - period / 2) < 2
    assert -period / 2 < out["yaw_deg"] <= period / 2
    assert out["n_points"] > 100 and out["valid_fraction"] == 1
    expected = [] if long == short else [q for q in out["quality"] if q.startswith("footprint not square")]
    assert out["depth_spread_mm"] < 1 and out["quality"] == expected and len(expected) == (long != short)
    assert np.array_equal(frame.color, before[0]) and np.array_equal(frame.depth_m, before[1])


def test_cube_on_box_uses_local_support():
    frame, cal = synthetic([(400, 0, 70, 160, 120, 12), (400, 0, 96, 26, 26, 31)])
    out = locate(frame, cal, pixel=(323, 243))
    assert out["top_center_mm"] == pytest.approx([400, 0, 96], abs=2)
    assert out["support_z_mm"] == pytest.approx(70, abs=2)
    assert out["height_mm"] == pytest.approx(26, abs=2)


def test_box_top_and_table_are_place_surfaces():
    frame, cal = synthetic([(400, 0, 70, 160, 120, 12)])
    out = locate(frame, cal)
    assert out["ok"] and out["kind"] == "surface"
    assert out["top_center_mm"] == pytest.approx([400, 0, 70], abs=2)
    assert out["footprint_mm"] == pytest.approx([160, 120], abs=2)
    assert out["yaw_deg"] == pytest.approx(12, abs=2)
    out = locate(*synthetic())
    assert out["ok"] and out["kind"] == "surface" and abs(out["height_mm"]) < 2


def test_depth_holes_noise_and_seed_outlier():
    frame, cal = synthetic([(400, 0, 26, 26, 26, 31)])
    rng = np.random.default_rng(2)
    frame.depth_m += rng.normal(0, 0.0005, frame.depth_m.shape)
    frame.depth_m[240, 320] = 0.5
    frame.depth_m[233:237, 316:320] = 0
    out = locate(frame, cal)
    assert out["ok"] and out["top_center_mm"] == pytest.approx([400, 0, 26], abs=2)
    assert out["height_mm"] == pytest.approx(26, abs=2)
    assert 0.8 < out["valid_fraction"] < 1


@pytest.mark.parametrize("pixel", [[-1, 0], [640, 240], [], None, [float('nan'), 2], [1, 'bad']])
def test_bad_pixels_return_errors(pixel):
    out = locate(*synthetic(), pixel=pixel)
    assert out["ok"] is False and out["error"]


def test_missing_geometry_returns_errors():
    frame, cal = synthetic()
    assert not locate(frame, None)["ok"]
    assert not locate(frame, HandEye())["ok"]
    frame.depth_m[:] = np.nan
    assert not locate(frame, cal)["ok"]
    frame.depth_m = None
    assert not locate(frame, cal)["ok"]
    assert not locate(None, cal)["ok"]


def test_roi_image_edge_and_workspace_warnings():
    frame, cal = synthetic([(400, 0, 26, 26, 26, 31)])
    out = locate(frame, cal, roi_radius_px=8, workspace=WorkspaceLimits(x=(0, 100)))
    assert "mask touches ROI border" in out["quality"]
    assert "outside workspace" in out["quality"]
    out = locate(*synthetic(), pixel=(0, 0))
    assert "cut by image edge" in out["quality"] and "support not found" in out["quality"]


def test_few_points_spread_and_missing_support_warnings():
    frame, cal = synthetic([(400, 0, 26, 2, 2, 0)])
    assert "few points" in locate(frame, cal, seed_radius_px=0)["quality"]
    frame, cal = synthetic([(400, 0, 26, 26, 26, 0)])
    top = frame.depth_m < 0.99
    frame.depth_m[~top] = 0
    out = locate(frame, cal)
    assert out["ok"] and "support not found" in out["quality"]
    assert out["support_z_mm"] == 0 and out["height_mm"] == pytest.approx(26, abs=2)
    frame, cal = synthetic([(400, 0, 26, 26, 26, 0)])
    top = frame.depth_m < 0.99
    frame.depth_m[top] += np.linspace(-0.005, 0.005, top.sum())
    assert "large spread" in locate(frame, cal)["quality"]


def test_scene_uses_last_env_frame_and_full_pixels():
    frame, cal = synthetic([(400, 0, 26, 26, 26, 31)])
    scene = Scene(PerceptionConfig(), {"env": NS(read=lambda: frame)}, cal)
    assert not scene.locate_object([320, 240])["ok"]
    scene.last_frames["env"] = frame, time.time()
    scene.last_views["env"] = (200, 100, 3)
    scene.sources["env"] = NS(read=lambda: pytest.fail("must reuse the last look"))
    out = scene.locate_object([320, 240], table_z=0)
    assert out["ok"] and out["frame"] == 7
    assert out["top_center_mm"] == pytest.approx([400, 0, 26], abs=2)
    assert "image_jpeg_b64" not in out
