"""Manual tasks API with synthetic frames and no hardware or HTTP server."""
from io import BytesIO
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from common.errors import RobotError, SafetyError, ValidationError
from ui.tasks_api import TasksAPI


class FakeScene:
    def __init__(self):
        self.last_frames = {}
        self.looks = []
        self.locations = []
        self.error = None
        self.frame = SimpleNamespace(
            color=np.full((48, 96, 3), [40, 80, 120], dtype=np.uint8),
            depth_m=np.ones((48, 96)), intrinsics=object(),
        )
        self.result = {"ok": True, "top_center_mm": [300, 10, 100], "height_mm": 26,
                       "footprint_mm": [26, 26], "yaw_deg": 20, "kind": "object",
                       "quality": ["few points"]}

    def look(self, camera):
        self.looks.append(camera)
        if self.error:
            raise self.error
        self.last_frames[camera] = (self.frame, 0)
        # A model-sized JPEG must not be used for the full-image pixel API.
        return {"image_jpeg_b64": "not the native frame"}

    def locate_object(self, pixel, *, table_z):
        self.locations.append((pixel, table_z, self.last_frames["env"][0]))
        return self.result


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.result = {"ok": False, "action": "pick_object", "result": {
            "ok": False, "steps": [{"phase": "above", "ok": True},
                                    {"phase": "descend", "ok": False}],
            "failed_step": "descend", "detail": "synthetic contact"}}

    def execute(self, name, args):
        self.calls.append((name, args))
        return self.result


@pytest.fixture
def setup_api():
    scene, executor = FakeScene(), FakeExecutor()
    created = []

    def factory():
        created.append(scene)
        return scene

    api = TasksAPI(executor, workspace=object(), scene_factory=factory,
                   workcell_loader=lambda: {"table_z_mm": -12})
    return api, scene, executor, created


def test_scene_is_lazy_and_frames_are_fresh_native_raw_jpegs(setup_api):
    api, scene, _, created = setup_api
    assert created == []
    before = scene.frame.color.copy()
    jpeg = api.frame_jpeg()
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert jpeg.startswith(b"\xff\xd8")
    assert decoded.shape == before.shape
    assert np.array_equal(scene.frame.color, before)
    scene.frame.color[:] = 220
    assert api.frame_jpeg() != jpeg
    assert created == [scene] and scene.looks == ["env", "env"]


def test_default_scene_uses_robot_workspace_lazily(monkeypatch):
    from ui import tasks_api

    scene, workspace, calls = FakeScene(), object(), []

    def from_config(**kwargs):
        calls.append(kwargs)
        return scene

    monkeypatch.setattr(tasks_api.Scene, "from_config", from_config)
    api = TasksAPI(FakeExecutor(), workspace)
    assert calls == []
    api.frame_jpeg()
    assert calls == [{"cameras": ("env",), "workspace": workspace}]


def test_locate_passes_full_image_pixel_and_workcell_height_for_last_capture(setup_api):
    api, scene, executor, _ = setup_api
    api.frame_jpeg()
    result = api.locate({"pixel": [90, 42]})
    assert result is scene.result
    assert scene.locations == [([90, 42], -12, scene.frame)]
    assert scene.looks == ["env"] and executor.calls == []


def test_locate_failure_is_returned_unchanged(setup_api):
    api, scene, _, _ = setup_api
    api.frame_jpeg()
    scene.result = {"ok": False, "error": "no depth at pixel"}
    assert api.locate({"pixel": [10, 10]}) is scene.result


def test_locate_requires_a_successful_capture(setup_api):
    api, scene, _, created = setup_api
    with pytest.raises(ValidationError, match="Capture env"):
        api.locate({"pixel": [10, 10]})
    assert created == [] and scene.locations == []


@pytest.mark.parametrize("pixel", [None, [], [1], [1, 2, 3], "1,2", [True, 1],
                                    [float("nan"), 1], [1, float("inf")]])
def test_invalid_pixels_are_rejected(setup_api, pixel):
    api, scene, _, _ = setup_api
    api.frame_jpeg()
    with pytest.raises(ValidationError, match="pixel"):
        api.locate({"pixel": pixel})
    assert scene.locations == []


@pytest.mark.parametrize("operation", ["pick", "place"])
@pytest.mark.parametrize("confirm", [None, False, 1, "true"])
def test_motion_requires_literal_true_before_executor_is_called(setup_api, operation, confirm):
    api, _, executor, created = setup_api
    with pytest.raises(SafetyError, match="Workspace clear"):
        getattr(api, operation)({"x": 300, "y": 10, "top_z": 100, "yaw_deg": 20,
                                 "confirm": confirm})
    assert executor.calls == [] and created == []


@pytest.mark.parametrize("operation,args", [
    ("pick", {"x": 300, "y": 10, "top_z": 100, "yaw_deg": 20, "height_mm": 26}),
    ("pick", {"x": 300, "y": 10, "top_z": 100, "yaw_deg": 20}),
    ("place", {"x": 300, "y": 10, "top_z": 70, "yaw_deg": -10, "held_height_mm": 30}),
    ("place", {"x": 300, "y": 10, "top_z": 70}),
])
def test_motion_uses_supplied_executor_and_preserves_result(setup_api, operation, args):
    api, _, executor, created = setup_api
    assert getattr(api, operation)({**args, "confirm": True}) is executor.result
    assert executor.calls == [(f"{operation}_object", args)]
    assert created == []


@pytest.mark.parametrize("error", [OSError("connection refused"), RobotError("helper unavailable")])
def test_depth_helper_down_has_clear_error_and_can_recover(setup_api, error):
    api, scene, _, _ = setup_api
    scene.error = error
    with pytest.raises(RobotError, match="D455 depth helper.*Capture env|D455 depth helper.*run_vision"):
        api.frame_jpeg()
    scene.error = None
    assert api.frame_jpeg().startswith(b"\xff\xd8")
    assert api.locate({"pixel": [10, 10]})["ok"]


def test_failed_capture_invalidates_old_frame(setup_api):
    api, scene, _, _ = setup_api
    api.frame_jpeg()
    scene.error = OSError("helper stopped")
    with pytest.raises(RobotError):
        api.frame_jpeg()
    with pytest.raises(ValidationError, match="Capture env"):
        api.locate({"pixel": [10, 10]})
    assert scene.locations == []


@pytest.mark.parametrize("missing", ["depth_m", "intrinsics"])
def test_color_only_frame_is_not_accepted(setup_api, missing):
    api, scene, _, _ = setup_api
    setattr(scene.frame, missing, None)
    with pytest.raises(RobotError, match="D455 depth helper"):
        api.frame_jpeg()


@pytest.fixture
def handler(setup_api):
    """Exercise routing and response methods without constructing a server/socket."""
    from ui.app import Handler

    api, _, _, _ = setup_api
    handler = Handler.__new__(Handler)
    handler.tasks = api
    handler.skills = SimpleNamespace(run=lambda data: data, system=lambda data: data,
                                     gripper_driver=lambda data: data)
    handler.responses, handler.headers_sent = [], {}
    handler.respond = lambda status, payload: handler.responses.append((status, payload))
    handler.send_response = lambda status: handler.responses.append((status, None))
    handler.send_header = lambda key, value: handler.headers_sent.update({key: value})
    handler.end_headers = lambda: None
    handler.wfile = BytesIO()
    return handler


def test_frame_route_sends_native_jpeg_and_no_cache(handler):
    handler.path = "/api/tasks/env/frame.jpg"
    handler.do_GET()
    assert handler.responses == [(200, None)]
    assert handler.headers_sent["Content-Type"] == "image/jpeg"
    assert handler.headers_sent["Cache-Control"] == "no-store"
    assert int(handler.headers_sent["Content-Length"]) == len(handler.wfile.getvalue())
    assert handler.wfile.getvalue().startswith(b"\xff\xd8")


def test_frame_route_returns_json_error_when_depth_helper_is_down(handler, setup_api):
    _, scene, _, _ = setup_api
    scene.error = OSError("helper refused connection")
    handler.path = "/api/tasks/env/frame.jpg"
    handler.do_GET()
    status, payload = handler.responses[0]
    assert status == 503 and "D455 depth helper" in payload["error"]
    assert handler.wfile.getvalue() == b""


@pytest.mark.parametrize("operation", ["locate", "pick", "place"])
def test_post_routes_use_tasks_api(handler, setup_api, operation):
    api, scene, executor, _ = setup_api
    if operation == "locate":
        api.frame_jpeg()
        request, expected = {"pixel": [90, 42]}, scene.result
    else:
        request = {"x": 300, "y": 10, "top_z": 100, "yaw_deg": 20, "confirm": True}
        expected = executor.result
    handler.path = f"/api/tasks/{operation}"
    handler.body = lambda: request
    handler.do_POST()
    assert handler.responses == [(200, expected)]


@pytest.mark.parametrize("operation", ["pick", "place"])
def test_post_routes_return_json_confirmation_error(handler, setup_api, operation):
    _, _, executor, _ = setup_api
    handler.path = f"/api/tasks/{operation}"
    handler.body = lambda: {"x": 300, "y": 10, "top_z": 100, "yaw_deg": 20}
    handler.do_POST()
    status, payload = handler.responses[0]
    assert status == 400 and "Workspace clear" in payload["error"]
    assert executor.calls == []


def test_insert_needs_confirm_and_goes_to_the_insert_tool(setup_api):
    api, _, executor, _ = setup_api
    with pytest.raises(SafetyError, match="Workspace clear"):
        api.insert({"depth_mm": 5})
    assert api.insert({"depth_mm": 5, "retract": True, "confirm": True}) is executor.result
    assert executor.calls == [("insert", {"depth_mm": 5, "retract": True})]


def test_enable_or_clear_lets_the_dashboard_move_again(prims):
    from ui.skills_api import SkillsAPI
    api = SkillsAPI(prims)
    api.executor.stopped = "CONTACT at fingertip Z=3"
    prims.a.system = lambda action: "ok"
    api.system({"action": "clear"})
    assert api.executor.stopped is None
