"""Debug-tab backend: real HTTP server + SkillsAPI, simulated robot and gripper."""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.app import Handler  # noqa: E402
from ui.skills_api import SkillsAPI  # noqa: E402


@pytest.fixture
def server(actions, prims):
    Handler.actions, Handler.skills = actions, SkillsAPI(prims)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_assets_served(server):
    status, page = get(server + "/")
    assert status == 200 and b'src="debug_skills.js"' in page and b'id="skStop"' in page
    assert get(server + "/debug_skills.js")[0] == 200
    assert get(server + "/debug_skills.css")[0] == 200


@pytest.mark.parametrize("path", ["/app.py", "/../config/robot.yaml", "/.env", "/x/debug_skills.js"])
def test_no_other_files_served(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + path)
    assert e.value.code == 404


def test_catalogue_and_state(server):
    cat = json.loads(get(server + "/api/skills")[1])
    names = {s["name"] for s in cat["skills"]}
    assert {"move_relative", "open_gripper", "close_gripper", "stop"} <= names
    assert cat["defaults"]["speed"] == 20
    state = json.loads(get(server + "/api/skills/state")[1])
    assert state["pose"][:3] == [300, 0, 400]


def test_motion_requires_confirm(server, sim):
    status, body = post(server + "/api/skills/run", {"name": "move_relative", "args": {"dz": -10}})
    assert status == 400 and "Workspace clear" in body["error"] and sim.motion_log == []


def test_button_moves_robot(server, sim):
    status, body = post(server + "/api/skills/run",
                        {"name": "move_relative", "args": {"dz": -10, "speed": 15}, "confirm": True})
    assert status == 200 and body["ok"], body
    assert sim.motion_log[-1].startswith("MovL(300.0000,0.0000,390.0000") and "SpeedL=15" in sim.motion_log[-1]


def test_gripper_buttons(server, gripper):
    _, body = post(server + "/api/skills/run", {"name": "close_gripper", "args": {"force": 25}, "confirm": True})
    assert body["ok"] and body["result"]["position"] == 0
    _, body = post(server + "/api/skills/run", {"name": "open_gripper", "args": {}, "confirm": True})
    assert body["ok"] and body["result"]["position"] == 1000


def test_stop_needs_no_confirm_and_skips_lock(server, sim, actions):
    with actions.lock:   # simulate a skill mid-motion holding the lock
        status, body = post(server + "/api/skills/run", {"name": "stop", "args": {}})
    assert status == 200 and body["ok"] and sim.dashboard_log[-1] == "ResetRobot()"


def test_skill_errors_are_reported(server):
    _, body = post(server + "/api/skills/run",
                   {"name": "move_to_pose", "args": {"x": 0, "y": 0, "z": -5, "rx": 0, "ry": 0, "rz": 0},
                    "confirm": True})
    assert body["ok"] is False and "workspace" in body["error"]
    status, _ = post(server + "/api/skills/run", {"name": "nope"})
    assert status == 400


def test_enable_requires_confirm_clear_does_not(server, sim):
    status, body = post(server + "/api/skills/system", {"action": "enable"})
    assert status == 400 and "Enable" in body["error"]
    assert post(server + "/api/skills/system", {"action": "enable", "confirm": True})[1]["ok"]
    assert post(server + "/api/skills/system", {"action": "clear"})[1]["ok"]
    assert sim.dashboard_log[-2:] == ["EnableRobot()", "ClearError()"]


def test_offline_state_message(server, sim):
    def unreachable():
        raise TimeoutError("timed out")
    sim.feedback = unreachable
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + "/api/skills/state")
    assert e.value.code == 503 and b"192.168.5.100/24" in e.value.read()
