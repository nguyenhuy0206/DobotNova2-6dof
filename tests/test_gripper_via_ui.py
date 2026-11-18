"""One Modbus master: other processes drive the gripper through ui/app.py (17 Nov NO RESPONSE)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from common.errors import RobotError
from drivers.gripper import PGCGripperDirect, PGCGripperRemote
from ui.skills_api import SkillsAPI


class OwnerGripper:
    """The app's own driver (stands in for PGCGripperDirect on the bus)."""
    def __init__(self):
        self.position, self.calls = 1000, []

    def status(self):
        return {"initialized": 1, "grip_state": 2 if self.position == 512 else 1, "position": self.position}

    def init(self, force_reinit=False, full=False):
        self.calls.append(("init", force_reinit))
        return self.status()

    def go_to(self, position, force=30, speed=50, wait=True):
        self.calls.append(("go_to", position, force, speed, wait))
        self.position = max(position, 512)                      # a cube stops the fingers at 512
        return self.status()


@pytest.fixture
def app():
    owner = OwnerGripper()
    api = SkillsAPI.__new__(SkillsAPI)
    api.primitives = SimpleNamespace(gripper=owner)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                code, out = 200, api.gripper_driver(data)
            except (RobotError, ValueError) as exc:
                code, out = 400, {"error": str(exc)}
            body = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield owner, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_remote_gripper_drives_the_apps_driver(app):
    owner, url = app
    g = PGCGripperRemote(url)
    assert g.connect()["position"] == 1000 and g.connected
    assert g.close(force=30, speed=10)["grip_state"] == 2 and g.is_holding()
    assert owner.calls[-1] == ("go_to", 0, 30, 10, True)
    assert g.open()["position"] == 1000


def test_errors_come_back_readable(app):
    _, url = app
    with pytest.raises(RobotError, match="unknown gripper op"):
        PGCGripperRemote(url)._call("spin")
    with pytest.raises(RobotError, match="cannot reach ui/app.py.*start it"):
        PGCGripperRemote("http://127.0.0.1:9", timeout=1).status()


def test_only_the_bus_owner_opens_the_port(prims):
    from actions.primitives import gripper_factory

    prims.a.config.gripper.via_ui = True
    assert isinstance(gripper_factory(prims.a)(), PGCGripperRemote)              # episodes, scripts
    assert isinstance(gripper_factory(prims.a, owner=True)(), PGCGripperDirect)  # ui/app.py
    prims.a.config.gripper.via_ui = False
    assert isinstance(gripper_factory(prims.a)(), PGCGripperDirect)


def test_motion_without_confirm_is_refused():
    api = SkillsAPI.__new__(SkillsAPI)
    api.primitives = SimpleNamespace(gripper=OwnerGripper())
    with pytest.raises(RobotError, match="confirm"):          # SafetyError, like every UI motion
        api.gripper_driver({"op": "go_to", "position": 0})


def test_slow_fingers_get_time_to_arrive():
    from drivers.gripper import motion_timeout

    assert motion_timeout(50) == 5.0                     # the old fixed value at normal speed
    assert motion_timeout(10) == 13.0 and motion_timeout(1) == 103.0
    g = PGCGripperDirect()
    g.move = lambda *a: None
    waited = {}
    g.wait_motion = lambda timeout=5.0, target=None: waited.setdefault("t", timeout) and {"grip_state": 1}
    g.go_to(500, 30, 10)
    assert waited["t"] == 13.0



def test_opening_while_holding_waits_for_the_fingers(monkeypatch):
    """grip_state 3 appears at once when opening around a cube: not the end of the motion."""
    import drivers.gripper as gm
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    readings = iter([(3, 521), (3, 640), (3, 820), (3, 990), (1, 1000)])
    g = PGCGripperDirect()
    g.status = lambda: (lambda st, pos: {"grip_state": st, "position": pos})(*next(readings))
    assert g.wait_motion(5.0, target=1000)["position"] == 990          # within ±20 of the target
    stuck = iter([(3, 700), (3, 700), (3, 700), (3, 700)] + [(3, 700)] * 50)
    g.status = lambda: (lambda st, pos: {"grip_state": st, "position": pos})(*next(stuck))
    clock = iter([float(i) * 0.2 for i in range(200)])
    monkeypatch.setattr(gm.time, "monotonic", lambda: next(clock))
    assert g.wait_motion(30.0, target=1000)["position"] == 700         # stopped short: report it
