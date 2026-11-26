"""Operator hover requires an explicit y; tests never create hardware clients."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def checker(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts" / "dev"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("locate_check", folder / "locate_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reply,hover,moves", [("n", True, 0), ("", True, 0), ("yes", True, 0),
                                              ("y", True, 2), ("y", False, 0)])
def test_hover_confirmation(checker, monkeypatch, tmp_path, reply, hover, moves):
    monkeypatch.setattr("builtins.input", lambda prompt: reply)
    calls, factories = [], []
    def factory():
        factories.append(True)
        return NS(execute=lambda name, args=None: calls.append((name, args)) or {"ok": True,
                   "result": {"pose": [300, 0, 400, 180, 0, 5]}})
    scene = NS(locate_object=lambda *a, **kw: {"ok": True, "top_center_mm": [400, 0, 26], "yaw_deg": 20})
    row = checker.check_pixel(scene, [320, 240], 0, hover=hover,
                              executor_factory=factory, log_path=tmp_path / "checks.jsonl")
    motion = [(name, args) for name, args in calls if name == "move_to_pose"]
    assert len(motion) == moves and len(factories) == (1 if moves else 0)
    if moves:      # joint-space to 150 mm above, then straight down to top+40
        assert motion[0][1] == dict(x=400, y=0, z=176, rx=180, ry=0, rz=5)
        assert motion[1][1] == dict(x=400, y=0, z=66, rx=180, ry=0, rz=5, linear=True)
    assert (tmp_path / "checks.jsonl").read_text().count("\n") == 1
    assert row["location"]["ok"]


def test_failed_location_never_hovers(checker, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not offer hover"))
    scene = NS(locate_object=lambda *a, **kw: {"ok": False, "error": "no depth"})
    checker.check_pixel(scene, [320, 240], 0, hover=True,
                        executor_factory=lambda: pytest.fail("must not connect"), log_path=tmp_path / "checks.jsonl")
