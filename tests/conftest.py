import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from actions.primitives import PrimitiveSkills  # noqa: E402
from actions.robot_actions import RobotActions  # noqa: E402
from common.config import RobotConfig  # noqa: E402
from memory.store import PositionStore  # noqa: E402
from sim_robot import FakeGripper, SimNova2  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_judge_data(tmp_path, monkeypatch):
    """Synthetic runs must never populate the operator's private runtime ledger."""
    monkeypatch.setattr("judge.ledger.LEDGER_FILE", tmp_path / "judge" / "ledger.jsonl")


@pytest.fixture
def sim():
    return SimNova2()


@pytest.fixture
def actions(sim, tmp_path):
    cfg = RobotConfig()
    cfg.motion.start_grace_s, cfg.motion.poll_s, cfg.motion.move_timeout_s = 0.0, 0.0, 0.2
    return RobotActions(sim, cfg, PositionStore(tmp_path / "positions.json"))


@pytest.fixture
def gripper():
    return FakeGripper()


@pytest.fixture
def prims(actions, gripper):
    return PrimitiveSkills(actions, lambda: gripper)
