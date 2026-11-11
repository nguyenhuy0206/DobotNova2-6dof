import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from actions.robot_actions import RobotActions  # noqa: E402
from common.config import RobotConfig  # noqa: E402
from memory.store import PositionStore  # noqa: E402
from sim_robot import SimNova2  # noqa: E402


@pytest.fixture
def sim():
    return SimNova2()


@pytest.fixture
def actions(sim, tmp_path):
    return RobotActions(sim, RobotConfig(), PositionStore(tmp_path / "positions.json"))
