import pytest

from actions import validation as v
from common.config import JointStepLimits, SpeedLimits, WorkspaceLimits
from common.errors import ValidationError


@pytest.mark.parametrize("speed", [0, 101, -5])
def test_speed_out_of_range(speed):
    with pytest.raises(ValidationError):
        v.check_speed(speed, SpeedLimits())


@pytest.mark.parametrize("joint,deg", [(0, 5), (7, 5), (1, 0.05), (1, 25), (3, -21)])
def test_joint_step_out_of_range(joint, deg):
    with pytest.raises(ValidationError):
        v.check_joint_step(joint, deg, JointStepLimits())


def test_joint_step_ok():
    v.check_joint_step(6, -20, JointStepLimits())


@pytest.mark.parametrize("joints", [[1, 2, 3], [1, 2, 3, 4, 5, "x"], "abc", [True, 0, 0, 0, 0, 0]])
def test_check_joints_rejects(joints):
    with pytest.raises(ValidationError):
        v.check_joints(joints)


def test_workspace():
    ws = WorkspaceLimits()
    v.check_pose_in_workspace([0, 0, 100, 0, 0, 0], ws)
    with pytest.raises(ValidationError, match="Z"):
        v.check_pose_in_workspace([0, 0, -10, 0, 0, 0], ws)


def test_jpeg():
    with pytest.raises(ValidationError):
        v.check_jpeg(b"\x89PNG" + b"0" * 1000)
    assert v.check_jpeg(b"\xff\xd8" + b"0" * 1000)
