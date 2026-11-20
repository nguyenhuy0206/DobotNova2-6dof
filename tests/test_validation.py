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


def test_reach_radius_and_keep_out_zone():
    ws = WorkspaceLimits(x=(-720, 720), y=(-720, 720), z=(0, 530), max_radius_mm=690,
                         keep_out=[{"name": "D455", "xy": [-45, -690], "radius_mm": 90, "z": [0, 250]}])
    v.check_pose_in_workspace([-480, -480, 100], ws)                   # r 679: inside the reach
    with pytest.raises(ValidationError, match="reach limit"):
        v.check_pose_in_workspace([-500, -500, 100], ws)               # r 707
    with pytest.raises(ValidationError, match="keep-out zone around D455"):
        v.check_pose_in_workspace([-60, -620, 150], ws)
    v.check_pose_in_workspace([-60, -620, 300], ws)                    # above the camera


def test_working_sector_20_nov_zone_touches():
    """zone_1 (-35.6, -709.8) at -92.9° … zone_5 (-695.2, 150.5) at 167.8°, counter-clockwise through
    ±180: the table side. The D455 side (camera3d_front at -76.6°) and the back of the robot are out."""
    ws = WorkspaceLimits(x=(-720, 720), y=(-720, 720), z=(0, 530), max_radius_mm=690,
                         sector_deg=(167.8, -92.9))
    for xy in ([-233, -610], [-480, -480], [-600, 0], [-650, 140], [-45, -600]):   # board, table, both ends
        v.check_pose_in_workspace([*xy, 100], ws)
    for xy in ([150, -630], [300, 300], [400, 0], [-600, 170]):          # D455 side, behind, past zone_5
        with pytest.raises(ValidationError, match="working sector"):
            v.check_pose_in_workspace([*xy, 100], ws)
    v.check_pose_in_workspace([60, 60, 400], ws)                          # next to the base axis: exempt


def test_floor_is_checked_on_the_lower_open_fingertip():
    from types import SimpleNamespace

    from actions.primitives import PrimitiveSkills
    from common.poses import pose_to_matrix
    me = SimpleNamespace(finger_half_gap_mm=31.6)
    down = pose_to_matrix([0, 0, 0, 180, 0, 0])[:3, :3]
    assert PrimitiveSkills._lowest_tip(me, [0, 0, 50], down)[2] == pytest.approx(50)
    tilted = pose_to_matrix([0, 0, 0, 135, 0, 0])[:3, :3]                 # 45° about X: fingers (flange Y) dip
    assert PrimitiveSkills._lowest_tip(me, [0, 0, 50], tilted)[2] == pytest.approx(50 - 31.6 * 0.7071, abs=0.1)


def test_scene_ignores_objects_outside_the_working_sector():
    """20 Nov: the robot's E-stop button (84, -18, 45) came back from look as a red object."""
    from perception.scene import Scene
    sc = Scene.__new__(Scene)
    sc.workspace = WorkspaceLimits(x=(-720, 720), y=(-720, 720), z=(0, 530), max_radius_mm=690,
                                   sector_deg=(167.8, -92.9))
    assert not sc.in_workspace([84.0, -18.2, 44.5])                  # E-stop: -12°
    assert sc.in_workspace([-233, -610, 20])                         # board area
    assert sc.in_workspace([-20, -600, 20])                          # 1.0° past zone_1, inside the 50 mm margin
    assert not sc.in_workspace([150, -600, 20])                      # D455 side
