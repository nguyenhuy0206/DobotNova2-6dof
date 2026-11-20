"""Self-collision guard: gripper vs the robot's own elbow (17 Nov auto_calibrate alarm)."""
import pytest

from actions import self_collision as sc
from common.errors import ValidationError

# joints from the 17 Nov auto_calibrate replay (rz 39.6, centre -305,-242,283; tip 117 mm)
POSE5 = [32.25, 88.45, -94.03, 75.73, 272.51, -276.91]     # ran fine
POSE6 = [22.7, 55.57, -74.13, 118.53, 251.32, -287.24]     # the target of the move that hit the elbow
POSE2 = [41.13, 65.45, -80.71, 105.82, 250.01, -268.37]    # ran fine, the closest safe one (79 mm)
ALARM = [26.2, 64.8, -82.1, 107.0, 259.0, -284.5]          # measured where the arm stopped
LIMIT = 65.0


def test_fk_matches_the_controller():
    # measured by the controller in an episode: joints -> flange pose
    pose = sc.flange_pose([37.62, 116.93, -81.43, 54.23, 270.0, -288.28])
    assert pose[:3] == pytest.approx([-324.85, -398.73, 192.28], abs=0.2)
    assert sc.ik_near(pose, [37.0, 116.0, -81.0, 54.0, 270.0, -288.0]) == pytest.approx(
        [37.62, 116.93, -81.43, 54.23, 270.0, -288.28], abs=0.01)


def test_the_move_into_the_elbow_is_refused_and_the_safe_ones_pass():
    assert sc.clearance(POSE6, 117)[0] < 60 < 75 < sc.clearance(POSE2, 117)[0]
    with pytest.raises(ValidationError, match="self-collision.*elbow"):
        sc.check_joint_path(POSE5, POSE6, 117, LIMIT)
    with pytest.raises(ValidationError, match="self-collision"):
        sc.check_linear_path(POSE5, sc.flange_pose(POSE5), sc.flange_pose(POSE6), 117, LIMIT)
    sc.check_linear_path(POSE5, sc.flange_pose(POSE5), sc.flange_pose(POSE2), 117, LIMIT)   # ran on 17 Nov


def test_getting_out_is_always_allowed():
    sc.check_joint_path(POSE6, POSE5, 117, LIMIT)          # from inside the limit, away from the elbow
    sc.check_joint_path(ALARM, POSE5, 117, LIMIT)


def test_primitives_refuse_before_moving(prims, sim):
    p = prims
    p.tool_offset_source = "controller FK"
    p.measure_tool_offset = lambda fb=None: None             # keep "controller FK" in the simulator
    p.m.self_clearance_mm, p.tool_length_mm = LIMIT, 117.0              # the real gripper (config/robot.yaml)
    sim.state["joints_deg"] = list(POSE5)
    sim.state["pose"] = sc.flange_pose(POSE5)                # what the real controller reports there
    sent = len(sim.motion_log)
    with pytest.raises(ValidationError, match="self-collision"):
        p.move_to_joints(POSE6)
    assert len(sim.motion_log) == sent                       # nothing was sent to the robot
    p.m.self_clearance_mm = 0                                # 0 = off
    p.move_to_joints(POSE6)
    assert len(sim.motion_log) == sent + 1


def test_off_when_the_model_does_not_describe_this_robot(prims, sim):
    prims.tool_offset_source = "controller FK"
    prims.measure_tool_offset = lambda fb=None: None
    prims.m.self_clearance_mm, prims.tool_length_mm = LIMIT, 117.0
    sim.state["joints_deg"] = list(POSE5)                  # the simulator's pose does not match Nova 2 FK
    prims.move_to_joints(POSE6)                             # not judged by the wrong model


# place_block_on_table 17 Nov: lifted blue at (-318, -334, 187), straight move toward (-200, -250, 165)
PLACE_START = [31.64, 100.2, -87.16, 76.96, 270.0, -313.86]
PLACE_ALARM = [30.26, 103.01, -120.5, 105.94, 270.01, -315.22]      # measured where the arm stopped


def _flange(tip):
    from common.poses import pose_to_matrix
    import numpy as np
    R = pose_to_matrix(tip)[:3, :3]
    return [*map(float, np.array(tip[:3]) - R[:, 2] * 118), *tip[3:]]


def test_the_elbow_folding_the_wrist_onto_the_upper_arm_is_refused():
    assert sc.wrist_clearance(PLACE_ALARM) < 190 < 196 < sc.wrist_clearance(PLACE_START)
    with pytest.raises(ValidationError, match="J3 -1[12]\\d°.*upper arm"):
        sc.check_linear_path(PLACE_START, sc.flange_pose(PLACE_START), _flange([-200, -250, 165, 180, 0, 75.5]),
                             118, LIMIT, wrist_min_mm=190)
    sc.check_linear_path(PLACE_START, sc.flange_pose(PLACE_START), _flange([-260, -300, 165, 180, 0, 75.5]),
                         118, LIMIT, wrist_min_mm=190)                 # farther from the base: fine


def test_the_wrist_check_only_counts_with_the_elbow_folded():
    tilted = sc.ik_near([-215.0, -303.0, 402.0, -165.0, -15.0, 76.0], [26.2, 64.8, -82.1, 107.0, 259.0, -284.5])
    assert tilted[2] > -100 and sc.wrist_clearance(tilted) < 190        # ran fine during calibration
    sc.check_joint_path(tilted, tilted, 118, LIMIT, wrist_min_mm=190)


def test_elbow_vs_table_19_nov_descent_near_the_base():
    """180814: descend_to z=20 at (-280,-360) folded the elbow onto the table (stall at elbow 57 -> 41)."""
    from actions import self_collision as sc
    from drivers.nova2_kinematics import flange_pose
    J = [43.72, 126.49, -111.86, 60.63, 256.33, -228.06]           # fingertip z 103, elbow centre 57 mm
    f = flange_pose(J)
    down, up = list(f), list(f)
    down[2] -= 19
    up[2] += 19
    with pytest.raises(ValidationError, match="arm vs table"):
        sc.check_linear_path(J, f, down, 127.1, 65, wrist_min_mm=190, table_z=0.3, elbow_min_mm=60)
    sc.check_linear_path(J, f, up, 127.1, 65, wrist_min_mm=190, table_z=0.3, elbow_min_mm=60)   # way out: allowed
    sc.check_linear_path(J, f, down, 127.1, 65, wrist_min_mm=190)                              # no table: no check
    # 12 Nov: the lowest elbow without contact (51 mm) — the far-out grasp poses of today stay allowed
    assert sc.elbow_height([21, 117, -59, 9, 278, -287], 0.3) > 90
