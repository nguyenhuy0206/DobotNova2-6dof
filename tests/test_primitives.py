import math
"""Primitive skills against the simulated Nova 2."""
import pytest

from common.errors import RobotError, SafetyError, ValidationError


def test_get_state(prims):
    s = prims.get_state()
    assert s["pose"] == [300, 0, 400, 180, 0, 0] and s["enabled"] and not s["error"]
    assert "gripper" not in s   # gripper is lazy


def test_move_to_pose_joint_and_linear(prims, sim):
    r = prims.move_to_pose(350, 50, 300, 180, 0, 90)
    assert sim.motion_log[-1] == ("MovJ(350.0000,50.0000,300.0000,180.0000,0.0000,90.0000,"
                                  "User=0,Tool=0,SpeedJ=20,AccJ=20)")
    assert r["pose"][:3] == [350, 50, 300]
    prims.move_to_pose(300, 0, 250, 180, 0, 90, linear=True, speed=10)
    assert sim.motion_log[-1].startswith("MovL(") and sim.motion_log[-1].endswith("SpeedL=10,AccL=10)")


def test_move_to_pose_outside_workspace_never_sent(prims, sim):
    with pytest.raises(ValidationError, match="workspace"):
        prims.move_to_pose(300, 0, -50, 180, 0, 0)
    assert sim.motion_log == []


def test_move_to_joints_limits(prims, sim):
    prims.move_to_joints([10, 20, 30, 0, 90, 0])
    assert sim.state["joints_deg"] == [10, 20, 30, 0, 90, 0]
    with pytest.raises(ValidationError, match="J3"):
        prims.move_to_joints([0, 0, 170, 0, 0, 0])


def test_move_relative_base(prims, sim):
    r = prims.move_relative(dz=-50)
    assert sim.motion_log[-1].startswith("MovL(300.0000,0.0000,350.0000,180.0000")
    assert sim.motion_log[-1].endswith("User=0,Tool=0,SpeedL=20,AccL=20)")
    assert r["pose"][2] == 350


def test_move_relative_tool_frame_points_down(prims, sim):
    # pose Rx=180: tool +Z points down in the base frame -> base Z decreases
    r = prims.move_relative(dz=30, frame="tool")
    assert sim.motion_log[-1].startswith("MovL(300.0000,0.0000,370.0000")
    assert r["pose"][2] == pytest.approx(370)


def test_move_relative_limits(prims, sim):
    with pytest.raises(ValidationError):
        prims.move_relative(dx=80, dy=80)          # 113 mm > 100
    with pytest.raises(ValidationError, match="Xoay"):
        prims.move_relative(drz=60)                 # 60° > 45°
    assert sim.motion_log == []


def test_move_relative_below_table_rejected(prims, sim):
    sim.state["pose"][2] = 50
    with pytest.raises(ValidationError, match="Z"):
        prims.move_relative(dz=-80)
    assert sim.motion_log == []


def test_jog_joint_checks_resulting_angle(prims, sim):
    prims.jog_joint(1, 10)
    assert sim.state["joints_deg"][0] == 10
    sim.state["joints_deg"][2] = 150
    with pytest.raises(ValidationError, match="J3"):
        prims.jog_joint(3, 10)


def test_save_and_go_to_position(prims, sim):
    prims.save_position("above_bin")
    prims.move_to_joints([0, 10, 80, 0, 90, 0])
    prims.save_position("above_bin")            # overwrite same name
    assert prims.get_state()["saved_positions"] == ["above_bin"]
    sim.state["joints_deg"] = [0, 0, 0, 0, 0, 0]
    prims.go_to_position("above_bin")
    assert sim.state["joints_deg"] == [0, 10, 80, 0, 90, 0]
    prims.go_to_position("home")                # config fallback
    assert sim.state["joints_deg"] == [0, 0, 90, 0, 90, 0]
    with pytest.raises(ValidationError):
        prims.go_to_position("nowhere")


def test_gripper_primitives(prims, gripper):
    assert prims.open_gripper()["position"] == 1000
    assert gripper.connected_with == (1, 115200, "N")   # lazily connected from config
    assert prims.close_gripper()["holding"] is False
    gripper.object_at = 400
    prims.open_gripper()
    assert prims.close_gripper()["holding"] is True
    assert prims.set_gripper(700)["position"] == 700
    assert "gripper" in prims.get_state()


def test_motion_refused_when_not_ready(prims, sim):
    sim.state["enabled"] = 0
    with pytest.raises(SafetyError):
        prims.move_relative(dz=10)
    assert sim.motion_log == []


def test_timeout_stops_robot(prims, sim):
    sim.stall = True
    with pytest.raises(RobotError, match="not arrived"):
        prims.move_to_pose(350, 0, 300, 180, 0, 0)
    assert sim.dashboard_log[-1] == "Pause()"


def test_alarm_during_motion_stops_robot(prims, sim):
    sim.stall = True
    sim.motion = lambda cmd: sim.state.__setitem__("error", 1)
    with pytest.raises(SafetyError, match="alarm"):
        prims.jog_joint(1, 5)
    assert sim.dashboard_log[-1] == "Pause()"


# ---------------------------------------------------------------- fingertip TCP + speed cap
@pytest.fixture
def gripper_prims(actions, gripper):
    from actions.primitives import PrimitiveSkills
    actions.config.gripper.tool_length_mm = 118
    return PrimitiveSkills(actions, lambda: gripper)


def test_move_to_pose_targets_fingertip(gripper_prims, sim):
    # pointing straight down: flange must be 118 mm ABOVE the requested fingertip point
    r = gripper_prims.move_to_pose(-450, -300, 50, 180, 0, 0)
    assert sim.motion_log[-1].startswith("MovJ(-450.0000,-300.0000,168.0000,180.0000")
    assert r["pose"][:3] == pytest.approx([-450, -300, 50], abs=0.01)        # reported as fingertip
    assert r["flange_pose"][2] == pytest.approx(168, abs=0.01)


def test_get_state_reports_fingertip(gripper_prims, sim):
    sim.state["pose"] = [300.0, 0.0, 400.0, 180.0, 0.0, 0.0]
    s = gripper_prims.get_state()
    assert s["pose"][2] == pytest.approx(282) and s["flange_pose"][2] == 400


def test_fingertip_below_table_rejected_even_if_flange_ok(gripper_prims, sim):
    with pytest.raises(ValidationError, match="Z"):
        gripper_prims.move_to_pose(-450, -300, -20, 180, 0, 0)     # flange z 98 ok, tip -20 not
    assert sim.motion_log == []


def test_speed_cap_clamps(gripper_prims, sim):
    gripper_prims.speed_cap = 10
    gripper_prims.move_relative(dz=10, speed=40)
    assert "SpeedL=10,AccL=10" in sim.motion_log[-1]
    gripper_prims.move_relative(dz=-10)                              # default 20 -> 10
    assert "SpeedL=10" in sim.motion_log[-1]
    from actions.registry import build_registry
    spec = next(s for s in build_registry(gripper_prims) if s.name == "move_to_pose")
    assert spec.parameters["properties"]["speed"]["maximum"] == 10


def test_a_path_from_above_the_ceiling_may_come_down_but_not_go_higher(prims):
    """20 Nov: hand-guided to the D455 (fingertip 555 mm, ceiling 530): go_to_position 'start' was refused
    at 'joint path 1%' — every move, even the way down, started above the ceiling."""
    import dataclasses
    prims.cfg.workspace = dataclasses.replace(prims.cfg.workspace, z=(0, 530), sector_deg=None, keep_out=[])
    down = [-300, -400, 400, 180, 0, 0]
    prims._check_linear_path([-300, -400, 700, 180, 0, 0], down)            # flange 700 -> 400: allowed
    with pytest.raises(ValidationError, match="Z="):
        prims._check_linear_path([-300, -400, 700, 180, 0, 0], [-300, -400, 760, 180, 0, 0])   # higher: no
    with pytest.raises(ValidationError, match="target"):
        prims._check_linear_path([-300, -400, 700, 180, 0, 0], [-300, -400, 650, 180, 0, 0])   # still above
    with pytest.raises(ValidationError, match="path"):
        prims._check_linear_path([-300, -400, 500, 180, 0, 0], [-300, -400, 700, 180, 0, 0])   # from under: normal
    assert prims._escape([-300, -400, 380], [-300, -400, 507]) is None                         # inside: no escape


def test_a_path_from_outside_the_working_sector_may_only_come_back(prims):
    """20 Nov, after the D455 touches: fingertip at -55° (sector 167.8° -> -92.9°): go_to_position refused at
    'joint path 1%'. Turning back toward the sector is allowed, turning further away is not."""
    import dataclasses
    prims.cfg.workspace = dataclasses.replace(prims.cfg.workspace, z=(0, 530), sector_deg=(167.8, -92.9),
                                              keep_out=[], max_radius_mm=690)
    at = lambda deg, z=400: [500 * math.cos(math.radians(deg)), 500 * math.sin(math.radians(deg)), z, 180, 0, 0]
    prims._check_linear_path(at(-55), at(-120))                    # back into the sector: allowed
    with pytest.raises(ValidationError, match="working sector"):
        prims._check_linear_path(at(-55), at(-20))                 # further out: refused on the path
    with pytest.raises(ValidationError, match="target"):
        prims._check_linear_path(at(-55), at(-80))                 # nearer, but the target is still outside
