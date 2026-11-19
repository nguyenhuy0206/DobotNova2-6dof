"""Explore mode: frames, table keep-out, view guard, 50 % grip, self-calibration, notes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from actions.primitives import PrimitiveSkills
from common.errors import SafetyError, ValidationError
from perception.transforms import pose_to_matrix


@pytest.fixture
def p(actions, gripper):
    return PrimitiveSkills(actions, lambda: gripper)


# ---------------------------------------------------------------- controller active tool vs flange
def test_active_tool_offset_measured_from_controller_fk(p, sim):
    # Feedback reports the ACTIVE tool, 50 mm beyond the flange along flange Z (pointing down);
    # Mov* with Tool=0 address the flange (what happens on the real controller after ResetRobot).
    sim.active_tool = [0.0, 0.0, 50.0]
    sim.state["pose"] = [300.0, 0.0, 400.0, 180.0, 0.0, 0.0]        # flange at 450
    p.tool_length_mm = 118
    s = p.get_state()
    assert p.active_tool_offset == pytest.approx([0, 0, 50])
    assert s["flange_pose"][2] == pytest.approx(450) and s["pose"][2] == pytest.approx(332)
    r = p.move_to_pose(300, 0, 200, 180, 0, 0, linear=True)
    # fingertip 200 -> flange 318, sent with Tool=0; arrival judged on the converted feedback
    assert sim.motion_log[-1].startswith("MovL(300.0000,0.0000,318.0000,180.0000") and "Tool=0" in sim.motion_log[-1]
    assert r["pose"][2] == pytest.approx(200) and sim.state["pose"][2] == pytest.approx(268)
    r = p.move_relative(dz=30, drz=20)                              # rotation pivots on the fingertip
    assert r["pose"][:3] == pytest.approx([300, 0, 230], abs=0.05)


def test_lost_fk_is_an_error_not_a_guess(p, sim):
    sim.active_tool = [0.0, 0.0, 50.0]
    p.get_state()
    sim.active_tool, sim.fk = None, None             # FK stops answering
    from common.errors import RobotError
    with pytest.raises(RobotError, match="refusing to guess"):
        p.get_state()


def test_rotated_active_tool_is_refused(p, sim):
    sim.fk = lambda j: [300.0, 0.0, 400.0, 180.0, 0.0, 30.0]
    with pytest.raises(SafetyError, match="rotated"):
        p.move_relative(dz=5)
    assert sim.motion_log == []


def test_no_fk_assumes_flange(p, sim):
    p.get_state()
    assert list(p.active_tool_offset) == [0, 0, 0] and "assumed" in p.tool_offset_source


# ---------------------------------------------------------------- table keep-out
@pytest.fixture
def table(p, sim):
    """Table at z=0, tool length 0; J2 lowers the flange 10 mm per degree."""
    sim.fk = lambda j: [300.0, 0.0, 400.0 - 10.0 * j[1], 180.0, 0.0, 0.0]
    sim.state["pose"] = [300.0, 0.0, 400.0, 180.0, 0.0, 0.0]
    p.cfg.workspace.z = (-500.0, 700.0)
    p.table_z_mm = 0.0                      # floors: tip 10, flange 60
    p.m.approach_height_mm = 0.0            # approach zone tested separately
    return p


def test_joint_path_checked_against_table(table, sim):
    table.jog_joint(2, 20)                                       # flange 400 -> 200: fine
    with pytest.raises(ValidationError, match="table"):
        table.move_to_joints([0, 36, 90, 0, 90, 0])              # flange 400 -> 40 < 60
    assert not sim.motion_log[-1].startswith("JointMovJ")


def test_cartesian_floor_and_straight_line_near_table(table, sim):
    with pytest.raises(ValidationError, match="table"):
        table.move_to_pose(300, 0, 5, 180, 0, 0)                 # tip below 10
    r = table.move_to_pose(300, 50, 80, 180, 0, 0)               # low: no IK, near table -> MovL
    assert sim.motion_log[-1].startswith("MovL(") and "straight line" in r["note"]


def test_move_relative_floor(table, sim):
    sim.state["pose"][2] = 70
    sim.fk = lambda j: list(sim.state["pose"])                   # active tool = flange here
    with pytest.raises(ValidationError, match="table"):
        table.move_relative(dz=-20)                             # flange 50 < 60
    assert table.get_state()["table"]["min_fingertip_z"] == 10


# ---------------------------------------------------------------- view guard
def test_view_guard_undoes_move(p, sim):
    p.view_guard = lambda: {"visible": False, "detail": "marker gone"}
    with pytest.raises(SafetyError, match="camera view"):
        p.move_relative(dx=50)
    assert sim.motion_log[-1].startswith("MovL(300.0000,0.0000,400.0000")   # moved back
    assert sim.state["pose"][:3] == [300, 0, 400]
    p.view_guard = lambda: {"visible": True, "marker_pixel": [10, 20]}
    assert p.jog_joint(1, 5)["robot_in_view"]["marker_pixel"] == [10, 20]


# ---------------------------------------------------------------- 50 % grip
def test_grip_closes_only_to_limit(p, gripper):
    p.grip_close_position = 500
    r = p.close_gripper()
    assert gripper.position == 500 and r["close_target"] == 500 and r["holding"] is False
    gripper.object_at = 620
    assert p.close_gripper()["holding"] is True
    with pytest.raises(ValidationError):
        p.set_gripper(100)


# ---------------------------------------------------------------- self-calibration


# ---------------------------------------------------------------- notes + wiring
def test_notes_reach_instructions(tmp_path):
    from agent.harness import AgentHarness
    from common.config import AgentConfig
    from memory.notes import NotesStore
    from memory.skills import SkillLibrary

    notes = NotesStore(tmp_path / "notes.jsonl")
    notes.add("box top at z=-20 mm", "geometry")
    with pytest.raises(ValidationError):
        notes.add("x")
    assert len(notes.add("y" * 2000)["saved"]["text"]) == 800
    assert notes.read("geometry")["total"] == 1
    h = AgentHarness(None, None, AgentConfig(), skills=SkillLibrary(tmp_path / "s.json"), notes=notes)
    assert "box top at z=-20 mm" in h._instructions("pick a block")


def test_explore_tools_registered(p, tmp_path):
    from agent.explore import explore_specs, load_explore_config
    from agent.toolkit import build_toolkit
    from common.config import AgentConfig, PerceptionConfig, RobotConfig
    from memory.knowledge import KnowledgeStore
    from memory.notes import NotesStore
    from memory.skills import SkillLibrary
    from perception.scene import Scene

    cfg = load_explore_config()
    assert cfg.gripper.close_position == 500
    # camera calibration is operator infrastructure (18 Nov), not something Astra calls
    assert not {"auto_calibrate", "verify_calibration", "record_calibration_point"} & set(cfg.allowed_tools)
    scene = Scene(PerceptionConfig(colors={"red": [[[0, 120, 70], [10, 255, 255]]]}), {})
    agent = AgentConfig(allowed_tools=cfg.allowed_tools)
    lib = SkillLibrary(tmp_path / "skills.json")
    p.grip_close_position = 500
    executor, _ = build_toolkit("dry_run", robot_cfg=RobotConfig(), agent_cfg=agent, perception_cfg=scene.cfg,
                                primitives=p, scene=scene, library=lib, seed_skills=False,
                                extra_specs=explore_specs(lambda: {"visible": True}, NotesStore(tmp_path / "n"),
                                                          knowledge=KnowledgeStore(tmp_path / "c.json")))
    names = {s.name for s in executor.list()}
    assert {"propose_hypothesis", "check_robot_in_view",
            "define_concept", "query_observations", "record_measurement", "read_history"} <= names
    assert not {"record_evidence", "generalize_hypothesis", "read_notes"} & names      # evidence is computed
    assert "write_note" not in names                                  # raw notes are not written any more
    assert "stop" not in names and "set_gripper" not in names          # only a controller alarm stops
    assert not {"auto_calibrate", "verify_calibration", "record_calibration_point"} & names   # operator calibrates
    assert "search_object" not in names and lib.all() == []          # no seeds in explore mode
    assert "500" in executor.specs["close_gripper"].description
    assert json.dumps(executor.specs["look"].parameters).count("zoom") == 1


def test_pure_lift_allowed_below_floor(table, sim):
    sim.state["pose"][2] = 5                                     # tip below the 10 mm floor
    sim.fk = lambda j: list(sim.state["pose"])
    with pytest.raises(ValidationError, match="table"):
        table.move_relative(dx=20, dz=30)                        # sideways at that height: no
    assert table.move_relative(dz=80)["pose"][2] == pytest.approx(85)


# ---------------------------------------------------------------- scripted auto-calibration


def test_unreachable_pose_refused_without_alarm(p, sim):
    from perception.transforms import matrix_to_rpy_deg

    assert matrix_to_rpy_deg(pose_to_matrix([0, 0, 0, 170, -20, 35])[:3, :3]) == pytest.approx([170, -20, 35])
    sim.fk = lambda j: list(sim.state["pose"])
    real_dashboard = sim.dashboard

    def dashboard(cmd):
        if cmd.startswith("InverseSolution"):
            from common.errors import RobotError
            sim.dashboard_log.append(cmd)
            raise RobotError("Controller từ chối InverseSolution: -40000")
        return real_dashboard(cmd)
    sim.dashboard = dashboard
    with pytest.raises(ValidationError, match="reachable"):
        p.move_relative(dx=50)
    assert sim.motion_log == []


def test_scene_ignores_coloured_tape_on_the_robot():
    from perception.scene import Scene
    from perception.transforms import HandEye
    from common.config import PerceptionConfig
    from drivers.camera import Frame, Intrinsics
    import cv2

    img = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(img, (40, 150), (70, 180), (0, 200, 0), -1)        # block on the table (small)
    cv2.rectangle(img, (200, 20), (260, 70), (0, 200, 0), -1)        # tape on the wrist (bigger)
    depth = np.full((240, 320), 1.0, np.float32)
    frame = Frame(color=img, depth_m=depth, intrinsics=Intrinsics(fx=300, fy=300, cx=160, cy=120, width=320, height=240))
    T = np.eye(4); T[:3, :3] = np.diag([1, -1, -1]); T[2, 3] = 1.07  # camera 1.07 m up, looking down
    scene = Scene(PerceptionConfig(colors={"green": [[[40, 80, 60], [85, 255, 255]]]}, min_area_px=50),
                  {"env": SimpleNamespace(read=lambda: frame)}, HandEye(T, eye_in_hand=False))
    scene.object_z_max = 150
    # tilt depth so the tape is "high": closer to the camera
    depth[20:71, 200:261] = 0.7
    found = scene.find("green")
    assert len(found) == 1 and found[0].pixel[1] > 140


# ---------------------------------------------------------------- approach zone + contact
def test_approach_zone_short_slow_steps(table, sim):
    table.m.approach_height_mm = 120                             # zone: fingertip below z=120
    sim.state["pose"] = [300.0, 0.0, 150.0, 180.0, 0.0, 0.0]
    sim.fk = lambda j: list(sim.state["pose"])
    with pytest.raises(ValidationError, match="approach zone"):
        table.move_to_pose(300, 0, 80, 180, 0, 0)               # 40 mm inside the zone
    table.move_to_pose(300, 0, 100, 180, 0, 0, speed=10)        # 20 mm inside: ok, slowed down
    assert sim.motion_log[-1].startswith("MovL(") and "SpeedL=3," in sim.motion_log[-1]
    table.move_relative(dz=-20, speed=10)
    assert "SpeedL=3," in sim.motion_log[-1]
    table.move_relative(dz=80, speed=10)                        # leaving straight up: full speed
    assert "SpeedL=10," in sim.motion_log[-1]


def test_contact_stops_and_backs_off(table, sim):
    from common.errors import ContactError

    table.m.approach_height_mm, table.m.stall_s, table.m.move_timeout_s = 120, 0.05, 5
    sim.state["pose"] = [300.0, 0.0, 100.0, 180.0, 0.0, 0.0]
    sim.fk = lambda j: list(sim.state["pose"])
    real = sim.motion
    blocked = {"n": 0}

    def motion(cmd):                       # first move: blocked after 5 mm; later moves normal
        if blocked["n"] == 0:
            blocked["n"] = 1
            sim.motion_log.append(cmd)
            sim.state["pose"][2] = 95.0
            sim.state["running"] = 1
            return "queued"
        sim.state["running"] = 0
        return real(cmd)
    sim.motion = motion
    orig_feedback = sim.feedback
    calls = {"n": 0}

    def feedback():                        # first poll sees the start pose, then the blocked one
        calls["n"] += 1
        fb = orig_feedback()
        if calls["n"] == 1 and blocked["n"] == 1:
            fb["pose"][2] = 99.0
        return fb
    sim.feedback = feedback
    with pytest.raises(ContactError, match=r"CONTACT at fingertip Z=95"):
        table.move_relative(dz=-20)
    assert sim.dashboard_log[-1] == "ResetRobot()" or "ResetRobot()" in sim.dashboard_log
    assert sim.state["pose"][2] == pytest.approx(110)            # backed off 15 mm


def _collision_world(sim, clears: bool):
    """First move: the controller raises a collision alarm at Z 95; ClearError works or not."""
    sim.state["pose"] = [300.0, 0.0, 100.0, 180.0, 0.0, 0.0]
    sim.fk = lambda j: list(sim.state["pose"])
    real_motion, real_dash = sim.motion, sim.dashboard
    hit = {"n": 0}

    def motion(cmd):
        if hit["n"] == 0:
            hit["n"] = 1
            sim.motion_log.append(cmd)
            sim.state.update(pose=[300.0, 0.0, 95.0, 180.0, 0.0, 0.0], error=1, mode=9, running=0)
            return "queued"
        return real_motion(cmd)

    def dashboard(cmd):
        if cmd.startswith("ClearError") and clears:
            sim.state.update(error=0, mode=5)
        return real_dash(cmd)
    sim.motion, sim.dashboard = motion, dashboard


def test_collision_alarm_in_the_approach_zone_is_a_contact(table, sim):
    from common.errors import ContactError

    table.m.approach_height_mm, table.m.move_timeout_s, table.m.poll_s = 120, 5, 0.01
    _collision_world(sim, clears=True)
    with pytest.raises(ContactError, match=r"CONTACT at fingertip Z=95.*collision detection"):
        table.move_relative(dz=-20)
    assert "ClearError()" in sim.dashboard_log
    assert sim.state["pose"][2] == pytest.approx(110) and sim.state["error"] == 0    # cleared, backed off


def test_collision_alarm_that_does_not_clear_stops(table, sim):
    from common.errors import SafetyError

    table.m.approach_height_mm, table.m.move_timeout_s, table.m.poll_s = 120, 5, 0.01
    table.m.recovery_timeout_s = 0.5
    _collision_world(sim, clears=False)
    with pytest.raises(SafetyError, match="robot alarm during motion"):
        table.move_relative(dz=-20)
    assert sim.state["pose"][2] == pytest.approx(95)                                  # did not move again


def test_open_gripper_partial(p, gripper):
    p.grip_open_position = 750
    assert p.open_gripper()["position"] == 750


def test_gripper_tools_are_fixed_open_full_close_half_speed_20(prims, gripper):
    from actions.registry import build_registry

    prims.grip_open_position, prims.grip_close_position, prims.grip_speed = 1000, 500, 20
    specs = {s.name: s for s in build_registry(prims)}
    assert "1000" in specs["open_gripper"].description and "speed 20 %" in specs["open_gripper"].description
    assert "500" in specs["close_gripper"].description
    assert specs["close_gripper"].parameters["properties"]["speed"]["enum"] == [20]
    seen = []
    gripper.go_to = lambda pos, force=30, speed=50, wait=True: seen.append((pos, speed)) or {"grip_state": 1,
                                                                                            "position": pos}
    prims.close_gripper(speed=10)                         # a direct call cannot pick another speed either
    prims.open_gripper()
    assert seen == [(500, 20), (1000, 20)]


def test_preflight_refuses_without_an_env_calibration_and_never_expires_one():
    """18 Nov: calibration is operator infrastructure; a calibration from another day stays in use."""
    from types import SimpleNamespace

    from agent.explore import session as explore
    from agent.explore import ExploreSession
    prims = SimpleNamespace(get_state=lambda: {"mode": 5, "error": False}, table_z_mm=0.3,
                            cfg=SimpleNamespace(gripper=SimpleNamespace(via_ui=False)))
    scene = SimpleNamespace(env_calibration=None, observe=lambda cam: None)
    me = SimpleNamespace(kit=SimpleNamespace(primitives=prims, scene=scene), view_guard=None)
    assert any("not calibrated" in p for p in ExploreSession.preflight(me))
    scene.env_calibration = object()
    assert ExploreSession.preflight(me) == []
    assert not hasattr(explore, "_from_another_day") and not hasattr(explore, "_set_aside")
