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
def test_self_calibration_recovers_camera(tmp_path, monkeypatch):
    from perception import robot_view
    from perception.robot_view import SelfCalibration

    rng = np.random.default_rng(1)
    T = pose_to_matrix([400, -200, 600, -120, 5, 30])                 # camera -> base (metres)
    o = np.array([0.03, -0.02, 0.09])                                  # marker on the gripper (m)
    poses = [[250 + rng.uniform(-120, 120), rng.uniform(-150, 150), 250 + rng.uniform(0, 200),
              180 + rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-90, 90)] for _ in range(8)]
    current = {}

    def cam_point(pose):
        P = pose_to_matrix(pose)
        base = P[:3, 3] + P[:3, :3] @ o
        return T[:3, :3].T @ (base - T[:3, 3])

    scene = SimpleNamespace(cfg=SimpleNamespace(calibration_path=tmp_path / "hand_eye_env.json"),
                            env_calibration=None, observe=lambda cam: "frame",
                            locate=lambda f, u, v: (1.0, list(cam_point(current["pose"])), None))
    prims = SimpleNamespace(get_state=lambda: {"flange_pose": current["pose"], "moving": False})
    monkeypatch.setattr(robot_view, "find_marker", lambda *a: SimpleNamespace(center=(320, 240)))
    cal = SelfCalibration(scene, prims, session_file=tmp_path / "session.json")

    with pytest.raises(ValidationError, match="need"):
        cal.solve()
    for pose in poses:
        current["pose"] = pose
        rec = cal.record()
    with pytest.raises(ValidationError, match="same pose"):
        cal.record()
    assert rec["points"] == 8 and rec["rotation_spread_deg"] > 30
    out = cal.solve()
    assert out["rms_mm"] < 1 and scene.env_calibration is not None
    assert np.allclose(scene.env_calibration.T, T, atol=1e-4)
    assert (tmp_path / "hand_eye_env.json").exists()
    assert len(SelfCalibration(scene, prims, session_file=tmp_path / "session.json").points) == 8  # resumes


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
    from agent.explore import _explore_specs, load_explore_config
    from agent.toolkit import build_toolkit
    from common.config import AgentConfig, PerceptionConfig, RobotConfig
    from memory.knowledge import KnowledgeStore
    from memory.notes import NotesStore
    from memory.skills import SkillLibrary
    from perception.robot_view import SelfCalibration
    from perception.scene import Scene

    cfg = load_explore_config()
    assert cfg.gripper.close_position == 500 and "auto_calibrate" in cfg.allowed_tools
    assert "record_calibration_point" not in cfg.allowed_tools            # manual calibration is off
    scene = Scene(PerceptionConfig(colors={"red": [[[0, 120, 70], [10, 255, 255]]]}), {})
    cal = SelfCalibration(scene, p, session_file=tmp_path / "session.json")
    agent = AgentConfig(allowed_tools=cfg.allowed_tools)
    lib = SkillLibrary(tmp_path / "skills.json")
    p.grip_close_position = 500
    executor, _ = build_toolkit("dry_run", robot_cfg=RobotConfig(), agent_cfg=agent, perception_cfg=scene.cfg,
                                primitives=p, scene=scene, library=lib, seed_skills=False,
                                extra_specs=_explore_specs(cal, lambda: {"visible": True}, NotesStore(tmp_path / "n"),
                                                           knowledge=KnowledgeStore(tmp_path / "c.json")))
    names = {s.name for s in executor.list()}
    assert {"auto_calibrate", "verify_calibration", "propose_hypothesis", "check_robot_in_view",
            "define_concept", "query_observations", "record_measurement", "read_history"} <= names
    assert not {"record_evidence", "generalize_hypothesis", "read_notes"} & names      # evidence is computed
    assert "write_note" not in names                                  # raw notes are not written any more
    assert "stop" not in names and "set_gripper" not in names          # only a controller alarm stops
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
class _FakeCal:
    def __init__(self):
        self.n = 0

    def record(self):
        self.n += 1
        if self.n == 3:
            raise ValidationError("marker not visible")
        return {"points": self.n, "marker_pixel": [1, 2], "depth_m": 0.5, "noise_mm": 1.0}

    def solve(self):
        return {"points": self.n - 1, "leave_one_out_mm": {"median": 5.0, "max": 9.0}}


def test_auto_calibration_lifts_then_visits_pattern(table, sim):
    from perception.auto_calibration import auto_calibrate

    sim.state["pose"] = [300.0, 0.0, 5.0, 180.0, 0.0, 0.0]        # below the safe floor (just taught)
    sim.fk = lambda j: list(sim.state["pose"])
    calls = []
    table.view_guard = lambda: ({"visible": False, "detail": "x"} if len(calls) == 5 else {"visible": True}) \
        if calls.append(1) is None else None
    report = auto_calibrate(table, _FakeCal(), speed=10, settle_s=0, log=lambda m: None, target_points=6)
    assert report["points"] == 6
    assert sim.motion_log[0].startswith("MovL(300.0000,0.0000,")            # straight lift first
    assert sim.motion_log[0].split(",")[3:6] == ["180.0000", "0.0000", "0.0000"]
    assert report["center"] == [300.0, 0.0, 150.0]
    reasons = [s["why"] for s in report["skipped"]]
    assert any("camera view" in r for r in reasons) and any("marker" in r for r in reasons)
    assert sim.state["pose"][:3] == pytest.approx([300, 0, 150])          # ends at the centre
    assert all("SpeedL=10" in c and "Tool=0" in c for c in sim.motion_log if c.startswith("MovL"))
    assert len(report["skipped"]) == 2


def test_auto_calibration_aborts_on_alarm(table, sim):
    from perception.auto_calibration import auto_calibrate

    sim.fk = lambda j: list(sim.state["pose"])
    sim.state["pose"] = [300.0, 0.0, 200.0, 180.0, 0.0, 0.0]

    class Alarm(_FakeCal):
        def record(self):
            sim.state["error"] = 1
            raise SafetyError("robot alarm during motion — stopped.")
    with pytest.raises(SafetyError, match="alarm"):
        auto_calibrate(table, Alarm(), speed=10, settle_s=0, log=lambda m: None)


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


def test_solve_drops_a_bad_point(tmp_path, monkeypatch):
    from perception import robot_view
    from perception.robot_view import SelfCalibration

    rng = np.random.default_rng(3)
    T = pose_to_matrix([400, -200, 600, -120, 5, 30])
    o = np.array([0.02, 0.03, 0.06])
    pts = []
    for k in range(12):
        pose = [250 + rng.uniform(-100, 100), rng.uniform(-100, 100), 250 + rng.uniform(0, 120),
                180 + rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-20, 20)]
        P = pose_to_matrix(pose)
        cam = T[:3, :3].T @ (P[:3, 3] + P[:3, :3] @ o - T[:3, 3]) + rng.normal(0, 0.002, 3)
        if k == 4:
            cam = cam + np.array([0.0, 0.0, 0.08])                   # 80 mm depth glitch
        pts.append({"camera_m": list(cam), "flange_pose": pose, "t": 0})
    (tmp_path / "s.json").write_text(json.dumps({"points": pts}))
    scene = SimpleNamespace(cfg=SimpleNamespace(calibration_path=tmp_path / "he.json"), env_calibration=None)
    out = SelfCalibration(scene, None, session_file=tmp_path / "s.json").solve()
    assert [d["point"] for d in out["dropped_outliers"]] == [4]
    assert out["points"] == 11 and out["leave_one_out_mm"]["median"] < 8


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


def _calibration_world(tmp_path, monkeypatch, seed=5, n=10):
    from perception import robot_view
    from perception.robot_view import SelfCalibration

    rng = np.random.default_rng(seed)
    T = pose_to_matrix([400, -200, 600, -120, 5, 30])
    o = np.array([0.03, -0.02, 0.09])
    world = {"T": T, "pose": None}

    def cam_point(pose):
        P = pose_to_matrix(pose)
        base = P[:3, 3] + P[:3, :3] @ o
        return world["T"][:3, :3].T @ (base - world["T"][:3, 3])

    scene = SimpleNamespace(cfg=SimpleNamespace(calibration_path=tmp_path / "hand_eye_env.json"),
                            env_calibration=None, observe=lambda cam: "frame",
                            locate=lambda f, u, v: (1.0, list(cam_point(world["pose"])), None))
    prims = SimpleNamespace(get_state=lambda: {"flange_pose": world["pose"], "moving": False})
    monkeypatch.setattr(robot_view, "find_marker", lambda *a: SimpleNamespace(center=(320, 240)))
    cal = SelfCalibration(scene, prims, session_file=tmp_path / "session.json")
    poses = [[250 + rng.uniform(-120, 120), rng.uniform(-150, 150), 220 + rng.uniform(0, 200),
              180 + rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-90, 90)] for _ in range(n)]
    return cal, scene, world, poses


def test_verify_calibration_catches_a_moved_camera(tmp_path, monkeypatch):
    cal, scene, world, poses = _calibration_world(tmp_path, monkeypatch)
    for pose in poses:
        world["pose"] = pose
        cal.record()
    cal.solve()
    world["pose"] = [300, 50, 260, 180, 10, -40]                      # a pose that was not recorded
    ok = cal.verify()
    assert ok["calibration_valid"] and ok["error_mm"] < 2
    world["T"] = pose_to_matrix([430, -180, 600, -120, 5, 30])         # camera bumped by ~36 mm
    bad = cal.verify()
    assert bad["calibration_valid"] is False and bad["deactivated"] and scene.env_calibration is None
    assert not (tmp_path / "hand_eye_env.json").exists()               # set aside as *.rejected-*.json
    assert any(p.name.startswith("hand_eye_env.rejected-") for p in tmp_path.iterdir())


def test_solve_rejects_points_mixed_from_two_camera_positions(tmp_path, monkeypatch):
    cal, scene, world, poses = _calibration_world(tmp_path, monkeypatch, n=8)
    for pose in poses[:5]:
        world["pose"] = pose
        cal.record()
    world["T"] = pose_to_matrix([520, -60, 600, -110, 5, 45])          # camera moved between points
    for pose in poses[5:]:
        world["pose"] = pose
        cal.record()
    with pytest.raises(ValidationError, match="REJECTED"):
        cal.solve()
    assert scene.env_calibration is None and not (tmp_path / "hand_eye_env.json").exists()


def test_calibration_advice_asks_for_height_spread(tmp_path, monkeypatch):
    cal, scene, world, poses = _calibration_world(tmp_path, monkeypatch, n=3)
    for i, pose in enumerate(poses):
        world["pose"] = [pose[0], pose[1], 230 + 5 * i, *pose[3:]]
        rec = cal.record()
    assert "HEIGHT" in rec["advice"]


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



def test_auto_calibrate_tool_runs_the_scripted_calibration(monkeypatch):
    from types import SimpleNamespace as NS

    from agent.explore import auto_calibrate_spec
    from perception import auto_calibration

    calls = {}
    prims = NS(table_z_mm=41.96, speed_cap=10)
    cal = NS(p=prims, reset=lambda: calls.setdefault("reset", True))

    def fake_auto(p, calibration, center=None, span_mm=100, speed=10, target_points=12, log=print):
        calls.update(center=center, span=span_mm, speed=speed, points=target_points)
        return {"points": 12, "rms_mm": 4.4, "leave_one_out_mm": {"median": 5.5, "max": 9.0},
                "marker_offset_in_flange_mm": [18.2, 1.0, 44.8], "skipped": [{"pose": "6", "why": "unreachable"},
                                                                                {"pose": "9", "why": "unreachable"}]}
    monkeypatch.setattr(auto_calibration, "auto_calibrate", fake_auto)
    monkeypatch.setattr(auto_calibration, "suggest_center", lambda p: [-300.0, -350.0])     # near the edge
    spec = auto_calibrate_spec(cal)
    assert spec.moves_robot and "whenever the calibration is in doubt" in spec.description
    out = spec.handler()
    assert calls["reset"] and calls["center"] == [-300.0, -350.0, 41.96 + 150] and calls["speed"] == 10
    assert out["calibrated"] and out["rms_mm"] == 4.4 and out["skipped_why"] == {"unreachable": 2}
    spec.handler(center=[-320, -360, 200], span_mm=80)
    assert calls["center"] == [-320.0, -360.0, 200.0] and calls["span"] == 80.0
