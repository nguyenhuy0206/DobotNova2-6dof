"""Perception + calibration + composite skills + toolkit (sim robot, synthetic camera)."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from actions.skill_runner import fill
from agent.harness import AgentHarness
from agent.toolkit import build_toolkit
from common.config import AgentConfig, PerceptionConfig
from common.errors import ValidationError
from drivers.camera import Frame, Intrinsics
from memory.skills import SkillLibrary
from memory.store import EpisodeStore
from calibration.geometry import solve_rigid_transform
from perception.scene import Scene
from perception.transforms import HandEye

INTR = Intrinsics(640, 480, 600.0, 600.0, 320.0, 240.0)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


# A camera looking straight down from 1 m above the base origin, rotated 30° about Z.
CAM_TO_BASE = np.eye(4)
CAM_TO_BASE[:3, :3] = rot_z(30) @ np.diag([1, -1, -1])
CAM_TO_BASE[:3, 3] = [0.40, 0.0, 1.0]


def test_calibration_recovers_transform():
    rng = np.random.default_rng(0)
    cam = rng.uniform([-0.3, -0.2, 0.6], [0.3, 0.2, 1.0], (10, 3))
    base_mm = (cam @ CAM_TO_BASE[:3, :3].T + CAM_TO_BASE[:3, 3]) * 1000 + rng.normal(0, 0.5, (10, 3))
    T, rms = solve_rigid_transform(cam, base_mm)
    assert np.allclose(T, CAM_TO_BASE, atol=2e-3) and rms < 2
    with pytest.raises(ValueError):
        solve_rigid_transform(cam[:3], base_mm[:3])


class StaticSource:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame


def scene_with(objects, calibrated=True):
    """objects: list of (bgr colour, (u, v), half size px, depth m)."""
    color = np.full((480, 640, 3), 90, np.uint8)
    depth = np.full((480, 640), 1.0, np.float32)
    for bgr, (u, v), r, d in objects:
        color[v - r:v + r, u - r:u + r] = bgr
        depth[v - r:v + r, u - r:u + r] = d
    frame = Frame(color=color, depth_m=depth, intrinsics=INTR)
    cfg = PerceptionConfig(colors={"red": [[[0, 120, 70], [10, 255, 255]], [[170, 120, 70], [179, 255, 255]]],
                                   "green": [[[40, 80, 60], [85, 255, 255]]]})
    return Scene(cfg, {"env": StaticSource(frame)}, HandEye(CAM_TO_BASE, eye_in_hand=False) if calibrated else None)


def test_scene_finds_red_block_in_robot_coordinates():
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95), ((0, 200, 0), (400, 290), 30, 0.99)])
    red = scene.find("red")
    assert len(red) == 1 and red[0].pixel == (320, 240)
    # straight below the camera, 0.95 m away -> 5 cm above the table, at the camera's x, y
    assert red[0].base_xyz_mm == pytest.approx([400, 0, 50], abs=1)
    green = scene.find("green")[0]
    assert green.base_xyz_mm[2] == pytest.approx(10, abs=1)


def test_look_is_the_picture_only_and_depth_at_answers_for_its_pixels():
    """20 Nov (operator): look detects nothing; Astra reads the picture and asks depth_at for pixels."""
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95)])
    with pytest.raises(ValueError, match="look at the env camera first"):
        scene.depth_at([[320, 240]])
    out = scene.look("env")
    assert out["has_depth"] and out["calibrated"] and "objects" not in out
    assert len(out["image_jpeg_b64"]) > 100
    d = scene.depth_at([[320, 240], [5000, 5]])["points"]
    assert d[0]["depth_mm"] == pytest.approx(950, abs=1) and d[0]["valid_fraction"] == 1.0
    assert d[0]["base_xyz_mm"] == pytest.approx([400, 0, 50], abs=1)      # same point find() gave before
    assert "outside the image" in d[1]["error"]
    p = scene.depth_at([[320, 240]], plane_z=50)["points"][0]
    assert p["at_plane_xyz_mm"] == pytest.approx([400, 0, 50], abs=1)    # same point, from the ray alone
    assert p["depth_spread_mm"] == 0


def test_skill_param_filling():
    p = {"x": 300.0, "z": 50.0, "tool_length_mm": 120, "hover_mm": 100, "speed": 20}
    assert fill({"x": "{x}", "z": "={z} + tool_length_mm + hover_mm", "s": "=speed / 2"}, p) == \
        {"x": 300.0, "z": 270.0, "s": 10}
    with pytest.raises(ValidationError):
        fill("={z} + __import__", p)
    with pytest.raises(ValidationError):
        fill("{missing}", p)


@pytest.fixture
def kit(prims, tmp_path):
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95), ((0, 200, 0), (400, 290), 30, 0.99)])
    robot_cfg = prims.cfg
    robot_cfg.gripper.tool_length_mm = 0
    agent_cfg = AgentConfig(allowed_tools=[])
    return build_toolkit("live", robot_cfg=robot_cfg, agent_cfg=agent_cfg, perception_cfg=scene.cfg,
                         primitives=prims, scene=scene, library=SkillLibrary(tmp_path / "skills.json"))


def test_toolkit_exposes_all_modules(kit):
    executor, _ = kit
    assert {"look", "find_objects", "list_skills", "run_skill", "save_skill", "move_to_pose",
            "close_gripper"} <= set(executor.specs)
    names = {s["name"] for s in executor.execute("list_skills")["result"]}
    assert {"pick_from_above", "place_from_above"} <= names


def test_red_block_into_green_frame_end_to_end(kit, sim, gripper):
    executor, _ = kit
    red = executor.execute("find_objects", {"color": "red"})["result"]["objects"][0]["base_xyz_mm"]
    frame = executor.execute("find_objects", {"color": "green"})["result"]["objects"][0]["base_xyz_mm"]
    gripper.object_at = 400

    pick = executor.execute("run_skill", {"name": "pick_from_above",
                                          "params": dict(zip("xyz", red))})
    assert pick["ok"] and pick["result"]["ok"], pick
    assert pick["result"]["last_result"]["pose"][:2] == pytest.approx(red[:2], abs=1)
    assert gripper.position == 400                      # holding the block

    place = executor.execute("run_skill", {"name": "place_from_above",
                                           "params": dict(zip("xyz", frame))})
    assert place["result"]["ok"] and gripper.position == 1000
    assert sim.state["pose"][:2] == pytest.approx(frame[:2], abs=1)
    lib = executor.specs["list_skills"].handler()
    assert next(s for s in lib if s["name"] == "pick_from_above")["successes"] == 0


def test_dry_run_skill_shows_filled_steps_without_moving(prims, sim, tmp_path):
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95)])
    ex, _ = build_toolkit("dry_run", robot_cfg=prims.cfg, agent_cfg=AgentConfig(allowed_tools=[]),
                          perception_cfg=scene.cfg, primitives=prims, scene=scene,
                          library=SkillLibrary(tmp_path / "s.json"))
    r = ex.execute("run_skill", {"name": "pick_from_above", "params": {"x": 400, "y": 0, "z": 50}})
    assert r["ok"] and r["result"]["steps_run"] == 5 and sim.motion_log == []


def test_skill_failure_stops_and_is_counted(kit, sim):
    executor, _ = kit
    r = executor.execute("run_skill", {"name": "pick_from_above", "params": {"x": 400, "y": 0, "z": -500}})
    assert r["ok"] and not r["result"]["ok"] and "workspace" in r["result"]["failed_step"]["error"]
    lib = executor.specs["list_skills"].handler()
    assert next(s for s in lib if s["name"] == "pick_from_above")["failures"] == 0


def test_save_skill_validates_steps(kit):
    executor, _ = kit
    ok = executor.execute("save_skill", {"name": "wiggle", "description": "jog J6 back and forth",
                                         "steps": [{"tool": "jog_joint", "args": {"joint": 6, "degrees": "{d}"}},
                                                   {"tool": "jog_joint", "args": {"joint": 6, "degrees": "=-d"}}],
                                         "params": {"d": 5}})
    assert ok["ok"]
    assert not executor.execute("save_skill", {"name": "bad", "description": "x",
                                               "steps": [{"tool": "teleport"}]})["ok"]
    assert executor.execute("run_skill", {"name": "wiggle"})["result"]["ok"]


def test_harness_attaches_look_image_and_saves_it(kit, tmp_path):
    executor, _ = kit

    class Client:
        def __init__(self):
            self.inputs = []
            self.script = iter([
                NS(id="r1", output=[NS(type="function_call", name="look", arguments="{}", call_id="c1")], output_text=""),
                NS(id="r2", output=[], output_text="I see a red block."),
            ])

        def create(self, input, **kw):
            self.inputs.append(input)
            return next(self.script)

    client = Client()
    episodes = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(client, executor, AgentConfig(), episodes, SkillLibrary(tmp_path / "s.json"))
    assert h.run("what do you see?") == "I see a red block."
    second = client.inputs[1]
    out = json.loads(second[0]["output"])
    assert "image_jpeg_b64" not in out["result"] and out["result"]["image_file"].endswith(".jpg")
    assert second[1]["role"] == "user" and second[1]["content"][1]["type"] == "input_image"


def test_astra_client_reads_key_from_env_file(tmp_path, monkeypatch):
    from agent.astra_client import AstraClient
    from common.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("gpt6-key=sk-test-123\n")
    monkeypatch.delenv("gpt6-key", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_dotenv(env)
    client = AstraClient(AgentConfig(api_key_env="gpt6-key"))
    assert client.key_source == "gpt6-key" and client.client.api_key == "sk-test-123"
    monkeypatch.delenv("gpt6-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
    assert AstraClient(AgentConfig(api_key_env="gpt6-key")).client.api_key == "sk-shell"


def test_old_seed_skills_are_upgraded_keeping_counts(tmp_path):
    from memory.skills import Skill
    lib = SkillLibrary(tmp_path / "s.json")
    lib.save(Skill(name="pick_from_above", description="old", successes=3, failures=1,
                   steps=[{"tool": "move_to_pose", "args": {"z": "={z} + tool_length_mm + hover_mm"}}]))
    lib.ensure_seeds()
    up = lib.get("pick_from_above")
    from memory.skills import SEED_SKILLS
    current = next(sd for sd in SEED_SKILLS if sd.name == "pick_from_above").version
    assert up.version == current > 1 and up.successes == 3 and up.failures == 1
    assert "tool_length_mm" not in json.dumps(up.steps)


def test_objects_outside_workspace_are_dropped():
    from common.config import WorkspaceLimits
    # second red blob is 0.4 m deeper -> far outside a small workspace box
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95), ((0, 0, 220), (500, 100), 20, 0.20)])
    assert len(scene.find("red")) == 2
    scene.workspace = WorkspaceLimits(x=(-600, 600), y=(-600, 600), z=(0, 700))
    kept = scene.find("red")
    assert len(kept) == 1 and kept[0].pixel == (320, 240)


# ---------------------------------------------------------------- hidden objects
class ArmDependentSource:
    """The block is hidden while J1 is at its start angle (arm in the way)."""

    def __init__(self, sim, visible, hidden, start_j1):
        self.sim, self.visible, self.hidden, self.start = sim, visible, hidden, start_j1

    def read(self):
        return self.hidden if abs(self.sim.state["joints_deg"][0] - self.start) < 1 else self.visible


def test_find_reports_last_seen_when_hidden(prims, sim):
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95)])
    seen = scene.find_report("red")
    assert seen["objects"] and "last_seen" not in seen
    scene.sources["env"] = StaticSource(scene_with([]).sources["env"].frame)        # arm now hides it
    hidden = scene.find_report("red")
    assert hidden["objects"] == [] and hidden["last_seen"]["base_xyz_mm"] == pytest.approx([400, 0, 50], abs=1)


def test_touch_calibration_solver():
    from calibration.geometry import solve_points
    rng = np.random.default_rng(3)
    base = rng.uniform([-550, -450, 10], [-350, -150, 60], (9, 3))
    inv = np.linalg.inv(CAM_TO_BASE)
    cam = (base / 1000) @ inv[:3, :3].T + inv[:3, 3] + rng.normal(0, 0.001, (9, 3))
    cam[4] += 0.04                                           # one bad touch
    T, rms, errors, dropped = solve_points(cam, base)
    assert dropped == [4] and rms < 3 and np.allclose(T[:3, 3], CAM_TO_BASE[:3, 3], atol=0.006)


def test_leave_one_out_reports_unseen_error():
    from calibration.geometry import leave_one_out_errors
    rng = np.random.default_rng(4)
    base = rng.uniform([-550, -450, -60], [-350, -150, 30], (8, 3))
    inv = np.linalg.inv(CAM_TO_BASE)
    cam = (base / 1000) @ inv[:3, :3].T + inv[:3, 3] + rng.normal(0, 0.002, (8, 3))
    loo = np.linalg.norm(leave_one_out_errors(cam, base), axis=1)
    assert 1 < loo.mean() < 10          # honest: bigger than the fit residual, same order as the noise


def test_wrist_overlay_marks_the_fingertips_where_the_rigid_camera_sees_them(tmp_path):
    """20 Nov 164552: the wrist camera looks 69° off the tool axis; the fingertip marks come from the
    calibration. Camera and fingers move together, so the marks do not depend on the robot pose."""
    import json
    import numpy as np
    from perception.wrist_overlay import WristOverlay
    X = np.eye(4)
    X[:3, 3] = [0.0, 0.0, 0.0]                         # camera at the flange, looking down the tool axis
    (tmp_path / "c.json").write_text(json.dumps({"wrist_intrinsics": {"K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                                                                       "dist": [0, 0, 0, 0, 0]},
                                                  "T_flange_wrist": X.tolist()}))
    ov = WristOverlay(tmp_path / "c.json", lambda: {"flange_pose": [0, 0, 300, 180, 0, 0], "gripper": {"position": 500}},
                      tool_length_mm=100.0, finger_half_gap_mm=30.0)
    tips = ov()["fingertips_px"]
    assert tips["centre"] == pytest.approx([320, 240], abs=0.5)
    assert tips["finger+"] == pytest.approx([320, 240 + 500 * 15 / 100], abs=0.5)     # half open: 15 mm off axis
    moved = ov.project([-300, -400, 120, 180, 0, 90], 500)
    assert moved == tips                                                             # pose-independent


def test_routines_and_skills_may_use_depth_at():
    """20 Nov: look no longer detects; without depth_at Astra's own code could not locate anything."""
    from actions import astra_code, skill_runner
    assert astra_code.SKILL_TOOLS is skill_runner.SKILL_TOOLS
    assert "depth_at" in astra_code.SKILL_TOOLS
    assert hasattr(astra_code.RoutineContext, "depth_at")


def test_depth_at_takes_full_image_pixels_whatever_was_shown():
    """23 Nov 154154: 'pixels of the picture last shown' flipped meaning with every zoom; one rule now."""
    scene = scene_with([((0, 0, 220), (320, 240), 20, 0.95)])
    full = scene.look("env")
    z = scene.look("env", zoom=[160, 120, 320, 240])                 # 2x enlarged crop around the centre
    assert z["picture_scale"] == 2.0 and z["picture_size"] == [640, 480]
    p = scene.depth_at([[320, 240]])["points"][0]
    assert p["pixel"] == [320, 240] and "picture_pixel" not in p
    r = scene.depth_at([[160, 120]])
    assert r["points"][0]["pixel"] == [160, 120] and "image_jpeg_b64" not in r
    assert full["image_size"] == [640, 480]
