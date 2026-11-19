"""Token-efficient harness: accounting, bounded context, chunking, selective images,
event router, local skill executor with validation gate. Mocks only — no network,
no robot."""
import json
import math
from types import SimpleNamespace as NS

import pytest

from actions import ActionExecutor, build_registry
from actions.local_steps import LocalSteps
from actions.skill_runner import SkillRunner
from agent.chunking import ChunkPolicy
from agent.context import ContextPolicy, WorkingMemory
from agent.events import EventRouter, EventType
from agent.harness import AgentHarness, HarnessOptions
from agent.metrics import EpisodeMetrics, usage_of
from agent.vision import ImagePolicy
from common.config import AgentConfig
from memory.skills import Skill, SkillLibrary
from memory.store import EpisodeStore


def usage(inp=1000, out=50, cached=800):
    return NS(input_tokens=inp, output_tokens=out, total_tokens=inp + out,
              input_tokens_details=NS(cached_tokens=cached), output_tokens_details=NS(reasoning_tokens=7))


def call(name, args, cid):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid)


class Client:
    def __init__(self, script):
        self.script, self.kw = iter(script), []

    def create(self, **kw):
        self.kw.append(kw)
        return next(self.script)


def harness(prims, tmp_path, script, variant="V0", **kw):
    client = Client(script)
    ex = ActionExecutor(build_registry(prims), mode="live")
    h = AgentHarness(client, ex, AgentConfig(max_steps=kw.pop("max_steps", 50)), EpisodeStore(tmp_path / "ep"),
                     SkillLibrary(tmp_path / "s.json"), options=HarnessOptions.from_variant(variant, **kw),
                     state_fn=kw.pop("state_fn", None))
    return h, client


# ---------------------------------------------------------------- Phase 1: accounting
def test_usage_parsed_from_api_response():
    u = usage_of(NS(usage=usage()))
    assert u == {"input_tokens": 1000, "cached_tokens": 800, "output_tokens": 50, "reasoning_tokens": 7,
                 "total_tokens": 1050}
    assert usage_of(NS()) is None                         # mock without usage: not invented


def test_every_request_logged_with_usage_and_episode_metrics(prims, tmp_path):
    h, _ = harness(prims, tmp_path, [NS(id="r1", output=[call("get_state", {}, "c1")], output_text="", usage=usage()),
                                     NS(id="r2", output=[], output_text="done", usage=usage(2000, 10, 1500))])
    assert h.run("x") == "done"
    log = h.episodes.load(h.last_episode)
    reqs = [e for e in log if e["event"] == "usage"]
    assert [r["input_tokens"] for r in reqs] == [1000, 2000] and all("latency_s" in r for r in reqs)
    m = log[-1]
    assert m["event"] == "metrics" and m["model_calls"] == 2 and m["total_tokens"] == 3060
    assert m["cached_tokens"] == 2300 and m["usage_complete"] and m["tool_calls"] == 1


def test_metrics_count_grasps_and_safety_stops():
    m = EpisodeMetrics()
    m.add_action("close_gripper", {"ok": True, "result": {"holding": False}})
    m.add_action("move_relative", {"ok": False, "error": "move_relative: CONTACT at fingertip Z=62.8"})
    m.add_action("run_skill", {"ok": True, "result": {"ok": False, "grasp_attempts": 2}})
    s = m.summary()
    assert s["grasp_attempts"] == 3 and s["safety_stops"] == 1 and s["skill_failures"] == 1


def test_token_budget_stops_episode(prims, tmp_path):
    script = [NS(id=f"r{i}", output=[call("get_state", {}, f"c{i}")], output_text="", usage=usage(5000))
              for i in range(10)]
    h, client = harness(prims, tmp_path, script, max_episode_tokens=12000)
    assert h.run("x").startswith("Stopped: token budget")
    assert len(client.kw) == 3


# ---------------------------------------------------------------- Phase 2: bounded context
def test_bounded_context_is_stateless_and_does_not_grow(prims, tmp_path):
    n = 80
    script = [NS(id=f"r{i}", output=[call("get_state", {}, f"c{i}")], output_text=f"step {i}") for i in range(n)]
    script.append(NS(id="end", output=[], output_text="done"))
    h, client = harness(prims, tmp_path, script, variant="V1", max_steps=100, summary_max_chars=1000)
    h.run("task")
    assert all("previous_response_id" not in kw for kw in client.kw)
    sizes = [len(json.dumps(kw["input"])) for kw in client.kw]
    tail = sizes[-12:]
    assert max(tail) - min(tail) < 0.05 * max(tail) and max(sizes) < 4000   # plateau: bounded by the caps
    text = client.kw[-1]["input"][0]["content"][0]["text"]
    assert text.startswith("TASK:\ntask") and "RESULTS OF YOUR LAST TOOL CALLS" in text
    assert "#0 get_state" not in text                       # old routine lines dropped
    assert len(h.episodes.load(h.last_episode)) > 2 * n      # full history kept locally


def test_chain_baseline_still_uses_previous_response_id(prims, tmp_path):
    h, client = harness(prims, tmp_path, [NS(id="r1", output=[call("get_state", {}, "c1")], output_text=""),
                                          NS(id="r2", output=[], output_text="ok")])
    h.run("x")
    assert client.kw[1]["previous_response_id"] == "r1" and client.kw[1]["input"][0]["call_id"] == "c1"


def test_digest_keeps_milestones_longer_than_routine_lines():
    wm = WorkingMemory("t", ContextPolicy(summary_max_chars=300))
    wm.record_turn(0, "", [], [], "", ["#0 close_gripper{} -> holding=True [grasp_caught]"])
    for i in range(1, 20):
        wm.record_turn(i, "", [], [], "", [f"#{i} get_state{{}} -> tip=1,2,3 rz=0"])
    lines = [line for _, line in wm.digest]
    assert lines[0].startswith("#0 close_gripper") and sum(len(x) + 1 for x in lines) <= 300


def test_state_invariants_come_from_harness_not_model():
    wm = WorkingMemory("t")
    wm.progress.append("green placed at (-230, -440) — skill verified")
    wm.record_turn(1, "", [("move_relative", {"dz": -5}, {"ok": False, "error": "CONTACT at Z=62"})], [], "", [])
    text = wm.build({"pose": [1, 2, 3, 180, 0, 0], "approval": {"execution_mode": "live"}})
    assert '"verified_progress"' in text and '"unresolved_error": "CONTACT at Z=62"' in text
    assert '"execution_mode": "live"' in text


# ---------------------------------------------------------------- Phase 4: chunking
def C(name, **args):
    return NS(name=name, parsed_args=args)


def test_free_space_allows_four_motions():
    run, deferred, _ = ChunkPolicy().split([C("move_to_pose", x=0, y=0, z=300), C("look", camera="env"),
                                            C("move_relative", dz=-20), C("move_relative", dx=10),
                                            C("move_relative", dy=10), C("move_relative", dy=10)],
                                           tip_z=300, zone_top=162)
    assert len(run) == 5 and len(deferred) == 1


def test_descending_in_approach_zone_is_one_step():
    run, deferred, why = ChunkPolicy().split([C("move_relative", dz=-19), C("move_relative", dz=-19)],
                                             tip_z=120, zone_top=162)
    assert len(run) == 1 and len(deferred) == 1 and "L=1" in why


def test_alignment_in_zone_two_steps_and_entering_zone_counts():
    run, deferred, _ = ChunkPolicy().split([C("move_relative", dx=5, frame="tool"), C("move_relative", dy=5, frame="tool"),
                                            C("move_relative", dy=5, frame="tool")], tip_z=99, zone_top=162)
    assert len(run) == 2 and len(deferred) == 1
    run, _, _ = ChunkPolicy().split([C("move_relative", dz=-10), C("move_relative", dz=-10)], tip_z=170, zone_top=162)
    assert len(run) == 1                                   # the move ends inside the zone, going down


def test_no_motion_after_gripper_boundary_but_reads_allowed():
    run, deferred, why = ChunkPolicy().split([C("close_gripper"), C("look", camera="wrist"), C("move_relative", dz=15)],
                                             tip_z=300, zone_top=162)
    assert [c.name for c in run] == ["close_gripper", "look"] and "boundary" in why


def test_chunk_aborts_after_error(prims, tmp_path):
    bad = call("move_relative", {"dx": 500}, "a")           # > max_relative_mm: validation error
    script = [NS(id="r1", output=[bad, call("get_state", {}, "b"), call("jog_joint", {"joint": 1, "degrees": 5}, "c")],
                 output_text=""), NS(id="r2", output=[], output_text="ok")]
    h, client = harness(prims, tmp_path, script, variant="V2",
                        state_fn=lambda: {"pose": [300, 0, 400, 180, 0, 0], "approach_zone_top": 162})
    h.run("x")
    outs = {o["call_id"]: json.loads(o["output"]) for o in client.kw[1]["input"] if o.get("call_id")}
    assert not outs["a"]["ok"] and outs["b"]["error"].startswith("not executed")
    assert outs["c"]["executed"] is False
    assert h.last_metrics["actions_deferred"] == 2


# ---------------------------------------------------------------- Phase 3: selective vision
def _jpeg(v):
    import base64

    import cv2
    import numpy as np
    return base64.b64encode(cv2.imencode(".jpg", np.full((40, 40, 3), v, np.uint8))[1]).decode()


def test_image_policy_first_then_only_when_useful():
    pol = ImagePolicy()
    res = {"camera": "wrist", "objects": [{"label": "green"}], "image_jpeg_b64": _jpeg(100)}
    assert pol.decide({"colors": ["green"]}, res, False)[0]
    pol.sent("wrist", res["image_jpeg_b64"])
    res2 = {**res, "image_jpeg_b64": _jpeg(180)}
    assert not pol.decide({"colors": ["green"]}, res2, False)[0]               # clear detection: data only
    assert pol.decide({"colors": ["green"], "image": "always"}, res2, False)[0]
    assert pol.decide({"colors": ["green"]}, res2, True)[0]                    # after a failed grasp
    assert not pol.decide({"colors": ["green"]}, {**res}, True)[0]             # identical to the one sent
    assert pol.decide({"colors": ["blue"]}, res2, False)[0]                    # requested colour missing
    assert ImagePolicy(mode="all").decide({}, res, False)[0]


# ---------------------------------------------------------------- Phase 6: events
def test_router_classifies_and_requests_images():
    r = EventRouter()
    ev = r.classify("close_gripper", {}, {"ok": True, "result": {"holding": True, "position": 870}})
    assert ev[0].type is EventType.GRASP_MISSED and r.wants_image("wrist")
    ev = r.classify("move_relative", {}, {"ok": False, "error": "CONTACT at fingertip Z=62.8"})
    assert ev[0].type is EventType.CONTACT
    ev = r.classify("get_state", {}, {"ok": True, "result": {"error": True}})
    assert ev[0].type is EventType.ROBOT_ALARM and r.stop_reason


def test_scene_change_only_when_robot_did_not_handle_blocks():
    r = EventRouter()
    look = lambda x: {"ok": True, "result": {"camera": "env", "objects": [{"label": "blue", "base_xyz_mm": [x, 0, 70]}]}}  # noqa: E731
    r.classify("look", {"camera": "env"}, look(0))
    assert r.classify("look", {"camera": "env"}, look(60))[0].type is EventType.SCENE_CHANGED
    r.classify("close_gripper", {}, {"ok": True, "result": {"holding": True, "position": 776}})
    assert r.classify("look", {"camera": "env"}, look(200))[0].type is EventType.ACTION_OK


# ---------------------------------------------------------------- Phase 5/7: skills
@pytest.fixture
def runner(prims, tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    r = SkillRunner(lib, require_validation=True)
    ex = ActionExecutor(build_registry(prims, runner=r), mode="live")
    r.executor = ex
    lib.save(Skill("up", "lift", steps=[{"tool": "move_relative", "args": {"dz": "{dz}"}}], params={"dz": 10}))
    return r


def test_candidate_skill_needs_operator(runner, sim):
    out = runner.executor.execute("run_skill", {"name": "up"})
    assert not out["ok"] and "operator approval" in out["error"] and sim.motion_log == []
    runner.supervisor = lambda name, params: True
    out = runner.executor.execute("run_skill", {"name": "up"})
    assert out["ok"] and out["result"]["ok"]
    ev = runner.library.get("up").evidence
    assert ev[-1]["supervised"] and ev[-1]["ok"]


def test_promotion_needs_supervised_successes_and_rollback(runner):
    runner.supervisor = lambda name, params: True
    lib = runner.library
    runner.run("up")
    assert not lib.promote("up", 2)["promoted"]
    runner.run("up")
    assert lib.promote("up", 2)["promoted"] and lib.get("up").validation_status == "validated"
    runner.supervisor = None
    assert runner.run("up")["ok"]                          # validated: no approval needed
    runner.save("up", "lift more", [{"tool": "move_relative", "args": {"dz": 20}}])
    assert lib.get("up").validation_status == "candidate"  # a new version starts untrusted
    out = lib.rollback("up", 1)
    assert out["rolled_back"] and lib.get("up").validation_status == "validated" and lib.get("up").version == 3


def test_saved_skills_cannot_touch_memory_or_go_down_in_recovery(runner):
    from common.errors import ValidationError
    with pytest.raises(ValidationError, match="not allowed"):
        runner.save("x", "d", [{"tool": "write_note", "args": {"text": "hi"}}])
    with pytest.raises(ValidationError, match="unknown check"):
        runner.save("x", "d", [{"check": "disable_safety"}])
    with pytest.raises(ValidationError, match="recovery"):
        runner.save("x", "d", [{"tool": "get_state"}], recovery=[{"tool": "move_relative", "args": {"dz": -30}}])


def test_failure_runs_recovery_and_reports_code(runner):
    runner.supervisor = lambda name, params: True
    runner.local = LocalSteps(runner.executor.specs["get_state"].handler.__self__, scene=None, executor=runner.executor)
    runner.library.save(Skill("grab", "g", steps=[{"tool": "close_gripper"}, {"check": "holding"}],
                              recovery=[{"tool": "open_gripper"}],
                              failure_conditions={"grasp_missed": "turn rz"}))
    out = runner.run("grab")
    assert not out["ok"] and out["failure"] == "grasp_missed" and out["meaning"] == "turn rz"
    assert out["recovery"] and out["recovery"][0].startswith("-1:tool:open_gripper ok")


# ---------------------------------------------------------------- local routines
def test_descend_routine_steps_and_lift(prims, sim):
    ex = ActionExecutor(build_registry(prims), mode="live")
    local = LocalSteps(prims, None, ex)
    sim.state["pose"] = [300.0, 0.0, 200.0, 180.0, 0.0, 0.0]
    out = local.routine("descend_to", {}, z=150, step_mm=19)
    assert out["ok"] and out["moves"] == 3 and out["z"] == pytest.approx(150)
    assert local.routine("lift_to", {}, z=200, step_mm=25)["moves"] == 2


class WristCam:
    """Wrist detections follow tool-frame moves with the 181222 px/mm mapping."""

    def __init__(self, u, v, sign=1.0):
        self.u, self.v, self.sign = u, v, sign

    def find(self, color, camera="env", frame=None):
        return [NS(summary=lambda: {"label": color, "pixel": [self.u, self.v],
                                    "bbox": [self.u - 50, self.v - 100, 100, 100], "area_px": 9000.0})]


class MoveRecorder:
    mode = "live"

    def __init__(self, cam):
        self.cam, self.moves = cam, []

    def execute(self, name, args):
        self.moves.append(args)
        self.cam.u += self.cam.sign * 5.6 * args.get("dy", 0)
        self.cam.v += self.cam.sign * -2.6 * args.get("dx", 0)
        return {"ok": True, "result": {}}


def test_wrist_centring_converges():
    cam = WristCam(420, 190)
    local = LocalSteps(None, cam, MoveRecorder(cam), settle_s=0)
    out = local.routine("wrist_center", {}, color="green")
    assert out["ok"] and abs(out["error_px"][0]) <= 12 and abs(out["error_px"][1]) <= 12
    assert all(abs(m["dx"]) <= 15 and abs(m["dy"]) <= 15 and m["frame"] == "tool" for m in local.executor.moves)


def test_wrist_centring_gives_up_when_mapping_is_wrong():
    cam = WristCam(420, 190, sign=-1.0)                    # image moves the other way
    out = LocalSteps(None, cam, MoveRecorder(cam), settle_s=0).routine("wrist_center", {}, color="green")
    assert not out["ok"] and ("converge" in out["detail"] or "would move" in out["detail"])


def test_builtin_pick_place_skills_validate_in_dry_run(prims, tmp_path):
    from agent.explore.pick_place_skills import ensure_pick_place_skills
    lib = SkillLibrary(tmp_path / "s.json")
    assert ensure_pick_place_skills(lib) == ["pick_block_wrist", "place_block_on_table"]
    assert ensure_pick_place_skills(lib) == []              # never overwrites
    r = SkillRunner(lib, constants={"table_z": 42.0}, require_validation=True)
    ex = ActionExecutor(build_registry(prims, runner=r), mode="dry_run")
    r.executor = ex
    r.local = LocalSteps(prims, WristCam(355, 140), ex, settle_s=0)
    pick = r.run("pick_block_wrist", {"color": "green", "x": -330, "y": -440, "z_top": 69.3, "rz": 70.9})
    assert pick["ok"], pick
    place = r.run("place_block_on_table", {"color": "green", "x": -230, "y": -440, "rz": 70.9})
    assert place["ok"], place


# ---------------------------------------------------------------- Phase 8: variants
def test_variants_are_reproducible_and_v3_is_cheapest(tmp_path):
    from mock_world import run_variant
    res = {v: run_variant(v, tmp_path / v) for v in ("V0", "V1", "V2", "V3")}
    again = run_variant("V3", tmp_path / "V3b")
    for r in (*res.values(), again):
        r.pop("report")
    assert {k: v for k, v in again.items()} == res["V3"]
    assert all(r["blocks_on_table"] == ["blue", "green", "yellow"] for r in res.values())
    assert all(r["motions_sent"] == res["V0"]["motions_sent"] for r in res.values())   # same robot actions
    assert res["V1"]["est_total_tokens"] < res["V0"]["est_total_tokens"] / 4
    assert res["V2"]["model_calls"] < res["V0"]["model_calls"]
    assert res["V3"]["est_total_tokens"] == min(r["est_total_tokens"] for r in res.values())


def test_skill_plan_moves_all_three_blocks_with_few_calls(tmp_path):
    from mock_world import run_variant
    r = run_variant("V3", tmp_path, plan="skills")
    assert r["blocks_on_table"] == ["blue", "green", "yellow"] and r["model_calls"] <= 12


def test_options_from_explore_yaml_dict():
    cfg = {"variant": "V3", "recent_turns": 3, "max_horizon": 9, "max_episode_tokens": 1000}
    o = HarnessOptions.from_variant("V2", **{k: v for k, v in cfg.items() if k != "variant"})
    assert o.variant == "V2" and o.context.mode == "chain" and o.chunk.enabled and o.chunk.max_horizon == 4
    assert o.max_episode_tokens == 1000
    with pytest.raises(ValueError):
        HarnessOptions.from_variant("V9")


# ---------------------------------------------------------------- fixes after episode 20261113-164841
def test_builtin_skill_upgrades_but_astra_versions_are_kept(tmp_path):
    import copy

    from agent.explore.pick_place_skills import PICK, ensure_pick_place_skills
    lib = SkillLibrary(tmp_path / "s.json")
    old = copy.deepcopy(PICK)
    old.recovery = [{"tool": "open_gripper"}]
    lib.save(old)
    assert "pick_block_wrist" in ensure_pick_place_skills(lib)
    s = lib.get("pick_block_wrist")
    assert s.version == 2 and s.recovery == PICK.recovery and s.history[0]["recovery"] == [{"tool": "open_gripper"}]
    lib.revise(Skill("pick_block_wrist", "Astra's own", steps=[{"tool": "get_state"}], tags=["pick"]))
    assert ensure_pick_place_skills(lib) == [] and lib.get("pick_block_wrist").description == "Astra's own"


def test_pick_has_no_measurement_lock_or_probe():
    from agent.explore.pick_place_skills import BUILTIN, PICK
    assert [s.name for s in BUILTIN] == ["pick_block_wrist", "place_block_on_table"]
    assert PICK.steps[0]["routine"] == "approach"
    routines = [s.get("routine") for s in PICK.steps]
    assert "probe_top" not in routines and "grasp_ladder" in routines
    assert not any(c["check"] == "contact_calibrated" for c in PICK.preconditions)


def test_z_top_must_match_the_taught_cube_top():
    local = LocalSteps(None, None, None, block_top_z=69.3, settle_s=0)
    assert local.check("z_top_plausible", {}, z_top=72)["ok"]
    out = local.check("z_top_plausible", {}, z_top=87.6)
    assert not out["ok"] and "+18.3" in out["detail"]
    assert LocalSteps(None, None, None, settle_s=0).check("z_top_plausible", {}, z_top=87.6)["ok"]


def test_wrist_reads_a_fresh_frame_before_saying_not_seen():
    class Laggy:
        n = 0

        def find(self, color, camera="env", frame=None):
            self.n += 1
            return [] if self.n < 3 else WristCam(355, 140).find(color)
    local = LocalSteps(None, Laggy(), None, settle_s=0, frame_retries=3)
    assert local.check("wrist_sees", {}, color="green")["ok"]


def test_small_specks_are_not_ambiguous_duplicates():
    from agent.vision import _ambiguous_duplicates
    assert not _ambiguous_duplicates([{"label": "blue", "area_px": 232}, {"label": "blue", "area_px": 52},
                                      {"label": "white", "area_px": 600}, {"label": "white", "area_px": 80}])
    assert _ambiguous_duplicates([{"label": "green", "area_px": 9000}, {"label": "green", "area_px": 5000}])


def test_list_skills_is_compact_unless_one_is_asked(runner):
    rows = runner.list()
    assert "steps" not in rows[0] and rows[0]["name"] == "up"
    assert runner.list("up")[0]["steps"] == [{"tool": "move_relative", "args": {"dz": "{dz}"}}]


def test_harness_attaches_image_itself_after_a_failure(tmp_path):
    """No extra model turn just to look after a failed grasp/skill."""
    from mock_world import build
    h, client, sim = build("V3", tmp_path, plan=[("get_state", {}), ("close_gripper", {})], object_at=None)
    h.run("x")
    log = h.episodes.load(h.last_episode)
    auto = [e for e in log if e.get("event") == "tool" and e.get("auto")]
    assert auto and auto[0]["args"]["camera"] == "wrist"
    assert h.last_metrics["auto_looks"] == 1


def test_wrist_search_finds_a_cube_beyond_the_image_top():
    class Cam:
        """Cube visible only after the gripper moved ~30 mm towards it (tool -dx)."""
        def __init__(self):
            self.dx = 0.0

        def find(self, color, camera="env", frame=None):
            return WristCam(355, 140).find(color) if self.dx <= -25 else []

    cam = Cam()

    class Ex:
        mode = "live"

        def execute(self, name, args):
            cam.dx += args.get("dx", 0)
            return {"ok": True, "result": {}}
    out = LocalSteps(None, cam, Ex(), settle_s=0, frame_retries=1).routine("wrist_search", {}, color="green")
    assert out["ok"] and out["offset"] == [-30, 0] and out["moves"] == 2 and cam.dx == -30


def test_wrist_search_returns_to_start_when_not_found():
    moves = []

    class Ex:
        mode = "live"

        def execute(self, name, args):
            moves.append((args.get("dx", 0), args.get("dy", 0)))
            return {"ok": True, "result": {}}

    class Empty:
        def find(self, *a, **k):
            return []
    out = LocalSteps(None, Empty(), Ex(), settle_s=0, frame_retries=1).routine("wrist_search", {}, color="green")
    assert not out["ok"] and out["ambiguous"]
    assert abs(sum(m[0] for m in moves)) < 0.1 and abs(sum(m[1] for m in moves)) < 0.1
    assert all(math.hypot(dx, dy) <= 20.01 for dx, dy in moves)


def test_probe_top_measures_the_cube_by_contact():
    calls = []

    class Ex:
        mode = "live"

        def execute(self, name, args):
            calls.append((name, args))
            if name == "move_relative" and len([c for c in calls if c[0] == "move_relative"]) == 3:
                return {"ok": False, "error": "move_relative: CONTACT at fingertip Z=71.4 (X=-1, Y=-2) — something"}
            return {"ok": True, "result": {}}

    class P:
        def get_state(self):
            return {"pose": [0, 0, 99.0, 180, 0, 0]}
    memo = {}
    out = LocalSteps(P(), None, Ex(), settle_s=0).routine("probe_top", memo, z_est=79)
    assert out["ok"] and out["measured_top"] == 71.4 and memo["vars"]["measured_top"] == 71.4
    assert calls[0] == ("set_gripper", {"position": 500})
    assert all(abs(a["dz"]) <= 8 for n, a in calls if n == "move_relative")


def test_probe_without_contact_fails():
    class Ex:
        mode = "live"

        def execute(self, name, args):
            return {"ok": True, "result": {}}

    class P:
        def get_state(self):
            return {"pose": [0, 0, 99.0, 180, 0, 0]}
    out = LocalSteps(P(), None, Ex(), settle_s=0).routine("probe_top", {}, z_est=79, below_mm=25)
    assert not out["ok"] and "no contact" in out["detail"]


def test_measured_top_drives_the_grasp_height(prims, tmp_path):
    """Later steps use the measured top, not the camera estimate."""
    lib = SkillLibrary(tmp_path / "s.json")
    r = SkillRunner(lib)
    ex = ActionExecutor(build_registry(prims, runner=r), mode="live")
    r.executor = ex

    class Probe(LocalSteps):
        def routine_probe_top(self, memo, z_est, **kw):
            memo.setdefault("vars", {})["measured_top"] = 300.0
            return {"ok": True, "measured_top": 300.0}
    r.local = Probe(prims, None, ex, settle_s=0)
    lib.save(Skill("t", "t", params={"z_top": None},
                   steps=[{"routine": "probe_top", "args": {"z_est": "{z_top}"}},
                          {"tool": "move_to_pose", "args": {"x": 300, "y": 0, "z": "={measured_top} - 8",
                                                            "rx": 180, "ry": 0, "rz": 0}}]))
    out = r.run("t", {"z_top": 380})
    assert out["ok"] and out["last_result"]["pose"][2] == pytest.approx(292)


# ---------------------------------------------------------------- autonomy guards
def test_wrist_centring_never_wanders_far():
    cam = WristCam(355 + 5.6 * 60, 140)                   # 60 mm away: further than any real centring needs
    local = LocalSteps(None, cam, MoveRecorder(cam), settle_s=0)
    out = local.routine("wrist_center", {}, color="green", max_iter=10, max_travel_mm=500)   # capped at 50
    moved = (sum(m["dx"] for m in local.executor.moves), sum(m["dy"] for m in local.executor.moves))
    assert not out["ok"] and "would move" in out["detail"]
    assert (moved[0] ** 2 + moved[1] ** 2) ** 0.5 <= 50


def test_z_top_below_table_is_refused():
    local = LocalSteps(NS(table_z_mm=42.0), None, None, settle_s=0)
    assert not local.check("z_top_plausible", {}, z_top=30.6)["ok"]
    assert local.check("z_top_plausible", {}, z_top=79)["ok"]


class ReachLimit:
    """move_relative executor: z above 160 is out of reach unless the tip moved towards the base."""
    mode = "live"

    def __init__(self):
        self.pose = [-358.0, -502.0, 150.0]

    def execute(self, name, args):
        x, y, z = self.pose
        nx, ny, nz = x + args.get("dx", 0), y + args.get("dy", 0), z + args.get("dz", 0)
        if nz > 160 and (nx ** 2 + ny ** 2) ** 0.5 > 600:
            return {"ok": False, "error": "pose not reachable (no inverse-kinematics solution)"}
        self.pose = [nx, ny, nz]
        return {"ok": True, "result": {"pose": self.pose + [180, 0, 0]}}


def test_lift_detours_towards_the_base_when_out_of_reach():
    ex = ReachLimit()
    local = LocalSteps(NS(get_state=lambda: {"pose": ex.pose + [180, 0, 0]}), None, ex, settle_s=0)
    out = local.routine("lift_to", {}, z=180, step_mm=23)
    assert out["ok"] and out["z"] == pytest.approx(180) and out["detours"] >= 1
    assert (ex.pose[0] ** 2 + ex.pose[1] ** 2) ** 0.5 <= 600


def test_autonomous_runner_runs_candidates_and_promotes(prims, tmp_path):
    from agent.explore.pick_place_skills import ensure_pick_place_skills
    lib = SkillLibrary(tmp_path / "s.json")
    ensure_pick_place_skills(lib)
    r = SkillRunner(lib, constants={"table_z": 42.0}, require_validation=False)
    r.auto_promote_after = 2
    ex = ActionExecutor(build_registry(prims, runner=r), mode="live")
    r.executor = ex
    lib.add_evidence("place_block_on_table", True, False)
    lib.add_evidence("place_block_on_table", True, False)
    assert lib.auto_promote("place_block_on_table", 2)["promoted"]
    assert lib.get("place_block_on_table").validation_status == "validated"
    lib.add_evidence("pick_block_wrist", True, False)
    lib.add_evidence("pick_block_wrist", False, False)
    assert lib.auto_promote("pick_block_wrist", 2) is None          # a failure among the last runs
    saved = r.save("my_pick", "draft", [{"tool": "open_gripper"}])
    assert "you may run it now" in saved["note"]
    assert lib.retire("my_pick", "never worked")["retired"] == "my_pick"
    assert lib.get("my_pick").validation_status == "deprecated"
    with pytest.raises(ValueError, match="built-in"):
        lib.retire("pick_block_wrist")


def test_prompt_carries_the_skill_index_not_descriptions(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    lib.save(Skill(name="pick_x", description="SECRET RAW TEXT from an old episode", steps=[], params={"x": None}))
    lib.save(Skill(name="old_one", description="old", steps=[], params={}))
    lib.retire("old_one", "never run")
    text = AgentHarness(None, None, AgentConfig(), skills=lib)._instructions("pick a block")
    assert "pick_x v1 [candidate] 0 ok / 0 failed · params x" in text
    assert "SECRET RAW TEXT" not in text and "old_one" not in text
