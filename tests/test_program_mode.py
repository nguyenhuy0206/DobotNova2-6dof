"""Program mode (23 Nov): Astra acts only through programs; each run carries a trace; the next attempt starts
from a report built from the outcome (verifier, else the operator's label) and that trace."""
import json

from actions import ActionExecutor, build_registry
from actions.astra_code import AstraCode
from actions.local_steps import LocalSteps


def cube_session():
    """The 23 Nov cube session (D455 + task tools + routines), whatever task explore.yaml holds now."""
    from agent.explore.config import load_explore_config
    from agent.explore.prompt import DEFAULT_SESSION_TOOLS
    cfg = load_explore_config()
    cfg.program_mode, cfg.program_tools, cfg.cameras = True, list(DEFAULT_SESSION_TOOLS), ["env", "wrist"]
    cfg.operator_rules = ['Use the wrist camera as your main view: look(camera="wrist") about 5 times.']
    return cfg

PROGRAM = '''"""Down 5 mm, then a gripper call that the executor refuses."""
KIND = "routine"
def run(ctx, memo, dz=5):
    a = ctx.move(dz=-float(dz), speed=1)
    b = ctx.execute("set_gripper", {"position": 100})
    return {"ok": a["ok"] and b["ok"], "detail": "done"}
'''


def test_a_program_run_returns_its_trace(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("down_then_grip", PROGRAM)
    ex = ActionExecutor(build_registry(prims), mode="dry_run")
    out = LocalSteps(prims, None, ex, settle_s=0, code=code).routine("down_then_grip", {})
    assert out["routine"] == "down_then_grip" and out["version"] == 1
    assert [t["tool"] for t in out["trace"]] == ["move_relative", "set_gripper"]
    assert out["trace"][0]["ok"] and out["trace"][0]["args"]["dz"] == -5.0


def test_a_crashing_program_still_returns_the_steps_it_took(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("crash_after_move", 'KIND = "routine"\ndef run(ctx, memo):\n    ctx.move(dz=-3, speed=1)\n'
                                   '    return {"ok": 1 / 0}\n')
    ex = ActionExecutor(build_registry(prims), mode="dry_run")
    out = LocalSteps(prims, None, ex, settle_s=0, code=code).routine("crash_after_move", {})
    assert not out["ok"] and out["code_error"] and [t["tool"] for t in out["trace"]] == ["move_relative"]


def test_colour_threshold_helpers_are_not_in_astras_api():
    """Operator, 23 Nov: pictures + depth_at only."""
    from actions.astra_code import RoutineContext
    assert not hasattr(RoutineContext, "wrist") and not hasattr(RoutineContext, "env")
    assert hasattr(RoutineContext, "depth_at")


def test_program_mode_tool_set_has_no_direct_motion():
    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    names = set(s.executor.specs)
    assert s.cfg.program_mode
    assert {"write_routine", "run_routine", "look", "depth_at"} <= names
    assert not {"move_to_pose", "move_relative", "close_gripper", "open_gripper", "run_skill"} & names
    assert "THIS SESSION" in s.harness._instructions("x")


class Ledger:
    def __init__(self, status):
        self.status = status

    def outcome(self, episode):
        return {"status": self.status}


def test_outcome_prefers_the_verifier_then_the_operator_label(tmp_path):
    from agent.explore.attempts import labels, outcome, record_label
    f = tmp_path / "labels.jsonl"
    record_label("ep1", "failure", "closed above the cube", path=f)
    rows = labels(f)
    assert outcome("ep1", Ledger("passed"), rows) == {"status": "passed", "source": "verifier", "detail": ""}
    o = outcome("ep1", Ledger("inconclusive"), rows)
    assert o["status"] == "failed" and o["source"] == "operator" and "above" in o["detail"]
    assert outcome("ep2", Ledger("inconclusive"), rows)["source"] == "none"


def test_report_names_the_program_outcome_and_first_failed_step(tmp_path):
    from agent.explore.attempts import report
    from memory.store import EpisodeStore
    st = EpisodeStore(tmp_path / "ep")
    ep = st.start("task", {"model": "gpt-6-astra", "driver": "astra"})
    trace = [{"i": 0, "tool": "move_relative", "ok": True}, {"i": 1, "tool": "close_gripper", "ok": False,
                                                              "error": "gripper offline"}]
    st.append(ep, {"event": "tool", "step": 1, "name": "run_routine", "args": {"name": "pick_blue", "args": {"u": 1}},
                   "result": {"ok": True, "result": {"ok": False, "detail": "no grip", "routine": "pick_blue",
                                                     "version": 2, "trace": trace}}})
    text = report(st, Ledger("inconclusive"), ep, {ep: {"label": "failure", "note": "missed"}})
    assert "outcome FAILED (source: operator) — missed" in text
    assert "pick_blue v2" in text and "FAILED after 2 steps" in text and "gripper offline" in text
    assert json.dumps({"i": 1, "tool": "close_gripper", "ok": False, "error": "gripper offline"}) in text


def test_programs_reach_motion_tools_the_model_is_not_offered(prims, tmp_path):
    """23 Nov 133140: program mode filtered the executor itself -> every program failed 'Unknown action open_gripper'."""
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("open_then_down", 'KIND = "routine"\ndef run(ctx, memo):\n    a = ctx.execute("open_gripper", {})\n'
                                 '    b = ctx.move(dz=-3, speed=1)\n    return {"ok": a["ok"] and b["ok"]}\n')
    ex = ActionExecutor(build_registry(prims), mode="dry_run", allowed=["look", "run_routine"])
    assert "open_gripper" not in ex.specs                         # the model is not offered it ...
    assert not ex.execute("open_gripper", {})["ok"]               # ... nor may it call it
    out = LocalSteps(prims, None, ex, settle_s=0, code=code).routine("open_then_down", {})
    assert out["ok"], out                                         # ... but a program may
    assert [t["tool"] for t in out["trace"]] == ["open_gripper", "move_relative"]


def test_unknown_ctx_attributes_are_refused_when_written(tmp_path):
    """23 Nov: three episodes lost to ctx.env (removed) failing only when run."""
    import pytest
    from common.errors import ValidationError
    code = AstraCode(tmp_path, core=LocalSteps)
    with pytest.raises(ValidationError, match="ctx has no 'env'"):
        code.write("finder", 'KIND = "routine"\ndef run(ctx, memo):\n    return {"ok": bool(ctx.env("blue"))}\n')


# ---------------------------------------------------------------- 23 Nov audit: what Astra is told is true
def test_after_a_safety_stop_no_motion_is_sent_and_the_harness_sees_it():
    from actions.executor import ActionSpec
    from common.errors import ContactError
    sent = []

    def blocked(**_):
        raise ContactError("move_relative: CONTACT at fingertip Z=12.0 — robot PAUSED")

    ex = ActionExecutor([ActionSpec("move_relative", "", blocked, moves_robot=True),
                         ActionSpec("open_gripper", "", lambda: sent.append(1), moves_robot=True),
                         ActionSpec("get_state", "", lambda: {"pose": []}, read_only=True)], mode="live")
    r = ex.execute("move_relative", {}, internal=True)
    assert not r["ok"] and r["executed"] and ex.stopped
    again = ex.execute("open_gripper", {}, internal=True)
    assert not again["ok"] and again["error_type"] == "MotionStopped" and not sent
    assert ex.execute("get_state", {})["ok"]                      # reading still works


def test_a_routine_returning_ok_false_is_a_failure_event():
    from agent.events import EventRouter, EventType, aborts_chunk
    res = {"ok": True, "result": {"ok": False, "detail": "missed", "trace": [
        {"i": 0, "tool": "close_gripper", "ok": True, "holding": False}]}}
    events = EventRouter().classify("run_routine", {"name": "pick"}, res)
    assert events[0].type is EventType.SKILL_FAILED and aborts_chunk(events)
    contact = {"ok": True, "result": {"ok": False, "trace": [
        {"i": 0, "tool": "move_relative", "ok": False, "error": "CONTACT at fingertip Z=10"}]}}
    assert EventRouter().classify("run_routine", {"name": "pick"}, contact)[0].type is EventType.CONTACT


def test_a_long_trace_keeps_its_failed_entry_and_says_what_was_left_out():
    from agent.context import compact_result
    trace = [{"i": i, "tool": "move_relative", "ok": True, "args": {"dz": -1}, "pose": [1.0, 2.0, 3.0]}
             for i in range(100)] + [{"i": 100, "tool": "move_relative", "ok": False, "error": "CONTACT at Z=4"}]
    text = compact_result({"ok": True, "observation": "O9", "result": {"ok": False, "trace": trace}}, 2000)
    assert "CONTACT at Z=4" in text and "of 101 entries shown" in text and '"observation": "O9"' in text


def test_depth_at_pixels_are_full_image_and_no_colour_detector(prims, tmp_path):
    code = AstraCode(tmp_path, core=LocalSteps)
    code.write("detect", 'KIND = "routine"\ndef run(ctx, memo):\n'
                         '    return ctx.execute("find_objects", {"color": "blue"})\n')
    ex = ActionExecutor(build_registry(prims), mode="dry_run")
    out = LocalSteps(prims, None, ex, settle_s=0, code=code).routine("detect", {})
    assert not out["ok"] and "not available to routines" in out["detail"]
    import inspect
    from actions.astra_code import RoutineContext
    assert "depth_at" in inspect.getsource(RoutineContext.depth_at)
    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    assert "full_image" not in s.executor.specs["depth_at"].parameters["properties"]
    assert "FULL-image pixels" in s.executor.specs["depth_at"].description


def test_review_gets_the_tools_its_instructions_name():
    from agent.explore import build_explore
    from agent.explore.learning import REFLECT_TOOLS_PROGRAM, OutcomeRecorder, reflect_executor
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    rev = reflect_executor(s.executor, OutcomeRecorder(s.harness.episodes), REFLECT_TOOLS_PROGRAM)
    assert {"write_routine", "read_routine", "propose_hypothesis", "record_episode_outcome"} <= set(rev.specs)
    assert "save_skill" not in rev.specs


def test_program_mode_prompt_names_only_offered_tools():
    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    text = s.harness._instructions("x")
    for absent in ("list_skills", "run_skill", "save_skill", "diagnose", "conclude", "(list_concepts)", "89 mm"):
        assert absent not in text, absent
    desc = s.executor.all_specs["write_routine"].description
    assert "move_to_pose(x: number" in desc and "find_objects" not in desc


def test_many_failures_still_keep_the_last_one_and_the_count():
    from agent.context import compact_result
    trace = [{"i": i, "tool": "move_relative", "ok": False, "error": f"refused {i} " + "x" * 150} for i in range(30)]
    trace[-1]["error"] = "CONTACT at fingertip Z=5"
    text = compact_result({"ok": True, "result": {"ok": False, "trace": trace}}, 1500)
    assert "CONTACT at fingertip Z=5" in text and "of 30 entries shown" in text


def test_a_contact_inside_a_nested_routine_is_seen():
    from agent.events import EventRouter, EventType
    res = {"ok": True, "result": {"ok": True, "trace": [{"i": 0, "tool": "routine:probe", "ok": True, "trace": [
        {"i": 0, "tool": "move_relative", "ok": False, "error": "CONTACT at fingertip Z=9"}]}]}}
    assert EventRouter().classify("run_routine", {"name": "outer"}, res)[0].type is EventType.CONTACT


def test_program_mode_offers_no_colour_detection_through_core_steps(prims, tmp_path):
    """Operator, 23 Nov: raw pictures only."""
    from actions.local_steps import COLOUR_STEPS
    assert {"routine:wrist_search", "routine:wrist_center", "check:placed"} <= COLOUR_STEPS
    code = AstraCode(tmp_path, core=LocalSteps)
    code.hide_colour_steps = True
    code.write("uses_core", 'KIND = "routine"\ndef run(ctx, memo):\n'
                            '    return ctx.routine("wrist_search", memo, color="blue")\n')
    ex = ActionExecutor(build_registry(prims), mode="live")
    local = LocalSteps(prims, None, ex, settle_s=0, code=code)
    local.colour_detection = False
    out = local.routine("uses_core", {})
    assert not out["ok"] and "colour detection is not available" in out["detail"]
    listed = code.list()["core"]
    assert "routine:wrist_search" not in listed and "routine:lift_to" in listed
    import pytest
    with pytest.raises(Exception):
        code.read("wrist_search")


def test_astra_gets_raw_pictures():
    """Operator, 23 Nov: no marks on wrist pictures, depth_at returns numbers only."""
    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    assert s.kit.scene.wrist_overlay is None
    text = s.harness._instructions("x")
    assert "fingertips_px" not in text and "nothing is detected or drawn" in text


def test_astra_sees_only_the_current_calibration(tmp_path):
    from memory.knowledge import KnowledgeStore, Observations, calibration_since
    cal = tmp_path / "cal.json"
    cal.write_text('{"created": "20261123-132353"}')
    since = calibration_since(cal)
    assert since == "2026-11-23 13:23:53"
    obs = Observations(tmp_path / "observations.jsonl")
    obs.add({"action": "look", "x": 1}, "e-old", t="2026-11-20 10:00:00")
    obs.add({"action": "look", "x": 2}, "e-new", t="2026-11-23 14:00:00")
    obs.since = since
    k = KnowledgeStore(tmp_path / "concepts.json", observations=obs)
    rows = k.query({})["rows"]
    assert [r["x"] for r in rows] == [2] and len(obs.items) == 2


def test_prompt_has_no_old_calibration():
    from agent.explore.prompt import instructions
    text = instructions(True)
    assert "118" not in text and "41.96" not in text and "tool118" not in text


def test_operator_rules_reach_the_prompt():
    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), config=cube_session())
    text = s.harness._instructions("x")
    assert "OPERATOR RULES" in text and 'look(camera="wrist")' in text


def test_wrist_only_threading_session_prompt_names_only_what_exists():
    """23 Nov: threading task — wrist camera only, step-by-step motion, no gripper tools."""
    from agent.explore import build_explore
    from agent.explore.config import load_explore_config
    cfg = load_explore_config()
    cfg.program_mode, cfg.cameras = True, ["wrist"]
    cfg.program_tools = ["get_state", "look", "move_relative", "move_to_pose", "insert", "list_routines",
                         "read_routine", "write_routine", "rollback_routine", "run_routine",
                         "query_observations", "read_history"]
    s = build_explore(mode="dry_run", client=object(), config=cfg)
    assert s.executor.specs["look"].parameters["properties"]["camera"]["enum"] == ["wrist"]
    text = s.harness._instructions("x")
    for absent in ("D455", "depth_at", "locate_object", "pick_object", "close_gripper closes", "env-camera"):
        assert absent not in text, absent
    assert "move_relative, move_to_pose, insert" in text and "only camera in this session" in text


def test_threading_session_locks_the_gripper_and_limits_steps():
    """23 Nov: the operator aligned the setup and closed the gripper on the object: nudges only."""
    from agent.explore import build_explore
    from agent.explore.config import load_explore_config
    cfg = load_explore_config()
    cfg.program_mode, cfg.cameras, cfg.gripper_locked = True, ["wrist"], True
    cfg.max_step_mm, cfg.max_step_deg = 10, 5
    cfg.locked_tools = ["move_to_pose", "move_to_joints", "jog_joint"]
    cfg.program_tools = ["get_state", "look", "move_relative", "go_to_position", "insert", "run_routine"]
    s = build_explore(mode="dry_run", client=object(), config=cfg)
    for name in ("open_gripper", "close_gripper", "set_gripper", "move_to_pose"):
        r = s.executor.execute(name, {}, internal=True)
        assert not r["ok"] and "not available in this session" in r["error"]
    assert "Max 10.0 mm and 5.0°" in s.executor.specs["move_relative"].description
    assert s.kit.primitives.m.max_relative_mm == 10 and s.kit.primitives.m.max_relative_deg == 5


def test_one_picture_per_motion():
    """24 Nov (operator): after a picture, the next one needs a motion first; auto looks are not gated."""
    from actions.executor import ActionSpec
    ex = ActionExecutor([ActionSpec("look", "", lambda camera="wrist": {"camera": camera}, read_only=True,
                                    parameters={"type": "object", "properties": {"camera": {"type": "string"}}}),
                         ActionSpec("move_relative", "", lambda dx=0: {"pose": [0] * 6}, moves_robot=True,
                                    parameters={"type": "object", "properties": {"dx": {"type": "number"}}})],
                        mode="live")
    ex.look_gate = {"look"}
    assert ex.execute("look", {})["ok"]                       # the first picture is free
    r = ex.execute("look", {"camera": "env"})
    assert not r["ok"] and "1 picture per motion" in r["error"]
    assert ex.execute("look", {}, internal=True)["ok"]         # harness / program looks are not gated
    assert ex.execute("move_relative", {"dx": 1})["ok"]
    assert ex.execute("look", {})["ok"] and not ex.execute("look", {})["ok"]


def test_two_pictures_per_motion():
    from actions.executor import ActionSpec
    ex = ActionExecutor([ActionSpec("look", "", lambda: {}, read_only=True),
                         ActionSpec("move_relative", "", lambda: {"pose": [0] * 6}, moves_robot=True)], mode="live")
    ex.look_gate, ex.look_limit = {"look"}, 2
    assert ex.execute("look")["ok"] and ex.execute("look")["ok"]
    r = ex.execute("look")
    assert not r["ok"] and "2 pictures per motion" in r["error"]
    assert ex.execute("move_relative")["ok"] and ex.execute("look")["ok"]


def test_insert_is_refused_until_both_sideways_axes_were_tried():
    from actions.executor import ActionSpec
    from agent.explore.session import _calibrate_before_insert
    ex = ActionExecutor([ActionSpec("move_relative", "", lambda dx=0, dy=0, frame="base": {"pose": [0] * 6},
                                    moves_robot=True, parameters={"type": "object", "properties": {
                                        "dx": {"type": "number"}, "dy": {"type": "number"},
                                        "frame": {"type": "string"}}}),
                         ActionSpec("insert", "Insert.", lambda depth_mm: {"inserted_mm": depth_mm}, moves_robot=True,
                                    parameters={"type": "object", "properties": {"depth_mm": {"type": "number"}}})],
                        mode="live")
    _calibrate_before_insert(ex)
    assert "sideways calibration" in ex.specs["insert"].description
    r = ex.execute("insert", {"depth_mm": 1})
    assert not r["ok"] and "dx and dy" in r["error"]
    assert ex.execute("move_relative", {"dx": 1})["ok"]               # base frame: does not count
    assert ex.execute("move_relative", {"dx": 1, "frame": "tool"})["ok"]
    r = ex.execute("insert", {"depth_mm": 1})
    assert not r["ok"] and "along dy" in r["error"]
    assert ex.execute("move_relative", {"dy": -1, "frame": "tool"})["ok"]
    assert ex.execute("insert", {"depth_mm": 1})["ok"]
    ex.episode_state = {}                                              # a new attempt calibrates again
    assert not ex.execute("insert", {"depth_mm": 1})["ok"]


def test_go_to_position_only_returns_to_the_start(monkeypatch):
    from agent.explore import build_explore
    from agent.explore.config import load_explore_config
    cfg = load_explore_config()
    cfg.allowed_positions = ["insert_start"]
    s = build_explore(mode="live", client=object(), config=cfg)
    p = s.kit.primitives
    monkeypatch.setattr(p.a.positions, "find_by_name", lambda n: {"joints_deg": [1, 2, 3, 4, 5, 6]})
    monkeypatch.setattr(p, "get_state", lambda: {"joints_deg": [1, 2, 3, 4, 5, 6.1]})
    spec = s.executor.all_specs["go_to_position"]
    import pytest
    from common.errors import ValidationError
    with pytest.raises(ValidationError, match="already at insert_start"):
        spec.handler("insert_start")
    with pytest.raises(ValidationError, match="not available"):
        spec.handler("home")
    assert "Return to the saved start position (insert_start)" in spec.description


def test_previous_attempt_carries_the_final_report_and_self_check():
    from agent.explore.attempts import report

    class Store:
        def load(self, ep):
            return [{"event": "tool", "name": "insert", "args": {"depth_mm": 2}, "result": {"ok": True}},
                    {"event": "model", "text": "dx +1 mm moves the tip 12 px right in env."},
                    {"event": "end", "reason": "done"},
                    {"event": "self_check", "status": "unclear", "evidence": "tip left of the eye",
                     "next_action": "get an oblique view"}]
    text = report(Store(), Ledger("inconclusive"), "e1", {})
    assert "dx +1 mm moves the tip 12 px right" in text and text.count("tip left of the eye") == 1
    assert "oblique" not in text
