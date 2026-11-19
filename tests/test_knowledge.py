"""Knowledge: harness-recorded observations -> concepts -> hypotheses whose evidence is computed."""
import pytest

from common.errors import ValidationError
from memory.knowledge import KnowledgeStore, observe_call

STATE = {"pose": [-340.0, -440.0, 112.9, 180.0, 0.0, 81.9], "gripper": {"position": 1000, "grip_status": "open"}}


def store(tmp_path):
    k = KnowledgeStore(tmp_path / "concepts.json")
    k.records = lambda name: True
    k.episode = "ep1"
    return k


def centre(k, color, rz, target_v, ok=True):
    """One harness-recorded wrist_center routine run."""
    state = {**STATE, "pose": STATE["pose"][:5] + [rz]}
    k.state_fn = lambda: state
    return k.observe("run_routine", {"name": "wrist_center", "args": {"color": color, "target_v": target_v}},
                     {"ok": True, "result": {"ok": ok, "error_px": 4 if ok else 60}})


def test_a_tool_call_becomes_numbers(tmp_path):
    v = observe_call("run_skill", {"name": "pick_block_wrist", "params": {"color": "blue", "rz": 81.9}},
                     {"ok": True, "result": {"ok": False, "failure": "grasp_missed", "detail": "fingers met air",
                                             "trace": ["5:routine:grasp_ladder FAIL grip=500 depth=6"]}}, STATE)
    assert v["skill"] == "pick_block_wrist" and v["color"] == "blue" and v["arg.rz"] == 81.9 and v["rz"] == 81.9
    assert v["ok"] is False and v["failure"] == "grasp_missed" and v["out.grip"] == 500 and v["out.depth"] == 6
    look = observe_call("look", {"camera": "wrist", "colors": ["blue", "green"]},
                        {"ok": True, "result": {"camera": "wrist", "objects": [
                            {"label": "blue", "pixel": [350, 120], "area_px": 900, "angle_deg": 12.0},
                            {"label": "blue", "pixel": [10, 10], "area_px": 40}]}}, STATE)
    assert (look["wrist.blue.u"], look["wrist.blue.v"], look["wrist.blue.angle"]) == (350, 120, 12.0)
    assert look["wrist.green.seen"] is False and look["z"] == 112.9
    hit = observe_call("move_relative", {"dz": -10}, {"ok": False, "error": "CONTACT at fingertip Z=73.4 …"})
    assert hit["contact_z"] == 73.4 and hit["failure"] == "contact" and hit["ok"] is False


def test_status_is_computed_and_needs_a_real_prediction(tmp_path):
    k = store(tmp_path)
    for rz in (80.0, 82.0, 85.0):
        centre(k, "blue", rz, 70)
    k.define_concept("finger-gap position in the wrist image", ["rz", "z"], "target_v ≈ 70 at rz ≈ 82")
    h = k.propose("C1", "the camera sits off the J6 axis, so the gap moves with rz",
                  {"routine": "wrist_center", "rz": [75, 90], "target_v": 70}, {"ok": True})
    assert (h["supports"], h["predicted"], h["status"]) == (3, 0, "open")        # a fit, not yet a prediction
    seen = centre(k, "green", 88.0, 70)
    assert "prediction_errors" not in seen
    assert k.list("C1")["hypotheses"][0]["status"] == "supported"
    assert "rz 80–88" in k.list("C1")["hypotheses"][0]["tested"]
    err = centre(k, "yellow", 78.0, 70, ok=False)                       # inside its conditions, wrong outcome
    assert err["prediction_errors"][0]["hypothesis"] == "C1.1"
    assert err["prediction_errors"][0]["observed"] == {"ok": False}
    assert k.list("C1")["hypotheses"][0]["status"] == "supported"     # 4 of 5 right is still >= 80 %
    centre(k, "blue", 120.0, 70, ok=False)                              # outside its conditions: not evidence
    assert k.list("C1")["hypotheses"][0]["contradicts"] == 1
    centre(k, "green", 76.0, 70, ok=False)
    assert k.list("C1")["hypotheses"][0]["status"] == "contested"


def test_refuted_hypotheses_leave_the_live_set(tmp_path):
    k = store(tmp_path)
    centre(k, "yellow", 82.0, 70)                                       # the variables exist (no match: target 70)
    k.define_concept("finger-gap position in the wrist image")
    k.propose("C1", "target_v 140 centres the gap at any rz", {"routine": "wrist_center", "target_v": 140},
              {"ok": True})
    centre(k, "blue", 82.0, 140, ok=False)
    centre(k, "green", 82.0, 140, ok=False)
    assert k.list("C1")["hypotheses"][0]["status"] == "refuted"
    assert "ruled out: C1.1" in k.digest()


def test_one_idea_stays_one_entry(tmp_path):
    k = store(tmp_path)
    k.define_concept("position of a cube after the gripper touched it", ["contact_z"])
    with pytest.raises(ValidationError, match="C1 already covers"):
        k.define_concept("cube position after the gripper touched it")
    k.record_measurement({"failure": "contact", "env.blue.seen": True, "case": 0})
    a = k.propose("C1", "a touch pushes the cube sideways so its stored position is stale",
                  {"failure": "contact"}, {"env.blue.seen": True})
    b = k.propose("C1", "a touch pushes the cube sideways, so the stored position is stale",
                  {"failure": "not_centred"}, {"ok": False})
    assert b["id"] == a["id"] and "existing" in b
    for i, why in enumerate(("the finger lands on the cube edge and tips it over",
                             "the env camera height estimate drifts after recalibration",
                             "vibration of the descent shakes the box surface"), 2):
        k.propose("C1", why, {"case": i}, {"ok": True})
    with pytest.raises(ValidationError, match="4 live hypotheses"):
        k.propose("C1", "a fifth competing explanation of the same quantity", {"case": 9}, {"ok": True})
    k.archive("C1.4", "superseded by C1.1")
    assert k.propose("C1", "a fifth competing explanation of the same quantity", {"case": 9}, {"ok": True})["id"] == "C1.5"


def test_generalizing_is_widening_the_conditions(tmp_path):
    k = store(tmp_path)
    for color in ("blue", "green"):
        k.record_measurement({"color": color, "top_error_mm": -6.0})
    k.record_measurement({"color": "yellow", "top_error_mm": 3.0})
    k.define_concept("error of the env-camera cube-top height", ["color", "place"])
    narrow = k.propose("C1", "depth reads the far edge of the cube on the box", {"color": "blue"},
                       {"top_error_mm": [-10, -2]})
    assert narrow["supports"] == 1 and narrow["tested"].endswith("1 episode")
    broad = k.propose("C1", "the env camera reads every cube top too high", {"action": "measurement"},
                      {"top_error_mm": [-10, -2]})
    assert (broad["supports"], broad["contradicts"]) == (2, 1)          # yellow is the counter-example
    assert broad["status"] == "contested"


def test_digest_holds_knowledge_not_observations(tmp_path):
    k = store(tmp_path)
    k.record_measurement({"color": "blue", "top_error_mm": -8.5}, note="blue touch 73.4 vs env 81.9")
    k.define_concept("error of the env-camera cube-top height", ["color"], "about -8 mm on the box")
    k.propose("C1", "depth reads the far edge of the cube on the box", {"color": "blue"},
              {"top_error_mm": [-10, -2]})
    k.define_concept("grip position when a cube is caught", ["color"])
    k.archive("C2", "not needed")
    d = k.digest()
    assert d.startswith("- C1 error of the env-camera cube-top height (depends on color) · model: about -8 mm")
    assert "C1.1 [open 1✓ 0✗, 0 predicted]" in d and "73.4" not in d and "C2" not in d
    reloaded = KnowledgeStore(tmp_path / "concepts.json")
    assert reloaded.list("C2")["archived"]["reason"] == "not needed"           # kept, not deleted
    assert len(reloaded.obs.items) == 1 and reloaded.obs.items[0]["note"] == "blue touch 73.4 vs env 81.9"


def test_measurement_carries_the_pose_and_episode_summary_points_at_gaps(tmp_path):
    k = store(tmp_path)
    k.state_fn = lambda: STATE
    m = k.record_measurement({"wrist.gap_v": 70})
    o = k.obs.items[-1]
    assert m["id"] == "O1" and o["vars"]["rz"] == 81.9 and o["vars"]["grip"] == 1000 and o["source"] == "astra"
    k.define_concept("finger-gap position in the wrist image")
    with pytest.raises(ValidationError, match=r"never observed: routine \(closest"):
        k.propose("C1", "the camera sits off the J6 axis, so the gap moves with rz",
                  {"routine": "wrist_center", "rz": [75, 90]}, {"ok": True})
    k.record_measurement({"routine": "none"})                          # now 'routine' is a variable observations carry
    k.propose("C1", "the camera sits off the J6 axis, so the gap moves with rz",
              {"routine": "wrist_center", "rz": [75, 90]}, {"ok": True})
    centre(k, "blue", 82.0, 70, ok=False)
    k.observe("run_skill", {"name": "pick_block_wrist"}, {"ok": True, "result": {"ok": False, "failure": "grasp_missed"}})
    k.observe("run_skill", {"name": "pick_block_wrist"}, {"ok": True, "result": {"ok": False, "failure": "grasp_missed"}})
    s = k.episode_summary("ep1")
    assert "O3 contradicts C1.1" in s
    assert "Failures no hypothesis covers: run_skill pick_block_wrist grasp_missed ×2" in s
    assert k.query({"skill": "pick_block_wrist"}, ["failure"])["total"] == 2


def test_only_selected_calls_are_recorded(tmp_path):
    k = KnowledgeStore(tmp_path / "concepts.json")
    assert k.observe("move_relative", {}, {"ok": True}) is None          # nothing recorded until wired
    k.records = lambda name: name != "get_state"
    assert k.observe("get_state", {}, {"ok": True}) is None
    assert k.observe("move_relative", {"dz": 5}, {"ok": True})["id"] == "O1"


def test_this_episode_means_the_reviewed_one_during_a_review(tmp_path):
    k = store(tmp_path)
    k.observe("move_relative", {"dz": 5}, {"ok": True})
    k.episode, k.focus_episode = "review-run", "ep1"
    assert k.query(this_episode=True)["total"] == 1


def test_diagnose_hypotheses_are_knowledge_hypotheses(tmp_path):
    from agent.explore.diagnosis import Investigation

    k = store(tmp_path)
    for v in (140, 140):
        centre(k, "blue", 82.0, v, ok=False)
    inv = Investigation(gate_after_repeats=2, knowledge=k)
    inv.after("run_skill", {"name": "pick_block_wrist"}, {"ok": True, "result": {"ok": False, "failure": "grasp_missed"}})
    with pytest.raises(ValidationError, match="prediction"):
        inv.diagnose("closed on air twice", [{"cause": "too high", "test": "grip lower", "expect": "caught"}])
    k.define_concept("finger gap position in the wrist image")
    with pytest.raises(ValidationError, match="already covers"):      # a new concept must be a new quantity
        inv.diagnose("fingers land beside the cube", [
            {"concept": "finger-gap position in the wrist image", "mechanism": "the gap is off the image centre",
             "when": {"routine": "wrist_center"}, "expect": {"ok": True}, "test": "look at the wrist"}])
    out = inv.diagnose("fingers land on the cube after centring", [
        {"concept_id": "C1", "mechanism": "target_v 140 puts the fingers on the cube at rz 82",
         "when": {"routine": "wrist_center", "target_v": 140}, "expect": {"ok": False}, "test": "rerun at target_v 70"}])
    h = out["hypotheses"][0]
    assert h["knowledge"]["id"] == "C1.1" and h["knowledge"]["supports"] == 2      # fits the past at once
    again = inv.diagnose("the same failure once more", [{"hypothesis_id": "C1.1", "test": "one more at 140"}])
    assert again["hypotheses"][0]["knowledge"]["id"] == "C1.1" and len(k.list("C1")["hypotheses"]) == 1
    centre(k, "green", 82.0, 140, ok=False)                                          # the test, recorded
    c = inv.conclude("H1", "fingers on the green cube top", "confirmed", "target_v 70")
    assert c["knowledge"]["status"] == "supported" and c["knowledge"]["predicted"] == 1


def test_episode_prompt_carries_the_knowledge_record(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from agent.explore import learning
    from agent.explore import ExploreSession

    k = store(tmp_path)
    centre(k, "blue", 82.0, 70)
    k.define_concept("finger-gap position in the wrist image")
    monkeypatch.setattr(learning, "history", lambda last=10: [])
    seen = {}
    harness = SimpleNamespace(run=lambda p: seen.setdefault("prompt", p) and "report", last_episode="e",
                              last_metrics={})
    s = ExploreSession(None, None, None, harness, None, None, None, k)
    s.run_episode("move the cubes", reflect=False)
    assert "Knowledge record: 1 recorded observations from 1 episodes (query_observations), 1 concepts, " \
           "0 hypotheses." in seen["prompt"]



def test_hypotheses_on_variables_no_observation_carries_are_refused(tmp_path):
    """17 Nov: target_XY, descend_to, grasp_depth_mm … could never get evidence (0 ✓ 0 ✗ forever)."""
    k = store(tmp_path)
    centre(k, "blue", 82.0, 70)
    k.define_concept("finger-gap position in the wrist image")
    with pytest.raises(ValidationError, match="target_XY.*record_measurement"):
        k.propose("C1", "some table spots fold the elbow", {"target_XY": "(-220,-300)"}, {"ok": False})
    assert k.propose("C1", "target 70 centres the gap at rz 82", {"routine": "wrist_center", "target_v": 70},
                     {"ok": True})["supports"] == 1
