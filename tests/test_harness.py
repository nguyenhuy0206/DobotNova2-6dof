"""Agent loop with a scripted fake model — no network."""
import json
from types import SimpleNamespace as NS

from actions import ActionExecutor, build_registry
from agent.harness import AgentHarness
from common.config import AgentConfig
from memory.skills import SkillLibrary
from memory.store import EpisodeStore


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = iter(script), []

    def create(self, input, **kw):
        self.calls.append(input)
        return next(self.script)


def call(name, args, cid):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid)


def test_harness_runs_tools_and_logs_episode(prims, sim, tmp_path):
    client = FakeClient([
        NS(id="r1", output=[call("get_state", {}, "c1")], output_text=""),
        NS(id="r2", output=[call("jog_joint", {"joint": 1, "degrees": 5, "speed": 10}, "c2")], output_text=""),
        NS(id="r3", output=[], output_text="done"),
    ])
    ex = ActionExecutor(build_registry(prims), mode="live")
    episodes = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(client, ex, AgentConfig(), episodes, SkillLibrary(tmp_path / "skills.json"))

    assert h.run("jog J1") == "done"
    assert sim.motion_log == ["RelJointMovJ(5,0,0,0,0,0,SpeedJ=10,AccJ=10)"]
    assert client.calls[1][0]["call_id"] == "c1"
    events = [r["event"] for r in episodes.load(episodes.list()[0]) if r["event"] != "usage"]
    assert events == ["start", "model", "tool", "model", "tool", "model", "end", "metrics"]


def test_instructions_sent_on_every_call(prims, tmp_path):
    """Responses API drops `instructions` across previous_response_id: resend them each turn."""
    class Rec(FakeClient):
        def create(self, input, **kw):
            self.calls.append(kw.get("instructions"))
            return next(self.script)
    client = Rec([NS(id="r1", output=[call("get_state", {}, "c1")], output_text=""),
                  NS(id="r2", output=[], output_text="done")])
    h = AgentHarness(client, ActionExecutor(build_registry(prims), mode="live"), AgentConfig(system_prompt="RULES"),
                     EpisodeStore(tmp_path / "ep"), SkillLibrary(tmp_path / "s.json"))
    h.run("x")
    assert len(client.calls) == 2 and all(c and c.startswith("RULES") for c in client.calls)


def test_truncated_tool_arguments_do_not_crash(prims, tmp_path):
    bad = NS(type="function_call", name="get_state", arguments='{"a": "unterminated', call_id="c1")
    client = FakeClient([NS(id="r1", output=[bad], output_text=""), NS(id="r2", output=[], output_text="ok")])
    h = AgentHarness(client, ActionExecutor(build_registry(prims), mode="live"), AgentConfig(),
                     EpisodeStore(tmp_path / "ep"), SkillLibrary(tmp_path / "s.json"))
    assert h.run("x") == "ok"
    assert "not valid JSON" in client.calls[1][0]["output"]


# ---------------------------------------------------------------- failure investigation
def test_failure_must_be_diagnosed_before_retry(tmp_path):
    """run_skill fails -> a blind retry is refused -> diagnose + test -> conclude -> retry runs."""
    from actions.executor import ActionSpec
    from actions.registry import _obj
    from agent.diagnosis import Investigation
    from agent.diagnosis import specs as diagnosis_specs
    from agent.harness import HarnessOptions
    runs = []

    def run_skill(name, params=None):
        runs.append(params)
        return ({"skill": name, "ok": False, "failure": "grasp_missed", "detail": "fingers met nothing"}
                if len(runs) == 1 else {"skill": name, "ok": True})

    inv = Investigation(gate_after_repeats=1)        # strict mode: every failure gated
    ex = ActionExecutor([ActionSpec("run_skill", "run", run_skill,
                                    _obj({"name": {"type": "string"}, "params": {"type": "object"}}, ["name"]),
                                    delegates=True),
                         ActionSpec("look", "look", lambda camera="wrist", colors=None: {"camera": camera, "objects": []},
                                    _obj({"camera": {"type": "string"}, "colors": {"type": "array"}}, []),
                                    read_only=True),
                         *diagnosis_specs(inv)], mode="live")
    pick = {"name": "pick_block_wrist", "params": {"color": "blue", "depth_mm": 8}}
    hyps = [{"cause": "grasp too high", "test": "retry with depth_mm 12", "expect": "holding=true"},
            {"cause": "cube not centred", "test": "look wrist", "expect": "cube off the finger gap"}]
    client = FakeClient([
        NS(id="r1", output=[call("run_skill", pick, "c1")], output_text=""),
        NS(id="r2", output=[call("run_skill", pick, "c2")], output_text=""),               # blind retry
        NS(id="r3", output=[call("diagnose", {"why": "closed on air at measured top - 8 mm",
                                               "hypotheses": hyps}, "c3"),
                            call("look", {"camera": "wrist"}, "c4")], output_text=""),
        NS(id="r4", output=[call("conclude", {"hypothesis_id": "H2", "observed": "cube centred in the gap",
                                              "verdict": "rejected"}, "c5"),
                            call("run_skill", {**pick, "params": {"color": "blue", "depth_mm": 12}}, "c6")],
           output_text=""),
        NS(id="r5", output=[call("conclude", {"hypothesis_id": "H1", "observed": "holding at depth 12, lifted",
                                              "verdict": "confirmed", "fix": "depth_mm 12"}, "c7")],
           output_text=""),
        NS(id="r6", output=[], output_text="done. Why it failed: grasp too high (depth 8) -> depth 12"),
    ])
    episodes = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(client, ex, AgentConfig(max_steps=10), episodes, SkillLibrary(tmp_path / "s.json"),
                     investigation=inv, state_fn=lambda: {"pose": [0, 0, 300]},
                     options=HarnessOptions.from_variant("V3"))
    h.run("pick blue")

    tools = [r for r in episodes.load(h.last_episode) if r["event"] == "tool" and not r.get("auto")]
    first, blind = tools[0]["result"], tools[1]["result"]
    assert first["diagnosis"]["failure_id"] == "F1"
    assert blind["executed"] is False and "without a diagnosis" in blind["error"]
    assert len(runs) == 2 and runs[1]["depth_mm"] == 12                  # only the diagnosed retry ran
    s = h.last_diagnosis
    assert s["diagnosed"] == 1 and s["confirmed_causes"] == 1 and s["untested"] == 0
    assert s["per_failure"][0]["cause"] == "grasp too high" and s["per_failure"][0]["fix"] == "depth_mm 12"
    # STATE showed the open failure while it was undiagnosed
    second_turn = client.calls[2][0]["content"][0]["text"]
    assert "undiagnosed_failures" in second_turn and "F1" in second_turn


def test_investigation_validates_hypotheses(tmp_path):
    import pytest

    from agent.diagnosis import Investigation
    from common.errors import ValidationError

    inv = Investigation()
    with pytest.raises(ValidationError, match="no failure"):
        inv.diagnose("something went wrong here", [{"cause": "a", "test": "b", "expect": "c"}])
    inv.after("verify_calibration", {}, {"ok": True, "result": {"calibration_valid": False, "advice": "moved"}})
    with pytest.raises(ValidationError, match="test"):
        inv.diagnose("camera disagrees by 245 mm", [{"cause": "camera moved"}])
    out = inv.diagnose("camera disagrees by 245 mm", [{"cause": "camera moved", "test": "recalibrate + verify",
                                                         "expect": "verify < 15 mm"}])
    assert out["failure"] == "F1" and inv.state()["untested_hypotheses"]
    with pytest.raises(ValidationError, match="verdict"):
        inv.conclude("H1", "verify 4 mm after recalibration", "maybe")
    assert "all hypotheses tested" in inv.conclude("H1", "still 200 mm off", "rejected")["next"]
    assert inv.state() is None and "NOT diagnosed" not in inv.report_text()


def test_default_gate_lets_one_failure_retry_and_stops_repeats():
    from agent.diagnosis import Investigation
    inv = Investigation()                                   # gate_after_repeats = 2
    fail = {"ok": True, "result": {"ok": False, "failure": "grasp_missed"}}
    pick = {"name": "pick_block_wrist"}
    assert inv.after("run_skill", pick, fail)["next"] == "change ONE thing and retry"
    assert inv.before("run_skill", pick) is None             # one failure: retry freely
    inv.after("run_skill", pick, {"ok": True, "result": {"ok": False, "failure": "not_centred"}})
    assert inv.before("run_skill", pick) is None             # different failure: still free
    assert "diagnose" in inv.after("run_skill", pick, fail)["next"]
    assert "grasp_missed 2×" in inv.before("run_skill", pick)
    inv.diagnose("closes on air twice", [{"cause": "target off", "test": "look wrist", "expect": "cube off-gap"}])
    assert inv.before("run_skill", pick) is None             # one diagnosis covers both repeats
    assert Investigation(gate_after_repeats=0).before("run_skill", pick) is None


def test_bounded_context_keeps_lookups_and_every_colour():
    from agent.context import WorkingMemory
    from agent.events import objects_digest

    objs = [{"label": "blue", "pixel": [560, 167], "base_xyz_mm": [55, -130, 176]},
            {"label": "blue", "pixel": [623, 93]}, {"label": "blue", "pixel": [262, 291],
                                                    "base_xyz_mm": [-349, -356, 70]},
            {"label": "blue", "pixel": [588, 147]}, {"label": "blue", "pixel": [489, 178]},
            {"label": "green", "pixel": [235, 296], "base_xyz_mm": [-377, -394, 68]},
            {"label": "yellow", "pixel": [206, 298], "base_xyz_mm": [-395, -439, 73]}]
    d = objects_digest(objs)
    assert d == "blue@55,-130,176|-349,-356,70(+3);green@-377,-394,68;yellow@-395,-439,73"
    wm = WorkingMemory("task")
    skill = {"ok": True, "result": [{"name": "pick_block_wrist", "params": {"target_v": 140}}]}
    wm.record_turn(1, "", [("list_skills", {"name": "pick_block_wrist"}, skill)], [], "", [])
    wm.record_turn(2, "", [("look", {"camera": "env"}, {"ok": True, "result": {"objects": []}})], [], "", [])
    wm.record_turn(3, "", [("look", {"camera": "env"}, {"ok": True, "result": {"objects": []}})], [], "", [])
    text = wm.build({})
    assert "LOOKUPS YOU MADE" in text and '"target_v": 140' in text            # still there 2 turns later
    wm.record_turn(4, "", [("list_skills", {"name": "pick_block_wrist"}, skill)], [], "", [])
    assert "LOOKUPS YOU MADE" not in wm.build({})                               # shown once, as latest result


def test_a_gripper_call_that_changes_nothing_does_not_hold_back_the_move():
    from types import SimpleNamespace as C

    from agent.chunking import ChunkPolicy

    calls = [C(name="open_gripper", parsed_args={}), C(name="move_relative", parsed_args={"dz": -10})]
    run, deferred, why = ChunkPolicy().split(calls, 120.0, 162.0, grip=1000)       # already fully open
    assert [c.name for c in run] == ["open_gripper", "move_relative"] and not deferred
    run, deferred, why = ChunkPolicy().split(calls, 120.0, 162.0, grip=500)        # really opens: look first
    assert [c.name for c in run] == ["open_gripper"] and deferred
    run, _, _ = ChunkPolicy().split([C(name="set_gripper", parsed_args={"position": 998})] + calls[1:],
                                    120.0, 162.0, grip=1000)
    assert len(run) == 2


def test_failure_ids_are_accepted_as_state_shows_them():
    from agent.diagnosis import Investigation

    inv = Investigation()
    inv.after("go_to_position", {"name": "home"}, {"ok": False, "error": "no FK"})
    shown = inv.state()["undiagnosed_failures"][0]
    assert shown == "F1 go_to_position:error"
    out = inv.diagnose("the joint path could not be checked", [{"cause": "FK connection reset", "test": "retry",
                                                                 "expect": "accepted"}], failure_id=shown)
    assert out["failure"] == "F1"
    assert inv.conclude("H1(F1) FK connection reset", "accepted on retry", "confirmed")["hypothesis"] == "H1"


def test_harness_ends_the_episode_on_the_third_identical_action(tmp_path):
    from actions.executor import ActionSpec
    from actions.registry import _obj
    from agent.harness import HarnessOptions

    moves = []
    ex = ActionExecutor([ActionSpec("open_gripper", "open", lambda force=30: moves.append(force) or {"position": 1000},
                                    _obj({"force": {"type": "integer"}}, []), moves_robot=True)], mode="live")
    same = lambda i: NS(id=f"r{i}", output=[call("open_gripper", {"force": 30}, f"c{i}")], output_text="")
    client = FakeClient([same(1), same(2), same(3), same(4), NS(id="r5", output=[], output_text="done")])
    opts = HarnessOptions.from_variant("V3", max_repeats=3)
    h = AgentHarness(client, ex, AgentConfig(max_steps=10), EpisodeStore(tmp_path / "ep"),
                     SkillLibrary(tmp_path / "s.json"), options=opts, state_fn=lambda: {"pose": [0, 0, 300]})
    out = h.run("open")
    assert len(moves) == 3 and out.startswith("Stopped: the same action 3× in a row: open_gripper")
