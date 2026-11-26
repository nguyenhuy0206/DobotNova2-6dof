"""Autonomous threading checks use fresh images, no motion tools and no judge credit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.explore.checking import check_task, retry_stop_reason
from common.config import AgentConfig
from memory.store import EpisodeStore
from test_harness import FakeClient, call


@pytest.fixture
def script(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("run_explore_retry_test", root / "scripts/run_explore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session(tmp_path, status="inconclusive"):
    store = EpisodeStore(tmp_path / "episodes")
    ep = store.start("Thread the object", {"driver": "astra"})
    store.append(ep, {"event": "end", "reason": "done"})
    s = NS(cfg=NS(cameras=["wrist"]), executor=NS(stopped=None),
           harness=NS(episodes=store, config=AgentConfig(),
                      judge=NS(ledger=NS(outcome=lambda _: {"status": status}))),
           preflight=lambda: [], kit=NS(scene=NS()))
    return s, ep


@pytest.mark.parametrize("status", ["success", "failure", "unclear"])
def test_checker_uses_fresh_image_and_returns_structured_assessment(tmp_path, status):
    s, ep = session(tmp_path)
    seen = []
    s.kit.scene.look = lambda camera: seen.append(camera) or {"camera": camera, "image_jpeg_b64": "fresh-final"}
    class Client(FakeClient):
        def create(self, input, **kw):
            assert [t["name"] for t in kw["tools"]] == ["record_task_check"]
            return super().create(input, **kw)
    s.harness.client = Client([
        NS(id="check1", output=[call("record_task_check", {
            "status": status, "evidence": "The tip is visible beside the eye.",
            "next_action": "Inspect alignment before correcting."}, "c")], output_text=""),
        NS(id="check1b", output=[call("record_task_check", {
            "status": "unclear", "evidence": "No camera images are attached.",
            "next_action": "x"}, "c2")], output_text=""),
        NS(id="check2", output=[], output_text="Checked.")])
    out = check_task(s, "Thread the object", ep)
    assert seen == ["wrist"] and out["status"] == status and out["retry_allowed"]
    assert out["evidence"] == "The tip is visible beside the eye."    # a second call does not overwrite it
    assert any(p.get("type") == "input_image" and "fresh-final" in p["image_url"]
               for p in s.harness.client.calls[0][0]["content"])
    assert s.harness.judge.ledger.outcome(ep)["status"] == "inconclusive"
    assert s.harness.episodes.load(ep)[-1]["event"] == "self_check"
    from agent.explore.attempts import report
    feedback = report(s.harness.episodes, s.harness.judge.ledger, ep, {})
    assert "ASTRA VISUAL SELF-CHECK" in feedback and "beside the eye" in feedback
    assert "Inspect alignment" not in feedback and feedback.count("beside the eye") == 1
    from agent.explore.learning import summarize_episode
    assert "ASTRA VISUAL SELF-CHECK" in summarize_episode(s.harness.episodes, ep)


def test_camera_failure_stops_retries_without_a_model_call(tmp_path):
    s, ep = session(tmp_path)
    def missing(**_):
        raise OSError("camera disconnected")
    s.kit.scene.look = missing
    s.harness.client = None
    out = check_task(s, "Thread", ep)
    assert out["status"] == "unclear" and not out["retry_allowed"]
    assert "disconnected" in out["evidence"]


def test_no_camera_cannot_produce_success(tmp_path):
    s, ep = session(tmp_path)
    s.cfg.cameras = []
    s.harness.client = None
    assert not check_task(s, "Thread", ep)["retry_allowed"]


def test_session_refuses_autonomous_motion_while_stop_is_latched(tmp_path):
    from agent.explore.session import ExploreSession
    from common.errors import RobotError
    s, _ = session(tmp_path)
    s.cfg.driver = "astra"
    s.executor.stopped = "CONTACT"
    # No run method: reaching the harness would also clear the latch and fail this test.
    with pytest.raises(RobotError, match="operator resume required"):
        ExploreSession.run_episode(s, "Thread", reflect=False, self_check=True)


def test_dry_run_never_claims_physical_success_or_calls_checker(tmp_path):
    from agent.explore.session import ExploreSession
    s, ep = session(tmp_path)
    s.cfg.driver, s.cfg.program_mode = "astra", False
    s.executor.mode, s.knowledge, s.investigation = "dry_run", None, None
    s.harness.run = lambda _: "simulated"
    s.harness.last_episode, s.harness.last_metrics = ep, {}
    out = ExploreSession.run_episode(s, "Thread", reflect=False, self_check=True)
    assert "dry_run" in out["retry_stop"] and "self_check" not in out


def test_no_structured_check_does_not_trigger_infinite_retry(tmp_path):
    s, ep = session(tmp_path)
    s.kit.scene.look = lambda camera: {"image_jpeg_b64": "image"}
    s.harness.client = FakeClient([NS(id="r", output=[], output_text="Looks good!")])
    out = check_task(s, "Thread", ep)
    assert out["status"] == "unclear" and not out["retry_allowed"]


@pytest.mark.parametrize("reason", ["robot_stopped", "robot_alarm"])
def test_retry_never_clears_previous_safety_stop(tmp_path, reason, script):
    s, ep = session(tmp_path)
    s.harness.episodes.append(ep, {"event": "end", "reason": reason})
    assert retry_stop_reason(s, ep) == reason
    assert script.autonomous_stop(s, {"episode": ep}) == reason
    s.executor.stopped = "CONTACT"
    assert script.autonomous_stop(s, {"episode": ep}) == "CONTACT"


def test_self_check_success_stops_but_never_overrides_judge_failure(tmp_path, script):
    s, ep = session(tmp_path)
    out = {"episode": ep, "self_check": {"status": "success", "retry_allowed": True}}
    assert "not judge credit" in script.autonomous_stop(s, out)
    s.harness.judge.ledger.outcome = lambda _: {"status": "failed"}
    assert script.autonomous_stop(s, out) is None
    s.preflight = lambda: ["robot not ready"]
    assert script.autonomous_stop(s, out) == "robot not ready"


@pytest.mark.parametrize("statuses,cap,expected", [(["failure", "unclear", "success"], 0, 3),
                                                  (["failure", "failure"], 2, 2)])
def test_cli_retries_without_label_or_reset_prompts(tmp_path, monkeypatch, script, statuses, cap, expected):
    s, ep = session(tmp_path)
    from agent.explore.config import ExploreConfig
    from agent.harness import HarnessOptions
    s.cfg = ExploreConfig(task="Thread", until_success=True, episodes=cap)
    s.harness.options = HarnessOptions.from_variant("V3")
    s.harness.skills = NS(all=lambda: [])
    s.kit.primitives = NS(active_tool_offset=[], tool_offset_source="mock", table_z_mm=0,
                          grip_close_position=500)
    s.kit.scene.env_calibration, s.view_guard = None, None
    s.executor.list = lambda: []
    s.arm_collision_detection = s.set_global_speed = lambda: "mock"
    calls = []
    def run(task, **kwargs):
        calls.append(kwargs)
        return {"episode": ep, "report": "Attempt", "self_check": {
            "status": statuses[len(calls) - 1], "retry_allowed": True, "evidence": "image", "next_action": "align"}}
    s.run_episode = run
    monkeypatch.setattr(script, "build_explore", lambda **_: s)
    monkeypatch.setattr(script, "setup_logging", lambda **_: None)
    monkeypatch.setattr("sys.argv", ["run_explore.py", "--yes", "--no-reflect"])
    def no_input(*_):
        pytest.fail("autonomous loop must not ask for a label or scene reset")
    monkeypatch.setattr("builtins.input", no_input)
    script.main()
    assert len(calls) == expected
    assert all(c == {"reflect": False, "self_check": True} for c in calls)


def test_retry_prompt_says_the_robot_is_where_the_last_attempt_left_it(tmp_path):
    from agent.explore.session import ExploreSession
    s, ep = session(tmp_path)
    s.cfg.driver, s.cfg.program_mode, s.cfg.allowed_positions = "astra", False, ["insert_start"]
    s.executor.mode, s.knowledge, s.investigation = "dry_run", None, None
    prompts = []
    s.harness.run = lambda p: prompts.append(p) or "simulated"
    s.harness.last_episode, s.harness.last_metrics = ep, {}
    ExploreSession.run_episode(s, "Thread", reflect=False, self_check=True)
    assert "where the previous attempt left it" in prompts[0]
    assert "go_to_position(insert_start) starts over" in prompts[0]


def test_keep_going_continues_unless_the_check_verifies_success(monkeypatch):
    from agent.explore import checking
    from agent.explore.session import ExploreSession
    s = NS(executor=NS(mode="live", stopped=None), _verified=None)
    results = iter([{"status": "unclear", "evidence": "tip beside the shaft", "next_action": "oblique view"},
                    {"status": "success", "evidence": "tip past the eye"}])
    monkeypatch.setattr(checking, "check_task", lambda session, task, ep: next(results))
    keep_going = ExploreSession._keep_going(s, "Thread")
    msg = keep_going("e1", "cannot confirm")
    assert "NOT FINISHED" in msg and "tip beside the shaft" in msg and "oblique" not in msg
    assert keep_going("e1", "threaded") is None and s._verified["status"] == "success"
    s.executor.stopped = "contact"
    assert keep_going("e1", "x") is None                  # a stopped robot is never told to continue
