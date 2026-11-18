"""Episodes -> self-review -> versioned skills, outcomes, notes (fake model, no network)."""
import json
from types import SimpleNamespace as NS

from actions import ActionExecutor, build_registry
from actions.skill_runner import SkillRunner
from agent.harness import AgentHarness
from agent.learning import OutcomeRecorder, history, reflect_executor, summarize_episode
from common.config import AgentConfig
from memory.notes import NotesStore
from memory.skills import Skill, SkillLibrary
from memory.store import EpisodeStore
from agent.explore import _explore_specs


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = iter(script), []

    def create(self, input, **kw):
        self.calls.append((input, kw))
        return next(self.script)


def call(name, args, cid):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid)


def test_skill_revisions_keep_history(tmp_path):
    lib = SkillLibrary(tmp_path / "s.json")
    lib.revise(Skill("grasp", "v1", steps=[{"tool": "close_gripper"}]))
    lib.record_outcome("grasp", False)
    lib.record_outcome("grasp", True)
    assert lib.get("grasp").version == 1                       # outcomes don't bump the version
    v2 = lib.revise(Skill("grasp", "deeper grasp", steps=[{"tool": "open_gripper"}, {"tool": "close_gripper"}]))
    assert v2.version == 2 and v2.successes == v2.failures == 0
    old = v2.history[0]
    assert {k: old[k] for k in ("version", "steps", "params", "successes", "failures")} == \
        {"version": 1, "steps": [{"tool": "close_gripper"}], "params": {}, "successes": 1, "failures": 1}
    assert v2.validation_status == "candidate"
    assert SkillLibrary(tmp_path / "s.json").get("grasp").history[0]["failures"] == 1


def test_episode_review_records_outcome_and_updates_skill(prims, sim, tmp_path):
    episodes = EpisodeStore(tmp_path / "ep")
    lib = SkillLibrary(tmp_path / "skills.json")
    notes = NotesStore(tmp_path / "notes.jsonl")
    from memory.knowledge import KnowledgeStore
    know = KnowledgeStore(tmp_path / "concepts.json")
    know.records = lambda name: name in ("move_relative", "run_skill")     # what the harness records
    runner = SkillRunner(lib)
    cal = NS(record=lambda: {}, solve=lambda: {}, reset=lambda: {}, verify=lambda: {})
    main = ActionExecutor(build_registry(prims, runner=runner) + _explore_specs(cal, None, notes, knowledge=know), mode="live")
    runner.executor = main

    # 1) an episode with the robot
    robot = FakeClient([NS(id="a", output=[call("get_state", {}, "1")], output_text=""),
                        NS(id="b", output=[call("move_relative", {"dz": -20}, "2")], output_text=""),
                        NS(id="c", output=[], output_text="moved down")])
    h = AgentHarness(robot, main, AgentConfig(), episodes, lib, notes=know, observer=know.observe)
    h.run("pick a block")
    ep = h.last_episode
    assert [o["vars"]["action"] for o in know.obs.items] == ["move_relative"]       # recorded by the harness
    assert know.obs.items[0]["episode"] == ep and know.obs.items[0]["vars"]["ok"] is True
    log = summarize_episode(episodes, ep)
    assert "TASK: pick a block" in log and "move_relative" in log and "END: done" in log

    # 2) the review: memory tools only
    recorder = OutcomeRecorder(episodes, tmp_path / "ep" / "index.jsonl")
    recorder.episode_id = ep
    review_ex = reflect_executor(main, recorder)
    assert set(review_ex.specs) == {"record_episode_outcome", "list_skills", "save_skill", "list_concepts",
                                    "define_concept", "update_model", "propose_hypothesis", "archive_knowledge",
                                    "query_observations", "read_history"}
    assert "move_relative" not in review_ex.specs and "get_state" not in review_ex.specs
    steps = [{"tool": "move_relative", "args": {"dz": "=-{step}"}}]
    reviewer = FakeClient([
        NS(id="r1", output=[call("record_episode_outcome", {"success": False, "blocks_moved": 0,
                                                            "summary": "only descended"}, "o"),
                            call("save_skill", {"name": "descend", "description": "step down",
                                                "steps": steps, "params": {"step": 20}}, "s"),
                            call("define_concept", {"quantity": "free travel above the block",
                                                    "depends_on": ["dz"]}, "c"),
                            call("propose_hypothesis", {"concept_id": "C1", "mechanism": "nothing is in the way "
                                                        "for the first 20 mm below the start pose",
                                                        "when": {"action": "move_relative", "dz": [-20, 0]},
                                                        "expect": {"ok": True}}, "n")],
           output_text=""),
        NS(id="r2", output=[], output_text="reviewed")])
    AgentHarness(reviewer, review_ex, AgentConfig(), EpisodeStore(tmp_path / "rev"), lib, notes=know).run(log)
    assert recorder.last["success"] is False and history(tmp_path / "ep" / "index.jsonl")[0]["episode"] == ep
    assert episodes.load(ep)[-1]["event"] == "outcome"
    hyp = know.list("C1")["hypotheses"][0]
    assert lib.get("descend").version == 1
    assert (hyp["supports"], hyp["predicted"], hyp["status"]) == (1, 0, "open")    # fits the past, predicts nothing yet
    # the next review improves the same skill -> version 2
    runner.save("descend", "smaller steps", [{"tool": "move_relative", "args": {"dz": -10}}])
    assert lib.get("descend").version == 2
