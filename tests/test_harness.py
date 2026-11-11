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


def test_harness_runs_tools_and_logs_episode(actions, sim, tmp_path):
    client = FakeClient([
        NS(id="r1", output=[call("get_robot_state", {}, "c1")], output_text=""),
        NS(id="r2", output=[call("joint_step", {"joint": 1, "degrees": 5, "speed": 10}, "c2")], output_text=""),
        NS(id="r3", output=[], output_text="done"),
    ])
    ex = ActionExecutor(build_registry(actions), mode="live")
    episodes = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(client, ex, AgentConfig(), episodes, SkillLibrary(tmp_path / "skills.json"))

    assert h.run("jog J1") == "done"
    assert sim.motion_log == ["RelJointMovJ(5,0,0,0,0,0,SpeedJ=10,AccJ=10)"]
    assert client.calls[1][0]["call_id"] == "c1"
    events = [r["event"] for r in episodes.load(episodes.list()[0])]
    assert events == ["start", "model", "tool", "model", "tool", "model", "end"]
