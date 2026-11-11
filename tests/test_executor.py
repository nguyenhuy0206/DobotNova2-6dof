from actions import ActionExecutor, build_registry
from agent.tools import to_openai_tools


def test_dry_run_never_moves(actions, sim):
    ex = ActionExecutor(build_registry(actions), mode="dry_run")
    r = ex.execute("joint_step", {"joint": 1, "degrees": 5, "speed": 10})
    assert r["ok"] and not r["executed"] and sim.motion_log == []
    # read-only tools still run in dry_run
    assert ex.execute("get_robot_state")["executed"]


def test_dry_run_still_validates_schema(actions):
    ex = ActionExecutor(build_registry(actions), mode="dry_run")
    assert not ex.execute("joint_step", {"joint": 1})["ok"]
    assert not ex.execute("joint_step", {"joint": 1, "degrees": 5, "speed": 10, "x": 1})["ok"]
    assert not ex.execute("nope")["ok"]


def test_confirm_mode(actions, sim):
    answers = iter([False, True])
    ex = ActionExecutor(build_registry(actions), mode="confirm", confirm=lambda n, a: next(answers))
    assert not ex.execute("joint_step", {"joint": 1, "degrees": 5, "speed": 10})["ok"]
    assert ex.execute("joint_step", {"joint": 1, "degrees": 5, "speed": 10})["executed"]
    assert len(sim.motion_log) == 1


def test_live_mode_reports_safety_errors(actions, sim):
    sim.state["error"] = 1
    r = ActionExecutor(build_registry(actions), mode="live").execute(
        "joint_step", {"joint": 1, "degrees": 5, "speed": 10})
    assert not r["ok"] and "Mode 5" in r["error"]


def test_allowed_filter_and_tool_schema(actions):
    ex = ActionExecutor(build_registry(actions), allowed=["get_robot_state", "stop"])
    tools = to_openai_tools(ex.list())
    assert {t["name"] for t in tools} == {"get_robot_state", "stop"}
    assert all(t["type"] == "function" and "parameters" in t for t in tools)
