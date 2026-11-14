from actions import ActionExecutor, build_registry
from agent.tools import to_openai_tools

PRIMITIVES = {"get_state", "move_to_pose", "move_to_joints", "move_relative", "jog_joint", "go_to_position",
              "save_position", "open_gripper", "close_gripper", "set_gripper", "stop"}


def test_registry_exposes_primitives(prims):
    specs = build_registry(prims)
    assert {s.name for s in specs} == PRIMITIVES
    tools = to_openai_tools(specs)
    assert all(t["type"] == "function" and t["description"] and "parameters" in t for t in tools)


def test_dry_run_never_moves(prims, sim):
    ex = ActionExecutor(build_registry(prims), mode="dry_run")
    r = ex.execute("move_relative", {"dz": -20})
    assert r["ok"] and not r["executed"] and sim.motion_log == []
    assert ex.execute("get_state")["executed"]   # read-only tools still run


def test_schema_validation(prims):
    ex = ActionExecutor(build_registry(prims), mode="live")
    assert not ex.execute("jog_joint", {"joint": 1})["ok"]
    assert not ex.execute("move_relative", {"dz": 5, "bogus": 1})["ok"]
    assert not ex.execute("move_relative", {"frame": "world"})["ok"]
    assert not ex.execute("nope")["ok"]


def test_confirm_mode(prims, sim):
    answers = iter([False, True])
    ex = ActionExecutor(build_registry(prims), mode="confirm", confirm=lambda n, a: next(answers))
    assert not ex.execute("jog_joint", {"joint": 1, "degrees": 5})["ok"]
    assert ex.execute("jog_joint", {"joint": 1, "degrees": 5})["executed"]
    assert len(sim.motion_log) == 1


def test_live_mode_reports_safety_errors(prims, sim):
    sim.state["error"] = 1
    r = ActionExecutor(build_registry(prims), mode="live").execute("jog_joint", {"joint": 1, "degrees": 5})
    assert not r["ok"] and "Mode 5" in r["error"]


def test_composite_skill_from_primitives(prims, sim, gripper):
    """A 'pick' skill is just a list of primitive steps."""
    gripper.object_at = 300
    pick = [
        {"tool": "open_gripper"},
        {"tool": "move_to_pose", "args": {"x": 350, "y": 0, "z": 200, "rx": 180, "ry": 0, "rz": 0}},
        {"tool": "move_relative", "args": {"dz": -50}},
        {"tool": "close_gripper"},
        {"tool": "move_relative", "args": {"dz": 50}},
    ]
    results = ActionExecutor(build_registry(prims), mode="live").run_sequence(pick)
    assert all(r["ok"] for r in results) and results[3]["result"]["holding"]
    assert sim.state["pose"][2] == 200


def test_sequence_stops_on_first_error(prims, sim):
    steps = [{"tool": "move_relative", "args": {"dz": 500}}, {"tool": "open_gripper"}]
    results = ActionExecutor(build_registry(prims), mode="live").run_sequence(steps)
    assert len(results) == 1 and not results[0]["ok"]


def test_allowed_filter(prims):
    ex = ActionExecutor(build_registry(prims), allowed=["get_state", "stop"])
    assert {s.name for s in ex.list()} == {"get_state", "stop"}
