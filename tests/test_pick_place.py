"""Pick/place delegation uses the real primitives over the simulated driver."""
import numpy as np
import pytest

from actions import ActionExecutor, build_registry
from common.errors import ContactError
from common.poses import pose_to_matrix
from perception.wrist_overlay import WristOverlay


def executor(prims, mode="live", **kwargs):
    return ActionExecutor(build_registry(prims), mode=mode, **kwargs)


@pytest.mark.parametrize("mode", ["dry_run", "live"])
def test_pick_order_grasp_z_and_report(prims, sim, gripper, mode):
    gripper.object_at = 700
    ex = executor(prims, mode)
    spec = ex.specs["pick_object"]
    assert spec.moves_robot and spec.delegates
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=100, yaw_deg=30, height_mm=26))
    assert out["ok"] and out["result"]["ok"]
    result = out["result"]
    assert [s["tool"] for s in result["steps"]] == [
        "open_gripper", "move_to_pose", "move_to_pose", "close_gripper", "move_to_pose"]
    poses = [s["args"] for s in result["steps"] if s["tool"] == "move_to_pose"]
    assert [p["z"] for p in poses] == [160, 87, 180]
    assert all(p["rx"] == 180 and p["ry"] == 0 for p in poses)
    if mode == "live":
        assert result["holding"] is True and result["position"] == 700
        assert len(sim.motion_log) == 3
        assert prims.get_state()["pose"][2] == pytest.approx(180)
    else:
        assert sim.motion_log == [] and gripper.position == 1000
        assert result["holding"] is None


@pytest.mark.parametrize("height,grasp", [(None, 87), (20, 90), (50, 85)])
def test_pick_height_default_and_cap(prims, height, grasp):
    args = dict(x=300, y=0, top_z=100, yaw_deg=0)
    if height is not None:
        args["height_mm"] = height
    result = executor(prims, "dry_run").execute("pick_object", args)["result"]
    assert result["steps"][2]["args"]["z"] == grasp


def test_gripper_y_axis_and_nearest_square_yaw(prims, sim):
    sim.state["pose"][5] = 82
    result = executor(prims, "dry_run").execute(
        "pick_object", dict(x=300, y=0, top_z=100, yaw_deg=10))["result"]
    rz = result["steps"][1]["args"]["rz"]
    assert rz == 100
    overlay = object.__new__(WristOverlay)
    overlay.tool, overlay.half_gap, overlay.open = 100, 30, 1000
    pts = overlay.points_flange(1000)
    axis = pts["finger+"] - pts["finger-"]
    assert axis == pytest.approx([0, 60, 0])
    base_axis = pose_to_matrix([0, 0, 0, 180, 0, rz])[:3, :3] @ axis
    edge = np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0])
    assert abs(np.dot(base_axis / 60, edge)) == pytest.approx(1)


@pytest.mark.parametrize("mode", ["dry_run", "live"])
def test_place_bottom_clearance_and_order(prims, sim, mode):
    out = executor(prims, mode).execute("place_object", dict(x=300, y=0, top_z=100))
    assert out["ok"] and out["result"]["ok"]
    steps = out["result"]["steps"]
    assert [s["tool"] for s in steps] == ["move_to_pose", "move_to_pose", "open_gripper", "move_to_pose"]
    assert [s["args"]["z"] for s in steps if s["tool"] == "move_to_pose"] == [186, 116, 186]
    assert len(sim.motion_log) == (3 if mode == "live" else 0)


def test_stop_at_first_failure_and_contact_latch(prims, sim):
    ex = executor(prims)
    moves = []
    original = ex.specs["move_to_pose"].handler
    def contact(**args):
        moves.append(args)
        if len(moves) == 2:
            sim.stop()
            raise ContactError("synthetic contact")
        return original(**args)
    ex.specs["move_to_pose"].handler = contact
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=100, yaw_deg=0))["result"]
    assert not out["ok"] and out["failed_step"] == "descend"
    assert "contact" in out["detail"] and len(out["steps"]) == 3
    assert ex.stopped and len(moves) == 2 and "Pause()" in sim.dashboard_log


def test_confirm_and_filtered_tools_still_delegate(prims, sim):
    calls = []
    ex = executor(prims, "confirm", allowed=["pick_object"],
                  confirm=lambda name, args: calls.append(name) or False)
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=100, yaw_deg=0))["result"]
    assert not out["ok"] and out["failed_step"] == "open"
    assert calls == ["open_gripper"] and sim.motion_log == []


def test_helper_failure_reaches_the_callers_status(prims):
    ex = executor(prims, "confirm", confirm=lambda name, args: False)
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=100, yaw_deg=0))
    assert not out["ok"] and not out["result"]["ok"]
    assert out["result"]["failed_step"] == "open" and "rejected" in out["error"]


def test_taught_table_splits_descent_and_refuses_floor(prims):
    prims.table_z_mm = 0
    prims.tool_length_mm = 118
    ex = executor(prims)
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=26, yaw_deg=0))["result"]
    # the simulator holds no cube: every motion runs, then the empty grasp is reported
    assert out["ok"] or out["failed_step"] == "grasp", (out.get("failed_step"), out.get("detail"))
    assert any(s.get("phase") == "descend" for s in out["steps"])
    out = ex.execute("pick_object", dict(x=300, y=0, top_z=5, yaw_deg=0))["result"]
    assert not out["ok"] and out["failed_step"] == "descend" and "table" in out["detail"]


def test_square_yaw_uses_nearest_solution_within_joint_limits(prims, sim, monkeypatch):
    sim.state["pose"][5] = 82
    prims.m.joint_limits_deg[5] = 90
    monkeypatch.setattr(prims, "_ik", lambda pose, near: [0, 0, 90, 0, 90, pose[5]])
    result = executor(prims, "dry_run").execute(
        "pick_object", dict(x=300, y=0, top_z=100, yaw_deg=10))["result"]
    assert result["ok"] and result["steps"][1]["args"]["rz"] == 10


def test_tools_available_to_programs_and_skills(prims):
    from actions.astra_code import ROUTINE_TOOLS
    from actions.skill_runner import SKILL_TOOLS
    from actions.registry import perception_specs
    from test_locate import synthetic
    from common.config import PerceptionConfig
    from perception.scene import Scene
    from types import SimpleNamespace as NS
    frame, cal = synthetic()
    scene = Scene(PerceptionConfig(), {"env": NS(read=lambda: frame)}, cal)
    spec = next(s for s in perception_specs(scene) if s.name == "locate_object")
    assert spec.read_only and not spec.moves_robot
    assert {"locate_object", "pick_object", "place_object"} <= SKILL_TOOLS & ROUTINE_TOOLS
    from agent.explore.prompt import DEFAULT_SESSION_TOOLS
    assert {"locate_object", "pick_object", "place_object"} <= set(DEFAULT_SESSION_TOOLS)


@pytest.mark.parametrize("name,phases", [("pick_object", ["open", "above", "descend", "close", "lift"]),
                                       ("place_object", ["above", "descend", "open", "lift"])])
def test_each_failure_stops_the_remaining_steps(prims, monkeypatch, name, phases):
    for index, phase in enumerate(phases):
        ex, calls = executor(prims, "dry_run"), []
        original = ex.execute
        def fail_step(tool, args=None, internal=False):
            if internal:
                calls.append(tool)
                if len(calls) == index + 1:
                    return {"ok": False, "error": "synthetic refusal", "executed": False}
            return original(tool, args, internal=internal)
        monkeypatch.setattr(ex, "execute", fail_step)
        args = dict(x=300, y=0, top_z=100, yaw_deg=0)
        out = ex.execute(name, args)
        assert not out["ok"] and out["result"]["failed_step"] == phase
        assert len(calls) == index + 1 and len(out["result"]["steps"]) == index + 1


def test_helpers_run_from_a_routine_and_a_saved_skill(prims, tmp_path):
    from actions.astra_code import AstraCode
    from actions.local_steps import LocalSteps
    from actions.skill_runner import SkillRunner
    from memory.skills import SkillLibrary
    runner = SkillRunner(SkillLibrary(tmp_path / "skills.json"))
    ex = ActionExecutor(build_registry(prims, runner=runner), mode="dry_run")
    runner.executor = ex
    code = AstraCode(tmp_path / "routines", core=LocalSteps)
    code.write("pick_one", 'KIND = "routine"\ndef run(ctx, memo):\n'
                           '    return ctx.execute("pick_object", {"x": 300, "y": 0, "top_z": 100, '
                           '"yaw_deg": 0})["result"]\n')
    local = LocalSteps(prims, None, ex, code=code)
    out = local.routine("pick_one", {})
    assert out["ok"] and out["trace"][0]["tool"] == "pick_object"
    runner.require_validation = False
    runner.save("place_one", "Place the held object", [{"tool": "place_object", "args": {
        "x": 300, "y": 0, "top_z": 100}}])
    assert runner.run("place_one")["ok"]


def test_an_empty_grasp_is_a_failure(prims):
    from actions import ActionExecutor, build_registry
    ex = ActionExecutor(build_registry(prims), mode="live")
    original = ex.execute

    def empty_close(tool, args=None, internal=False):
        r = original(tool, args, internal=internal)
        if tool == "close_gripper" and r.get("ok"):
            r = {**r, "result": {**r["result"], "holding": False, "position": 500}}
        return r
    ex.execute = empty_close
    out = ex.execute("pick_object", {"x": 300, "y": 0, "top_z": 100, "yaw_deg": 0})
    assert not out["ok"] and out["result"]["failed_step"] == "grasp"
