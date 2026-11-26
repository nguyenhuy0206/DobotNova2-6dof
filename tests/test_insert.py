"""Needle insertion (23 Nov, operator): a slow straight stroke along the gripper's own +Z."""
import pytest

from actions import ActionExecutor, build_registry


def test_insert_moves_along_tool_z_at_the_insert_speed(prims, sim):
    prims.fixed_speed = 50
    tip0 = prims.get_state()["pose"]
    ex = ActionExecutor(build_registry(prims), mode="live")
    sent = []
    movl = prims.robot.movl
    prims.robot.movl = lambda pose, speed, **kw: sent.append(speed) or movl(pose, speed, **kw)
    r = ex.execute("insert", {"depth_mm": 10})
    assert r["ok"], r
    tip1 = r["result"]["pose"]
    import numpy as np
    from common.poses import pose_to_matrix
    axis = pose_to_matrix(tip0)[:3, 2]                       # tool Z in the base frame
    assert np.allclose(np.array(tip1[:3]) - tip0[:3], axis * 10, atol=1.5)
    assert sent == [prims.m.insert_speed] and r["result"]["inserted_mm"] == 10


def test_insert_retract_comes_back(prims):
    tip0 = prims.get_state()["pose"]
    r = ActionExecutor(build_registry(prims), mode="live").execute("insert", {"depth_mm": 8, "retract": True})
    assert r["ok"] and r["result"]["retracted"]
    assert r["result"]["pose"][:3] == pytest.approx(tip0[:3], abs=1.5)


def test_insert_depth_is_bounded(prims):
    r = ActionExecutor(build_registry(prims), mode="live").execute("insert", {"depth_mm": prims.m.insert_max_mm + 1})
    assert not r["ok"]


def test_insert_is_offered_to_programs_and_the_agent():
    from actions.astra_code import ROUTINE_TOOLS
    from agent.explore import build_explore
    assert "insert" in ROUTINE_TOOLS
    s = build_explore(mode="dry_run", client=object())
    assert "insert" in s.executor.specs


@pytest.mark.parametrize("axis,col,sign", [("+x", 0, 1), ("-y", 1, -1)])
def test_insert_along_another_gripper_axis(prims, axis, col, sign):
    import numpy as np
    from common.poses import pose_to_matrix
    tip0 = prims.get_state()["pose"]
    r = ActionExecutor(build_registry(prims), mode="live").execute("insert", {"depth_mm": 6, "axis": axis})
    assert r["ok"], r
    want = np.array(tip0[:3]) + pose_to_matrix(tip0)[:3, col] * sign * 6
    assert np.allclose(r["result"]["pose"][:3], want, atol=1.5) and r["result"]["axis"] == f"tool {axis}"


def test_insert_smallest_stroke_is_a_tenth_of_a_millimetre(prims):
    ex = ActionExecutor(build_registry(prims), mode="live")
    assert ex.execute("insert", {"depth_mm": 0.05})["ok"] is False
    assert ex.execute("insert", {"depth_mm": 0.1})["ok"] is True


def test_a_stroke_inside_the_arrival_tolerance_waits_for_the_move(prims):
    """24 Nov (episode 163425): a 0.5 mm insert counted as arrived before the controller started it."""
    prims.m.start_grace_s, prims.m.poll_s = 0.3, 0.01
    tip0 = prims.get_state()["pose"]
    movl, feedback, pending, polls = prims.robot.movl, prims.robot.feedback, [], []

    def late_movl(pose, speed, **kw):                  # the controller starts the queued move a few polls later
        pending.append((pose, speed, kw))

    def slow_feedback():
        polls.append(1)
        if pending and len(polls) > 5:
            pose, speed, kw = pending.pop()
            movl(pose, speed, **kw)
        return feedback()
    prims.robot.movl, prims.robot.feedback = late_movl, slow_feedback
    r = ActionExecutor(build_registry(prims), mode="live").execute("insert", {"depth_mm": 0.5})
    assert r["ok"], r
    import numpy as np
    assert np.linalg.norm(np.array(r["result"]["pose"][:3]) - tip0[:3]) == pytest.approx(0.5, abs=0.1)
