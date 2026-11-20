"""Motion must be refused unless the robot is idle, enabled and alarm-free."""
import pytest

from common.errors import SafetyError, ValidationError


@pytest.mark.parametrize("field,value", [("mode", 9), ("enabled", 0), ("error", 1), ("running", 1)])
def test_motion_refused_when_not_ready(actions, sim, field, value):
    sim.state[field] = value
    with pytest.raises(SafetyError):
        actions.joint_step(1, 5, 10)
    with pytest.raises(SafetyError):
        actions.move_joints([0] * 6, 10)
    assert sim.motion_log == []


def test_joint_step_sends_relative_move(actions, sim):
    actions.joint_step(2, -3.5, 15)
    assert sim.motion_log == ["RelJointMovJ(0,-3.5,0,0,0,0,SpeedJ=15,AccJ=15)"]


def test_validation_happens_before_feedback(actions, sim):
    sim.state["mode"] = 9  # would be a SafetyError, but limits are checked first
    with pytest.raises(ValidationError):
        actions.joint_step(1, 50, 10)


def test_speed_zero_locks_motion(actions, sim):
    actions.set_speed(0)
    assert "SpeedFactor" not in " ".join(sim.dashboard_log)
    with pytest.raises(SafetyError, match="Motion lock"):
        actions.joint_step(1, 5, 10)
    actions.set_speed(20)
    actions.joint_step(1, 5, 10)
    assert len(sim.motion_log) == 1


def test_saved_position_roundtrip(actions, sim):
    positions = actions.capture_position("home")
    actions.move_to_saved(positions[0]["id"], 10)
    assert sim.motion_log[0].startswith("JointMovJ(0.0000,0.0000,90.0000")
    assert actions.delete_position(positions[0]["id"]) == []


def test_gripper_requires_connect(actions):
    from common.errors import RobotError
    with pytest.raises(RobotError):
        actions.gripper_open()
    actions.gripper_connect()
    actions.gripper_open()


def test_stop_does_not_wait_for_lock(actions, sim):
    with actions.lock:
        import threading
        done = threading.Event()
        threading.Thread(target=lambda: (actions.system("stop"), done.set()), daemon=True).start()
        assert done.wait(1.0)
    assert sim.dashboard_log[-1] == "Pause()"


def test_stop_never_resets_unless_every_stop_command_is_refused():
    """19 Nov (operator): no ResetRobot() to the control box; Stop() is not in this firmware."""
    import pytest

    from common.errors import RobotError
    from drivers.nova2 import Nova2
    sent = []

    class Box(Nova2):
        def __init__(self, accept):
            self.accept = accept

        def dashboard(self, command):
            sent.append(command)
            if command not in self.accept:
                raise RobotError(f"refused {command}")
            return f"0,{{}},{command};"

    assert Box({"Pause()"}).stop() == "0,{},Pause();" and sent == ["Pause()"]
    sent.clear()
    assert Box({"ResetRobot()"}).stop() == "0,{},ResetRobot();" and sent == ["Pause()", "ResetRobot()"]
    with pytest.raises(RobotError):
        Box(set()).stop()


def test_a_stopped_move_ends_the_episode():
    from agent.harness import motion_stopped
    assert motion_stopped({"ok": False, "error_type": "MotionStopped", "error": "x: not arrived after 30s — stopped."})
    assert motion_stopped({"ok": True, "result": {"ok": False, "detail": "move_relative: not arrived after 30s — stopped."}})
    assert motion_stopped({"ok": False, "error": "move_relative: CONTACT at fingertip Z=68.3"}) is None   # contact goes on
    assert motion_stopped({"ok": True, "result": {"pose": [0, 0, 0]}}) is None
