"""Going round in circles ends the episode: same action / hypothesis 3× in a row."""

from agent.repeats import RepeatGuard

OK = {"ok": True, "result": {}}


def test_three_identical_actions_in_a_row_stop():
    g = RepeatGuard(3)
    assert g.add("open_gripper", {"force": 30}, OK) is None
    assert g.add("look", {"camera": "wrist"}, OK) is None            # looking neither counts nor breaks the run
    assert g.add("open_gripper", {"force": 30}, OK) is None
    assert "action 3× in a row: open_gripper" in g.add("open_gripper", {"force": 30.0}, OK)


def test_a_change_resets_the_count():
    g = RepeatGuard(3)
    for dz in (-25, -25, 25, -25, -25):                                 # up / down is not the same action
        assert g.add("move_relative", {"dz": dz}, OK) is None
    assert g.add("move_relative", {"dz": -25.04}, OK)                    # 0.1 rounding: same as -25


def test_progress_is_not_a_repeat_but_an_unchanged_state_is():
    g = RepeatGuard(3)
    for z in (99.0, 124.0, 149.0):                                       # lifting in 25 mm steps (17 Nov)
        assert g.add("move_relative", {"dz": 25}, OK, {"pose": [-311.4, -305.2, z, 180, 0, 75.5]}) is None
    at = {"pose": [-318.0, -334.0, 170.0, 180, 0, 75.5], "gripper": {"position": 1000}}
    g.add("open_gripper", {"force": 30}, OK, at)
    g.add("open_gripper", {"force": 30}, OK, at)
    assert "open_gripper" in g.add("open_gripper", {"force": 30}, OK, at)


def test_refused_calls_count_too_and_hypotheses_by_their_knowledge_id():
    g = RepeatGuard(3)
    bad = {"ok": False, "error": "no failure 'F3 go_to_position:error'"}
    g.add("diagnose", {"failure_id": "F3 x", "why": "first wording"}, bad)
    g.add("diagnose", {"failure_id": "F3 x", "why": "second wording"}, bad)
    assert "diagnose refused" in g.add("diagnose", {"failure_id": "F3 x", "why": "third"}, bad)
    h = RepeatGuard(3)
    same = {"ok": True, "result": {"hypotheses": [{"knowledge": {"id": "C2.1"}}]}}
    h.add("diagnose", {"why": "a"}, same)
    h.add("propose_hypothesis", {"concept_id": "C2"}, {"ok": True, "result": {"id": "C2.1"}})
    assert "hypothesis 3× in a row: hypothesis C2.1" in h.add("diagnose", {"why": "b"}, same)


def test_off_by_default():
    assert RepeatGuard(0).add("stop", {}, OK) is None
