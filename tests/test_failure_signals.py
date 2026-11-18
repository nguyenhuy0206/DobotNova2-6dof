"""Harness gaps found on 17 Nov: a missed grasp must count as a failure, the review must see the
wrist images around it, and an image angle is only reported when it can be measured."""
import base64

import numpy as np

from agent.diagnosis import Investigation
from memory.knowledge import observe_call
from perception.detection import detect_colored_objects

MISS = {"ok": True, "result": {"grip_state": 1, "position": 500, "holding": False}}


def _square(img, cx, cy, side, angle):
    import cv2
    box = cv2.boxPoints(((cx, cy), (side, side), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], (255, 0, 0))                      # BGR blue


def test_angle_only_when_measurable():
    img = np.full((480, 640, 3), 255, np.uint8)
    _square(img, 320, 240, 120, 30)                            # big, inside the frame: an angle
    _square(img, 100, 100, 22, 30)                             # env-sized blob: no angle
    _square(img, 600, 20, 120, 30)                             # cut by the top/right border
    dets = {(d.center[0] // 100): d for d in detect_colored_objects(img, ["blue"], min_area=300)}
    assert dets[3].angle_deg is not None and abs(dets[3].angle_deg - 30) < 3 and not dets[3].cut_by_edge
    assert dets[1].angle_deg is None                           # too small to show its edges
    edge = next(d for d in dets.values() if d.cut_by_edge)
    assert edge.angle_deg is None


def test_a_missed_grasp_is_a_failure_for_diagnosis_and_observations():
    inv = Investigation(gate_after_repeats=2)
    f = inv.after("close_gripper", {"force": 30}, MISS)
    assert f["failure_id"] == "F1" and inv.failures[0]["code"] == "grasp_missed"
    assert "without holding" in inv.failures[0]["detail"]
    assert inv.diagnose("closed on nothing at z 76", [{"cause": "c", "test": "t", "expect": "e"}])["failure"] == "F1"
    assert inv.after("close_gripper", {}, {"ok": True, "result": {"holding": True, "position": 657}}) is None
    v = observe_call("close_gripper", {"force": 30}, MISS)
    assert v["ok"] is False and v["failure"] == "grasp_missed"


def test_review_sees_the_wrist_images_before_and_after_a_missed_grasp(tmp_path):
    from agent.learning import failure_images
    from memory.store import EpisodeStore

    def jpg(name):
        p = tmp_path / name
        p.write_bytes(b"\xff\xd8" + name.encode())
        return str(p)

    store = EpisodeStore(tmp_path / "ep")
    ep = store.start("t", {})
    look = lambda step, f, auto=False: {"event": "tool", "step": step, "name": "look", "auto": auto,
                                        "args": {"camera": "wrist"}, "result": {"ok": True, "result": {"image_file": f}}}
    for e in (look(1, jpg("hover.jpg")),
              {"event": "tool", "step": 2, "name": "move_relative", "args": {"dz": -20}, "result": {"ok": True}},
              {"event": "tool", "step": 3, "name": "close_gripper", "args": {"force": 30}, "result": MISS},
              look(3, jpg("after.jpg"), auto=True)):
        store.append(ep, e)
    shots = failure_images(store, ep)
    assert [c.split("[")[-1] for c, _ in shots] == ["BEFORE]", "AFTER]"]
    assert "grasp missed (closed to 500)" in shots[0][0]
    assert base64.b64decode(shots[1][1]).endswith(b"after.jpg")


def test_post_lift_wrist_image_is_recorded_but_not_shown_live(tmp_path):
    """E/F: once per grasp, after lifting while holding — into the log and the review, not the context."""
    import json
    from types import SimpleNamespace as NS

    from actions.executor import ActionExecutor, ActionSpec
    from actions.registry import _obj
    from agent.harness import AgentHarness, HarnessOptions
    from agent.learning import post_lift_images
    from common.config import AgentConfig
    from memory.skills import SkillLibrary
    from memory.store import EpisodeStore

    img = base64.b64encode(b"\xff\xd8wrist").decode()
    st = {"z": 76.0, "grip": "arrived (no object)", "looks": 0}

    def look(camera="env", image="auto", colors=None):
        st["looks"] += 1
        return {"camera": camera, "objects": [], "image_jpeg_b64": img}

    def close(force=30, speed=20):
        st["grip"] = "object caught"
        return {"position": 657, "holding": True, "grip_state": 2}

    def move(dz=0):
        st["z"] += dz
        return {"pose": [0, 0, st["z"], 180, 0, 0]}

    ex = ActionExecutor([
        ActionSpec("look", "look", look, _obj({"camera": {"type": "string"}, "image": {"type": "string"},
                                               "colors": {"type": "array"}}, []), read_only=True),
        ActionSpec("close_gripper", "close", close, _obj({"force": {"type": "integer"}, "speed": {"type": "integer"}}, []),
                   moves_robot=True),
        ActionSpec("move_relative", "move", move, _obj({"dz": {"type": "number"}}, []), moves_robot=True)], mode="live")

    def call(name, args, cid):
        return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid)

    class Client:
        def __init__(self, rs):
            self.rs, self.inputs = list(rs), []

        def create(self, **kw):
            self.inputs.append(json.dumps(kw.get("input"), default=str))
            return self.rs.pop(0)

    turn = lambda i, *c: NS(id=f"r{i}", output=list(c), output_text="", usage=None)
    client = Client([turn(1, call("close_gripper", {}, "a")), turn(2, call("move_relative", {"dz": 25}, "b")),
                     turn(3, call("move_relative", {"dz": 25}, "c")), turn(4, call("move_relative", {"dz": 25}, "d")),
                     NS(id="r5", output=[], output_text="done", usage=None)])
    episodes = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(client, ex, AgentConfig(max_steps=8), episodes, SkillLibrary(tmp_path / "s.json"),
                     options=HarnessOptions.from_variant("V3", post_lift_image=True),
                     state_fn=lambda: {"pose": [0, 0, st["z"], 180, 0, 0], "gripper": {"position": 657,
                                                                                      "grip_status": st["grip"]}})
    h.run("lift it")
    assert st["looks"] == 1                                        # once per grasp, after ≥ 30 mm
    shots = post_lift_images(episodes, h.last_episode)
    assert len(shots) == 1 and "after lifting 50.0 mm, gripper position 657" in shots[0][0]
    assert not any("post-lift" in i or "image_url" in i for i in client.inputs)      # never in the live context
