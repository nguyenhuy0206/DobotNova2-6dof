"""Mock world for harness evaluation: simulated Nova 2 + gripper, a fake two-camera
scene and a scripted "Astra" that replays a fixed action plan. No network, no robot.

The scripted model is the same for every variant (same actions, same order); only
the harness differs (context, chunking, images). Token counts here are ESTIMATES
(characters / 4 + a fixed cost per image) because there is no API — they compare the
size of what each variant would send, not billed tokens.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace as NS

import numpy as np

from actions import ActionExecutor, build_registry
from actions.primitives import PrimitiveSkills
from actions.robot_actions import RobotActions
from actions.skill_runner import SkillRunner
from agent.harness import AgentHarness, HarnessOptions
from common.config import AgentConfig, RobotConfig
from memory.skills import SkillLibrary
from memory.store import EpisodeStore, PositionStore
from perception.scene import SceneObject
from sim_robot import FakeGripper, SimNova2

IMAGE_TOKENS = {"low": 85, "high": 765, "auto": 765, None: 765}
ZONE_TOP = 162.0


class FakeScene:
    """look/find over a synthetic image that changes with the robot pose."""

    colors = ["blue", "green", "yellow"]
    sources = {"env": None, "wrist": None}

    def __init__(self, robot: SimNova2, gripper: FakeGripper | None = None):
        self.robot, self.gripper = robot, gripper
        self.blocks = {"blue": [-345.0, -480.0, 69.0], "green": [-328.0, -440.0, 69.0], "yellow": [-300.0, -360.0, 70.0]}
        self.held: str | None = None
        self.env_calibration = True

    def _update(self):
        """A caught block follows the fingertips; an opened gripper drops it."""
        tip = self.robot.state["pose"]
        caught = self.gripper is not None and self.gripper.status()["grip_state"] == 2
        if caught and self.held is None:
            near = [c for c, b in self.blocks.items() if np.hypot(b[0] - tip[0], b[1] - tip[1]) < 30]
            self.held = near[0] if near else None
        if not caught and self.held is not None:          # released where the fingers are now
            self.blocks[self.held] = [tip[0], tip[1], 42.0 + 27]
            self.held = None
        if self.held:
            self.blocks[self.held] = [tip[0], tip[1], tip[2] + 8]

    def find(self, color, camera="env", frame=None):
        from common.poses import pose_to_matrix
        self._update()
        x, y, z = self.blocks[color]
        if camera == "env":
            return [SceneObject(color, (int(400 + x / 10), int(300 + y / 10 - z / 2)), (0, 0, 16, 16), 220.0,
                                base_xyz_mm=[x, y, z], angle_deg=12.0)]
        pose = self.robot.state["pose"]
        R = pose_to_matrix(pose)[:3, :3]
        d = R.T @ (np.array([x, y, z]) - np.array(pose[:3]))     # block in the tool frame
        if abs(d[0]) > 60 or abs(d[1]) > 60:
            return []
        u, v = 355 - 5.6 * d[1], 140 + 2.6 * d[0]
        return [SceneObject(color, (int(u), int(v - 50)), (int(u - 55), int(v - 100), 110, 100), 9000.0,
                            angle_deg=12.0)]

    def _image(self, camera) -> str:
        import cv2
        pose = self.robot.state["pose"]
        img = np.zeros((120, 160, 3), np.uint8)
        cx, cy = int(80 + pose[1] / 8) % 160, int(60 + pose[2] / 8) % 120
        cv2.rectangle(img, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 200, 0) if camera == "wrist" else (200, 0, 0), -1)
        return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()

    def look(self, camera="env", colors=None, zoom=None, image="auto"):
        objs = [o.summary() for c in (colors or self.colors) for o in self.find(c, camera)]
        return {"camera": camera, "calibrated": camera == "env", "objects": objs,
                "image_jpeg_b64": self._image(camera)}

    def find_report(self, color, camera="env"):
        return {"objects": [o.summary() for o in self.find(color, camera)]}


def pick_place_plan(blocks=("green", "blue", "yellow")) -> list[tuple[str, dict]]:
    """The 181222 primitive sequence (per block), as the baseline Astra issued it."""
    plan = [("get_state", {}), ("look", {"camera": "env"})]
    for i, b in enumerate(blocks):
        x, y = {"blue": (-345, -480), "green": (-328, -440), "yellow": (-300, -360)}[b]
        plan += [("open_gripper", {}),
                 ("move_to_pose", {"x": x, "y": y, "z": 164, "rx": 180, "ry": 0, "rz": 70.9, "speed": 3}),
                 ("look", {"camera": "wrist"})]
        for dz in (-25, -25, -15):
            plan += [("move_relative", {"dz": dz, "speed": 2}), ("look", {"camera": "wrist"})]
        for dx, dy in ((0, 10), (10, 0), (0, -15), (-15, 0)):
            plan += [("move_relative", {"dx": dx, "dy": dy, "frame": "tool", "speed": 1}),
                     ("look", {"camera": "wrist"})]
        for dz in (-19, -19):
            plan += [("move_relative", {"dz": dz, "speed": 1}), ("look", {"camera": "wrist"})]
        plan += [("close_gripper", {}), ("move_relative", {"dz": 15, "speed": 2}),
                 ("look", {"camera": "env"})]
        for _ in range(3):
            plan += [("move_relative", {"dz": 23, "speed": 2}), ("look", {"camera": "wrist"})]
        plan += [("move_to_pose", {"x": -230, "y": -440 + 70 * i, "z": 165, "rx": 180, "ry": 0, "rz": 70.9,
                                   "linear": True, "speed": 3})]
        for _ in range(4):
            plan += [("move_relative", {"dz": -25, "speed": 2}), ("look", {"camera": "wrist"})]
        plan += [("open_gripper", {}), ("move_relative", {"dz": 25, "speed": 2}),
                 ("look", {"camera": "env"})]
        plan += [("move_relative", {"dz": 85, "speed": 3})]
    return plan


class ScriptedAstra:
    """Replays a plan. With chunking it proposes up to 4 next actions per turn and
    continues from whatever the harness actually executed."""

    def __init__(self, plan, chunk: int = 1):
        self.plan, self.chunk, self.pos, self.proposed = plan, chunk, 0, 0
        self.harness: AgentHarness | None = None
        self.requests: list[dict] = []
        self._n = 0

    def _estimate(self, kw) -> NS:
        images, text = 0, 0
        detail_tokens = 0

        def walk(x):
            nonlocal images, text, detail_tokens
            if isinstance(x, dict):
                if x.get("type") == "input_image":
                    images += 1
                    detail_tokens += IMAGE_TOKENS.get(x.get("detail"), 765)
                    return
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, str):
                text += len(x)

        walk(kw.get("input"))
        fixed = (len(kw.get("instructions") or "") + len(json.dumps(kw.get("tools") or []))) // 4
        new = text // 4 + detail_tokens
        prev = self.requests[-1]["context"] if (self.requests and kw.get("previous_response_id")) else 0
        context = prev + new + 60                          # chained: everything before is billed again
        inp = fixed + context
        self.requests.append({"input": inp, "context": context, "images": images})
        return NS(input_tokens=inp, output_tokens=60, total_tokens=inp + 60,
                  input_tokens_details=NS(cached_tokens=fixed if len(self.requests) > 1 else 0),
                  output_tokens_details=NS(reasoning_tokens=0))

    def create(self, **kw):
        usage = self._estimate(kw)
        if self.harness is not None and self.harness.last_turn and self.proposed:
            self.pos += self.harness.last_turn["executed"]     # deferred ones are proposed again
        elif self.proposed:
            self.pos += self.proposed
        self._n += 1
        rid = f"r{self._n}"
        if self.pos >= len(self.plan):
            self.proposed = 0
            return NS(id=rid, output=[], output_text="done: 3 blocks on the table", usage=usage)
        batch = self.plan[self.pos:self.pos + self.chunk]
        self.proposed = len(batch)
        calls = [NS(type="function_call", name=n, arguments=json.dumps(a), call_id=f"{rid}-{i}")
                 for i, (n, a) in enumerate(batch)]
        return NS(id=rid, output=calls, output_text="", usage=usage)


def skill_plan(blocks=("green", "blue", "yellow")) -> list[tuple[str, dict]]:
    """The same task with the distilled skills: one call per pick and per place."""
    pos = {"blue": (-345, -480, 69), "green": (-328, -440, 69), "yellow": (-300, -360, 70)}
    plan = [("get_state", {}), ("look", {"camera": "env"})]
    for i, b in enumerate(blocks):
        x, y, z = pos[b]
        plan += [("run_skill", {"name": "pick_block_wrist", "params": {"color": b, "x": x + 6, "y": y - 4,
                                                                         "z_top": z, "rz": 70.9}}),
                 ("run_skill", {"name": "place_block_on_table", "params": {"color": b, "x": -230,
                                                                             "y": -440 + 70 * i, "rz": 70.9}})]
    plan += [("look", {"camera": "env"})]
    return plan


def build(variant: str, tmp, plan=None, object_at: int | None = 776):
    sim = SimNova2()
    sim.active_tool = [0.0, 0.0, 0.0]          # controller FK available (feedback = flange)
    cfg = RobotConfig()
    cfg.motion.start_grace_s, cfg.motion.poll_s, cfg.motion.move_timeout_s = 0.0, 0.0, 0.2
    cfg.workspace.z = (-500.0, 700.0)
    cfg.workspace.x = cfg.workspace.y = (-700.0, 700.0)
    gripper = FakeGripper(object_at=object_at)
    prims = PrimitiveSkills(RobotActions(sim, cfg, PositionStore(tmp / "positions.json")), lambda: gripper)
    prims.grip_close_position, prims.grip_open_position = 500, 800
    scene = FakeScene(sim, gripper)
    runner = SkillRunner(SkillLibrary(tmp / "skills.json"), constants={"table_z": 42.0})
    ex = ActionExecutor(build_registry(prims, scene, runner), mode="live")
    runner.executor = ex
    from actions.local_steps import LocalSteps
    from agent.explore.pick_place_skills import ensure_pick_place_skills
    class SimTouch(LocalSteps):
        """The sim has no force feedback: touching = the nearest cube's top, as a real contact would report."""
        def routine_probe_top(self, memo, z_est, **kw):
            tip = sim.state["pose"]
            c = min(scene.blocks.values(), key=lambda b: np.hypot(b[0] - tip[0], b[1] - tip[1]))
            memo.setdefault("vars", {})["measured_top"] = c[2]
            sim.state["pose"] = [tip[0], tip[1], c[2] + 15, *tip[3:]]
            return {"ok": True, "measured_top": c[2]}
    runner.local = SimTouch(prims, scene, ex, settle_s=0)
    ensure_pick_place_skills(runner.library)
    opts = HarnessOptions.from_variant(variant)
    client = ScriptedAstra(plan or pick_place_plan(), chunk=opts.chunk.max_horizon if opts.chunk.enabled else 1)
    state = lambda: {"pose": [round(v, 1) for v in sim.state["pose"]], "approach_zone_top": ZONE_TOP,  # noqa: E731
                     "gripper": {"position": gripper.position}}
    h = AgentHarness(client, ex, AgentConfig(max_steps=400), EpisodeStore(tmp / "ep"), runner.library,
                     options=opts, state_fn=state)
    client.harness = h
    sim.state["pose"] = [-300.0, -400.0, 300.0, 180.0, 0.0, 70.9]
    h.scene = scene
    return h, client, sim


def run_variant(variant: str, tmp, plan: str = "primitives") -> dict:
    h, client, sim = build(variant, tmp, skill_plan() if plan == "skills" else None)
    report = h.run("Pick the blue, green and yellow cubes from the box and put them on the metal table.")
    m = h.last_metrics
    on_table = sorted(c for c, b in h.scene.blocks.items() if b[0] > -260)
    return {"variant": variant, "plan": plan, "report": report, "blocks_on_table": on_table, "model_calls": m["model_calls"],
            "est_input_tokens": m["input_tokens"], "est_total_tokens": m["total_tokens"],
            "max_context_tokens": m["max_context_tokens"], "images_sent": m["images_sent"],
            "images_withheld": m["images_withheld"], "actions": m["actions_executed"],
            "actions_per_call": m["actions_per_call"], "deferred": m["actions_deferred"],
            "failed": m["actions_failed"], "motions_sent": len(sim.motion_log)}
