"""Agent harness: model ⇄ tools loop with logging to an episode file.

    task ─► Astra ─► tool call(s) ─► chunk policy ─► ActionExecutor ─► event router
              ▲                                                            │
              └──── context (chain | bounded) + selected images ◄──────────┘

Variants (config: harness.variant, see HarnessOptions.from_variant):
  V0 baseline     previous_response_id chain, every image, every proposed call runs
  V1 compression  bounded context (state + digest + last turns), selective images
  V2 chunking     baseline context, adaptive action horizon (agent/chunking.py)
  V3 combined     V1 + V2
Safety never depends on the variant: every action still goes through the executor
and the primitives' checks. Every step and every model request's token usage is
appended to data/episodes/<id>.jsonl.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from actions.executor import ActionExecutor
from agent.astra_client import AstraClient
from agent.chunking import ChunkPolicy
from agent.context import ContextPolicy, WorkingMemory
from agent.repeats import RepeatGuard, signature
from agent.events import EventRouter, aborts_chunk, digest_line
from agent.metrics import EpisodeMetrics, usage_of
from agent.tools import to_openai_tools
from agent.vision import ImagePolicy
from common.config import AgentConfig
from common.log import get_logger
from memory.skills import SkillLibrary
from memory.store import EpisodeStore

log = get_logger(__name__)

DEFAULT_INSTRUCTIONS = (
    "You control a Dobot Nova 2 six-axis arm with a PGC gripper and an RGB-D camera. "
    "Only act through the provided tools; each motion tool blocks until it has finished. "
    "Perceive before acting: use look / find_objects to locate things (base_xyz_mm is in robot "
    "coordinates), and get_state before moving. Prefer existing skills (list_skills, run_skill) "
    "over long chains of primitives. After a task succeeds, verify with look, then save_skill "
    "the sequence that worked. Use conservative speeds. If a tool returns an error, do not retry "
    "blindly: call get_state and look, then explain or adjust."
    " Poses are FINGERTIP positions (the gripper length is handled for you). To work on an object, "
    "point the gripper straight down (rx=180, ry=0) and use the object's base_xyz_mm: hover at "
    "z = object z + 100, never below the object's top surface unless grasping."
    " The arm often hides objects from the env camera when it is above them: then use last_seen "
    "from find_objects, or call search_object (it rotates J1 back and forth and looks again)."
)

CHUNK_HINT = (
    "\n\nACTION CHUNKS: you may return up to {L} tool calls in one turn; they run in order, one after "
    "another. The harness runs fewer when that is safer (inside the approach zone, after a gripper "
    "action or skill, after any error) and tells you which calls were NOT executed. Batch routine "
    "steps (e.g. free-space move + look) instead of one call per turn."
)
BOUNDED_HINT = (
    "\n\nCONTEXT: every request is self-contained: TASK, STATE measured by the harness, a digest of the "
    "episode so far, your last turns and the results of your last calls. Earlier images are not "
    "repeated; look results always contain the detected objects as data, and an image is attached "
    "only when it adds information (look with image='always' if you need to see)."
)


STOP_TEXTS = ("not arrived after", "robot alarm during motion")


def motion_stopped(result: dict) -> str | None:
    """The safety layer stopped the robot during this call (primitives.MotionStopped), directly or
    inside a routine/skill that turned the error into text. Returns the message, or None."""
    if result.get("error_type") == "MotionStopped":
        return str(result.get("error"))[:200]
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    for text in (result.get("error"), inner.get("detail"), inner.get("error")):
        if text and any(t in str(text) for t in STOP_TEXTS):
            return str(text)[:200]
    return None


@dataclass
class HarnessOptions:
    variant: str = "V0"
    context: ContextPolicy = field(default_factory=lambda: ContextPolicy(mode="chain"))
    chunk: ChunkPolicy = field(default_factory=lambda: ChunkPolicy(enabled=False))
    images: ImagePolicy = field(default_factory=lambda: ImagePolicy(mode="all"))
    max_episode_tokens: int = 0        # 0 = no cap; else stop the episode when reached
    max_repeats: int = 0               # same action / hypothesis this many times in a row -> stop (0 = off)
    post_lift_image: bool = False      # record a wrist image once per grasp after lifting (log/review only)

    @classmethod
    def from_variant(cls, variant: str = "V0", **cfg) -> "HarnessOptions":
        variant = variant.upper()
        if variant not in ("V0", "V1", "V2", "V3"):
            raise ValueError("harness variant must be V0, V1, V2 or V3")
        bounded, chunked = variant in ("V1", "V3"), variant in ("V2", "V3")
        return cls(variant=variant,
                   context=ContextPolicy(mode="bounded" if bounded else "chain",
                                         recent_turns=int(cfg.get("recent_turns", 4)),
                                         summary_max_chars=int(cfg.get("summary_max_chars", 4000))),
                   chunk=ChunkPolicy(enabled=chunked, max_horizon=max(1, min(4, int(cfg.get("max_horizon", 4))))),
                   images=ImagePolicy(mode="selective" if bounded else "all",
                                      env_detail=cfg.get("env_image_detail", "low"),
                                      wrist_detail=cfg.get("wrist_image_detail", "high")),
                   max_episode_tokens=int(cfg.get("max_episode_tokens", 0) or 0),
                   max_repeats=int(cfg.get("max_repeats", 0) or 0),
                   post_lift_image=bool(cfg.get("post_lift_image", False)))


class AgentHarness:
    def __init__(self, client: AstraClient, executor: ActionExecutor, config: AgentConfig,
                 episodes: EpisodeStore | None = None, skills: SkillLibrary | None = None,
                 send_images: bool = True, notes=None, options: HarnessOptions | None = None,
                 state_fn=None, skill_runner=None, investigation=None, observer=None):
        self.client = client
        self.notes = notes          # memory.knowledge.KnowledgeStore (or legacy NotesStore): what earlier runs found
        self.observer = observer    # (name, args, result) -> {"id", "prediction_errors"?}: the harness records
                                    # every robot action as an observation, the model does not have to
        self.executor = executor
        self.config = config
        self.episodes = episodes or EpisodeStore()
        self.skills = skills or SkillLibrary()
        self.send_images = send_images
        self.options = options or HarnessOptions()
        self.state_fn = state_fn        # () -> compact measured state for the bounded context
        self.skill_runner = skill_runner
        self.investigation = investigation   # agent.explore.diagnosis.Investigation: failures -> hypotheses -> tests
        self.last_episode: str | None = None
        self.last_metrics: dict | None = None
        self.last_turn: dict = {}
        self.last_diagnosis: dict | None = None

    # ------------------------------------------------------------------ helpers
    def _take_image(self, episode: str, step: int, name: str, result: dict) -> str | None:
        """Remove a camera image from a tool result (keeps the text output small),
        save it next to the episode, and return it."""
        payload = result.get("result")
        if not isinstance(payload, dict) or "image_jpeg_b64" not in payload:
            return None
        image = payload.pop("image_jpeg_b64")
        path = self.episodes.save_image(episode, f"{step:03d}-{name}.jpg", image)
        payload["image_file"] = str(path)
        return image

    def _instructions(self, task: str) -> str:
        text = self.config.system_prompt or DEFAULT_INSTRUCTIONS
        # the library as verified facts only (name, version, status, runs of the current version):
        # descriptions are written experience and stay out of the prompt (list_skills NAME shows them)
        index = self.skills.index() if hasattr(self.skills, "index") else []
        if index:
            text += ("\n\nYour skill library (runs = camera-verified outcomes of the CURRENT version; "
                     "list_skills NAME shows its steps):\n" + "\n".join(
                         f"- {s['name']} v{s['version']} [{s['status']}] {s['ok']} ok / {s['fail']} failed · "
                         f"params {', '.join(s['params'])}" for s in index))
        if self.notes:
            lessons = (self.notes.digest() if self.options.context.mode == "bounded" and hasattr(self.notes, "digest")
                       else self.notes.recent_text())
            if lessons:
                header = getattr(self.notes, "prompt_header",
                                 "Your notes from earlier runs (facts you measured; re-check if they look wrong):")
                text += "\n\n" + header + "\n" + lessons
        if self.options.chunk.enabled:
            text += CHUNK_HINT.format(L=self.options.chunk.max_horizon)
        if self.options.context.mode == "bounded":
            text += BOUNDED_HINT
        return text

    def _moves(self, name: str) -> bool:
        spec = self.executor.specs.get(name)
        return spec is not None and (spec.moves_robot or spec.delegates)

    def _grip_status(self) -> str:
        st = self._state() or {}
        return str((st.get("gripper") or {}).get("grip_status") or "")

    def _holding(self) -> bool:
        return "caught" in self._grip_status()

    def _mark_drop(self, result: dict) -> None:
        """The gripper held an object before this call and reports it dropped after: a failure
        signal (gripper_reported_drop), even when the motion itself arrived (18 Nov: lift_to ok=true
        three times while the cube stayed behind, and diagnose had no failure to work on). It is the
        gripper's own report, not proof: once on 18 Nov it came while the cube was still carried."""
        st = (self._state() or {}).get("gripper") or {}
        if "dropped" not in str(st.get("grip_status") or ""):
            return
        inner = result.get("result")
        if not isinstance(inner, dict):
            inner = result["result"] = {}
        inner["ok"] = False
        inner["failure"] = "gripper_reported_drop"
        note = (f"the gripper held an object before this call and now reports 'object dropped' (fingers at "
                f"position {st.get('position')}); this is the gripper's own report — a camera can confirm it")
        inner["detail"] = f"{inner['detail']} — {note}" if inner.get("detail") else note

    def _post_lift_image(self, episode: str, step: int, held: dict) -> None:
        """Once per grasp, when the gripper holds something and has been lifted ≥ 30 mm since it
        caught it: a wrist image of how the object sits in the fingers. For the log, the
        observations and the review only — it is NOT put into the model's context."""
        st = self._state() or {}
        grip, pose = st.get("gripper") or {}, st.get("pose")
        z = pose[2] if isinstance(pose, list) and len(pose) > 2 else None
        if "caught" not in str(grip.get("grip_status", "")) or z is None:
            held.clear()
            return
        if "z0" not in held:
            held.update(z0=z, position=grip.get("position"))
            return
        if held.get("taken") or z - held["z0"] < 30 or "look" not in self.executor.specs:
            return
        held["taken"] = True
        args = {"camera": "wrist", "image": "always"}
        r = self.executor.execute("look", args)
        self._take_image(episode, step, "post-lift-wrist", r)
        self.episodes.append(episode, {"event": "tool", "step": step, "name": "look", "auto": True, "post_lift": True,
                                       "grip_position": held.get("position"), "lift_mm": round(z - held["z0"], 1),
                                       "args": args, "result": r})
        self._observe("look", {"camera": "wrist", "purpose": "post_lift"}, r)

    def _observe(self, name: str, args: dict, result: dict) -> dict | None:
        if self.observer is None:
            return None
        try:
            return self.observer(name, args, result)
        except Exception as exc:  # noqa: BLE001 — recording must never break the run
            log.warning("observation not recorded for %s: %s", name, exc)
            return None

    def _state(self) -> dict | None:
        if self.state_fn is None:
            return None
        try:
            st = self.state_fn()
        except Exception as exc:  # noqa: BLE001 — the state is context, never a reason to crash
            return {"state_error": str(exc)[:160]}
        diag = self.investigation.state() if self.investigation is not None else None
        if diag and isinstance(st, dict):
            st = {**st, "diagnosis": diag}
        return st

    def _create(self, episode: str, step: int, metrics: EpisodeMetrics, n_images: int, **kw):
        t0 = time.time()
        response = self.client.create(**kw)
        rec = metrics.add_request(usage_of(response), time.time() - t0, n_images)
        self.episodes.append(episode, {"event": "usage", "step": step, **rec})
        return response

    def _tool_zone(self) -> tuple[float | None, float | None, float | None]:
        """(fingertip z, approach-zone top, gripper position) from the measured state."""
        st = self._state() or {}
        pose = st.get("pose")
        grip = (st.get("gripper") or {}).get("position")
        return ((pose[2] if isinstance(pose, list) and len(pose) > 2 else None), st.get("approach_zone_top"),
                grip if isinstance(grip, (int, float)) else None)

    # ------------------------------------------------------------------ main loop
    def run(self, task: str, images: list[str] | None = None) -> str:
        """images: base64 JPEGs shown with the task (e.g. evidence for a review)."""
        opt = self.options
        from memory.store import driver_of
        episode = self.episodes.start(task, {"model": self.config.model, "mode": self.executor.mode,
                                             "variant": opt.variant, "driver": driver_of({"model": self.config.model})})
        self.last_episode = episode
        if hasattr(self.client, "start_episode"):       # a local model logs its own input/output per episode
            self.client.start_episode(self.episodes.root / episode)
        if self.skill_runner is not None:
            self.skill_runner.episode = episode
        if self.notes is not None and hasattr(self.notes, "episode"):
            self.notes.episode = episode            # evidence recorded now is tagged with this episode
        inv = self.investigation
        self.last_diagnosis = None
        if inv is not None:
            inv.start(episode)
        metrics = EpisodeMetrics(variant=opt.variant)
        router = EventRouter()
        repeats, repeat_stop = RepeatGuard(opt.max_repeats), None
        robot_stopped = None              # a move the safety layer stopped: the episode ends, no further model turn
        held: dict = {}                     # the current grasp: z when caught, whether its image was taken
        images_policy = ImagePolicy(**{k: getattr(opt.images, k) for k in ("mode", "env_detail", "wrist_detail",
                                                                           "same_threshold")})
        tools = to_openai_tools(self.executor.list())
        # The Responses API does NOT carry `instructions` over through previous_response_id,
        # so they are sent with every call (identical text -> served from the prompt cache).
        instructions = self._instructions(task)
        bounded = opt.context.mode == "bounded"
        wm = WorkingMemory(task, opt.context)

        def image_part(img: str, camera: str | None) -> dict:
            part = {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}"}
            if images_policy.mode == "selective":
                part["detail"] = images_policy.detail(camera or "env")
            return part

        start_images = [image_part(i, None) for i in images or []]
        if bounded:
            st0 = self._state()
            self.episodes.append(episode, {"event": "state", "step": 0, "state": st0})
            first = [{"role": "user", "content": [{"type": "input_text", "text": wm.build(st0)},
                                                  *start_images]}]
        else:
            first = task if not images else [{"role": "user", "content": [
                {"type": "input_text", "text": task}, *start_images]}]
        response = self._create(episode, 0, metrics, len(start_images), input=first, tools=tools,
                                instructions=instructions)
        reason, final = "max_steps", None

        for step in range(self.config.max_steps):
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            self.episodes.append(episode, {"event": "model", "step": step, "response_id": getattr(response, "id", None),
                                           "text": response.output_text, "calls": len(calls)})
            if not calls:
                reason, final = "done", response.output_text
                break
            metrics.tool_calls += len(calls)

            parsed = []
            for call in calls:
                try:
                    args, bad = json.loads(call.arguments or "{}"), None
                except json.JSONDecodeError:
                    # usually output cut off by max_output_tokens: tell the model, don't crash
                    args, bad = {"_raw": (call.arguments or "")[:200]}, True
                parsed.append(SimpleNamespace(name=call.name, call_id=call.call_id, parsed_args=args, bad=bad))
            tip_z, zone_top, grip = self._tool_zone() if opt.chunk.enabled else (None, None, None)
            run_now, deferred, why = opt.chunk.split(parsed, tip_z, zone_top, grip)

            outputs, sent_images, executed, digests = [], [], [], []
            aborted = False
            for i, call in enumerate(run_now):
                if aborted:
                    deferred = list(run_now[i:]) + deferred
                    why = why or "an earlier call in this turn failed or needs your attention"
                    break
                if call.bad:
                    result = {"ok": False, "action": call.name, "executed": False,
                              "error": "arguments were not valid JSON (cut off?) — send a shorter call"}
                else:
                    if call.name == "run_skill":
                        router.handling = ((call.parsed_args.get("params") or {}).get("color") or router.handling)
                    refusal = inv.before(call.name, call.parsed_args) if inv is not None else None
                    if refusal:
                        result = {"ok": False, "action": call.name, "executed": False, "error": refusal}
                    else:
                        holding = self._moves(call.name) and call.name != "open_gripper" and self._holding()
                        result = self.executor.execute(call.name, call.parsed_args)
                        if holding:
                            self._mark_drop(result)
                        diag = inv.after(call.name, call.parsed_args, result) if inv is not None else None
                        if diag:
                            result["diagnosis"] = diag
                        seen = self._observe(call.name, call.parsed_args, result)
                        if seen:
                            result["observation"] = seen["id"]
                            if seen.get("prediction_errors"):
                                result["prediction_errors"] = seen["prediction_errors"]
                metrics.add_action(call.name, result)
                if opt.post_lift_image and result.get("executed", True) and self._moves(call.name):
                    self._post_lift_image(episode, step, held)
                if repeats.limit > 0 and signature(call.name, call.parsed_args, result) is not None:
                    repeat_stop = repeat_stop or repeats.add(call.name, call.parsed_args, result, self._state())
                image = self._take_image(episode, step, call.name, result)
                events = router.classify(call.name, call.parsed_args, result)
                if image is not None and self.send_images:
                    camera = (result.get("result") or {}).get("camera")
                    send, img_reason = images_policy.decide(call.parsed_args, result.get("result") or {},
                                                            router.wants_image(camera))
                    if send:
                        images_policy.sent(camera, image)
                        router.image_delivered(camera)
                        sent_images.append((image, camera))
                    else:
                        metrics.images_withheld += 1
                    result["result"]["image"] = ("attached" if send else "not attached") + f" ({img_reason})"
                log.info("tool %s(%s) -> ok=%s", call.name, call.parsed_args, result.get("ok"))
                self.episodes.append(episode, {"event": "tool", "step": step, "name": call.name,
                                               "args": call.parsed_args, "result": result,
                                               "events": [e.as_dict() for e in events]})
                executed.append((call.name, call.parsed_args, result))
                digests.append(digest_line(step, call.name, call.parsed_args, result, events))
                if result.get("ok") and call.name == "run_skill" and (result.get("result") or {}).get("ok"):
                    p = (call.parsed_args.get("params") or {})
                    if "place" in call.parsed_args.get("name", "") and p.get("color"):
                        wm.progress.append(f"{p['color']} placed at ({p.get('x')}, {p.get('y')}) — skill verified")
                outputs.append({"type": "function_call_output", "call_id": call.call_id,
                                "output": json.dumps(result, ensure_ascii=False, default=str)})
                robot_stopped = robot_stopped or motion_stopped(result)
                if (opt.chunk.enabled and aborts_chunk(events)) or repeat_stop or robot_stopped:
                    aborted = True
            self.last_turn = {"executed": len(executed), "deferred": len(deferred), "reason": why}
            if deferred:
                metrics.actions_deferred += len(deferred)
                self.episodes.append(episode, {"event": "deferred", "step": step, "reason": why,
                                               "calls": [c.name for c in deferred]})
                for c in deferred:
                    outputs.append({"type": "function_call_output", "call_id": c.call_id,
                                    "output": json.dumps({"ok": False, "executed": False,
                                                          "error": f"not executed: {why}"})})
            # camera images the router asked for but no look delivered: say so in the context
            if robot_stopped:
                reason, final = "robot_stopped", f"Stopped: the safety layer stopped the robot ({robot_stopped})"
                break
            if router.stop_reason:
                reason, final = "robot_alarm", f"Stopped: {router.stop_reason}"
                break
            if repeat_stop:
                reason, final = "repeated", f"Stopped: {repeat_stop}"
                break
            if opt.max_episode_tokens and metrics.tokens_so_far() >= opt.max_episode_tokens:
                reason, final = "token_budget", (f"Stopped: token budget {opt.max_episode_tokens} reached "
                                                 f"({metrics.tokens_so_far()} used).")
                break

            # Something unexpected and no fresh image of it yet: the harness looks itself
            # (read-only, no model call) instead of making Astra spend a turn on it.
            auto = []
            if bounded and self.send_images and "look" in self.executor.specs:
                for camera in sorted(router.pending_image):
                    colors = [router.handling] if router.handling else None
                    r = self.executor.execute("look", {"camera": camera, **({"colors": colors} if colors else {})})
                    img = self._take_image(episode, step, f"auto-look-{camera}", r)
                    router.classify("look", {"camera": camera, "colors": colors or []}, r)
                    router.image_delivered(camera)
                    self.episodes.append(episode, {"event": "tool", "step": step, "name": "look", "auto": True,
                                                   "args": {"camera": camera, "colors": colors}, "result": r})
                    if img is not None:
                        images_policy.sent(camera, img)
                        sent_images.append((img, camera))
                        r["result"]["image"] = "attached (taken by the harness after the unexpected event)"
                    auto.append((f"look[auto:{camera}]", {"camera": camera}, r))
                metrics.auto_looks += len(auto)

            img_parts = [image_part(img, cam) for img, cam in sent_images]
            if bounded:
                wm.record_turn(step, response.output_text, executed + auto, deferred, why, digests)
                extra = ""
                st = self._state()
                self.episodes.append(episode, {"event": "state", "step": step + 1, "state": st})
                msg = [{"role": "user", "content": [{"type": "input_text", "text": wm.build(st, extra)},
                                                    *img_parts]}]
                response = self._create(episode, step + 1, metrics, len(img_parts), input=msg, tools=tools,
                                        instructions=instructions)
            else:
                if img_parts:
                    # Images go in a user message right after the tool outputs.
                    outputs.append({"role": "user", "content": [
                        {"type": "input_text", "text": "Camera image(s) from the tool call(s) above:"}, *img_parts]})
                response = self._create(episode, step + 1, metrics, len(img_parts), input=outputs, tools=tools,
                                        previous_response_id=response.id, instructions=instructions)

        if final is None:
            final = f"Stopped after max_steps={self.config.max_steps}. Last: {response.output_text}"
        if inv is not None and inv.failures:
            self.last_diagnosis = inv.summary()
            self.episodes.append(episode, {"event": "diagnosis", **self.last_diagnosis})
        self.episodes.append(episode, {"event": "end", "reason": reason})
        self.last_metrics = metrics.summary()
        self.episodes.append(episode, {"event": "metrics", **self.last_metrics})
        return final
