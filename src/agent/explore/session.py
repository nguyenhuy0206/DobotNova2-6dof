"""Explore mode: Astra learns the cell by itself from primitive tools.

    session = build_explore()          # config/explore.yaml + agent.yaml + robot.yaml
    problems = session.preflight()     # robot ready, table taught, env calibrated, cameras; marker if guard enabled
    session.harness.run(session.cfg.task)

What is different from the plain agent (agent/toolkit.py):
  * only primitives + seeing + learning tools (no seed skills): every composite skill is one
    Astra wrote itself,
    kept in its own file (skills_explore.json)
  * hard limits enforced in code, whatever the model asks:
      agent.yaml fixed_speed (50 %), or max_speed cap when unset · fingertip/flange above
      the taught table along the whole path · contact only sends Pause() (20 Nov) · gripper
      never closes beyond gripper.close_position; optional view guard is disabled in explore.yaml
  * the cameras, table and limits are calibrated by the operator (scripts/calibrate.sh); the
    calibration stays in use until the operator calibrates again
  * knowledge (memory/knowledge.py): the harness records every robot action as an observation;
    Astra builds physical concepts and hypotheses with predictions, checked against observations
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from actions.local_steps import LocalSteps
from actions.primitives import create_primitives
from agent.astra_client import AstraClient
from agent.explore.config import ExploreConfig, load_explore_config
from agent.explore.diagnosis import Investigation
from agent.explore.diagnosis import specs as diagnosis_specs
from agent.explore.prompt import for_driver, instructions
from agent.explore.tools import reference_spec, code_specs, explore_specs
from agent.harness import AgentHarness, HarnessOptions
from agent.toolkit import Toolkit, build_toolkit
from calibration.workcell import load_workcell
from common.config import load_agent_config, load_perception_config, load_robot_config
from common.errors import RobotError, ValidationError
from common.log import get_logger
from common.paths import REPO_ROOT
from memory.knowledge import KnowledgeStore
from memory.notes import NotesStore
from memory.skills import SkillLibrary
from perception.robot_view import RobotViewGuard
from perception.scene import Scene

log = get_logger(__name__)


@dataclass
class ExploreSession:
    cfg: ExploreConfig
    kit: Toolkit
    executor: object
    harness: AgentHarness
    view_guard: RobotViewGuard | None
    notes: NotesStore                      # raw notes: read-only archive, never put into prompts
    investigation: Investigation | None = None
    knowledge: KnowledgeStore | None = None

    def preflight(self) -> list[str]:
        """Everything that must be true before Astra may move the robot. [] = ready."""
        problems = []
        p = self.kit.primitives
        try:
            state = p.get_state()
            if state["mode"] != 5 or state["error"]:
                problems.append(f"robot not ready (mode {state['mode']}, alarm={state['error']}): "
                                "Power on → Enable (Debug tab), clear alarms")
        except (RobotError, OSError) as exc:
            problems.append(f"robot unreachable: {exc}")
        if p.table_z_mm is None:
            problems.append("table height not taught: run  scripts/calibrate.sh table  (or survey table)")
        needs_depth = {"depth_at", "locate_object", "pick_object", "place_object"} & set(self.cfg.program_tools or [])
        if "env" in self.cfg.cameras and needs_depth and self.kit.scene.env_calibration is None:
            problems.append("env camera not calibrated: run  scripts/calibrate.sh cameras")
        if getattr(p.cfg.gripper, "via_ui", False):
            try:
                p.gripper.status()
            except RobotError as exc:
                problems.append(str(exc))
        for name in getattr(self.cfg, "required_positions", []) or []:
            if self.kit.primitives.a.positions.find_by_name(name) is None:
                problems.append(f"saved position '{name}' missing: put the robot at the aligned start pose and "
                                f"save it as '{name}' (Debug tab -> Save position)")
        task = getattr(self.cfg, "reference_task", "")
        if task:
            from perception.reference import LABELS, load_references
            refs = load_references(task)
            empty = [label for label in LABELS if not refs[label]]
            if empty:
                problems.append(f"no reference pictures for {empty} ({task}): at the start pose run  "
                                f".venv/bin/python scripts/dev/capture_reference.py --task {task} --label <success|fail>")
        for camera in self.cfg.cameras:
            try:
                self.kit.scene.observe(camera)
            except (RobotError, OSError) as exc:
                problems.append(f"{camera} camera: {exc}  (is scripts/run_vision.sh running?)")
        if self.view_guard is not None and not problems:
            info = self.view_guard()
            if not info["visible"]:
                problems.append(f"{info['detail']} — wrap {self.cfg.view_guard.marker_color} tape around the "
                                "gripper body so the D455 sees it (or view_guard.enabled: false)")
        return problems

    def run_episode(self, task: str, reflect: bool = True, self_check: bool = False) -> dict:
        """One attempt (robot moves), then Astra reviews it (no robot). Success comes from the judge ledger;
        the review records Astra's claim only. Program mode: the prompt starts with the previous attempt's report."""
        prompt = task.strip()
        if self_check:
            if self.cfg.driver != "astra":
                raise ValueError("autonomous self-check requires the Astra driver")
            if self.executor.stopped:
                raise RobotError(f"autonomous retry refused: {self.executor.stopped}; operator resume required")
            # 24 Nov: the task says the robot starts at the start position; on a retry it does not
            starts = list(getattr(self.cfg, "allowed_positions", None) or [])
            restart = (f" go_to_position({starts[0]}) starts over from the start position." if len(starts) == 1
                       else "")
            prompt += ("\n\nAUTONOMOUS RETRY: No operator label or scene reset between episodes. The robot is "
                       "where the previous attempt left it, not at the start position the task names; observe "
                       f"first.{restart} Inspect after an attempt and report visual evidence. A separate "
                       "read-only Astra check decides whether to continue.")
        if getattr(self.cfg, "program_mode", False) and self.cfg.driver == "astra":
            from agent.explore.attempts import last_astra_episode, report
            prev = last_astra_episode(self.harness.episodes)
            start = next((e for e in self.harness.episodes.load(prev) if e.get("event") == "start"), {}) if prev else {}
            # 23 Nov: a new task must not start from an attempt at a different task
            if prev and " ".join(str(start.get("task", "")).split()).startswith(" ".join(task.split())):
                prompt += "\n\n" + report(self.harness.episodes, self.harness.judge.ledger, prev)
        if self.knowledge is not None and self.knowledge.obs.current and getattr(self.cfg, "show_knowledge", True):
            items = self.knowledge.obs.current
            concepts = [c for c in self.knowledge.visible() if not c.get("archived")]
            live = [h for c in concepts for h in c["hypotheses"] if not h.get("archived")]
            prompt += (f"\nKnowledge record: {len(items)} recorded observations from "
                       f"{len({o.get('episode') for o in items})} episodes (query_observations), "
                       f"{len(concepts)} concepts, {len(live)} hypotheses.")
        images = []
        if getattr(self.cfg, "goal_picture", False) and getattr(self.cfg, "reference_task", ""):
            from perception.reference import load_references
            success = load_references(self.cfg.reference_task)["success"]
            if success:
                images = [self.kit.scene.encode_image(success[-1][1], [])]       # newest, 640 px wide
                prompt += ("\nThe attached picture is the operator's reference of the finished state, taken with "
                           "the wrist camera (sent once, here).")
        self.harness.keep_going = self._keep_going(task) if getattr(self.cfg, "keep_going", False) else None
        self._verified = None
        report = self.harness.run(prompt, images=images) if images else self.harness.run(prompt)
        self.harness.keep_going = None
        out = {"episode": self.harness.last_episode, "report": report, "metrics": self.harness.last_metrics,
               "diagnosis": self.investigation.report_text() if self.investigation else ""}
        if self_check:
            from agent.explore.checking import check_task, retry_stop_reason
            stopped = retry_stop_reason(self, out["episode"])
            if stopped or self.executor.mode == "dry_run":
                out["retry_stop"] = stopped or "dry_run does not verify physical success"
            else:
                out["self_check"] = self._verified or check_task(self, task, out["episode"])
        if reflect and self.cfg.driver == "astra":     # the review writes Astra's memory: Astra's runs only
            out.update(self.reflect(out["episode"]))
        return out

    def _keep_going(self, task: str):
        """24 Nov (operator): "keep trying until the calls or tokens run out; stopping is waste — verify, then keep
        doing". When Astra answers without a tool call, a separate check of fresh pictures decides: verified
        success ends the episode, anything else goes back to Astra with the check's evidence."""
        from agent.explore.checking import check_task

        def keep_going(episode: str, text: str) -> str | None:
            if self.executor.mode == "dry_run" or self.executor.stopped:
                return None
            check = check_task(self, task, episode)
            if check.get("status") == "success":
                self._verified = check
                return None
            return ("NOT FINISHED — keep going. You stopped, so an independent check looked at fresh pictures: "
                    f"{check.get('status')} — {' '.join(str(check.get('evidence', '')).split())[:600]} "
                    "The attempt ends only when the model calls or tokens run out, or when a check verifies "
                    "success. Make the next correction now with a motion, not a report.")
        return keep_going

    def reflect(self, episode_id: str) -> dict:
        import copy

        from agent.explore.learning import (REFLECT_INSTRUCTIONS, REFLECT_INSTRUCTIONS_PROGRAM, REFLECT_TOOLS,
                                            REFLECT_TOOLS_PROGRAM, OutcomeRecorder, failure_images, post_lift_images,
                                            reflect_executor, summarize_episode)
        from memory.store import EpisodeStore

        if self.harness.episodes.driver(episode_id) != "astra":
            raise ValueError(f"{episode_id} was not run by Astra (driver {self.harness.episodes.driver(episode_id)}): "
                             "it is not reviewed into Astra's memory")
        recorder = OutcomeRecorder(self.harness.episodes, ledger=self.harness.judge.ledger)
        recorder.episode_id = episode_id
        cfg = copy.copy(self.harness.config)
        # ONE reflection pass with a few memory calls; never more than reflect_max_steps requests
        program = bool(getattr(self.cfg, "program_mode", False))
        cfg.system_prompt = REFLECT_INSTRUCTIONS_PROGRAM if program else REFLECT_INSTRUCTIONS
        cfg.max_steps, cfg.allowed_tools = self.cfg.reflect_max_steps, []
        cfg.max_output_tokens = max(cfg.max_output_tokens, 6000)
        tools = REFLECT_TOOLS_PROGRAM if program else REFLECT_TOOLS
        reviewer = AgentHarness(self.harness.client, reflect_executor(self.executor, recorder, tools), cfg,
                                episodes=EpisodeStore(self.harness.episodes.root / "reflections"),
                                skills=self.harness.skills, send_images=False, notes=self.knowledge, judge=False,
                                options=HarnessOptions.from_variant("V3" if self.harness.options.variant in ("V1", "V3")
                                                                    else "V0"))
        shots = failure_images(self.harness.episodes, episode_id)
        log_text = "Episode log:\n" + summarize_episode(self.harness.episodes, episode_id)
        if self.knowledge is not None:
            log_text += "\n\nKnowledge check of this episode:\n" + self.knowledge.episode_summary(episode_id)
        held = post_lift_images(self.harness.episodes, episode_id)
        if shots:
            log_text += ("\n\nWrist images around failed grasps / failed runs (BEFORE = the last wrist image taken "
                         "before it, AFTER = right after it; gripper body at the bottom, fingers left/right):\n"
                         + "\n".join(f"#{i + 1}: {c}" for i, (c, _) in enumerate(shots)))
        if held:
            log_text += ("\n\nWrist images taken by the harness after each lift while holding (how the object sits "
                         "in the fingers):\n" + "\n".join(f"#{len(shots) + i + 1}: {c}" for i, (c, _) in enumerate(held)))
        if self.knowledge is not None:
            self.knowledge.focus_episode = episode_id      # this_episode = the reviewed run, not the review
        try:
            text = reviewer.run(log_text, images=[b for _, b in shots] + [b for _, b in held])
        finally:
            if self.knowledge is not None:
                self.knowledge.focus_episode = None
        return {"reflection": text, "outcome": recorder.last, "reflection_metrics": reviewer.last_metrics}

    def set_global_speed(self) -> str:
        """Controller SpeedFactor (global %, multiplies every command's speed) to the explore setting.
        20 Nov 182631: it had been lowered on the controller; commands at 2 % then crawled at ~0.1 %."""
        level = getattr(self.cfg, "speed_factor", 0)
        if not level:
            return "unchanged"
        try:
            self.kit.primitives.robot.speed_factor(int(level))
            return f"{int(level)} %"
        except (RobotError, OSError) as exc:
            log.warning("SpeedFactor failed: %s", exc)
            return f"NOT set ({exc})"

    def arm_collision_detection(self) -> str:
        if not self.cfg.collision_level:
            return "unchanged"
        try:
            self.kit.primitives.robot.set_collision_level(self.cfg.collision_level)
            return f"level {self.cfg.collision_level}"
        except (RobotError, OSError) as exc:
            log.warning("SetCollisionLevel failed: %s", exc)
            return f"NOT set ({exc})"

def compact_state(p, executor=None, runner=None) -> dict:
    """Measured state for the bounded context (read locally, no model call)."""
    s = p.get_state()
    g = s.get("gripper") or {}
    out = {"robot": {"mode": s["mode"], "alarm": s["error"], "moving": s["moving"]},
           "pose": [round(v, 1) for v in s["pose"]],
           "table_z": p.table_z_mm, "approach_zone_top": p.approach_top(),
           "block_top_z": getattr(getattr(runner, "local", None), "block_top_z", None),
           "gripper": {k: g.get(k) for k in ("position", "grip_status") if k in g} or None}
    if executor is not None:
        out["approval"] = {"execution_mode": executor.mode,
                           "candidate_skills_need_operator": bool(runner and runner.require_validation)}
    return out

def _calibrate_before_insert(executor) -> None:
    """24 Nov (operator): three attempts only pushed along the insert axis and never tried a sideways move.
    insert is refused until a tool-frame move along x and one along y ran in this episode, so the agent
    has seen how each sideways axis moves the tip in the picture before it pushes."""
    move, insert = executor.all_specs["move_relative"], executor.all_specs["insert"]
    move_original, insert_original = move.handler, insert.handler

    def move_relative(**kw):
        result = move_original(**kw)
        if kw.get("frame", "base") == "tool":
            done = executor.episode_state.setdefault("calibrated_axes", set())
            done.update(a for a in ("x", "y") if abs(float(kw.get("d" + a) or 0)) > 0)
        return result

    def insert_after_calibration(**kw):
        missing = [a for a in ("x", "y") if a not in executor.episode_state.get("calibrated_axes", set())]
        if missing:
            raise ValidationError("calibrate first: run move_relative with frame=\"tool\" along "
                                  + " and ".join("d" + a for a in missing)
                                  + " (about 1 mm each, a look after each) before the first insert of this attempt")
        return insert_original(**kw)

    move.handler, insert.handler = move_relative, insert_after_calibration
    insert.description += (" Refused in an attempt until move_relative with frame=\"tool\" has moved along dx and "
                           "along dy (sideways calibration).")


def build_explore(confirm=None, mode: str | None = None,
                  client: AstraClient | None = None, variant: str | None = None,
                  driver: str | None = None, config=None) -> ExploreSession:
    cfg = config if config is not None else load_explore_config()
    cfg.driver = driver or cfg.driver
    robot_cfg, perception_cfg = load_robot_config(), load_perception_config()
    agent_cfg = load_agent_config()
    agent_cfg.execution_mode = mode or cfg.execution_mode
    agent_cfg.max_steps = cfg.max_steps
    agent_cfg.max_output_tokens = cfg.max_output_tokens
    if cfg.reasoning_effort:
        agent_cfg.reasoning_effort = cfg.reasoning_effort
    agent_cfg.allowed_tools = cfg.program_tools if cfg.program_mode else cfg.allowed_tools
    from perception.scene import _env_is_depth
    agent_cfg.system_prompt = for_driver(cfg.driver, instructions(cfg.program_mode, tools=cfg.program_tools,
                                                                  cameras=tuple(cfg.cameras),
                                                                  env_depth=_env_is_depth()))
    if cfg.operator_rules:
        agent_cfg.system_prompt += "\nOPERATOR RULES (from the operator, for this session):\n" + "\n".join(
            f"- {r}" for r in cfg.operator_rules) + "\n"
    agent_cfg.truncation = "auto"        # long runs with images: let the API drop the oldest turns

    if cfg.max_step_mm:
        robot_cfg.motion.max_relative_mm = float(cfg.max_step_mm)
    if cfg.max_step_deg:
        robot_cfg.motion.max_relative_deg = float(cfg.max_step_deg)
    primitives = create_primitives(robot_cfg)
    primitives.grip_close_position = cfg.gripper.close_position
    primitives.grip_open_position = cfg.gripper.open_position
    primitives.grip_speed = cfg.gripper.speed
    # The env calibration is operator infrastructure (scripts/calibrate.sh cameras): it is used until
    # the operator calibrates again; preflight refuses to start without one.
    scene = Scene.from_config(perception_cfg, cameras=tuple(cfg.cameras), workspace=robot_cfg.workspace)
    if primitives.table_z_mm is not None:      # tape on the arm/wrist camera is not a block
        scene.object_z_max = primitives.table_z_mm + 150
    # 23 Nov (operator): Astra gets raw pictures — no fingertip marks drawn into the wrist picture
    # (perception/wrist_overlay.py stays for the operator's tools)
    guard = (RobotViewGuard(scene, cfg.view_guard.marker_color, cfg.view_guard.min_area_px)
             if cfg.view_guard.enabled else None)
    primitives.view_guard = guard
    notes = NotesStore()                  # raw archive (read_history); prompts get the knowledge digest
    if cfg.driver == "qwen":
        # Qwen's runs are not Astra's experience: its observations go to their own file
        from common.paths import STUDENT_RUNS_DIR
        from memory.knowledge import Observations
        knowledge = KnowledgeStore(observations=Observations(STUDENT_RUNS_DIR / "observations.jsonl"))
    else:
        knowledge = KnowledgeStore()
    if cfg.driver != "qwen":
        # operator, 23 Nov: Astra sees only what was measured in the current calibration
        from memory.knowledge import calibration_since
        knowledge.obs.since = calibration_since()
    investigation = Investigation(gate_after_repeats=cfg.diagnosis.gate_after_repeats, knowledge=knowledge)
    path = REPO_ROOT / cfg.skills_file
    library = SkillLibrary(path)
    executor, kit = build_toolkit(agent_cfg.execution_mode, confirm, robot_cfg=robot_cfg, agent_cfg=agent_cfg,
                                  perception_cfg=perception_cfg, primitives=primitives, scene=scene,
                                  library=library, seed_skills=False,
                                  extra_specs=explore_specs(guard, notes, library, knowledge)
                                  + diagnosis_specs(investigation) + code_specs(lambda: runner.local)
                                  + ([reference_spec(scene, cfg.reference_task)] if cfg.reference_task else []))
    if "write_routine" in executor.all_specs:
        from agent.explore.tools import routine_api
        executor.all_specs["write_routine"].description += routine_api(executor)
    if cfg.allowed_positions and "go_to_position" in executor.all_specs:
        spec, allowed = executor.all_specs["go_to_position"], list(cfg.allowed_positions)
        original = spec.handler

        def go_to_allowed(name, speed=None):
            if name not in allowed:
                raise ValidationError(f"position '{name}' is not available in this session (available: {allowed})")
            entry = primitives.a.positions.find_by_name(name)
            now = primitives.get_state().get("joints_deg") or []
            if entry and now and all(abs(a - b) < 0.3 for a, b in zip(entry["joints_deg"], now)):
                # 24 Nov: a "motion" that moves nothing must not count as one (one picture per motion)
                raise ValidationError(f"already at {name}: nothing to undo")
            return original(name, speed)
        spec.handler = go_to_allowed
        spec.description = (f"Return to the saved start position ({', '.join(allowed)}), e.g. after moving past "
                            "the target or losing the alignment. Refused when the arm is already there.")
    if cfg.one_look_per_motion:
        executor.look_gate = {"look", "compare_reference"}
        executor.look_limit = max(1, int(cfg.looks_per_motion))
    if cfg.calibrate_before_insert and {"insert", "move_relative"} <= set(executor.all_specs):
        _calibrate_before_insert(executor)
    locked = list(cfg.locked_tools)
    if cfg.gripper_locked:
        locked += ["open_gripper", "close_gripper", "set_gripper", "pick_object", "place_object"]
    executor.locked.update({n: "not available in this session" for n in locked})
    if cfg.driver == "qwen":                # nor may it change Astra's skills, routines, knowledge, positions
        for name in QWEN_NO_WRITE:
            executor.specs.pop(name, None)
    block_top = load_workcell().get("block_top_z_mm")
    runner = kit.skills
    runner.local = LocalSteps(primitives, scene, executor, block_top_z=block_top)
    if cfg.program_mode:
        # operator, 23 Nov: raw pictures only — the core steps that detect colours are neither listed nor run
        runner.local.colour_detection = False
        runner.local.code.hide_colour_steps = True
    if primitives.table_z_mm is not None:
        runner.constants["table_z"] = primitives.table_z_mm
    if block_top is not None:
        runner.constants["block_top_z"] = block_top
    runner.require_validation = cfg.skills.require_validation
    runner.supervisor = (lambda name, params: confirm(f"CANDIDATE skill {name} (supervised run)", params)) \
        if confirm is not None and cfg.skills.require_validation else None
    # observations: every live robot action, look and skill run, with the measured state after it
    knowledge.state_fn = lambda: _snapshot(primitives)
    knowledge.records = lambda name: executor.mode in ("live", "confirm") and _observed(executor, name)
    harness_cfg = {k: v for k, v in cfg.harness.items() if k != "variant"}
    if cfg.driver == "qwen":
        # a local model costs nothing per call: no token budget and no call limit (the robot's safety
        # guards and the stop rules still apply; Ctrl+C stops the episode)
        harness_cfg["max_episode_tokens"] = 0
        agent_cfg.max_steps = int((cfg.qwen or {}).get("max_steps", 0) or 100_000)
    options = HarnessOptions.from_variant(variant or cfg.harness.get("variant", "V3"), **harness_cfg)
    if client is None and cfg.driver == "qwen":
        from student.client import LocalQwenClient
        q = cfg.qwen or {}
        client = LocalQwenClient(**{k: q[k] for k in ("model", "adapter", "max_output_tokens", "images", "dump_dir")
                                    if q.get(k)},
                                 log=log.info)
        agent_cfg.model = f"qwen-local:{Path(client.model_path).name}" + (f"+{Path(client.adapter).name}"
                                                                          if client.adapter else "")
    harness = AgentHarness(client if client is not None else AstraClient(agent_cfg), executor, agent_cfg, skills=library,
                           send_images=perception_cfg.send_images_to_model, notes=knowledge, options=options,
                           state_fn=lambda: compact_state(primitives, executor, runner), skill_runner=runner,
                           # program mode offers no diagnose/conclude: failure ids and the diagnosis gate would point
                           # at tools that do not exist there (23 Nov audit)
                           investigation=None if cfg.program_mode else investigation, observer=knowledge.observe)
    harness.show_notes = cfg.show_knowledge
    return ExploreSession(cfg, kit, executor, harness, guard, notes, investigation, knowledge)


OBSERVED_READS = {"look", "find_objects", "run_routine"}

# Tools that write Astra's memory (or count runs of Astra's skills / routines): not offered to Qwen.
QWEN_NO_WRITE = ("save_skill", "rollback_skill", "retire_skill", "run_skill", "write_routine", "rollback_routine",
                 "run_routine", "define_concept", "update_model", "propose_hypothesis", "archive_knowledge",
                 "record_measurement", "diagnose", "conclude", "save_position")


def _observed(executor, name: str) -> bool:
    spec = executor.specs.get(name)
    return spec is not None and (spec.moves_robot or spec.delegates or name in OBSERVED_READS)


def _snapshot(p) -> dict:
    s = p.get_state()
    return {"pose": s.get("pose"), "gripper": s.get("gripper") or {}}
