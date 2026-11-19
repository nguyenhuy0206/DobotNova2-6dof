"""Explore mode: Astra learns the cell by itself from primitive tools.

    session = build_explore()          # config/explore.yaml + agent.yaml + robot.yaml
    problems = session.preflight()     # robot ready, table taught, env calibrated, cameras, marker visible
    session.harness.run(session.cfg.task)

What is different from the plain agent (agent/toolkit.py):
  * only primitives + seeing + learning tools (no seed skills,
    no search_object): every composite skill is one Astra wrote itself,
    kept in its own file (skills_explore.json)
  * hard limits enforced in code, whatever the model asks:
      speed ≤ agent.yaml max_speed · fingertip/flange above the taught table along the
      whole path · the env camera must still see the robot after every move (else the
      move is undone) · gripper never closes beyond gripper.close_position
  * the cameras, table and limits are calibrated by the operator (scripts/calibrate.sh); the
    calibration stays in use until the operator calibrates again
  * knowledge (memory/knowledge.py): the harness records every robot action as an observation;
    Astra builds physical concepts and hypotheses with predictions, checked against observations
"""
from __future__ import annotations

from dataclasses import dataclass

from actions.local_steps import LocalSteps
from actions.primitives import create_primitives
from agent.astra_client import AstraClient
from agent.explore.config import ExploreConfig, load_explore_config
from agent.explore.diagnosis import Investigation
from agent.explore.diagnosis import specs as diagnosis_specs
from agent.explore.pick_place_skills import ensure_pick_place_skills
from agent.explore.prompt import EXPLORE_INSTRUCTIONS
from agent.explore.tools import code_specs, explore_specs
from agent.harness import AgentHarness, HarnessOptions
from agent.toolkit import Toolkit, build_toolkit
from calibration.workcell import load_workcell
from common.config import load_agent_config, load_perception_config, load_robot_config
from common.errors import RobotError
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
        if self.kit.scene.env_calibration is None:
            problems.append("env camera not calibrated: run  scripts/calibrate.sh cameras")
        if getattr(p.cfg.gripper, "via_ui", False):
            try:
                p.gripper.status()
            except RobotError as exc:
                problems.append(str(exc))
        for camera in ("env", "wrist"):
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

    def run_episode(self, task: str, reflect: bool = True) -> dict:
        """One attempt (robot moves), then Astra reviews it (no robot): outcome, skills, knowledge."""
        from agent.explore.learning import history

        # the record as counts only: the free-text summaries are experience, not knowledge
        past = history(last=10000)
        prompt = task.strip()
        if past:
            wins = sum(1 for h in past if h["success"])
            moved = sum(int(h.get("blocks_moved") or 0) for h in past)
            last = past[-1]
            prompt += (f"\n\nEpisode record (all tasks so far): {len(past)} reviewed episodes, {wins} successful, "
                       f"{moved} objects moved as their task asked; last ({last['at'][:16]}): "
                       f"{'SUCCESS' if last['success'] else 'FAIL'}, {last['blocks_moved']} moved.")
        if self.knowledge is not None and self.knowledge.obs.items:
            items = self.knowledge.obs.items
            live = [h for c in self.knowledge.concepts if not c.get("archived") for h in c["hypotheses"]
                    if not h.get("archived")]
            prompt += (f"\nKnowledge record: {len(items)} recorded observations from "
                       f"{len({o.get('episode') for o in items})} episodes (query_observations), "
                       f"{sum(not c.get('archived') for c in self.knowledge.concepts)} concepts, "
                       f"{len(live)} hypotheses.")
        report = self.harness.run(prompt)
        out = {"episode": self.harness.last_episode, "report": report, "metrics": self.harness.last_metrics,
               "diagnosis": self.investigation.report_text() if self.investigation else ""}
        if reflect:
            out.update(self.reflect(out["episode"]))
        return out

    def reflect(self, episode_id: str) -> dict:
        import copy

        from agent.explore.learning import (REFLECT_INSTRUCTIONS, OutcomeRecorder, failure_images, post_lift_images,
                                    reflect_executor, summarize_episode)
        from memory.store import EpisodeStore

        recorder = OutcomeRecorder(self.harness.episodes)
        recorder.episode_id = episode_id
        cfg = copy.copy(self.harness.config)
        # ONE reflection pass with a few memory calls; never more than reflect_max_steps requests
        cfg.system_prompt, cfg.max_steps, cfg.allowed_tools = REFLECT_INSTRUCTIONS, self.cfg.reflect_max_steps, []
        cfg.max_output_tokens = max(cfg.max_output_tokens, 6000)
        reviewer = AgentHarness(self.harness.client, reflect_executor(self.executor, recorder), cfg,
                                episodes=EpisodeStore(self.harness.episodes.root / "reflections"),
                                skills=self.harness.skills, send_images=False, notes=self.knowledge,
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
                           "candidate_skills_need_operator": bool(runner and runner.require_validation),
                           **({"auto_promote_after": runner.auto_promote_after}
                              if runner and not runner.require_validation and runner.auto_promote_after else {})}
    return out

def build_explore(confirm=None, mode: str | None = None,
                  client: AstraClient | None = None, variant: str | None = None) -> ExploreSession:
    cfg = load_explore_config()
    robot_cfg, perception_cfg = load_robot_config(), load_perception_config()
    agent_cfg = load_agent_config()
    agent_cfg.execution_mode = mode or cfg.execution_mode
    agent_cfg.max_steps = cfg.max_steps
    agent_cfg.max_output_tokens = cfg.max_output_tokens
    agent_cfg.allowed_tools = cfg.allowed_tools
    agent_cfg.system_prompt = EXPLORE_INSTRUCTIONS
    agent_cfg.truncation = "auto"        # long runs with images: let the API drop the oldest turns

    primitives = create_primitives(robot_cfg)
    primitives.grip_close_position = cfg.gripper.close_position
    primitives.grip_open_position = cfg.gripper.open_position
    primitives.grip_speed = cfg.gripper.speed
    # The env calibration is operator infrastructure (scripts/calibrate.sh cameras): it is used until
    # the operator calibrates again; preflight refuses to start without one.
    scene = Scene.from_config(perception_cfg, workspace=robot_cfg.workspace)
    if primitives.table_z_mm is not None:      # tape on the arm/wrist camera is not a block
        scene.object_z_max = primitives.table_z_mm + 150
    guard = (RobotViewGuard(scene, cfg.view_guard.marker_color, cfg.view_guard.min_area_px)
             if cfg.view_guard.enabled else None)
    primitives.view_guard = guard
    notes = NotesStore()                  # raw archive (read_history); prompts get the knowledge digest
    knowledge = KnowledgeStore()
    investigation = Investigation(gate_after_repeats=cfg.diagnosis.gate_after_repeats, knowledge=knowledge)
    path = REPO_ROOT / cfg.skills_file
    library = SkillLibrary(path)
    executor, kit = build_toolkit(agent_cfg.execution_mode, confirm, robot_cfg=robot_cfg, agent_cfg=agent_cfg,
                                  perception_cfg=perception_cfg, primitives=primitives, scene=scene,
                                  library=library, seed_skills=False,
                                  extra_specs=explore_specs(guard, notes, library, knowledge)
                                  + diagnosis_specs(investigation) + code_specs(lambda: runner.local))
    block_top = load_workcell().get("block_top_z_mm")
    runner = kit.skills
    runner.local = LocalSteps(primitives, scene, executor, block_top_z=block_top)
    if primitives.table_z_mm is not None:
        runner.constants["table_z"] = primitives.table_z_mm
    if block_top is not None:
        runner.constants["block_top_z"] = block_top
    runner.require_validation = cfg.skills.require_validation
    runner.auto_promote_after = cfg.skills.auto_promote_after
    runner.supervisor = (lambda name, params: confirm(f"CANDIDATE skill {name} (supervised run)", params)) \
        if confirm is not None and cfg.skills.require_validation else None
    if cfg.skills.builtin_pick_place:
        ensure_pick_place_skills(library)
    # observations: every live robot action, look and skill run, with the measured state after it
    knowledge.state_fn = lambda: _snapshot(primitives)
    knowledge.records = lambda name: executor.mode in ("live", "confirm") and _observed(executor, name)
    harness_cfg = {k: v for k, v in cfg.harness.items() if k != "variant"}
    options = HarnessOptions.from_variant(variant or cfg.harness.get("variant", "V3"), **harness_cfg)
    harness = AgentHarness(client if client is not None else AstraClient(agent_cfg), executor, agent_cfg, skills=library,
                           send_images=perception_cfg.send_images_to_model, notes=knowledge, options=options,
                           state_fn=lambda: compact_state(primitives, executor, runner), skill_runner=runner,
                           investigation=investigation, observer=knowledge.observe)
    return ExploreSession(cfg, kit, executor, harness, guard, notes, investigation, knowledge)


OBSERVED_READS = {"look", "find_objects", "run_routine"}


def _observed(executor, name: str) -> bool:
    spec = executor.specs.get(name)
    return spec is not None and (spec.moves_robot or spec.delegates or name in OBSERVED_READS)


def _snapshot(p) -> dict:
    s = p.get_state()
    return {"pose": s.get("pose"), "gripper": s.get("gripper") or {}}
