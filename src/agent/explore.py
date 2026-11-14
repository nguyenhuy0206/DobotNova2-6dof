"""Explore mode: Astra learns the cell by itself from primitive tools.

    session = build_explore()          # config/explore.yaml + agent.yaml + robot.yaml
    problems = session.preflight()     # robot ready, table taught, cameras, marker visible
    session.harness.run(session.cfg.task)

What is different from the plain agent (agent/toolkit.py):
  * only primitives + seeing + self-calibration + learning tools (no seed skills,
    no search_object): every composite skill is one Astra wrote itself,
    kept in its own file (skills_explore.json)
  * hard limits enforced in code, whatever the model asks:
      speed ≤ agent.yaml max_speed · fingertip/flange above the taught table along the
      whole path · the env camera must still see the robot after every move (else the
      move is undone) · gripper never closes beyond gripper.close_position
  * the env camera is calibrated by Astra at runtime (record_calibration_point /
    solve_calibration) — no calibration from another day is reused unless kept
  * notes (memory/notes.py) carry measured facts to the next run
"""
from __future__ import annotations

from dataclasses import dataclass, field

from actions.executor import ActionSpec
from actions.primitives import create_primitives
from actions.registry import _obj
from agent.astra_client import AstraClient
from agent.harness import AgentHarness
from agent.toolkit import Toolkit, build_toolkit
from common.config import (_build, _read_yaml, load_agent_config,
                           load_perception_config, load_robot_config)
from common.errors import RobotError
from common.log import get_logger
from common.paths import CONFIG_DIR, REPO_ROOT
from memory.notes import NotesStore
from memory.skills import SkillLibrary
from perception.robot_view import RobotViewGuard, SelfCalibration
from perception.scene import Scene

log = get_logger(__name__)


@dataclass
class ExploreGripper:
    close_position: int | None = 500
    open_position: int | None = 800     # not fully open: wide fingers knock neighbouring blocks


@dataclass
class ViewGuardConfig:
    enabled: bool = True
    marker_color: str = "red"
    min_area_px: float = 40


@dataclass
class SkillPolicy:
    require_validation: bool = True       # candidate skills run live only with operator approval
    min_supervised_successes: int = 2     # for --promote
    builtin_pick_place: bool = True       # add pick_block_wrist / place_block_on_table if missing


@dataclass
class ExploreConfig:
    task: str = ""
    execution_mode: str = "live"
    max_steps: int = 150
    episodes: int = 1
    max_output_tokens: int = 2000
    gripper: ExploreGripper = field(default_factory=ExploreGripper)
    view_guard: ViewGuardConfig = field(default_factory=ViewGuardConfig)
    collision_level: int = 3
    skills_file: str = "data/memory/skills_explore.json"
    allowed_tools: list[str] = field(default_factory=list)
    harness: dict = field(default_factory=lambda: {"variant": "V3"})   # see agent.harness.HarnessOptions
    skills: SkillPolicy = field(default_factory=SkillPolicy)
    reflect_max_steps: int = 6


def load_explore_config(path=CONFIG_DIR / "explore.yaml") -> ExploreConfig:
    return _build(ExploreConfig, _read_yaml(path))


EXPLORE_INSTRUCTIONS = """\
You are Astra, controlling a real Dobot Nova 2 arm with a 2-finger PGC gripper, a fixed Intel
RealSense D455 RGB-D camera looking at the cell ("env"), and a small webcam on the wrist ("wrist").
Each run is one EPISODE; after it you review the log and update your skills and notes, so use
your skills (list_skills / run_skill) and notes from earlier episodes first, and improve them.
Nobody answers questions during the run: decide, act, observe, finish with a short report.

Hard limits enforced by the tools (an error tells you which one hit):
- speed is capped; every move blocks until finished
- fingertips/flange never closer to the metal table than get_state.table shows (whole path).
  If the fingertip is already below table.min_fingertip_z, first lift straight up (move_relative dz>0).
- APPROACH ZONE: below Z = table + 120 mm every Cartesian move is slow and at most 25 mm long.
  Stop above it, then descend / shift in small steps, looking with the wrist camera in between.
- CONTACT: if something blocks a move in the approach zone, the robot stops, backs off 15 mm up
  and you get "CONTACT at fingertip Z=…". That Z is a measured surface (block top / box top):
  use it — e.g. grasp a few mm below a block's contact height, never push further.
- the env camera must always see the robot (red tape on the gripper); a move that hides it is undone.
- open_gripper opens to 80 % (wide enough for a block, narrow enough to miss its neighbours);
  close_gripper closes to 50 % at most. holding=true means caught. A grip caught near the open
  position (e.g. 850+) is only an edge contact and will slip.

Coordinates: all poses are FINGERTIP positions in the robot base frame (mm, deg). rx=180, ry=0
points the gripper straight down; rz turns it about the vertical axis.

Gripping a cube on two opposite FACES is the skill to learn here. The final centring and turning
must be done with the WRIST camera — env-camera coordinates are only good to ~2–3 cm:
- Reading the wrist image: the camera sits on the gripper and looks past it. The gripper body is
  at the bottom of the image, the two black fingers stick up on the left and right, and they close
  left↔right. A block ready to grasp is BETWEEN the two fingertips, its front/back edges horizontal
  in the image. A block that appears above/beyond the fingertips is not between them yet.
- Centre with move_relative(frame="tool", dx/dy ≤ 10–15 mm): in the TOOL frame a given direction
  always moves the image the same way, whatever rz is. Learn once which tool direction moves the
  block left/right/up/down in the wrist image (one 10 mm test move), then write it in a note.
- Turn with move_relative(drz=…, frame="tool") — it pivots around the fingertips — or jog_joint(6, …).
  Turn by the angle between the block's edges and the image horizontal (never more than ±45°: a cube
  repeats every 90°). Learn the sign once with a 10° test turn and note it.
- Repeat look(wrist) → small correction until centred and square, THEN descend in steps.
- Grasp height: below the block top (contact Z or camera top) by about 10–12 mm — closing at the
  top only grabs air: position 500 with holding=false means the fingers met nothing.
- Grip position: ≈ 530 = face-to-face grip; 700+ = corner/edge grip (will slip): open, lift 15 mm,
  re-centre/turn, retry.
- Put the centring + turning loop into a new version of your pick skill when it works, and note
  the tool-direction ↔ image-direction mapping and the sign of drz.

Camera coordinates (look / find_objects base_xyz_mm) come from an env-camera calibration; they can
be off by a few cm in Z and XY away from where it was calibrated. Trust contacts and the wrist
camera over them, and note the corrections you measure.

How to work (skills first — every model call is expensive; a skill runs many steps locally):
1. Start: look env once (zoom on the box) to get every cube's base_xyz_mm; read_notes / list_skills only if
   the context does not already show them. If look says calibrated=false: calibrate first
   (record_calibration_point at 8–12 varied poses, then solve_calibration).
2. Per cube: run_skill pick_block_wrist {color, x, y, z_top, rz}. One call = one grasp attempt (lift out of
   the zone, hover, descend, wrist search + centring, touch-measure of the real cube top, grasp, lift, env
   check — all local). x, y, z_top straight from the env camera (estimates are fine: the skill searches
   ±30 mm sideways and measures the top itself; result.measured_top / offset tell you the real values —
   reuse them for the other cubes). rz: the cube's edges square to the fingers (a cube repeats every 90°).
   If it fails, read failure + detail + meaning (+ the attached image) and call pick_block_wrist AGAIN
   directly with ONE thing changed (rz ±15–30, depth_mm 8/12/16, x/y, target_u/target_v) — it lifts out of
   the approach zone by itself; never lift step by step yourself. At most 4 attempts per cube.
3. Then run_skill place_block_on_table {color, x, y, rz} on a FREE table spot ≥ 60 mm from other cubes
   (reachable, inside the D455 view). Verified placements appear in STATE.verified_progress.
4. If a skill is refused (needs operator approval) or does not fit, use primitives in short chunks.
5. When all cubes are on the table (or attempts are used up): one env look to confirm, write_note one line
   with what worked (rz, depth, failures), and finish with a short report (no tool call).
Calls are expensive: STATE already holds the pose, gripper and zone — do not call get_state or list_skills
to re-read them; do not look again when a skill result already says what happened.
A failed grasp is DATA: never repeat the exact same parameters. Only a robot alarm, a person in the
workspace or a broken gripper reading (grip_state 3) are reasons to stop.
"""


@dataclass
class ExploreSession:
    cfg: ExploreConfig
    kit: Toolkit
    executor: object
    harness: AgentHarness
    calibration: SelfCalibration
    view_guard: RobotViewGuard | None
    notes: NotesStore

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
            problems.append("table height not taught: run  .venv/bin/python scripts/teach_table.py")
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
        """One attempt (robot moves), then Astra reviews it (no robot): outcome, skills, notes."""
        from agent.learning import history

        past = history(last=5)
        prompt = task.strip()
        if past:
            prompt += "\n\nPrevious episodes (newest last):\n" + "\n".join(
                f"- {h['at']}: {'SUCCESS' if h['success'] else 'FAIL'}, {h['blocks_moved']} moved — {h['summary'][:120]}"
                for h in past)
        report = self.harness.run(prompt)
        out = {"episode": self.harness.last_episode, "report": report, "metrics": self.harness.last_metrics}
        if reflect:
            out.update(self.reflect(out["episode"]))
        return out

    def reflect(self, episode_id: str) -> dict:
        import copy

        from agent.learning import REFLECT_INSTRUCTIONS, OutcomeRecorder, reflect_executor, summarize_episode
        from memory.store import EpisodeStore

        from agent.harness import HarnessOptions

        recorder = OutcomeRecorder(self.harness.episodes)
        recorder.episode_id = episode_id
        cfg = copy.copy(self.harness.config)
        # ONE reflection pass with a few memory calls; never more than reflect_max_steps requests
        cfg.system_prompt, cfg.max_steps, cfg.allowed_tools = REFLECT_INSTRUCTIONS, self.cfg.reflect_max_steps, []
        cfg.max_output_tokens = max(cfg.max_output_tokens, 6000)
        reviewer = AgentHarness(self.harness.client, reflect_executor(self.executor, recorder), cfg,
                                episodes=EpisodeStore(self.harness.episodes.root / "reflections"),
                                skills=self.harness.skills, send_images=False, notes=self.notes,
                                options=HarnessOptions.from_variant("V3" if self.harness.options.variant in ("V1", "V3")
                                                                    else "V0"))
        text = reviewer.run("Episode log:\n" + summarize_episode(self.harness.episodes, episode_id))
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


def _explore_specs(calibration: SelfCalibration, guard: RobotViewGuard | None, notes: NotesStore) -> list[ActionSpec]:
    specs = [
        ActionSpec("record_calibration_point",
                   "Env-camera calibration: record the current robot pose together with where the env "
                   "camera sees the red gripper marker (3D from depth). The robot must stand still. "
                   "Use 8–12 varied poses (positions AND orientations), then solve_calibration.",
                   calibration.record, _obj({}, [])),
        ActionSpec("solve_calibration",
                   "Compute the env camera → robot transform from the recorded points (≥ 6) and start "
                   "using it: look / find_objects then return base_xyz_mm. Returns rms and worst error (mm).",
                   calibration.solve, _obj({}, [])),
        ActionSpec("reset_calibration", "Forget all recorded calibration points (only if the camera moved).",
                   calibration.reset, _obj({}, [])),
        ActionSpec("write_note",
                   "Save a short factual lesson for future runs (e.g. measured heights, what failed and why).",
                   notes.add, _obj({"text": {"type": "string"}, "topic": {"type": "string"}}, ["text"])),
        ActionSpec("read_notes", "Read your saved notes (optionally one topic).",
                   notes.read, _obj({"topic": {"type": "string"}}, []), read_only=True),
    ]
    if guard is not None:
        specs.append(ActionSpec("check_robot_in_view",
                                "Is the robot (red gripper marker) visible to the env camera right now?",
                                guard, _obj({}, []), read_only=True))
    return specs


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


def build_explore(confirm=None, mode: str | None = None, recalibrate: bool = False,
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
    # The camera setup changes from day to day: a calibration from another day is never used.
    stale = not recalibrate and _from_another_day(perception_cfg.calibration_path)
    if recalibrate or stale:
        _set_aside(perception_cfg.calibration_path)
        if stale:
            log.warning("env calibration is from another day — set aside, Astra must calibrate again")
    scene = Scene.from_config(perception_cfg, workspace=robot_cfg.workspace)
    if recalibrate or stale:
        scene.env_calibration = None
    if primitives.table_z_mm is not None:      # tape on the arm/wrist camera is not a block
        scene.object_z_max = primitives.table_z_mm + 150
    guard = (RobotViewGuard(scene, cfg.view_guard.marker_color, cfg.view_guard.min_area_px)
             if cfg.view_guard.enabled else None)
    primitives.view_guard = guard
    calibration = SelfCalibration(scene, primitives, cfg.view_guard.marker_color, cfg.view_guard.min_area_px)
    if recalibrate or (calibration.points and not _points_from_today(calibration.points)):
        calibration.reset()
    notes = NotesStore()
    path = REPO_ROOT / cfg.skills_file
    library = SkillLibrary(path)

    executor, kit = build_toolkit(agent_cfg.execution_mode, confirm, robot_cfg=robot_cfg, agent_cfg=agent_cfg,
                                  perception_cfg=perception_cfg, primitives=primitives, scene=scene,
                                  library=library, seed_skills=False,
                                  extra_specs=_explore_specs(calibration, guard, notes))
    from actions.local_steps import LocalSteps
    from agent.harness import HarnessOptions
    from agent.pick_place_skills import ensure_pick_place_skills

    from memory.workcell import load_workcell

    block_top = load_workcell().get("block_top_z_mm")
    runner = kit.skills
    runner.local = LocalSteps(primitives, scene, executor, block_top_z=block_top)
    if primitives.table_z_mm is not None:
        runner.constants["table_z"] = primitives.table_z_mm
    if block_top is not None:
        runner.constants["block_top_z"] = block_top
    runner.require_validation = cfg.skills.require_validation
    runner.supervisor = (lambda name, params: confirm(f"CANDIDATE skill {name} (supervised run)", params)) \
        if confirm is not None else None
    if cfg.skills.builtin_pick_place:
        ensure_pick_place_skills(library)
    harness_cfg = {k: v for k, v in cfg.harness.items() if k != "variant"}
    options = HarnessOptions.from_variant(variant or cfg.harness.get("variant", "V3"), **harness_cfg)
    harness = AgentHarness(client if client is not None else AstraClient(agent_cfg), executor, agent_cfg, skills=library,
                           send_images=perception_cfg.send_images_to_model, notes=notes, options=options,
                           state_fn=lambda: compact_state(primitives, executor, runner), skill_runner=runner)
    return ExploreSession(cfg, kit, executor, harness, calibration, guard, notes)


def _from_another_day(path) -> bool:
    import datetime as dt
    return path.exists() and dt.date.fromtimestamp(path.stat().st_mtime) != dt.date.today()


def _points_from_today(points: list[dict]) -> bool:
    import datetime as dt
    return all(dt.date.fromtimestamp(p.get("t", 0)) == dt.date.today() for p in points)


def _set_aside(path) -> None:
    """Keep an old calibration file for reference but stop using it."""
    if path.exists():
        import time
        path.rename(path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.old.json"))
