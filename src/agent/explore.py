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
  * knowledge (memory/knowledge.py): the harness records every robot action as an observation;
    Astra builds physical concepts and hypotheses with predictions, checked against observations
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from actions.executor import ActionSpec
from actions.primitives import create_primitives
from actions.registry import _obj
from agent.astra_client import AstraClient
from agent.diagnosis import Investigation
from agent.diagnosis import specs as diagnosis_specs
from agent.harness import AgentHarness
from agent.toolkit import Toolkit, build_toolkit
from common.config import (_build, _read_yaml, load_agent_config,
                           load_perception_config, load_robot_config)
from common.errors import RobotError, ValidationError
from common.log import get_logger
from common.paths import CONFIG_DIR, REPO_ROOT
from memory.hypotheses import HypothesisStore
from memory.knowledge import MAX_LIVE_HYPOTHESES, KnowledgeStore
from memory.notes import NotesStore
from memory.skills import SkillLibrary
from perception.robot_view import RobotViewGuard, SelfCalibration
from perception.scene import Scene

log = get_logger(__name__)


@dataclass
class ExploreGripper:
    close_position: int | None = 500
    open_position: int | None = 1000    # fully open (80 % pressed fingertips on the cube, 16 Nov)
    speed: int | None = 20              # fixed finger speed %: 73/73 ok at >= 20, 0/3 at 10 (17 Nov)


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
    auto_promote_after: int = 0           # autonomous (require_validation false): promote after N verified successes


@dataclass
class DiagnosisPolicy:
    gate_after_repeats: int = 2   # refuse run_skill/save_skill only when the same failure repeats this often (0 = never)


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
    diagnosis: DiagnosisPolicy = field(default_factory=DiagnosisPolicy)
    reflect_max_steps: int = 6


def load_explore_config(path=CONFIG_DIR / "explore.yaml") -> ExploreConfig:
    return _build(ExploreConfig, _read_yaml(path))


EXPLORE_INSTRUCTIONS = """\
You are Astra, controlling a real Dobot Nova 2 arm with a 2-finger PGC gripper, a fixed Intel
RealSense D455 RGB-D camera looking at the cell ("env"), and a small webcam on the wrist ("wrist").
Each run is one EPISODE; after it you review the log and may update your skills, code and knowledge.
Nobody answers questions during the run: decide, act, observe, finish with a short report.
How to reach the goal is up to you.

Hard limits enforced by the tools (an error tells you which one hit):
- speed is capped; every move blocks until finished
- fingertips/flange never closer to the metal table than get_state.table shows (whole path).
- APPROACH ZONE: below Z = table + 120 mm every Cartesian move is slow and at most 25 mm long.
- CONTACT: if something blocks a move in the approach zone (the move stalls, or the controller's collision
  detection raises an alarm), the robot stops, clears the alarm, backs off 15 mm up and the error says
  "CONTACT at fingertip Z=…". An alarm that does not clear stops the robot.
- the env camera must always see the robot (red tape on the gripper); a move that hides it is undone.
- open_gripper opens fully (1000); close_gripper closes until it holds something or reaches 50 %
  (position 500 of 1000); both at a fixed finger speed of 20 %.
  close_gripper reports position and holding. Gripper grip_state: 0 moving, 1 arrived (no object),
  2 object caught, 3 object dropped.

Coordinates: all poses are FINGERTIP positions in the robot base frame (mm, deg). rx=180, ry=0 points the
gripper straight down; rz turns it about the vertical axis. move_relative frame="tool" moves in the
gripper's own frame. Camera coordinates (look / find_objects base_xyz_mm) come from the env-camera
calibration, which the operator maintains.
Calibration changed on 18 Nov 2026: the fingertip point is now the centre between the fingertips,
127.1 mm from the flange (before: 118 mm), and the table top is at z 0.3 (before: 41.96). Observations
(variable calib), episode history, skills and routines made before carry the old definition
(calib=tool118_table42); new ones carry calib=tool127_table0.
Wrist image: the camera sits on the gripper; the gripper body is at the bottom of the image and the two
fingers are on the left and right, closing left<->right.

What you can build on:
- skills (list_skills / run_skill / save_skill / rollback_skill / retire_skill): step lists that run locally
  without model calls; their routines and checks are Python code you own (src/agent/astra_skills:
  list_routines / read_routine / write_routine / run_routine / rollback_routine). Only the primitive tools
  and their safety checks are fixed.
- knowledge (list_concepts / define_concept / update_model / propose_hypothesis / archive_knowledge /
  query_observations / record_measurement) carries what you learn to the next episodes. Every robot action,
  look and skill run is recorded as an observation (numbers: pose, gripper, detections, parameters, outcome;
  its id comes back as "observation"). A concept is a physical quantity and your model of it; its hypotheses
  are competing explanations, each with a prediction (when -> expect) that the code checks against every
  matching observation, past and future. supported = >= 3 matching, >= 80 % right, >= 1 observed after it
  was proposed; refuted = >= 2 wrong and <= 30 % right; contested = both; open = otherwise. A result that
  contradicts a live hypothesis carries prediction_errors.
- read_history: brief of earlier episodes, the old hypothesis list and old notes (not in this prompt).
- every model call costs tokens and the episode has a token budget; STATE already holds pose, gripper and
  zone.

Rules of the episode:
- The workspace is clear and safe. The episode stops early only when the controller reports an alarm, or
  when the same action (same tool and arguments) or the same hypothesis comes 3 times in a row; a failed
  or refused call, a gripper error or an odd measurement does not end it. Looking (look, get_state,
  list_*, query_observations, verify_calibration) is not counted.
- A failed result carries a failure_id. When the SAME failure repeats without a diagnosis, run_skill /
  save_skill are refused until you call diagnose (why + hypotheses, each a knowledge hypothesis with a
  prediction, and the test you will run); record test outcomes with conclude.
- Finish with a short report (no tool call). If anything failed, it MUST have a "Why it failed" part: per
  failure the confirmed cause and the fix, or the open hypotheses and the test that would decide them.
"""


@dataclass
class ExploreSession:
    cfg: ExploreConfig
    kit: Toolkit
    executor: object
    harness: AgentHarness
    calibration: SelfCalibration
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
            problems.append("table height not taught: run  .venv/bin/python scripts/teach_table.py")
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
        from agent.learning import history

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

        from agent.learning import (REFLECT_INSTRUCTIONS, OutcomeRecorder, failure_images, post_lift_images,
                                    reflect_executor, summarize_episode)
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


def _explore_specs(calibration: SelfCalibration, guard: RobotViewGuard | None, notes: NotesStore,
                   library: SkillLibrary | None = None,
                   knowledge: KnowledgeStore | None = None) -> list[ActionSpec]:
    knowledge = knowledge if knowledge is not None else KnowledgeStore()
    specs = [
        ActionSpec("record_calibration_point",
                   "Env-camera calibration: record the current robot pose together with where the env "
                   "camera sees the red gripper marker (3D from depth). The robot must stand still. "
                   "solve_calibration needs ≥ 6 points and rejects a solve with rms > 10 mm or leave-one-out "
                   "median > 15 mm.",
                   calibration.record, _obj({}, [])),
        ActionSpec("solve_calibration",
                   "Compute the env camera → robot transform from the recorded points (≥ 6) and start "
                   "using it: look / find_objects then return base_xyz_mm. Returns rms and worst error (mm).",
                   calibration.solve, _obj({}, [])),
        ActionSpec("reset_calibration", "Forget all recorded calibration points (only if the camera moved).",
                   calibration.reset, _obj({}, [])),
        ActionSpec("verify_calibration",
                   "Check the ACTIVE env calibration at the current pose (no motion; robot still, marker visible): "
                   "camera vs robot position of the marker in mm. If the error is over the limit the calibration "
                   "is switched off (camera moved) and you must calibrate again.",
                   calibration.verify, _obj({}, []), read_only=True),
        auto_calibrate_spec(calibration),
        *knowledge_specs(knowledge),
        history_spec(notes),
    ]
    if library is not None:
        specs += [
            ActionSpec("rollback_skill",
                       "Restore an earlier version of a skill as its newest version (when a revision made it worse). "
                       "list_skills {name} shows the versions and their success/failure counts.",
                       library.rollback, _obj({"name": {"type": "string"}, "version": {"type": "integer", "minimum": 1}},
                                              ["name", "version"])),
            ActionSpec("retire_skill",
                       "Retire a draft skill that never worked and will not be used (kept with its history, "
                       "run_skill refuses it). Built-in skills cannot be retired.",
                       library.retire, _obj({"name": {"type": "string"}, "reason": {"type": "string"}}, ["name"])),
        ]
    if guard is not None:
        specs.append(ActionSpec("check_robot_in_view",
                                "Is the robot (red gripper marker) visible to the env camera right now?",
                                guard, _obj({}, []), read_only=True))
    return specs


def auto_calibrate_spec(calibration: SelfCalibration) -> ActionSpec:
    """scripts/run_calibration.sh as a tool: the scripted env-camera calibration."""
    def run(center: list | None = None, span_mm: float = 100, points: int = 12) -> dict:
        from collections import Counter

        from perception.auto_calibration import auto_calibrate, suggest_center
        p = calibration.p
        z_mid = (p.table_z_mm + 150) if p.table_z_mm is not None else None
        if center is not None:
            if not (isinstance(center, list) and len(center) == 3):
                raise ValidationError("center: [x, y, z] fingertip mm")
            center = [float(v) for v in center]
        elif (inner := suggest_center(p)) is not None and z_mid is not None:
            center = [inner[0], inner[1], z_mid]          # near the edge of the reach: pulled toward the base
        calibration.reset()                                # fresh points; the active calibration stays until
        report = auto_calibrate(p, calibration, center=center, span_mm=float(span_mm),   # a better one is accepted
                                speed=p.speed_cap or 10, target_points=int(points), log=log.info)
        why = Counter(str(s.get("why", ""))[:60] for s in report.get("skipped", []))
        keep = ("points", "rms_mm", "leave_one_out_mm", "worst_error_mm", "marker_offset_in_flange_mm",
                "rotation_spread_deg", "z_spread_mm", "center", "dropped_outliers")
        return {"calibrated": True, **{k: report[k] for k in keep if k in report},
                "skipped_poses": sum(why.values()), "skipped_why": dict(why)}

    return ActionSpec(
        "auto_calibrate",
        "Calibrate the env camera end to end by itself (scripts/run_calibration.sh): the robot visits up to 24 "
        "poses within ±span_mm around a centre (fingertip; default the current x/y at table + 150 mm, pulled "
        "toward the base near the edge of the reach), tilting and twisting the gripper up to 20°, records the "
        "red marker at each, and solves once `points` good points are in. Poses that are unreachable, leave the "
        "camera view or would self-collide are skipped. The new calibration replaces the active one only if it "
        "passes the quality gate; otherwise the error says why and the old one stays. Run it whenever the "
        "calibration is in doubt (verify_calibration fails, positions or heights look wrong). Takes about a "
        "minute.",
        run, _obj({"center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                   "span_mm": {"type": "number", "minimum": 40, "maximum": 150},
                   "points": {"type": "integer", "minimum": 8, "maximum": 20}}, []),
        moves_robot=True)


def knowledge_specs(store: KnowledgeStore) -> list[ActionSpec]:
    cond = {"type": "object", "description": "variable -> value, [lo, hi] (numeric range) or [one, of, these]. "
                                             "Variable names as in query_observations"}
    return [
        ActionSpec("list_concepts",
                   "Your concepts with their model and hypotheses (status, counts, tested range). concept_id=… "
                   "shows one with the ids of its supporting / contradicting observations.",
                   store.list, _obj({"concept_id": {"type": "string"}, "include_archived": {"type": "boolean"}}, []),
                   read_only=True),
        ActionSpec("define_concept",
                   "A physical quantity or relation you want to understand (e.g. where something appears, how high "
                   "something is, what a measurement depends on), the variables it may depend on and your model of "
                   "it so far. A quantity that an existing concept already covers is refused.",
                   store.define_concept, _obj({"quantity": {"type": "string"},
                                               "depends_on": {"type": "array", "items": {"type": "string"}},
                                               "model": {"type": "string"}}, ["quantity"])),
        ActionSpec("update_model",
                   "Replace your model of a concept (numbers, formula, ranges) and optionally what it depends on.",
                   store.update_model, _obj({"concept_id": {"type": "string"}, "model": {"type": "string"},
                                             "depends_on": {"type": "array", "items": {"type": "string"}}},
                                            ["concept_id", "model"])),
        ActionSpec("propose_hypothesis",
                   "An explanation within a concept: mechanism (why) and a prediction — `when` (conditions an "
                   "observation must meet) and `expect` (what it must then show). Evidence is computed from every "
                   "matching observation, past and future; the result shows how it fits the past ones. Variables "
                   "must be ones observations carry (query_observations lists them; record_measurement adds one). "
                   "A broader "
                   "hypothesis is the same expectation with fewer or wider conditions. The same prediction or "
                   f"mechanism returns the existing one; at most {MAX_LIVE_HYPOTHESES} live hypotheses per concept.",
                   store.propose, _obj({"concept_id": {"type": "string"}, "mechanism": {"type": "string"},
                                        "when": cond, "expect": cond}, ["concept_id", "mechanism", "when", "expect"])),
        ActionSpec("archive_knowledge",
                   "Take a concept (C#) or hypothesis (C#.n) out of the prompts when it no longer applies; it is kept.",
                   store.archive, _obj({"knowledge_id": {"type": "string"}, "reason": {"type": "string"}},
                                       ["knowledge_id", "reason"])),
        ActionSpec("query_observations",
                   "Recorded observations (newest last) whose variables match `where`; fields = which variables "
                   "to show (default all). Without matches or fields it also lists the variable names in use.",
                   store.query, _obj({"where": cond, "fields": {"type": "array", "items": {"type": "string"}},
                                      "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                                      "this_episode": {"type": "boolean"}}, []), read_only=True),
        ActionSpec("record_measurement",
                   "Record values you measured yourself (e.g. read from an image) as an observation, together "
                   "with the current pose and gripper. Single values, your own variable names.",
                   store.record_measurement, _obj({"values": {"type": "object"}, "note": {"type": "string"}},
                                                  ["values"])),
    ]


def history_spec(notes: NotesStore) -> ActionSpec:
    def read_history(kind: str = "episodes", query: str = "", limit: int = 15) -> dict:
        from agent.learning import history
        limit = max(1, min(int(limit), 60))
        q = str(query or "").lower()
        if kind == "episodes":
            from memory.knowledge import CALIBRATION, OLD_CALIBRATION_SINCE
            rows = [{**{k: h.get(k) for k in ("episode", "at", "success", "blocks_moved", "summary")},
                     "calib": "tool118_table42" if str(h.get("at") or "") < OLD_CALIBRATION_SINCE else CALIBRATION}
                    for h in history(last=10000) if not q or q in json.dumps(h).lower()]
            return {"episodes": rows[-limit:], "total": len(rows)}
        if kind == "old_hypotheses":
            old = HypothesisStore()
            rows = [{"id": h["id"], "status": h["status"], "claim": h["claim"][:200], "scope": h["scope"]}
                    for h in old.list(query=q or None, limit=10000)["hypotheses"]]
            return {"old_hypotheses": rows[-limit:], "total": len(rows),
                    "note": "from before concepts existed; not checked against observations"}
        if kind == "notes":
            return notes.read(q or None)
        raise ValidationError("kind: episodes | old_hypotheses | notes")

    return ActionSpec("read_history",
                      "Brief of the past, not in the prompt: kind=episodes (outcome + summary written at review), "
                      "old_hypotheses (the earlier free-text hypotheses, incl. archived), notes (raw notes). "
                      "query filters by text.",
                      read_history, _obj({"kind": {"type": "string", "enum": ["episodes", "old_hypotheses", "notes"]},
                                          "query": {"type": "string"},
                                          "limit": {"type": "integer", "minimum": 1, "maximum": 60}}, []),
                      read_only=True)


def code_specs(get_local) -> list[ActionSpec]:
    """Astra writes, tests and rolls back its own routines/checks (actions/astra_code.py)."""
    from actions.astra_code import AstraCode
    code = AstraCode.default()

    def run_routine(name: str, args: dict | None = None, kind: str = "routine") -> dict:
        local, memo = get_local(), {}
        out = (local.check if kind == "check" else local.routine)(name, memo, **(args or {}))
        out = {k: v for k, v in out.items() if k != "error_result"}
        if memo.get("vars"):
            out["vars"] = memo["vars"]
        return out

    return [
        ActionSpec("list_routines",
                   "Your own routines/checks (src/agent/astra_skills, with version + params) and the core ones you "
                   "can replace. Skill steps call them as {routine: NAME} / {check: NAME}.",
                   code.list, _obj({}, []), read_only=True),
        ActionSpec("read_routine", "Python source of a routine/check (yours or core) and its version history.",
                   code.read, _obj({"name": {"type": "string"}}, ["name"]), read_only=True),
        ActionSpec("write_routine",
                   "Create or replace a routine/check with Python (same name as a core one replaces it; lift_to is "
                   "protected). File: docstring, KIND = 'routine'|'check', def run(ctx, memo, **params) -> dict with "
                   "'ok' (+ 'detail', numbers). ctx: execute(tool, args) / move(dx,dy,dz,drz,frame,speed) / "
                   "vertical(z, step_mm, speed) / tool_hop(dx,dy,speed) — all through the safety-checked executor; "
                   "tip(), state(), wrist(color,min_area), env(color), anchor(blob), dry, routine(name, memo, **a), "
                   "check(...), contact (status/add/dz/target), table_z, block_top_z. memo['vars'][k] = v passes a "
                   "value to later skill steps as {k}. math, statistics, re available; no imports/files. The old "
                   "version is kept (rollback_routine); run_routine runs it on its own.",
                   code.write, _obj({"name": {"type": "string"}, "code": {"type": "string"},
                                     "note": {"type": "string"}}, ["name", "code"])),
        ActionSpec("rollback_routine",
                   "Restore version N of your routine as the newest; version 0 removes your file (the core one of "
                   "that name, if any, runs again).",
                   code.rollback, _obj({"name": {"type": "string"}, "version": {"type": "integer", "minimum": 0}},
                                       ["name", "version"])),
        ActionSpec("run_routine",
                   "Run one routine (or kind=check) directly to test it, outside a skill. Moves the robot in live "
                   "mode (all limits apply).",
                   run_routine, _obj({"name": {"type": "string"}, "args": {"type": "object"},
                                      "kind": {"type": "string", "enum": ["routine", "check"]}}, ["name"])),
    ]


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
    primitives.grip_speed = cfg.gripper.speed
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
    notes = NotesStore()                  # raw archive (read_history); prompts get the knowledge digest
    knowledge = KnowledgeStore()
    investigation = Investigation(gate_after_repeats=cfg.diagnosis.gate_after_repeats, knowledge=knowledge)
    path = REPO_ROOT / cfg.skills_file
    library = SkillLibrary(path)
    # a calibration file without an accepted, quality-checked solve from this session is not trusted
    if scene.env_calibration is not None and not calibration.solution:
        log.warning("env calibration has no quality record — switched off, Astra must calibrate and verify")
        _set_aside(perception_cfg.calibration_path)
        scene.env_calibration = None

    executor, kit = build_toolkit(agent_cfg.execution_mode, confirm, robot_cfg=robot_cfg, agent_cfg=agent_cfg,
                                  perception_cfg=perception_cfg, primitives=primitives, scene=scene,
                                  library=library, seed_skills=False,
                                  extra_specs=_explore_specs(calibration, guard, notes, library, knowledge)
                                  + diagnosis_specs(investigation) + code_specs(lambda: runner.local))
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
    return ExploreSession(cfg, kit, executor, harness, calibration, guard, notes, investigation, knowledge)


OBSERVED_READS = {"look", "find_objects", "verify_calibration", "solve_calibration", "run_routine"}


def _observed(executor, name: str) -> bool:
    spec = executor.specs.get(name)
    return spec is not None and (spec.moves_robot or spec.delegates or name in OBSERVED_READS)


def _snapshot(p) -> dict:
    s = p.get_state()
    return {"pose": s.get("pose"), "gripper": s.get("gripper") or {}}


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
