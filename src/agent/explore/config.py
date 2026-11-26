"""Explore-mode settings (config/explore.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field

from common.config import _build, _read_yaml
from common.paths import CONFIG_DIR

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

@dataclass
class DiagnosisPolicy:
    gate_after_repeats: int = 2   # refuse run_skill/save_skill only when the same failure repeats this often (0 = never)

@dataclass
class ExploreConfig:
    task: str = ""
    execution_mode: str = "live"
    max_steps: int = 150
    episodes: int = 1
    until_success: bool = False   # autonomous visual check/retry; episodes=0 means no attempt cap
    max_output_tokens: int = 2000
    gripper: ExploreGripper = field(default_factory=ExploreGripper)
    view_guard: ViewGuardConfig = field(default_factory=ViewGuardConfig)
    collision_level: int = 3
    speed_factor: int = 0          # controller global SpeedFactor % set at start (0 = leave)
    skills_file: str = "data/memory/skills_explore.json"
    allowed_tools: list[str] = field(default_factory=list)
    # 23 Nov (operator): Astra acts on the robot only through programs it writes (write_routine / run_routine);
    # program_tools replace allowed_tools then. Off = the turn-by-turn tool set (diagnosis, comparison runs).
    program_mode: bool = False
    program_tools: list[str] = field(default_factory=list)
    # rules the operator gives Astra on purpose (appended to the system prompt as "OPERATOR RULES")
    operator_rules: list[str] = field(default_factory=list)
    # cameras of this session (23 Nov threading task: wrist only); look offers only these
    cameras: list[str] = field(default_factory=lambda: ["env", "wrist"])
    # session limits (23 Nov threading task: the setup is aligned by the operator, Astra only nudges)
    gripper_locked: bool = False         # open/close/set_gripper (+ pick/place) refused, also inside routines
    max_step_mm: float = 0               # move_relative per call (0 = robot.yaml motion.max_relative_mm)
    max_step_deg: float = 0
    required_positions: list[str] = field(default_factory=list)   # saved positions preflight insists on
    locked_tools: list[str] = field(default_factory=list)          # refused for the model AND for routines
    reference_task: str = ""             # data/references/<task>: compare_reference (24 Nov threading)
    goal_picture: bool = False           # attach the newest 'success' reference to the task (once, at the start)
    show_knowledge: bool = True          # concept/hypothesis digest in the system prompt
    allowed_positions: list[str] = field(default_factory=list)    # go_to_position only these ([] = any)
    one_look_per_motion: bool = False    # after look/compare_reference, the next one needs a motion first
    looks_per_motion: int = 1            # with one_look_per_motion: pictures allowed between two motions
    keep_going: bool = False              # a text-only answer is verified, then sent back to continue
    calibrate_before_insert: bool = False  # insert refused until tool-frame dx and dy moves ran this episode
    reasoning_effort: str = ""           # low / medium / high for this session ("" = agent.yaml / model default)
    harness: dict = field(default_factory=lambda: {"variant": "V3"})   # see agent.harness.HarnessOptions
    skills: SkillPolicy = field(default_factory=SkillPolicy)
    diagnosis: DiagnosisPolicy = field(default_factory=DiagnosisPolicy)
    reflect_max_steps: int = 6
    # who drives the robot in an episode: "astra" (API model) or "qwen" (the local student, src/student/)
    driver: str = "astra"
    qwen: dict = field(default_factory=dict)       # model, adapter, max_output_tokens, images

def load_explore_config(path=CONFIG_DIR / "explore.yaml") -> ExploreConfig:
    return _build(ExploreConfig, _read_yaml(path))
