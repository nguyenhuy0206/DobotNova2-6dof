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
    max_output_tokens: int = 2000
    gripper: ExploreGripper = field(default_factory=ExploreGripper)
    view_guard: ViewGuardConfig = field(default_factory=ViewGuardConfig)
    collision_level: int = 3
    speed_factor: int = 0          # controller global SpeedFactor % set at start (0 = leave)
    skills_file: str = "data/memory/skills_explore.json"
    allowed_tools: list[str] = field(default_factory=list)
    harness: dict = field(default_factory=lambda: {"variant": "V3"})   # see agent.harness.HarnessOptions
    skills: SkillPolicy = field(default_factory=SkillPolicy)
    diagnosis: DiagnosisPolicy = field(default_factory=DiagnosisPolicy)
    reflect_max_steps: int = 6
    # who drives the robot in an episode: "astra" (API model) or "qwen" (the local student, src/student/)
    driver: str = "astra"
    qwen: dict = field(default_factory=dict)       # model, adapter, max_output_tokens, images

def load_explore_config(path=CONFIG_DIR / "explore.yaml") -> ExploreConfig:
    return _build(ExploreConfig, _read_yaml(path))
