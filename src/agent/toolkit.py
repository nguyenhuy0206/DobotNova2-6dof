"""One place that wires everything Astra can use.

    executor, parts = build_toolkit(mode="dry_run")
    executor.execute("find_objects", {"color": "red"})

Tools come from three modules:
  actions/primitives.py   robot + gripper primitives (move_to_pose, close_gripper, ...)
  perception/scene.py     look, find_objects (camera -> robot coordinates)
  actions/skill_runner.py list_skills, run_skill, save_skill (composite skills in data/memory/skills.json)
"""
from __future__ import annotations

from dataclasses import dataclass

from actions.executor import ActionExecutor, ConfirmFn
from actions.primitives import PrimitiveSkills, create_primitives
from actions.registry import build_registry
from actions.skill_runner import SkillRunner
from common.config import (AgentConfig, PerceptionConfig, RobotConfig, load_agent_config,
                           load_perception_config, load_robot_config)
from memory.skills import SkillLibrary
from perception.scene import Scene


@dataclass
class Toolkit:
    primitives: PrimitiveSkills
    scene: Scene
    skills: SkillRunner
    robot_cfg: RobotConfig
    agent_cfg: AgentConfig
    perception_cfg: PerceptionConfig


def build_toolkit(mode: str | None = None, confirm: ConfirmFn | None = None, *, robot_cfg=None,
                  agent_cfg=None, perception_cfg=None, primitives=None, scene=None,
                  library: SkillLibrary | None = None, seed_skills: bool = True,
                  extra_specs: list | None = None) -> tuple[ActionExecutor, Toolkit]:
    robot_cfg = robot_cfg or load_robot_config()
    agent_cfg = agent_cfg or load_agent_config()
    perception_cfg = perception_cfg or load_perception_config()
    primitives = primitives or create_primitives(robot_cfg)
    primitives.speed_cap = agent_cfg.max_speed
    primitives.fixed_speed = agent_cfg.fixed_speed
    scene = scene or Scene.from_config(perception_cfg, workspace=robot_cfg.workspace)
    library = library or SkillLibrary()
    if seed_skills:
        library.ensure_seeds()
    runner = SkillRunner(library)

    kwargs = {"mode": mode or agent_cfg.execution_mode, "allowed": agent_cfg.allowed_tools}
    if confirm is not None:
        kwargs["confirm"] = confirm
    specs = build_registry(primitives, scene, runner) + list(extra_specs or [])
    executor = ActionExecutor(specs, **kwargs)
    runner.executor = executor
    from judge.capture import EvidenceCapture
    from judge.harness import HarnessJudge
    from judge.ledger import LEDGER_FILE
    from judge.spec import load_spec
    spec = load_spec()
    executor.judge = HarnessJudge(EvidenceCapture(scene, primitives.get_state, LEDGER_FILE.parent / "evidence", spec),
                                  spec=spec)
    return executor, Toolkit(primitives, scene, runner, robot_cfg, agent_cfg, perception_cfg)
