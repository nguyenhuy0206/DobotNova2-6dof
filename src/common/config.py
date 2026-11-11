"""Load config/robot.yaml and config/agent.yaml into typed dataclasses.

Environment variables override a few fields so you can switch robots without
editing YAML:  NOVA2_IP, NOVA2_UI_PORT, ASTRA_MODEL, ASTRA_EXECUTION_MODE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.paths import CONFIG_DIR, REPO_ROOT


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); does not override real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return value


@dataclass
class SpeedLimits:
    min: int = 1
    max: int = 100


@dataclass
class JointStepLimits:
    min_deg: float = 0.1
    max_deg: float = 20.0


@dataclass
class WorkspaceLimits:
    """Axis-aligned box in the robot base frame, millimetres."""
    x: tuple[float, float] = (-600.0, 600.0)
    y: tuple[float, float] = (-600.0, 600.0)
    z: tuple[float, float] = (0.0, 700.0)


@dataclass
class GripperConfig:
    slave_id: int = 1
    baud: int = 115200
    parity: str = "N"
    force: int = 30
    speed: int = 50
    open_position: int = 1000
    closed_position: int = 0


@dataclass
class RobotConfig:
    ip: str = "192.168.5.1"
    timeout_s: float = 3.0
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765
    speed: SpeedLimits = field(default_factory=SpeedLimits)
    joint_step: JointStepLimits = field(default_factory=JointStepLimits)
    workspace: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    gripper: GripperConfig = field(default_factory=GripperConfig)


@dataclass
class AgentConfig:
    model: str = "gpt-6-astra"
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 1000
    max_steps: int = 20
    # dry_run: tools are validated but never sent to the robot
    # confirm: each motion tool asks a human on stdin before running
    # live:    tools run directly (still subject to validation + safety checks)
    execution_mode: str = "dry_run"
    allowed_tools: list[str] = field(default_factory=list)  # empty = all registered
    system_prompt: str = ""


def _build(cls, data: dict):
    """Instantiate a dataclass, recursing into nested dataclass fields."""
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in data:
            continue
        value = data[name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if hasattr(default, "__dataclass_fields__") and isinstance(value, dict):
            value = _build(type(default), value)
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def load_robot_config(path: Path = CONFIG_DIR / "robot.yaml") -> RobotConfig:
    cfg = _build(RobotConfig, _read_yaml(path))
    cfg.ip = os.environ.get("NOVA2_IP", cfg.ip)
    cfg.ui_port = int(os.environ.get("NOVA2_UI_PORT", cfg.ui_port))
    return cfg


def load_agent_config(path: Path = CONFIG_DIR / "agent.yaml") -> AgentConfig:
    load_dotenv()
    cfg = _build(AgentConfig, _read_yaml(path))
    cfg.model = os.environ.get("ASTRA_MODEL", cfg.model)
    cfg.execution_mode = os.environ.get("ASTRA_EXECUTION_MODE", cfg.execution_mode)
    if cfg.execution_mode not in {"dry_run", "confirm", "live"}:
        raise ValueError("execution_mode must be dry_run, confirm or live")
    return cfg
