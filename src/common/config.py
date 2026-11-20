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
    """Robot base frame, millimetres: an axis-aligned box, optionally cut to a horizontal reach
    radius around the base axis, minus vertical keep-out cylinders (fixed things in the cell)."""
    x: tuple[float, float] = (-600.0, 600.0)
    y: tuple[float, float] = (-600.0, 600.0)
    z: tuple[float, float] = (0.0, 700.0)
    max_radius_mm: float | None = None      # sqrt(x² + y²) limit; None = box only
    # [{"name": "D455", "xy": [x, y], "radius_mm": r, "z": [lo, hi]}, ...]
    keep_out: list = field(default_factory=list)


@dataclass
class MotionConfig:
    """Limits/tolerances for the primitive skills (actions/primitives.py)."""
    # Nova 2 joint ranges, degrees (J1..J6)
    joint_limits_deg: list[float] = field(default_factory=lambda: [360, 180, 156, 360, 360, 360])
    max_relative_mm: float = 100.0      # biggest single relative translation
    max_relative_deg: float = 45.0      # biggest single relative rotation
    default_speed: int = 20             # % used when a skill call omits speed
    arrive_tol_mm: float = 1.0
    arrive_tol_deg: float = 0.5
    move_timeout_s: float = 30.0
    start_grace_s: float = 0.5          # time for the controller to start a queued move
    poll_s: float = 0.05
    # Approach zone (needs a taught table): fingertip below table + approach_height_mm.
    # Inside it every Cartesian move is short and slow, and a stalled move = contact:
    # the robot stops, backs off and the call fails with "contact at z=…".
    approach_height_mm: float = 120.0
    approach_max_step_mm: float = 25.0
    approach_speed: int = 3
    stall_s: float = 0.3           # no progress for this long while moving -> contact
    stall_mm: float = 0.3
    contact_backoff_mm: float = 15.0
    # a controller collision alarm during an approach-zone move is a contact too: clear it,
    # check the robot is enabled again, back off and fail with "CONTACT …" (false = stop as before)
    collision_recovery: bool = True
    # gripper vs the robot's own elbow / upper arm / forearm (actions/self_collision.py): moves whose
    # path comes closer than this (gripper axis to link axis) are refused before moving; 0 = off
    self_clearance_mm: float = 65.0
    # wrist links vs the upper arm when the elbow folds (same module); 0 = off
    wrist_clearance_mm: float = 190.0
    # elbow / wrist joint centres vs the table (same module): reaching low close to the base folds the
    # elbow down onto the table (19 Nov); 0 = off
    elbow_table_clearance_mm: float = 60.0
    recovery_timeout_s: float = 12.0
    home_joints: list[float] = field(default_factory=lambda: [0.0, 0.0, 90.0, 0.0, 90.0, 0.0])


@dataclass
class GripperConfig:
    # Flange -> fingertip distance along the tool Z axis. Motion commands use Tool 0
    # (the flange), so pick/place heights add this. MEASURE IT before picking.
    tool_length_mm: float = 0.0
    finger_half_gap_mm: float = 0.0     # open fingertips off the tool axis (along flange Y); 0 = ignore
    transport: str = "direct"           # direct (USB–RS485) | controller
    # true: only ui/app.py opens the bus; every other process (episodes, scripts) sends its gripper
    # calls to the app (drivers.gripper.PGCGripperRemote) — one Modbus master, no clashes
    via_ui: bool = False
    port: str | None = None             # None = auto-detect /dev/cu.usbserial-*
    slave_id: int = 1
    baud: int = 115200
    parity: str = "N"
    force: int = 30
    speed: int = 50
    open_position: int = 1000
    closed_position: int = 0


@dataclass
class TableConfig:
    """Keep-out above the table (table height itself: data/memory/workcell.json)."""
    tip_clearance_mm: float = 10.0      # fingertips never closer to the table than this
    flange_clearance_mm: float = 60.0   # flange/gripper body (matters when the gripper tilts)
    movj_min_height_mm: float = 120.0   # below this above the table, Cartesian moves go straight (MovL)
    path_step_deg: float = 2.0          # joint moves: FK check every N degrees along the path


@dataclass
class CameraConfig:
    """One camera window in the Debug tab (see drivers/camera_stream.py)."""
    label: str = "Camera"
    source: str = "opencv"      # opencv (any UVC cam, colour) | realsense (D455 colour + depth)
    index: int = 0              # OpenCV device number (macOS order != name order; pick in the UI)
    width: int = 640            # requested capture size
    height: int = 480
    fps: float = 10.0           # frames streamed to the browser
    jpeg_quality: int = 70
    helper_url: str = "http://127.0.0.1:8766"   # source=realsense: depth helper (scripts/run_depth.sh)


def _default_cameras() -> dict:
    return {"env": CameraConfig(label="Environment · RealSense D455", index=1, width=1280, height=720),
            "wrist": CameraConfig(label="Wrist · USB webcam", index=0)}


@dataclass
class RobotConfig:
    ip: str = "192.168.5.1"
    timeout_s: float = 3.0
    ui_host: str = "127.0.0.1"
    ui_port: int = 8765
    speed: SpeedLimits = field(default_factory=SpeedLimits)
    joint_step: JointStepLimits = field(default_factory=JointStepLimits)
    workspace: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    motion: MotionConfig = field(default_factory=MotionConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    table: TableConfig = field(default_factory=TableConfig)
    cameras: dict = field(default_factory=_default_cameras)   # name -> CameraConfig


@dataclass
class AgentConfig:
    model: str = "gpt-6-astra"
    # Variable holding the API key, looked up in the environment after .env is loaded.
    # Falls back to the SDK default (OPENAI_API_KEY exported in your shell).
    api_key_env: str = "gpt6-key"
    # OpenAI-compatible endpoint; "" = the SDK default (OPENAI_BASE_URL or api.openai.com)
    base_url: str = ""
    max_output_tokens: int = 1000
    max_steps: int = 20
    max_speed: int = 10        # % cap on every robot motion the agent commands
    # dry_run: tools are validated but never sent to the robot
    # confirm: each motion tool asks a human on stdin before running
    # live:    tools run directly (still subject to validation + safety checks)
    execution_mode: str = "dry_run"
    allowed_tools: list[str] = field(default_factory=list)  # empty = all registered
    system_prompt: str = ""
    truncation: str = ""       # "auto": let the API drop old turns in long runs (explore mode)


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
    data = _read_yaml(path)
    cameras = data.pop("cameras", None)
    cfg = _build(RobotConfig, data)
    if cameras:
        cfg.cameras = {name: _build(CameraConfig, value or {}) for name, value in cameras.items()}
    cfg.ip = os.environ.get("NOVA2_IP", cfg.ip)
    cfg.ui_port = int(os.environ.get("NOVA2_UI_PORT", cfg.ui_port))
    return cfg


def load_agent_config(path: Path = CONFIG_DIR / "agent.yaml") -> AgentConfig:
    load_dotenv()
    cfg = _build(AgentConfig, _read_yaml(path))
    cfg.model = os.environ.get("ASTRA_MODEL", cfg.model)
    cfg.base_url = os.environ.get("ASTRA_BASE_URL", cfg.base_url)
    cfg.execution_mode = os.environ.get("ASTRA_EXECUTION_MODE", cfg.execution_mode)
    if cfg.execution_mode not in {"dry_run", "confirm", "live"}:
        raise ValueError("execution_mode must be dry_run, confirm or live")
    return cfg


@dataclass
class PerceptionConfig:
    colors: dict = field(default_factory=dict)          # name -> [[lo HSV], [hi HSV]] list
    min_area_px: float = 400.0
    depth_window_px: int = 5
    env_helper_url: str = "http://127.0.0.1:8766"
    ui_url: str = "http://127.0.0.1:8765"
    env_calibration_file: str = "data/memory/hand_eye_env.json"
    send_images_to_model: bool = True
    image_max_width: int = 640
    calibration: dict = field(default_factory=dict)    # ChArUco board + tape-measured cell (perception.charuco_calibration)

    @property
    def calibration_path(self) -> Path:
        path = Path(self.env_calibration_file)
        return path if path.is_absolute() else REPO_ROOT / path


def load_perception_config(path: Path = CONFIG_DIR / "perception.yaml") -> PerceptionConfig:
    return _build(PerceptionConfig, _read_yaml(path))
