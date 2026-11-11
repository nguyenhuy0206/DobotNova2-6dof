"""Filesystem locations, all relative to the repository root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
UI_DIR = REPO_ROOT / "ui"
DATA_DIR = REPO_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
LOGS_DIR = DATA_DIR / "logs"
EPISODES_DIR = DATA_DIR / "episodes"
MEMORY_DIR = DATA_DIR / "memory"
POSITIONS_FILE = MEMORY_DIR / "positions.json"
