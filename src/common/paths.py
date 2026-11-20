"""Filesystem locations, all relative to the repository root.

data/ holds what the system produces (git-ignored). data/memory/ holds only the files in use;
anything replaced or set aside goes to data/archive/<date>/ through archive(), never next to them.
"""
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
UI_DIR = REPO_ROOT / "ui"
DATA_DIR = REPO_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
LOGS_DIR = DATA_DIR / "logs"
EPISODES_DIR = DATA_DIR / "episodes"
MEMORY_DIR = DATA_DIR / "memory"
ARCHIVE_DIR = DATA_DIR / "archive"
POSITIONS_FILE = MEMORY_DIR / "positions.json"
# what the local student model (Qwen) produced while driving the robot: kept apart from Astra's memory
STUDENT_RUNS_DIR = DATA_DIR / "student" / "robot_runs"
REALSENSE_LIB_DIR = REPO_ROOT / "third_party" / "librealsense" / "lib"


def archive(path: Path, tag: str = "", copy: bool = False) -> Path | None:
    """Move (or copy) a data file out of the way into data/archive/<YYYY-MM-DD>/<name>.<tag or HHMMSS><ext>.
    Returns where it went, or None if there was no file."""
    path = Path(path)
    if not path.exists():
        return None
    folder = ARCHIVE_DIR / time.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{path.stem}.{tag or time.strftime('%H%M%S')}{path.suffix}"
    n = 1
    while target.exists():
        target = folder / f"{path.stem}.{tag or time.strftime('%H%M%S')}-{n}{path.suffix}"
        n += 1
    (shutil.copy2 if copy else shutil.move)(str(path), str(target))
    return target
