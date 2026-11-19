"""Where every calibration result lives (all under data/, git-ignored; one place to look).

    data/memory/hand_eye_env.json        D455 -> robot base (Scene reads it; path in config/perception.yaml)
    data/memory/hand_eye_wrist.json      flange -> wrist camera
    data/memory/calibration_charuco.json the last accepted ChArUco calibration, everything about it
    data/memory/workcell.json            table height (and cube-top height) the motion guards use
    data/memory/workcell_survey.json     hand-guided touch survey: gripper contact model, points
    data/calibration/<stamp>/            each ChArUco run: images, views.json, report.json
A file that gets replaced goes to data/archive/<date>/ (common.paths.archive).
"""
from __future__ import annotations

from common.paths import DATA_DIR, MEMORY_DIR, archive

SESSIONS_DIR = DATA_DIR / "calibration"
CALIBRATION_FILE = MEMORY_DIR / "calibration_charuco.json"
WRIST_FILE = MEMORY_DIR / "hand_eye_wrist.json"
WORKCELL_FILE = MEMORY_DIR / "workcell.json"
SURVEY_FILE = MEMORY_DIR / "workcell_survey.json"


def backup(path, stamp: str) -> None:
    """Move the file being replaced to data/archive/<date>/<name>.<stamp>.json."""
    archive(path, tag=stamp)
