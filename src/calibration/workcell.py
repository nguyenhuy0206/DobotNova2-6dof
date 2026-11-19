"""Measured facts about the work cell (data/memory/workcell.json).

    table_z_mm   fingertip Z (robot base frame) when the fingertips just touch the
                 metal table. Taught with scripts/calibrate.py table (or the touch
                 survey, scripts/calibrate.py survey). Every motion
                 primitive refuses to bring the fingertip / flange closer to it than
                 the clearances in config/robot.yaml `table:`.
    block_top_z_mm  fingertip Z touching the top of a cube on the box (scripts/calibrate.py
                 table --block-top). Pick skills use it instead of the env camera's Z, which
                 can be off by 1–2 cm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from calibration.files import WORKCELL_FILE


def load_workcell(path: Path = WORKCELL_FILE) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def save_block_top(z_mm: float, path: Path = WORKCELL_FILE, **extra) -> dict:
    """Fingertip Z when the (closed) fingertips just touch the TOP of a cube on the box."""
    data = load_workcell(path)
    data.update({"block_top_z_mm": round(float(z_mm), 2), "block_top_taught_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 **extra})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def save_table_height(z_mm: float, path: Path = WORKCELL_FILE, **extra) -> dict:
    data = load_workcell(path)
    data.update({"table_z_mm": round(float(z_mm), 2), "table_taught_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 **extra})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
