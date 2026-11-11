"""Shared data shapes passed between drivers, actions, agent and UI."""
from __future__ import annotations

from typing import TypedDict


class Feedback(TypedDict):
    """Decoded V3 realtime feedback (port 30004)."""
    mode: int          # 5 = enabled & idle, 7 = running, 9 = error …
    enabled: int
    running: int
    error: int
    robot_type: int    # Nova 2 = 101
    joints_deg: list[float]   # J1..J6
    pose: list[float]         # X, Y, Z (mm), Rx, Ry, Rz (deg)


class SavedPosition(TypedDict):
    id: str
    name: str
    joints_deg: list[float]
    pose: list[float]


class ActionResult(TypedDict, total=False):
    ok: bool
    action: str
    executed: bool     # False in dry-run mode
    result: object
    error: str
