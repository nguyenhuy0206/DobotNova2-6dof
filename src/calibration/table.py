"""Teach a height by lowering the fingertips onto it step by step (scripts/calibrate.py table).

    table      the metal table (workcell.json table_z_mm): every motion primitive keeps the
               fingertips / flange above it by the clearances in config/robot.yaml `table:`
    cube top   a cube on the box (block_top_z_mm), with the fingers closed

Gripper pointing straight down, ~5–10 cm above the surface. Commands (Enter after each):
    d down 5 mm   f down 1 mm   u up 5 mm   s save this height   q quit
Speed 5 %, at most 300 mm down in total. The touch survey (calibration.survey) measures the
table too, with a tilted gripper and several points; this is the quick one-point way.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from calibration.workcell import load_workcell, save_block_top, save_table_height
from common.errors import RobotError
from perception.transforms import pose_to_matrix

SPEED = 5
MAX_DOWN_MM = 300
STEPS = {"d": -5.0, "f": -1.0, "u": 5.0}


def teach(block_top: bool = False, read: Callable[[str], str] = input, out: Callable[[str], None] = print) -> None:
    from actions.primitives import create_primitives
    from common.config import load_robot_config

    what, key = ("cube top", "block_top_z_mm") if block_top else ("table height", "table_z_mm")
    cfg = load_robot_config()
    # teaching needs the fingertip below the workspace floor; nothing else changes
    cfg.workspace = dataclasses.replace(cfg.workspace, z=(-600.0, cfg.workspace.z[1]))
    p = create_primitives(cfg)
    if block_top:
        out("Close the gripper fully (Debug tab) so the two fingertips touch the cube top together.")
    else:
        p.table_z_mm = None                       # no keep-out while we look for the table

    state = p.get_state()
    tilt = float(abs(pose_to_matrix(state["flange_pose"])[2, 2]))
    out(f"Active tool offset from flange: {[round(x, 1) for x in p.active_tool_offset]} mm ({p.tool_offset_source})")
    out(f"Fingertip now: {state['pose'][:3]}   (gripper "
        f"{'pointing down' if tilt > 0.98 else 'TILTED — point it down first'})")
    old = load_workcell().get(key)
    if old is not None:
        out(f"Currently saved {what}: {old} mm")
    start_z = state["pose"][2]

    while True:
        cmd = read("[d/f/u/s/q] > ").strip().lower()
        if cmd == "q":
            out("Quit without saving.")
            return
        if cmd == "s":
            z = p.get_state()["pose"][2]
            if block_top:
                data = save_block_top(z, method="touch with closed fingertips")
                out(f"Saved block_top_z_mm = {data['block_top_z_mm']} -> data/memory/workcell.json. Move up now (u).")
                continue
            data = save_table_height(z, method="touch with fingertips")
            out(f"Saved table_z_mm = {data['table_z_mm']} -> data/memory/workcell.json")
            out(f"Fingertips will stay ≥ {z + cfg.table.tip_clearance_mm:.1f}, flange ≥ "
                f"{z + cfg.table.flange_clearance_mm:.1f}. Move up now (u) or from the Debug tab.")
            continue
        if cmd not in STEPS:
            continue
        dz = STEPS[cmd]
        z = p.get_state()["pose"][2]
        if dz < 0 and start_z - (z + dz) > MAX_DOWN_MM:
            out(f"!! more than {MAX_DOWN_MM} mm below the start — refusing. Start closer to the table.")
            continue
        try:
            r = p.move_relative(dz=dz, speed=SPEED)
            out(f"   fingertip z = {r['pose'][2]:.1f} mm")
        except RobotError as exc:
            out(f"!! {exc}")
