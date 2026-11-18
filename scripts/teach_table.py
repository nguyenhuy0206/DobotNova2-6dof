#!/usr/bin/env python3
"""Teach the metal table height once (data/memory/workcell.json -> table_z_mm).

After this, every motion primitive (Debug tab and Astra) keeps the fingertips and
the flange above the table by the clearances in config/robot.yaml `table:`.

    .venv/bin/python scripts/teach_table.py
    .venv/bin/python scripts/teach_table.py --block-top   cube top on the box instead
                     (close the gripper fully first: the closed fingertips are the probe)

1. Bring the gripper (pointing straight down) over a FREE spot of the table,
   ~5–10 cm above it (Debug tab), then run this script.
2. Lower step by step until the fingertips JUST touch the table:
     d + Enter   down 5 mm        f + Enter   down 1 mm
     u + Enter   up 5 mm          s + Enter   save this height     q + Enter   quit
   Speed 5 %, at most 300 mm down in total. Robot alarm/collision = stop.
"""
from __future__ import annotations

import dataclasses

import _bootstrap  # noqa: F401
from actions.primitives import create_primitives
from common.config import load_robot_config
from common.errors import RobotError
from memory.workcell import load_workcell, save_block_top, save_table_height
from perception.transforms import pose_to_matrix

SPEED = 5
MAX_DOWN_MM = 300
STEPS = {"d": -5.0, "f": -1.0, "u": 5.0}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-top", action="store_true", help="teach the top of a cube on the box")
    block = ap.parse_args().block_top
    what, key = ("cube top", "block_top_z_mm") if block else ("table height", "table_z_mm")
    cfg = load_robot_config()
    # Teaching needs the tip below the placeholder workspace floor; nothing else changes.
    cfg.workspace = dataclasses.replace(cfg.workspace, z=(-600.0, cfg.workspace.z[1]))
    p = create_primitives(cfg)
    if not block:
        p.table_z_mm = None                 # no keep-out while we look for the table
    else:
        print("Close the gripper fully (Debug tab) so the two fingertips touch the cube top together.")

    state = p.get_state()
    tilt = float(abs(pose_to_matrix(state["flange_pose"])[2, 2]))
    print(f"Active tool offset from flange: {[round(x, 1) for x in p.active_tool_offset]} mm "
          f"({p.tool_offset_source})")
    print(f"Fingertip now: {state['pose'][:3]}   (gripper {'pointing down' if tilt > 0.98 else 'TILTED — point it down first'})")
    old = load_workcell().get(key)
    if old is not None:
        print(f"Currently saved {what}: {old} mm")
    start_z = state["pose"][2]

    while True:
        cmd = input("[d/f/u/s/q] > ").strip().lower()
        if cmd == "q":
            print("Quit without saving.")
            return
        if cmd == "s":
            z = p.get_state()["pose"][2]
            if block:
                data = save_block_top(z, method="touch with closed fingertips")
                print(f"Saved block_top_z_mm = {data['block_top_z_mm']}  -> data/memory/workcell.json. Move up now (u).")
                continue
            data = save_table_height(z, method="touch with fingertips")
            print(f"Saved table_z_mm = {data['table_z_mm']}  -> data/memory/workcell.json")
            print(f"Fingertips will stay ≥ {z + cfg.table.tip_clearance_mm:.1f}, flange ≥ "
                  f"{z + cfg.table.flange_clearance_mm:.1f}. Move up now (u) or from the Debug tab.")
            continue
        if cmd not in STEPS:
            continue
        dz = STEPS[cmd]
        z = p.get_state()["pose"][2]
        if dz < 0 and start_z - (z + dz) > MAX_DOWN_MM:
            print(f"!! more than {MAX_DOWN_MM} mm below the start — refusing. Start closer to the table.")
            continue
        try:
            r = p.move_relative(dz=dz, speed=SPEED)
            print(f"   fingertip z = {r['pose'][2]:.1f} mm")
        except RobotError as exc:
            print(f"!! {exc}")


if __name__ == "__main__":
    main()
