#!/usr/bin/env python3
"""Env-camera calibration, done by the robot itself at 10 % speed (no Astra needed).

    scripts/run_calibration.sh                    centre = current x/y, table + 150 mm
    scripts/run_calibration.sh --center -350,-300,200
    scripts/run_calibration.sh --span 80          bigger pattern (default ±60 mm)

Before: scripts/run_vision.sh running, robot enabled, red tape visible to the D455,
gripper above a FREE area in the middle of where Astra will work.
Old calibration is set aside (it was for another camera setup). Ctrl+C = stop robot.
"""
from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from agent.explore import build_explore
from common.log import setup_logging
from perception.auto_calibration import auto_calibrate, suggest_center


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", help="fingertip x,y,z in mm (default: current x,y; table + 150)")
    ap.add_argument("--span", type=float, default=100, help="± mm around the centre (default 100)")
    ap.add_argument("--points", type=int, default=12, help="stop after this many good points (default 12)")
    ap.add_argument("--yes", action="store_true")
    # "--center -300,-385,192": argparse would read "-300,…" as an option -> glue it on
    argv = sys.argv[1:]
    for i, a in enumerate(argv[:-1]):
        if a == "--center":
            argv[i:i + 2] = [f"--center={argv[i + 1]}"]
            break
    args = ap.parse_args(argv)
    setup_logging(filename="calibration.log")

    s = build_explore(mode="live", recalibrate=True, client=object())   # no model needed
    p = s.kit.primitives
    problems = s.preflight()
    if problems:
        print("Not ready:\n" + "\n".join(f"  ✗ {m}" for m in problems))
        raise SystemExit(1)
    center = [float(v) for v in args.center.split(",")] if args.center else None
    tip = p.get_state()["pose"]
    print(f"Speed {p.speed_cap}% · table z {p.table_z_mm} · fingertip now {[round(v) for v in tip[:3]]}")
    if center is None and (better := suggest_center(p)):
        print(f"!! the fingertip is near the edge of the reach (radius {round((tip[0]**2 + tip[1]**2) ** .5)} mm): "
              f"many poses will be unreachable. Better: --center {better[0]:.0f},{better[1]:.0f},"
              f"{p.table_z_mm + 150:.0f}  (if the D455 still sees the marker there)")
    print(f"Up to 24 poses within ±{args.span:g} mm (tilts/twists ≤ 20°), stops at {args.points} good points. "
          "Unreachable poses are skipped before moving; poses that leave the camera view are undone. "
          "Collision detection:", s.arm_collision_detection())
    if not args.yes:
        input("Robot will move by itself. Hand on the E-stop. Enter = start, Ctrl+C = cancel ")
    try:
        report = auto_calibrate(p, s.calibration, center=center, span_mm=args.span,
                                speed=p.speed_cap or 10, target_points=args.points)
    except KeyboardInterrupt:
        p.stop()
        print("\n!! stopped (ResetRobot)")
        raise SystemExit(130)

    print("\n" + json.dumps({k: v for k, v in report.items() if k != "skipped"}, indent=1))
    if report["skipped"]:
        print(f"skipped {len(report['skipped'])} pose(s)")
    if report["dropped_outliers"]:
        print(f"dropped {len(report['dropped_outliers'])} bad point(s): {report['dropped_outliers']}")
    loo = report["leave_one_out_mm"]["median"]
    verdict = "GOOD" if loo <= 10 else "OK — verify with the wrist camera" if loo <= 20 else "POOR — re-run"
    print(f"\nAccuracy (leave-one-out median): {loo} mm  →  {verdict}")
    s.notes.add(f"Env camera auto-calibrated: {report['points']} points, leave-one-out median {loo} mm, "
                f"max {report['leave_one_out_mm']['max']} mm ({verdict}).", "calibration")

    # Sanity check: block tops must be above the table.
    s.kit.scene.object_z_max = (p.table_z_mm + 150) if p.table_z_mm is not None else None
    look = s.kit.scene.look("env", ["blue", "green", "yellow"])
    for o in look["objects"]:
        z = o["base_xyz_mm"][2] if o["base_xyz_mm"] else None
        flag = "" if z is None or p.table_z_mm is None or z > p.table_z_mm else "  ← BELOW TABLE?!"
        print(f"  {o['label']:6s} pixel {o['pixel']} → base {o['base_xyz_mm']}{flag}")


if __name__ == "__main__":
    main()
