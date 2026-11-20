#!/usr/bin/env python3
"""Every calibration of the cell, one command (run through scripts/calibrate.sh: project venv).

  cameras   both cameras from the ChArUco board (calibration.cameras)
      calibrate.sh cameras --check            what each camera sees now (no motion)
      calibrate.sh cameras [--dry]            robot aims the wrist camera at the board and orbits it
      calibrate.sh cameras --solve DIR [--apply]   solve a saved run again (no robot)
  survey    the cell measured by hand-guided touches (calibration.survey); positions are names
            saved in the Debug tab (Drag on -> move by hand -> Save position)
      calibrate.sh survey contact VERTICAL TILT_A TILT_B   gripper contact model (tool length, fingertips)
      calibrate.sh survey points NAME...                    where those touches were (lowest contact)
      calibrate.sh survey table NAME... [--apply]           table height from touches on it
      calibrate.sh survey keepout NAME... [--name D455]     keep-out cylinder around an obstacle
  table     teach the table (or --block-top) height by lowering the fingertips (calibration.table)
  colors    save each colour mask of the env camera to data/captures/ (check the HSV ranges)

Needs scripts/run_vision.sh running for anything that uses the cameras or the robot.
Results: calibration.files lists every file written (all under data/, git-ignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from common.log import setup_logging


# --------------------------------------------------------------------------------------------- cameras

def _show(rep: dict) -> None:
    from calibration.cameras import public
    print("\n" + json.dumps(public({k: v for k, v in rep.items() if k not in ("problems", "warnings")}),
                            indent=1, default=str))
    for w in rep["warnings"]:
        print(f"  ! {w}")
    for p in rep["problems"]:
        print(f"  ✗ {p}")
    print(f"\nwrist: {'ACCEPTED' if rep['wrist_accepted'] else 'rejected'} · "
          f"D455: {'ACCEPTED' if rep['env_accepted'] else 'rejected'}")


def _check(scene, cfg) -> None:
    from calibration import board
    from calibration.cameras import camera_matrix_of
    for cam in (cfg.env_camera, cfg.wrist_camera):
        f = scene.observe(cam)
        det = board.detect(f.color, cfg.board, scales=(1, 2, 3))
        n = 0 if det is None else len(det)
        need = cfg.gates.min_env_corners if cam == cfg.env_camera else board.MIN_CORNERS
        line = f"  {cam:6s} {n:2d} / {(cfg.board.squares[0] - 1) * (cfg.board.squares[1] - 1)} corners"
        if det is not None and cam == cfg.env_camera and f.intrinsics is not None:
            T, err = board.board_pose(det, cfg.board, camera_matrix_of(f.intrinsics))
            d = (T[:3, :3] @ cfg.board.center_m + T[:3, 3]) * 1000
            line += f" · board {float((d ** 2).sum() ** .5):.0f} mm away · reprojection {err:.2f} px"
        print(line + ("  ✓" if n >= need else f"  ✗ need {need}"))


def cameras(args) -> None:
    from calibration import cameras as cc
    from common.config import load_perception_config
    pcfg = load_perception_config()
    cfg = cc.CalibrationConfig.from_dict(pcfg.calibration)

    if args.solve:
        session = Path(args.solve)
        rep = cc.solve_session(session, cfg)
        _show(rep)
        if args.apply and (rep["wrist_accepted"] or rep["env_accepted"]):
            cc.save(rep, pcfg.calibration_path, session.name)
            print("saved:", [k for k in ("wrist", "env") if rep[f"{k}_accepted"]])
        return
    if args.check:
        from perception.scene import Scene
        _check(Scene.from_config(pcfg), cfg)
        return

    from agent.explore import build_explore
    s = build_explore(mode="live", client=object())          # no model needed
    p = s.kit.primitives
    problems = s.preflight()
    if problems:
        print("Not ready:\n" + "\n".join(f"  ✗ {m}" for m in problems))
        raise SystemExit(1)
    prior_dir = cc.last_session() if args.prior == "last" else (Path(args.prior) if args.prior else None)
    prior = cc.prior_from_session(prior_dir, cfg) if prior_dir else None
    print(f"Board {cfg.board.squares[0]}x{cfg.board.squares[1]} · square {cfg.board.square_mm} mm · marker "
          f"{cfg.board.marker_mm} mm · speed {p.speed_cap}% · fingertip {[round(v) for v in p.get_state()['pose'][:3]]}")
    print(f"Aim / orbit from: {prior_dir or 'none (pivot about the tool axis; the wrist camera must see the board)'}")
    _check(s.kit.scene, cfg)
    if not args.yes:
        input("Robot will move by itself (10 %). Hand on the E-stop. Enter = start, Ctrl+C = cancel ")
    try:
        rep = cc.run(p, s.kit.scene, cfg, pcfg.calibration_path, speed=p.speed_cap or 10, max_views=args.views,
                     apply=not args.dry, prior=prior)
    except KeyboardInterrupt:
        p.stop()
        print("\n!! stopped (Pause)")
        raise SystemExit(130)
    _show(rep)
    print("saved:", rep["saved"] or "nothing", "· session:", rep["session"])


# ---------------------------------------------------------------------------------------------- survey

def survey(args) -> None:
    from calibration import survey as sv
    from memory.store import PositionStore
    saved = {p["name"]: p for p in PositionStore().all()}

    def flange(name: str):
        if name not in saved:
            raise SystemExit(f"no saved position '{name}' (have: {', '.join(saved)})")
        return sv.flange_from_joints(saved[name]["joints_deg"])

    if args.what == "contact":
        if len(args.names) != 3:
            raise SystemExit("contact needs 3 names: VERTICAL TILT_A TILT_B (one dot; centre, then each fingertip)")
        m = sv.fit_contact(*(flange(n) for n in args.names))
        print(json.dumps(m.to_dict(), indent=1))
        print(f"dot {m.dot_mm.round(1).tolist()} · residual per touch {[round(r, 1) for r in m.residual_mm]} mm")
        if max(m.residual_mm) > 3:
            print("!! residual over 3 mm: the three touches were not on one dot — redo them; nothing saved")
            return
        sv.save({"gripper": m.to_dict(), "contact_touches": args.names})
        print(f"saved -> {sv.SURVEY_FILE}\nconfig/robot.yaml gripper: tool_length_mm: {m.tool_length_mm:.1f}, "
              f"finger_half_gap_mm: {m.finger_half_gap_mm:.1f}")
        return

    model = sv.saved_model()
    if model is None:
        raise SystemExit("no contact model yet: run  survey contact VERTICAL TILT_A TILT_B  first")
    pts = {n: sv.lowest_contact(flange(n), model) for n in args.names}
    if args.what == "points":
        for n, q in pts.items():
            print(f"  {n:24s} {q.round(1).tolist()}  r={float((q[0] ** 2 + q[1] ** 2) ** .5):.0f}")
        sv.save({"points_mm": {**sv.load().get("points_mm", {}), **{n: q.round(1).tolist() for n, q in pts.items()}}})
    elif args.what == "table":
        h = sv.table_height([q[2] for q in pts.values()])
        print(f"table z {h}")
        sv.save({"table_z_mm": h})
        if args.apply:
            from calibration.workcell import save_table_height
            save_table_height(h["mean"], method=f"touch survey ({h['n']} points, {h['min']}..{h['max']})")
            print("applied -> data/memory/workcell.json")
    elif args.what == "keepout":
        cands = [q for n in args.names for q in sv.contact_points(flange(n), model).values()]
        c = sv.enclosing_cylinder(cands, args.margin)
        top = c["z_touched"][1] + args.margin
        print(json.dumps(c))
        print(f"config/robot.yaml workspace.keep_out:\n  - {{name: {args.name}, xy: {c['xy']}, "
              f"radius_mm: {c['radius_mm']}, z: [0, {round(top)}]}}")
        sv.save({f"keepout_{args.name}": c})


# ---------------------------------------------------------------------------------------- table, colors

def table(args) -> None:
    from calibration.table import teach
    teach(block_top=args.block_top)


def colors(args) -> None:
    import cv2
    import numpy as np

    from common.paths import CAPTURES_DIR
    from perception.scene import Scene
    scene = Scene.from_config()
    frame = scene.observe("env")
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_BGR2HSV)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    for color, ranges in scene.ranges.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        path = CAPTURES_DIR / f"tune_{color}.jpg"
        cv2.imwrite(str(path), np.hstack([frame.color, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        print(f"{color:7s} {len(scene.find(color, frame=frame))} blob(s)  -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cameras", help="both cameras from the ChArUco board")
    c.add_argument("--check", action="store_true", help="only report what each camera sees (no motion)")
    c.add_argument("--solve", help="solve a saved session directory again (no robot)")
    c.add_argument("--apply", action="store_true", help="with --solve: save the result if it passes")
    c.add_argument("--dry", action="store_true", help="full run, but save nothing")
    c.add_argument("--views", type=int, default=16, help="stop after this many good wrist views")
    c.add_argument("--prior", default="last",
                   help="session dir to aim the wrist camera from ('last' = the last saved calibration, "
                        "'' = none: the wrist camera must already see the board)")
    c.add_argument("--yes", action="store_true", help="no confirmation prompt")
    c.set_defaults(fn=cameras)
    s = sub.add_parser("survey", help="the cell measured by hand-guided touches")
    s.add_argument("what", choices=["contact", "points", "table", "keepout"])
    s.add_argument("names", nargs="+", help="saved position names")
    s.add_argument("--apply", action="store_true", help="table: write the height to workcell.json")
    s.add_argument("--name", default="obstacle", help="keepout: name of the obstacle")
    s.add_argument("--margin", type=float, default=40.0, help="keepout: margin around the touches (mm)")
    s.set_defaults(fn=survey)
    t = sub.add_parser("table", help="teach the table height by lowering the fingertips")
    t.add_argument("--block-top", action="store_true", help="the top of a cube on the box instead")
    t.set_defaults(fn=table)
    sub.add_parser("colors", help="save the colour masks of the env camera").set_defaults(fn=colors)
    args = ap.parse_args()
    setup_logging(filename="calibration.log")
    args.fn(args)


if __name__ == "__main__":
    main()
