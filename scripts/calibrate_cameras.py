#!/usr/bin/env python3
"""Calibrate BOTH cameras from the ChArUco board (config/perception.yaml calibration:).

    scripts/calibrate_cameras.py --check          no robot motion: does each camera see the board now?
    scripts/calibrate_cameras.py                  full run (robot pivots the wrist camera around the board)
    scripts/calibrate_cameras.py --solve data/calibration/<stamp>     solve a saved run again, offline
    scripts/calibrate_cameras.py --solve <dir> --apply                ... and save it if it passes

Before a full run: scripts/run_vision.sh running, robot enabled, the board where BOTH cameras see it
(lying flat, or leaning toward the D455 — the D455 sees the table at a grazing angle), and the gripper
above the board with the wrist camera looking at it from ~20–30 cm. Ctrl+C = stop robot.
Results: data/memory/hand_eye_env.json (the D455, used by Astra), data/memory/hand_eye_wrist.json,
data/memory/calibration_charuco.json (everything), data/calibration/<stamp>/ (images + report).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from common.config import load_perception_config
from common.log import setup_logging
from perception import charuco as ch
from perception import charuco_calibration as cc


def show(rep: dict) -> None:
    print("\n" + json.dumps(cc.public({k: v for k, v in rep.items() if k not in ("problems", "warnings")}),
                            indent=1, default=str))
    for w in rep["warnings"]:
        print(f"  ! {w}")
    for p in rep["problems"]:
        print(f"  ✗ {p}")
    print(f"\nwrist: {'ACCEPTED' if rep['wrist_accepted'] else 'rejected'} · "
          f"D455: {'ACCEPTED' if rep['env_accepted'] else 'rejected'}")


def check(scene, cfg: cc.CalibrationConfig) -> None:
    for cam in (cfg.env_camera, cfg.wrist_camera):
        f = scene.observe(cam)
        det = ch.detect(f.color, cfg.board, scales=(1, 2, 3))
        n = 0 if det is None else len(det)
        need = cfg.gates.min_env_corners if cam == cfg.env_camera else ch.MIN_CORNERS
        line = f"  {cam:6s} {n:2d} / {(cfg.board.squares[0] - 1) * (cfg.board.squares[1] - 1)} corners"
        if det is not None and cam == cfg.env_camera and f.intrinsics is not None:
            T, err = ch.board_pose(det, cfg.board, cc._K_from(f.intrinsics))
            d = (T[:3, :3] @ cfg.board.center_m + T[:3, 3]) * 1000
            line += f" · board {float((d ** 2).sum() ** .5):.0f} mm away · reprojection {err:.2f} px"
        print(line + ("  ✓" if n >= need else f"  ✗ need {need}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what each camera sees (no motion)")
    ap.add_argument("--solve", help="solve a saved session directory again (no robot)")
    ap.add_argument("--apply", action="store_true", help="with --solve: save the result if it passes")
    ap.add_argument("--views", type=int, default=16, help="stop after this many good wrist views")
    ap.add_argument("--dry", action="store_true", help="full run, but do not save the result")
    ap.add_argument("--prior", help="an earlier session dir: aims the wrist camera at the board and orbits it "
                                    "(needed when the camera looks at an angle to the gripper)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    setup_logging(filename="calibration.log")
    pcfg = load_perception_config()
    cfg = cc.CalibrationConfig.from_dict(pcfg.calibration)

    if args.solve:
        session = Path(args.solve)
        rep = cc.solve_session(session, cfg)
        show(rep)
        if args.apply and (rep["wrist_accepted"] or rep["env_accepted"]):
            cc.save(rep, pcfg.calibration_path, session.name)
            print("saved:", [k for k in ("wrist", "env") if rep[f"{k}_accepted"]])
        return

    if args.check:
        from perception.scene import Scene
        check(Scene.from_config(pcfg), cfg)
        return
    from agent.explore import build_explore
    s = build_explore(mode="live", client=object())          # no model needed
    p = s.kit.primitives
    problems = s.preflight()
    if problems:
        print("Not ready:\n" + "\n".join(f"  ✗ {m}" for m in problems))
        raise SystemExit(1)
    print(f"Board {cfg.board.squares[0]}x{cfg.board.squares[1]} · square {cfg.board.square_mm} mm · marker "
          f"{cfg.board.marker_mm} mm · speed {p.speed_cap}% · fingertip {[round(v) for v in p.get_state()['pose'][:3]]}")
    check(s.kit.scene, cfg)
    print(f"The robot tilts the gripper up to 18° and twists up to 40° about the board centre ({len(cc.VIEWS)} poses, "
          "never closer than the start). Collision detection:", s.arm_collision_detection())
    if not args.yes:
        input("Robot will move by itself. Hand on the E-stop. Enter = start, Ctrl+C = cancel ")
    try:
        prior = cc.prior_from_session(Path(args.prior), cfg) if args.prior else None
        rep = cc.run(p, s.kit.scene, cfg, pcfg.calibration_path, speed=p.speed_cap or 10, max_views=args.views,
                     apply=not args.dry, prior=prior)
    except KeyboardInterrupt:
        p.stop()
        print("\n!! stopped (ResetRobot)")
        raise SystemExit(130)
    show(rep)
    print("saved:", rep["saved"] or "nothing", "· session:", rep["session"])
    s.knowledge.obs.add({"action": "calibrate_cameras", "ok": bool(rep["accepted"]),
                         "calibration.wrist_views": rep["wrist_views"],
                         "calibration.wrist_px": (rep.get("wrist") or {}).get("hand_eye_rms_px"),
                         "calibration.env_px": (rep.get("env") or {}).get("reproj_px")},
                        episode=None, source="calibrate_cameras", note="; ".join(rep["problems"])[:300])


if __name__ == "__main__":
    main()
