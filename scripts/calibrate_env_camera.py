#!/usr/bin/env python3
"""Calibrate the D455 (env camera) to the robot base, so detections get base_xyz_mm.

Needs: depth helper + UI running (scripts/run_ui.sh --depth), robot reachable.

RECOMMENDED — touch mode (--touch): uses the same red block and detection that
picking uses, so their biases cancel, and the gripper length doesn't matter.
For each of 8+ spots spread over the table:
  1. put the block there, move the arm OUT of the camera's view, press Enter
     (the camera measures the block)
  2. jog the fingertips to the block's TOP CENTRE, just touching, press Enter
     (the robot position is recorded; any gripper tilt is fine, it is taken into account)
Then `s` to solve.

    python scripts/calibrate_env_camera.py --touch --color red
    python scripts/calibrate_env_camera.py --check --color red     # test accuracy on new spots

Marker mode (no --touch, older): a marker fixed somewhere on the gripper; its
position is solved too, which needs large tool rotations between points.

1. Put a small coloured marker on the gripper fingertips (e.g. a red sticker, or
   hold the red block). Set gripper.tool_length_mm in config/robot.yaml first.
2. Run this script, then for 8+ different positions spread over the work area
   (different X, Y and heights):
     - jog the robot in the Debug tab so the marker is clearly visible to the D455
     - press Enter here to record (robot TCP + where the camera sees the marker)
3. Type `s` to solve. Error under ~5 mm is good. The result goes to
   data/memory/hand_eye_env.json and `look` / `find_objects` start returning base_xyz_mm.

The marker's exact place on the gripper is solved too (vary the tool rotation
between points so it is observable), and points far off the rest (e.g. recorded
while the robot was still moving) are dropped automatically.

    python scripts/calibrate_env_camera.py --color red [--marker-offset-mm 0]
    python scripts/calibrate_env_camera.py --tune        # save colour masks to check HSV ranges
    python scripts/calibrate_env_camera.py --solve-only  # re-solve from the saved points (no robot needed)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from common.config import load_perception_config, load_robot_config
from common.paths import CAPTURES_DIR, MEMORY_DIR
from drivers.nova2 import Nova2
from perception.calibration import leave_one_out_errors, outliers, solve_points, solve_with_marker_offset
from perception.scene import Scene
from perception.transforms import HandEye, pose_to_matrix

POINTS_FILE = MEMORY_DIR / "calibration_points_env.json"
TOUCH_POINTS_FILE = MEMORY_DIR / "calibration_points_touch.json"


def marker_in_base_mm(pose: list[float], offset_mm: float) -> list[float]:
    """TCP pose + offset along the tool Z axis -> marker position in the base frame."""
    T = pose_to_matrix(pose)
    return list(T[:3, 3] * 1000.0 + T[:3, :3] @ np.array([0.0, 0.0, offset_mm]))


def tune(scene: Scene) -> None:
    import cv2
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


def solve_and_save(pairs: list[dict], initial_offset_mm: float, out_path) -> None:
    cam = [pt["camera_m"] for pt in pairs]
    poses = [pt["tcp_pose"] for pt in pairs]
    T, offset, rms, errors = solve_with_marker_offset(cam, poses, (0.0, 0.0, initial_offset_mm))
    dropped = outliers(errors)
    if dropped and len(pairs) - len(dropped) >= 6:
        keep = [i for i in range(len(pairs)) if i not in dropped]
        print(f"dropping outlier point(s) {dropped} (errors {np.round(errors[dropped], 1)} mm)")
        T, offset, rms, errors_kept = solve_with_marker_offset([cam[i] for i in keep], [poses[i] for i in keep],
                                                               tuple(offset))
        errors = np.full(len(pairs), np.nan)
        errors[keep] = errors_kept
    print("per-point error (mm):", np.round(errors, 1))
    print(f"marker offset in tool frame: {np.round(offset, 1)} mm")
    verdict = "good" if rms < 5 else "usable, add more varied points" if rms < 10 else "poor — re-record"
    print(f"RMS error: {rms:.1f} mm  ({verdict})")
    HandEye(T, eye_in_hand=False).save(out_path)
    report = {"rms_mm": round(rms, 2), "points_used": int(np.sum(~np.isnan(errors))),
              "points_dropped": dropped, "marker_offset_tool_mm": np.round(offset, 2).tolist(),
              "camera_position_base_mm": np.round(T[:3, 3] * 1000, 1).tolist()}
    out_path.with_name(out_path.stem + "_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved -> {out_path}")


def measure_block(scene: Scene, color: str, frames: int = 7) -> list[float] | None:
    """Median camera position of the largest `color` blob over several frames."""
    pts = []
    for _ in range(frames):
        objs = scene.find(color, "env")
        if objs and objs[0].camera_xyz_m is not None:
            pts.append(objs[0].camera_xyz_m)
    if len(pts) < frames // 2 + 1:
        return None
    return np.median(np.array(pts), axis=0).tolist()


def save_touch_solution(pairs: list[dict], out_path) -> None:
    T, rms, errors, dropped = solve_points([p["camera_m"] for p in pairs], [p["tip_mm"] for p in pairs])
    if dropped:
        print(f"dropped outlier spot(s) {dropped}")
    print("per-spot error (mm):", np.round(errors, 1))
    verdict = "good" if rms < 5 else "usable" if rms < 10 else "poor — check the touches / detections"
    print(f"fit RMS: {rms:.1f} mm over {len(pairs) - len(dropped)} spots  ({verdict})")
    keep = [i for i in range(len(pairs)) if i not in dropped]
    loo_mean = None
    if len(keep) >= 6:
        loo = np.linalg.norm(leave_one_out_errors([pairs[i]["camera_m"] for i in keep],
                                                  [pairs[i]["tip_mm"] for i in keep]), axis=1)
        loo_mean = float(loo.mean())
        print(f"ACCURACY ON UNSEEN SPOTS (leave-one-out): mean {loo.mean():.1f} mm, worst {loo.max():.1f} mm "
              f"-> this is what Astra will get on a new block position")
    else:
        print("add spots until 6+ are used to get an accuracy estimate on unseen spots")
    HandEye(T, eye_in_hand=False).save(out_path)
    report = {"method": "touch", "rms_mm": round(rms, 2), "unseen_spot_error_mm": loo_mean,
              "points_used": len(pairs) - len(dropped),
              "points_dropped": dropped, "camera_position_base_mm": np.round(T[:3, 3] * 1000, 1).tolist()}
    out_path.with_name(out_path.stem + "_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved -> {out_path}")


def check_calibration(scene: Scene, color: str) -> None:
    """Fair test of the saved calibration: touch the block, compare with the prediction.
    These spots are NOT added to the calibration."""
    from actions.primitives import create_primitives

    if scene.env_calibration is None:
        print("no calibration saved yet — run --touch first")
        return
    prims = create_primitives()
    errors = []
    print("CHECK mode: Enter = test a spot · q = quit and show the summary")
    while True:
        if input(f"\n[{len(errors)} checks] put the block somewhere, arm OUT of view, Enter > ").strip().lower() == "q":
            break
        cam = measure_block(scene, color)
        if cam is None:
            print("block not seen clearly — try again")
            continue
        predicted = scene.env_calibration.camera_to_base(np.array(cam))
        print(f"calibration predicts the block top at {np.round(predicted, 1)} mm")
        if input("now jog the FINGERTIPS to the block's top centre, Enter (or 'c' to cancel) > ").strip().lower() == "c":
            continue
        tip = np.array(prims.get_state()["pose"][:3])
        e = predicted - tip
        errors.append(e)
        print(f"error: {np.linalg.norm(e):.1f} mm  (X {e[0]:+.1f}, Y {e[1]:+.1f}, Z {e[2]:+.1f}; "
              f"horizontal {np.linalg.norm(e[:2]):.1f} mm)")
    if errors:
        n = np.linalg.norm(errors, axis=1)
        print(f"\nSUMMARY over {len(errors)} checks: mean {n.mean():.1f} mm, worst {n.max():.1f} mm, "
              f"mean bias X {np.mean(errors, 0)[0]:+.1f} Y {np.mean(errors, 0)[1]:+.1f} Z {np.mean(errors, 0)[2]:+.1f} mm")


def touch_calibration(scene: Scene, color: str, out_path, resume: bool) -> None:
    from actions.primitives import create_primitives

    prims = create_primitives()
    pairs = json.loads(TOUCH_POINTS_FILE.read_text()) if resume and TOUCH_POINTS_FILE.exists() else []
    print(f"TOUCH calibration with the {color} block. Spots so far: {len(pairs)} (8+ spread over the table).")
    print("Commands at the prompt: Enter = next step · u = undo last spot · s = solve & save · q = quit")
    while True:
        cmd = input(f"\n[{len(pairs)} spots] put the block on a new spot, move the arm OUT of view, Enter > ")
        cmd = cmd.strip().lower()
        if cmd == "q":
            break
        if cmd == "u":
            if pairs:
                pairs.pop()
                print("removed last spot")
        elif cmd == "s":
            if len(pairs) < 5:
                print("need at least 5 spots (8+ recommended)")
            else:
                save_touch_solution(pairs, out_path)
            continue
        elif cmd == "":
            cam = measure_block(scene, color)
            if cam is None:
                print("block not seen clearly (or no depth) — is the arm out of view? try again")
                continue
            print(f"camera sees the block at {np.round(cam, 3)} m")
            if input("now jog the FINGERTIPS to the block's top centre, Enter "
                     "(or 'c' to cancel) > ").strip().lower() == "c":
                continue
            state = prims.get_state()
            # Any gripper orientation is accepted: the fingertip is computed along the real
            # tool axis (flange + tool_length_mm). Tilt is only reported.
            axis = prims._tool_axis(state["flange_pose"])
            tilt = float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))
            pairs.append({"camera_m": cam, "tip_mm": state["pose"][:3], "flange_pose": state["flange_pose"],
                          "tilt_deg": round(tilt, 1)})
            print(f"recorded spot {len(pairs)}: fingertip {np.round(state['pose'][:3], 1)} mm "
                  f"(gripper tilt {tilt:.0f}° from vertical)")
            # Also keep it as a named position: ground truth for testing Astra later
            # (put the block back on this spot, compare Astra's estimate, go_to_position).
            used = [int(n.rsplit("_", 1)[1]) for n in (q["name"] for q in prims.a.positions.all())
                    if n.startswith("touch_spot_") and n.rsplit("_", 1)[1].isdigit()]
            name = f"touch_spot_{max(used, default=0) + 1}"   # never overwrite an earlier spot
            prims.save_position(name)
            print(f"   saved as named position '{name}'")
        TOUCH_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOUCH_POINTS_FILE.write_text(json.dumps(pairs, indent=2))


def main() -> None:
    robot_cfg, pcfg = load_robot_config(), load_perception_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--color", default="red", help="marker colour (config/perception.yaml)")
    p.add_argument("--marker-offset-mm", type=float, default=0.0,
                   help="extra distance from fingertip to marker centre along tool Z")
    p.add_argument("--tune", action="store_true", help="only save colour masks and exit")
    p.add_argument("--resume", action="store_true", help="continue from saved points")
    p.add_argument("--solve-only", action="store_true", help="solve from saved points and exit")
    p.add_argument("--touch", action="store_true", help="touch calibration with the block (recommended)")
    p.add_argument("--check", action="store_true", help="test the saved calibration by touching the block")
    args = p.parse_args()

    scene = Scene.from_config(pcfg, cameras=("env",))
    if args.tune:
        tune(scene)
        return
    if args.check:
        check_calibration(scene, args.color)
        return
    if args.touch:
        if args.solve_only:
            save_touch_solution(json.loads(TOUCH_POINTS_FILE.read_text()), pcfg.calibration_path)
        else:
            touch_calibration(scene, args.color, pcfg.calibration_path, args.resume)
        return

    offset = robot_cfg.gripper.tool_length_mm + args.marker_offset_mm
    if args.solve_only:
        solve_and_save(json.loads(POINTS_FILE.read_text()), offset, pcfg.calibration_path)
        return
    robot = Nova2(robot_cfg.ip, robot_cfg.timeout_s)
    if robot_cfg.gripper.tool_length_mm == 0:
        print("!! gripper.tool_length_mm is 0 in config/robot.yaml — measure it first for accurate picks.")
    pairs = json.loads(POINTS_FILE.read_text()) if args.resume and POINTS_FILE.exists() else []
    print(f"Marker colour: {args.color}. Offset along tool Z: {offset} mm. Points so far: {len(pairs)}")
    print("Enter = record point · u = undo last · s = solve & save · q = quit")

    while True:
        cmd = input(f"[{len(pairs)} pts] > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "u" and pairs:
            pairs.pop()
            print("removed last point")
        elif cmd == "s":
            if len(pairs) < 6:
                print("need at least 6 points (10+ recommended, with different tool rotations)")
                continue
            solve_and_save(pairs, offset, pcfg.calibration_path)
        elif cmd == "":
            frame = scene.observe("env")
            objs = scene.find(args.color, "env", frame)
            if not objs or objs[0].camera_xyz_m is None:
                print("marker not found or no depth there — adjust and retry")
                continue
            pose = robot.feedback()["pose"]
            pt = {"camera_m": objs[0].camera_xyz_m, "base_mm": marker_in_base_mm(pose, offset), "tcp_pose": pose}
            pairs.append(pt)
            print(f"recorded: camera {np.round(pt['camera_m'], 3)} m  <->  base {np.round(pt['base_mm'], 1)} mm")
        POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        POINTS_FILE.write_text(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
