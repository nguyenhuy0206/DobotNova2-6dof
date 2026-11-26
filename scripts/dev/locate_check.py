#!/usr/bin/env python3
"""Operator-only D455 geometry check. This script reads the running env depth helper.

    .venv/bin/python scripts/dev/locate_check.py --pixels 320,240 410,280
    .venv/bin/python scripts/dev/locate_check.py --hover

Without --pixels, ask for pixels by clicking the raw frame (q to finish).
Clicking prints a full-image pixel pair without drawing on the picture. --hover
offers a fingertip move to top+40 mm, separately confirmed by typing y for each
pixel. Measure the XY error with a ruler and enter it in the notes.
"""
from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401
from common.paths import DATA_DIR

LOG = DATA_DIR / "logs" / "locate_checks.jsonl"


def _executor():
    from actions import ActionExecutor, build_registry
    from actions.primitives import create_primitives
    from common.config import load_agent_config, load_robot_config
    p = create_primitives(load_robot_config())
    cfg = load_agent_config()
    p.fixed_speed, p.speed_cap = cfg.fixed_speed, cfg.max_speed
    return ActionExecutor(build_registry(p), mode="live")


def check_pixel(scene, pixel, table_z, *, hover=False, executor_factory=_executor, log_path=LOG):
    result = scene.locate_object(pixel, table_z=table_z)
    print(json.dumps(result, indent=2))
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pixel": pixel, "location": result}
    if hover and result.get("ok"):
        x, y, z = result["top_center_mm"]
        prompt = f"Move fingertips to [{x:.1f}, {y:.1f}, {z + 40:.1f}] mm (top+40)? Type y: "
        if input(prompt).strip().lower() == "y":
            executor = executor_factory()
            state = executor.execute("get_state")
            if state["ok"]:
                # review 23 Nov: first a joint-space move to 150 mm above the top (a straight line from
                # anywhere can cross the base), then straight down to top+40 (contact-sensed near the table)
                rz = state["result"]["pose"][5]
                above = executor.execute("move_to_pose", dict(x=x, y=y, z=z + 150, rx=180, ry=0, rz=rz))
                row["hover"] = above if not above["ok"] else executor.execute(
                    "move_to_pose", dict(x=x, y=y, z=z + 40, rx=180, ry=0, rz=rz, linear=True))
            else:
                row["hover"] = state
            print(json.dumps(row["hover"], indent=2))
            row["operator_notes"] = input("Measured XY error (mm) and notes: ")
        else:
            row["hover"] = "not confirmed"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def _pixel(text):
    try:
        u, v = text.split(",")
        return [int(u), int(v)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pixel must be u,v in full-image coordinates") from exc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pixels", nargs="+", type=_pixel)
    ap.add_argument("--hover", action="store_true")
    args = ap.parse_args()
    from common.config import load_robot_config
    from calibration.workcell import load_workcell
    from perception.scene import Scene
    scene = Scene.from_config(workspace=load_robot_config().workspace)
    scene.look("env")
    frame, _ = scene.last_frames["env"]
    table_z = load_workcell().get("table_z_mm")
    executor = None
    def motion_executor():
        nonlocal executor
        if executor is None:
            executor = _executor()
        return executor
    print(f"Frame {frame.sequence_id}, full image {frame.color.shape[1]}x{frame.color.shape[0]}, table Z {table_z} mm")
    if args.pixels:
        for pixel in args.pixels:
            check_pixel(scene, pixel, table_z, hover=args.hover, executor_factory=motion_executor)
    else:
        import cv2
        window = "env (raw, full-image pixels)"
        pixels = []
        def click(event, u, v, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Pixel: {u},{v}")
                pixels.append([u, v])
        cv2.imshow(window, frame.color)
        cv2.setMouseCallback(window, click)
        print("Click a full-image pixel near the object; press q in the picture to finish.")
        try:
            while True:
                if cv2.waitKey(50) & 0xff in (ord("q"), 27):
                    break
                if pixels:
                    check_pixel(scene, pixels.pop(0), table_z, hover=args.hover, executor_factory=motion_executor)
        finally:
            cv2.destroyWindow(window)
    print(f"Results: {LOG}")


if __name__ == "__main__":
    main()
