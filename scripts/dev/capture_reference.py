#!/usr/bin/env python3
"""Save the current camera picture as an operator reference for compare_reference (24 Nov, threading).

    .venv/bin/python scripts/dev/capture_reference.py --task threading --label success
    .venv/bin/python scripts/dev/capture_reference.py --task threading --label fail

    .venv/bin/python scripts/dev/capture_reference.py --label fail --from-file data/captures/wrist-....jpg

Put the scene in the state the label names (success = threaded, fail = not threaded) with the robot at the
start pose the agent works from (insert_start), then run. Several pictures per label are fine (the closest
one is used). Reads the camera through scripts/run_vision.sh; never moves the robot.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="threading")
    ap.add_argument("--label", required=True, choices=["success", "fail"])
    ap.add_argument("--camera", default="wrist", choices=["wrist", "env"])
    ap.add_argument("--from-file", help="use a picture already saved (e.g. the UI's Capture button) instead")
    args = ap.parse_args()
    from perception.reference import load_references, save_reference
    from perception.scene import Scene
    if args.from_file:
        import cv2
        image = cv2.imread(args.from_file)
        if image is None:
            raise SystemExit(f"cannot read {args.from_file}")
    else:
        image = Scene.from_config(cameras=(args.camera,)).observe(args.camera).color
    path = save_reference(image, args.task, args.label)
    counts = {k: len(v) for k, v in load_references(args.task).items()}
    print(f"saved {path}\nreferences now: {counts}")


if __name__ == "__main__":
    main()
