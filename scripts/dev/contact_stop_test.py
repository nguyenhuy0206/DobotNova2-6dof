#!/usr/bin/env python3
"""How hard does the arm hit, and where does it stop, when a move inside the approach zone meets an object?
(Operator decision 23 Nov: measure before freezing 50 % / 300 mm steps near the table.)

    .venv/bin/python scripts/dev/contact_stop_test.py --speeds 10 30 50 --depth 40

Put a SOFT object (empty cardboard box, foam) on the table under the gripper; the gripper must already be
above it, inside the approach zone (fingertip less than 120 mm above the table) or just above it. For each
speed the script moves straight down by --depth mm (never below the table floor) with the normal contact
detection, and records:
  • where the contact was detected (fingertip z) and where the arm finally stood 1 s later (overshoot)
  • whether the code's stall detection or the controller's collision alarm stopped it
  • how long the move ran
After a contact the robot is PAUSED with the rest of the move still queued: pressing Continue would resume the
move into the object. The script offers DisableRobot() to drop the queue; the operator enables the robot again
(UI / teach pendant) and lifts it before the next speed. Results go to data/logs/contact_tests.jsonl.

Hand on the E-stop. Nothing moves before you confirm each speed.
"""
from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401
from common.errors import ContactError, MotionStopped, ValidationError
from common.paths import DATA_DIR

LOG = DATA_DIR / "logs" / "contact_tests.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--depth", type=float, default=40.0, help="mm to go down per test (stops on contact)")
    args = ap.parse_args()

    from agent.explore import build_explore
    s = build_explore(mode="live", client=object())
    p = s.kit.primitives
    p.fixed_speed, p.speed_cap = None, max(args.speeds)          # this script sets the speed per test
    problems = [m for m in s.preflight() if "marker" not in m.lower()]
    if problems:
        raise SystemExit("Not ready:\n" + "\n".join(f"  ✗ {m}" for m in problems))
    zone_top = p.approach_top()
    LOG.parent.mkdir(parents=True, exist_ok=True)

    for speed in args.speeds:
        tip = p.get_state()["pose"]
        floor = p.floor_limits()[0]
        target = max(floor + 1.0, tip[2] - args.depth)
        print(f"\n=== speed {speed} % · fingertip z {tip[2]:.1f} -> {target:.1f} (zone top {zone_top:.0f}, floor {floor:.1f})")
        if tip[2] > zone_top + 30:
            print("   ! the fingertip is well above the approach zone: the contact detection only runs inside it")
        if input("   Object under the gripper, hand on the E-stop. Enter = go, s = skip, q = quit: ").strip().lower() in ("s", "q"):
            if input("   quit? [y/N] ").strip().lower() == "y":
                break
            continue
        old_speed = p.m.approach_speed
        p.m.approach_speed = speed                       # the zone runs at min(speed, approach_speed)
        t0, row = time.monotonic(), {"speed": speed, "start_z": round(tip[2], 1), "target_z": round(target, 1)}
        try:
            p.move_relative(dz=target - tip[2], speed=speed)
            row["result"] = "arrived (no contact)"
        except ContactError as exc:
            row["result"] = "contact: " + ("controller collision alarm" if "collision detection" in str(exc) else "stall")
            row["message"] = str(exc)[:200]
        except (MotionStopped, ValidationError) as exc:
            row["result"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            p.m.approach_speed = old_speed
        row["seconds"] = round(time.monotonic() - t0, 2)
        time.sleep(1.0)
        fb = p.robot.feedback()
        row["final_z"] = round(p._summary(fb)["pose"][2], 1)
        row["mode_after"] = fb["mode"]
        print(f"   {row['result']} · ran {row['seconds']} s · fingertip now z {row['final_z']} · controller mode {fb['mode']}")
        if fb["mode"] == 10:
            print("   Robot PAUSED, rest of the move still queued. Do NOT press Continue (it would resume the move).")
            if input("   Send DisableRobot() now to drop the queue? [y/N] ").strip().lower() == "y":
                p.robot.disable()
                time.sleep(1.0)
                fb = p.robot.feedback()
                row["after_disable_mode"] = fb["mode"]
                print(f"   disabled (mode {fb['mode']}). Enable it again (UI/pendant), then check it does NOT move.")
            row["operator_notes"] = input("   What did you see (deformation / sound / did it move after enable)? ")
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
        input("   Lift the gripper above the object again (UI), then Enter for the next speed ")
    print(f"\nresults: {LOG}")


if __name__ == "__main__":
    main()
