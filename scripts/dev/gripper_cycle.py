#!/usr/bin/env python3
"""Open/close the PGC gripper on the Nova 2 every N seconds (default 3 s).

Assumes the gripper is already initialized (register 0x0100 = 0xA5).
Defaults come from config/robot.yaml (gripper section).

Usage:
    python scripts/dev/gripper_cycle.py                     # cycle forever, 3 s interval
    python scripts/dev/gripper_cycle.py --interval 2 --cycles 10
    python scripts/dev/gripper_cycle.py --init              # re-initialize first

Stop with Ctrl+C — the Modbus master is closed on exit.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
from common.config import load_robot_config
from common.errors import RobotError
from drivers import Nova2, PGCGripper


def main() -> None:
    cfg = load_robot_config()
    g = cfg.gripper
    p = argparse.ArgumentParser(description="Cycle Nova 2 PGC gripper open/close")
    p.add_argument("--ip", default=cfg.ip)
    p.add_argument("--interval", type=float, default=3.0, help="seconds between moves")
    p.add_argument("--cycles", type=int, default=0, help="open+close cycles (0 = forever)")
    p.add_argument("--force", type=int, default=g.force, help="20–100 %%")
    p.add_argument("--speed", type=int, default=g.speed, help="1–100 %%")
    p.add_argument("--slave-id", type=int, default=g.slave_id)
    p.add_argument("--baud", type=int, default=g.baud)
    p.add_argument("--parity", default=g.parity)
    p.add_argument("--init", action="store_true", help="re-initialize gripper before cycling")
    args = p.parse_args()

    gripper = PGCGripper(Nova2(args.ip, cfg.timeout_s))
    info = gripper.connect(args.slave_id, args.baud, args.parity.upper())
    print(f"Connected: master={info['master_id']} via {info['transport']}")

    try:
        if args.init:
            print("Initializing gripper…")
            gripper.initialize()
            time.sleep(3)

        cycle = 0
        target_open = False  # start by closing
        while args.cycles == 0 or cycle < args.cycles:
            pos = g.open_position if target_open else g.closed_position
            gripper.move(pos, args.force, args.speed)
            label = "OPEN " if target_open else "CLOSE"
            print(f"[{time.strftime('%H:%M:%S')}] {label} -> pos {pos}")
            if target_open:
                cycle += 1  # one cycle = close + open
            target_open = not target_open
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RobotError as exc:
        print(f"Robot error: {exc}")
    finally:
        try:
            gripper.disconnect()
            print("Modbus master closed.")
        except RobotError as exc:
            print(f"Disconnect failed: {exc}")


if __name__ == "__main__":
    main()
