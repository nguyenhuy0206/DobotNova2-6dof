#!/usr/bin/env python3
"""Drive the PGC gripper directly over a USB–RS485 adapter (no Nova 2 controller).

    python scripts/gripper_direct.py status
    python scripts/gripper_direct.py init            # then waits until initialized
    python scripts/gripper_direct.py open
    python scripts/gripper_direct.py close --force 30 # close until object / fully closed
    python scripts/gripper_direct.py move 500
    python scripts/gripper_direct.py demo            # init (if needed) -> open -> close -> open

Port is auto-detected (/dev/cu.usbserial-*) unless --port is given.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from common.config import load_robot_config
from common.errors import RobotError
from drivers.gripper import PGCGripperDirect


def main() -> None:
    g = load_robot_config().gripper
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "init", "open", "close", "grip", "move", "demo"])
    p.add_argument("position", nargs="?", type=int, help="for 'move': 0 closed … 1000 open")
    p.add_argument("--port")
    p.add_argument("--slave-id", type=int, default=g.slave_id)
    p.add_argument("--baud", type=int, default=g.baud)
    p.add_argument("--force", type=int, default=g.force)
    p.add_argument("--speed", type=int, default=g.speed)
    p.add_argument("--full", action="store_true", help="init with 0xA5 full recalibration")
    a = p.parse_args()

    gripper = PGCGripperDirect(a.port)
    try:
        info = gripper.connect(a.slave_id, a.baud, "N")
        print(f"Connected: {info['transport']}")
        print("Status:", gripper.status())

        if a.command == "init":
            print("Init  ->", gripper.init(force_reinit=True, full=a.full))
        elif a.command == "open":
            print("Open  ->", gripper.open(a.force, a.speed))
        elif a.command in ("close", "grip"):
            print("Close ->", gripper.close(a.force, a.speed))
        elif a.command == "move":
            if a.position is None:
                p.error("move needs a position 0–1000")
            print("Move  ->", gripper.go_to(a.position, a.force, a.speed))
        elif a.command == "demo":
            print("Init  ->", gripper.init(full=a.full))
            print("Open  ->", gripper.open(a.force, a.speed))
            print("Close ->", gripper.close(a.force, a.speed))
            print("Open  ->", gripper.open(a.force, a.speed))
    except RobotError as exc:
        print("Gripper error:", exc)
        raise SystemExit(1)
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    main()
