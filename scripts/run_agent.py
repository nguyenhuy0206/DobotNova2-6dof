#!/usr/bin/env python3
"""Run one task through the Astra agent harness.

    python scripts/run_agent.py "open the gripper, then jog J1 by +5 degrees"
    python scripts/run_agent.py --mode confirm "..."     # ask before each motion
    python scripts/run_agent.py --mode live "..."        # no prompts (careful)

Default mode comes from config/agent.yaml (dry_run).
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from agent import AgentHarness, AstraClient
from agent.toolkit import build_toolkit
from common.config import load_agent_config, load_robot_config
from common.log import setup_logging


def ask_operator(name: str, args: dict) -> bool:
    return input(f"\n>> Agent wants {name}({args}). Execute? [y/N] ").strip().lower() == "y"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task")
    p.add_argument("--mode", choices=["dry_run", "confirm", "live"])
    args = p.parse_args()

    setup_logging(filename="agent.log")
    robot_cfg, agent_cfg = load_robot_config(), load_agent_config()
    if args.mode:
        agent_cfg.execution_mode = args.mode

    executor, kit = build_toolkit(agent_cfg.execution_mode, ask_operator,
                                  robot_cfg=robot_cfg, agent_cfg=agent_cfg)
    harness = AgentHarness(AstraClient(agent_cfg), executor, agent_cfg, skills=kit.skills.library,
                           send_images=kit.perception_cfg.send_images_to_model)
    print(f"[{agent_cfg.execution_mode}] {agent_cfg.model} · max speed {agent_cfg.max_speed}% · "
          "Ctrl+C = stop agent AND robot")
    try:
        print(harness.run(args.task))
    except KeyboardInterrupt:
        # A motion already sent keeps going unless the controller is told to stop.
        print("\n!! Ctrl+C — stopping the robot")
        try:
            kit.primitives.stop()
            print("Robot stopped (Pause). Check it in the Debug tab before continuing.")
        except Exception as exc:  # noqa: BLE001 — still exit; tell the operator loudly
            print(f"!! STOP FAILED: {exc} — use the Debug tab STOP / emergency button NOW")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
