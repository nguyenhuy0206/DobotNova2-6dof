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
from actions import ActionExecutor, RobotActions, build_registry
from agent import AgentHarness, AstraClient
from common.config import load_agent_config, load_robot_config
from common.log import setup_logging
from drivers import Nova2


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

    actions = RobotActions(Nova2(robot_cfg.ip, robot_cfg.timeout_s), robot_cfg)
    executor = ActionExecutor(build_registry(actions), mode=agent_cfg.execution_mode,
                              confirm=ask_operator, allowed=agent_cfg.allowed_tools)
    harness = AgentHarness(AstraClient(agent_cfg), executor, agent_cfg)
    print(f"[{agent_cfg.execution_mode}] {agent_cfg.model}")
    print(harness.run(args.task))


if __name__ == "__main__":
    main()
