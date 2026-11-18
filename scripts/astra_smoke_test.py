#!/usr/bin/env python3
"""Check the Astra model is reachable. Sends text only — no tools, no robot."""
import _bootstrap  # noqa: F401
from agent.astra_client import AstraClient
from common.config import load_agent_config

PROMPT = (
    "You are controlling a Dobot Nova 2 robot. The robot has a gripper and an RGB camera. "
    "Your task is to pick up a red cube and place it in a target area. "
    "Explain your first three steps. Do not execute any robot actions."
)

if __name__ == "__main__":
    cfg = load_agent_config()
    try:
        client = AstraClient(cfg)
        print("Key from:", client.key_source, "| endpoint:", client.endpoint)
        response = client.create(PROMPT)
        print("Model:", response.model)
        print("Response:", response.output_text)
        print("Usage:", response.usage)
    except Exception as e:  # noqa: BLE001 — smoke test: show any failure
        print("API ERROR:", e)
