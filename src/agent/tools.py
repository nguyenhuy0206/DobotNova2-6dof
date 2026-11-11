"""Convert ActionSpecs into OpenAI Responses-API function tools."""
from __future__ import annotations

from actions.executor import ActionSpec


def to_openai_tools(specs: list[ActionSpec]) -> list[dict]:
    return [{"type": "function", "name": s.name, "description": s.description,
             "parameters": s.parameters} for s in specs]
