"""Thin wrapper around the OpenAI Responses API for the Astra model."""
from __future__ import annotations

import os

from common.config import AgentConfig


class AstraClient:
    def __init__(self, config: AgentConfig):
        from openai import OpenAI   # imported lazily so the UI runs without it

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {config.api_key_env} in .env or the environment")
        self.config = config
        self.client = OpenAI(api_key=api_key)

    def create(self, input, tools: list[dict] | None = None, previous_response_id: str | None = None,
               instructions: str | None = None):
        kwargs = {"model": self.config.model, "input": input,
                  "max_output_tokens": self.config.max_output_tokens}
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if instructions:
            kwargs["instructions"] = instructions
        return self.client.responses.create(**kwargs)
