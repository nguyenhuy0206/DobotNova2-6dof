"""Thin wrapper around the OpenAI Responses API for the Astra model.

Key: the variable named by ``agent.yaml: api_key_env`` (``gpt6-key`` in .env,
loaded by common.config). Endpoint: ``agent.yaml: base_url`` (an OpenAI-compatible
proxy such as modelapi.vn; empty = api.openai.com). If the key isn't set, ``OpenAI()`` falls back to its own
defaults (OPENAI_API_KEY exported in your shell), like the original test script.
"""
from __future__ import annotations

import os

from common.config import AgentConfig


class QuotaExhausted(RuntimeError):
    """The API key's organisation/project has no credits left (HTTP 429 insufficient_quota).
    Retrying cannot help: someone has to add credits or use another key."""


class AstraClient:
    def __init__(self, config: AgentConfig):
        from openai import OpenAI   # imported lazily so the UI runs without it

        self.config = config
        key = os.environ.get(config.api_key_env) if config.api_key_env else None
        self.key_source = config.api_key_env if key else "OPENAI_API_KEY (SDK default)"
        # More retries than the default 2: the API answers 429 (rate limit) when busy.
        kwargs = {"max_retries": 6}
        if key:
            kwargs["api_key"] = key
        if getattr(config, "base_url", ""):
            kwargs["base_url"] = config.base_url
        self.endpoint = config.base_url or "api.openai.com (SDK default)"
        self.client = OpenAI(**kwargs)

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
        if self.config.truncation:
            kwargs["truncation"] = self.config.truncation
        if instructions:
            # same instructions + tools every turn: one cache key so they are billed as cached input
            import hashlib
            kwargs["prompt_cache_key"] = "astra-" + hashlib.sha1(instructions.encode()).hexdigest()[:16]
        from openai import RateLimitError
        try:
            return self.client.responses.create(**kwargs)
        except RateLimitError as exc:
            body = str(getattr(exc, "body", "") or exc)
            if "insufficient_quota" in body or "credit_balance_exhausted" in body:
                raise QuotaExhausted(
                    f"API key in {self.key_source}: no credits remaining at {self.endpoint} "
                    "(429 insufficient_quota). Top up with that provider (modelapi.vn wallet, or "
                    "platform.openai.com billing for an OpenAI key) or put another key in .env.") from exc
            raise
