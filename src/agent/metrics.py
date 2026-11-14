"""Token and execution accounting for every model request and every episode.

Numbers come from the API's own ``response.usage`` (never estimated from text
length when the API reports them). They are appended to the episode log as

    {"event": "usage", "step", "input_tokens", "cached_tokens", "output_tokens",
     "reasoning_tokens", "total_tokens", "latency_s", "images", "tool_calls"}

and summed into one {"event": "metrics", ...} line at the end of the episode,
which scripts/episode_report.py turns into a table.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def _get(obj, *path, default=0):
    for key in path:
        if obj is None:
            return default
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if obj is None else obj


def usage_of(response) -> dict | None:
    """Token usage reported by the Responses API, or None when the response has none
    (mock clients). Cached / reasoning counts are 0 when the API does not report them."""
    u = getattr(response, "usage", None)
    if u is None:
        return None
    inp, out = int(_get(u, "input_tokens")), int(_get(u, "output_tokens"))
    return {"input_tokens": inp,
            "cached_tokens": int(_get(u, "input_tokens_details", "cached_tokens")),
            "output_tokens": out,
            "reasoning_tokens": int(_get(u, "output_tokens_details", "reasoning_tokens")),
            "total_tokens": int(_get(u, "total_tokens", default=inp + out))}


@dataclass
class EpisodeMetrics:
    variant: str = "V0"
    started: float = field(default_factory=time.time)
    model_calls: int = 0
    calls_with_usage: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    images_sent: int = 0
    images_withheld: int = 0
    auto_looks: int = 0                 # images the harness took itself after an unexpected event
    tool_calls: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    actions_deferred: int = 0           # proposed in a chunk but beyond the safe horizon
    grasp_attempts: int = 0
    safety_stops: int = 0               # contact, table, view guard, IK, operator rejection
    skill_runs: int = 0
    skill_failures: int = 0
    model_latency_s: float = 0.0
    max_context_tokens: int = 0         # largest single-request input (shows context growth)

    def add_request(self, usage: dict | None, latency_s: float, images: int) -> dict:
        self.model_calls += 1
        self.model_latency_s += latency_s
        self.images_sent += images
        rec = {"latency_s": round(latency_s, 2), "images": images}
        if usage is not None:
            self.calls_with_usage += 1
            for key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
                setattr(self, key, getattr(self, key) + usage[key])
            self.max_context_tokens = max(self.max_context_tokens, usage["input_tokens"])
            rec.update(usage)
        return rec

    def add_action(self, name: str, result: dict) -> None:
        self.actions_executed += 1
        ok = bool(result.get("ok"))
        if not ok:
            self.actions_failed += 1
            err = str(result.get("error", "")).lower()
            if any(w in err for w in ("contact", "table", "camera view", "not reachable", "approach zone",
                                      "operator rejected", "safety", "undone")):
                self.safety_stops += 1
        if name == "close_gripper":
            self.grasp_attempts += 1
        if name == "run_skill":
            self.skill_runs += 1
            inner = result.get("result") or {}
            if not ok or not inner.get("ok", True):
                self.skill_failures += 1
            self.grasp_attempts += int(inner.get("grasp_attempts", 0) or 0)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("started")
        d["execution_s"] = round(time.time() - self.started, 1)
        d["model_latency_s"] = round(self.model_latency_s, 1)
        d["actions_per_call"] = round(self.actions_executed / self.model_calls, 2) if self.model_calls else 0.0
        d["usage_complete"] = self.calls_with_usage == self.model_calls
        return d

    def tokens_so_far(self) -> int:
        return self.total_tokens
