#!/usr/bin/env python3
"""Per-episode table: outcome, model calls, tokens, images, actions, grasps, time.

    .venv/bin/python scripts/episode_report.py              all robot episodes
    .venv/bin/python scripts/episode_report.py --last 5
    .venv/bin/python scripts/episode_report.py --md out.md  also write a Markdown table

Token numbers come only from the API usage logged per request ({"event": "usage"}).
Episodes recorded before usage logging show "n/a" — never an estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter

import _bootstrap  # noqa: F401
from judge.ledger import JudgeLedger
from memory.store import EpisodeStore

COLUMNS = ["episode", "variant", "outcome", "moved", "calls", "tokens", "cached", "max_ctx", "images",
           "actions", "act/call", "failed", "grasps", "safety", "skills", "minutes"]


def row(store: EpisodeStore, ep: str, verdicts: dict) -> dict:
    log = store.load(ep)
    tools = [e for e in log if e.get("event") == "tool"]
    usage = [e for e in log if e.get("event") == "usage"]
    metrics = next((e for e in reversed(log) if e.get("event") == "metrics"), None)
    calls = metrics["model_calls"] if metrics else sum(1 for e in log if e.get("event") == "model")
    v = verdicts.get(ep) or {"status": "inconclusive", "success_credit": False}
    names = Counter(e["name"] for e in tools)
    failed = sum(1 for e in tools if not (e.get("result") or {}).get("ok"))
    tok = sum(u.get("total_tokens", 0) for u in usage) if usage and all("total_tokens" in u for u in usage) else None
    return {
        "episode": ep, "variant": (log[0].get("meta") or {}).get("variant", "V0*"),
        "outcome": v["status"].upper(),
        "moved": int(v["success_credit"]), "calls": calls,
        "tokens": "n/a" if tok is None else f"{tok:,}",
        "cached": "n/a" if tok is None else f"{sum(u.get('cached_tokens', 0) for u in usage):,}",
        "max_ctx": "n/a" if tok is None else f"{max(u.get('input_tokens', 0) for u in usage):,}",
        "images": metrics["images_sent"] if metrics else names.get("look", 0),
        "actions": len(tools), "act/call": round(len(tools) / calls, 2) if calls else 0, "failed": failed,
        "grasps": metrics["grasp_attempts"] if metrics else names.get("close_gripper", 0),
        "safety": metrics["safety_stops"] if metrics else sum(
            1 for e in tools if "contact" in str((e.get("result") or {}).get("error", "")).lower()),
        "skills": names.get("run_skill", 0),
        "minutes": round((log[-1]["t"] - log[0]["t"]) / 60, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int)
    ap.add_argument("--md", help="write a Markdown table to this file")
    args = ap.parse_args()
    store = EpisodeStore()
    eps = store.list()[-args.last:] if args.last else store.list()
    ledger = JudgeLedger()
    verdicts = {ep: ledger.outcome(ep) for ep in eps}
    rows = [row(store, ep, verdicts) for ep in eps]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c) for c in COLUMNS}
    print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS))
    print("\nV0* = recorded before variants existed (baseline harness). n/a = no API usage logged for it.")
    if args.md:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
