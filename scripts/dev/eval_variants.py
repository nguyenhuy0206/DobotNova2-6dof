#!/usr/bin/env python3
"""Reproducible V0–V3 comparison in the MOCK world (simulated robot + scripted Astra).

    .venv/bin/python scripts/dev/eval_variants.py            both plans, all variants
    .venv/bin/python scripts/dev/eval_variants.py --md data/logs/eval_variants.md

Same task, same actions, same robot limits for every variant; only the harness
differs. Tokens are ESTIMATES of request size (chars/4 + per-image cost) because no
API is called — use scripts/episode_report.py for billed tokens of real episodes.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from mock_world import run_variant  # noqa: E402

COLS = ["plan", "variant", "blocks_on_table", "model_calls", "est_total_tokens", "max_context_tokens",
        "images_sent", "images_withheld", "actions", "actions_per_call", "deferred", "failed", "motions_sent"]


def main() -> None:
    import logging
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    args = ap.parse_args()
    rows = []
    for plan in ("primitives", "skills"):
        for v in ("V0", "V1", "V2", "V3"):
            r = run_variant(v, Path(tempfile.mkdtemp()), plan)
            r["blocks_on_table"] = len(r["blocks_on_table"])
            rows.append(r)
    base = {r["plan"]: r["est_total_tokens"] for r in rows if r["variant"] == "V0"}
    w = {c: max(len(c), *(len(f"{r[c]:,}" if isinstance(r[c], int) else str(r[c])) for r in rows)) for c in COLS}
    fmt = lambda r, c: f"{r[c]:,}" if isinstance(r[c], int) else str(r[c])  # noqa: E731
    print("  ".join(c.ljust(w[c]) for c in COLS) + "  vs V0")
    for r in rows:
        print("  ".join(fmt(r, c).ljust(w[c]) for c in COLS)
              + f"  {r['est_total_tokens'] / base[r['plan']]:.1%}")
    if args.md:
        lines = ["| " + " | ".join(COLS) + " | tokens vs V0 |", "|" + "---|" * (len(COLS) + 1)]
        lines += ["| " + " | ".join(fmt(r, c) for c in COLS) + f" | {r['est_total_tokens'] / base[r['plan']]:.1%} |"
                  for r in rows]
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
