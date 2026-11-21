#!/usr/bin/env python3
"""Episode reports — who ran it (driver/model), what the judge says, what the driver itself claimed.

    .venv/bin/python scripts/episode_report.py                   table of every robot episode (Astra and Qwen)
    .venv/bin/python scripts/episode_report.py --last 5          the last 5
    .venv/bin/python scripts/episode_report.py --driver qwen     only one driver
    .venv/bin/python scripts/episode_report.py --md out.md       also write the table as Markdown
    .venv/bin/python scripts/episode_report.py <episode id> [--turns]   one episode in detail
                                                                  (--turns: every local-model turn)

Columns: driver = who drove the robot (meta.driver, or the model name for older episodes);
judge = the verifier's verdict (data/judge/ledger.jsonl); claim = the driver's own review of the episode.
Token numbers come only from the API usage logged per request ({"event": "usage"}); "n/a" = none logged.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st

import _bootstrap  # noqa: F401
from judge.ledger import JudgeLedger
from memory.store import EpisodeStore, driver_of

COLUMNS = ["episode", "driver", "variant", "judge", "claim", "calls", "tokens", "max_ctx", "images",
           "actions", "failed", "grasps", "safety", "minutes"]
MOVES = {"move_to_pose", "move_relative", "move_to_joints", "jog_joint", "go_to_position", "run_routine", "run_skill"}
GRIP = {"open_gripper", "close_gripper"}


def _meta(log: list) -> dict:
    return (log[0].get("meta") or {}) if log else {}


def _claim(log: list) -> str:
    """The driver's own review (record_episode_outcome), if it wrote one."""
    out = next((e for e in reversed(log) if e.get("event") == "outcome"), None)
    if not out:
        return "-"
    claim = out.get("claim") or {"success": out.get("success")}
    return "success" if claim.get("success") else "fail"


def row(store: EpisodeStore, ep: str, ledger: JudgeLedger) -> dict:
    log = store.load(ep)
    meta = _meta(log)
    tools = [e for e in log if e.get("event") == "tool"]
    usage = [e for e in log if e.get("event") == "usage"]
    metrics = next((e for e in reversed(log) if e.get("event") == "metrics"), None)
    calls = metrics["model_calls"] if metrics else sum(1 for e in log if e.get("event") == "model")
    names = collections.Counter(e["name"] for e in tools)
    tok = sum(u.get("total_tokens", 0) for u in usage) if usage and all("total_tokens" in u for u in usage) else None
    return {
        "episode": ep, "driver": driver_of(meta), "variant": meta.get("variant", "V0*"),
        "judge": ledger.outcome(ep)["status"], "claim": _claim(log), "calls": calls,
        "tokens": "n/a" if tok is None else f"{tok:,}",
        "max_ctx": "n/a" if tok is None else f"{max(u.get('input_tokens', 0) for u in usage):,}",
        "images": (metrics or {}).get("images_sent", names.get("look", 0)),
        "actions": len(tools), "failed": sum(1 for e in tools if not (e.get("result") or {}).get("ok")),
        "grasps": names.get("close_gripper", 0),
        "safety": sum(1 for e in tools if "contact" in str((e.get("result") or {}).get("error", "")).lower()),
        "minutes": round((log[-1]["t"] - log[0]["t"]) / 60, 1) if log else 0,
    }


def table(args) -> None:
    store, ledger = EpisodeStore(), JudgeLedger()
    eps = store.list(driver=args.driver)
    eps = eps[-args.last:] if args.last else eps
    rows = [row(store, ep, ledger) for ep in eps]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) if rows else len(c) for c in COLUMNS}
    print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS))
    print("\nV0* = recorded before variants existed. judge = verifier verdict; claim = the driver's own review.")
    if args.md:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {args.md}")


def _sig(e: dict) -> str:
    return e["name"] + json.dumps(e.get("args") or {}, sort_keys=True)


def detail(ep: str, turns: bool) -> None:
    """One episode: who ran it, latency, what it called, loops, what the robot did, judge vs claim."""
    store = EpisodeStore()
    ev = store.load(ep)
    meta = _meta(ev)
    end = next((e for e in ev if e.get("event") == "end"), {})
    tools = [e for e in ev if e.get("event") == "tool" and not e.get("auto")]
    usage = [e for e in ev if e.get("event") == "usage"]
    states = [e for e in ev if e.get("event") == "state"]
    print(f"episode {ep} · driver {driver_of(meta)} ({meta.get('model')}) · mode {meta.get('mode')} · "
          f"{(ev[-1]['t'] - ev[0]['t']) / 60:.1f} min · ended: {end.get('reason', 'no end event (stopped?)')}")
    print(f"task: {str(ev[0].get('task', '')).splitlines()[0][:160]}")
    print(f"judge: {JudgeLedger().outcome(ep)['status']} · claim by the driver: {_claim(ev)}")

    lat = [u["latency_s"] for u in usage if u.get("latency_s") is not None]
    if lat:
        print(f"\nlatency: {len(lat)} requests · median {st.median(lat):.1f} s · max {max(lat):.1f} s · "
              f"prompt median {st.median(u.get('input_tokens', 0) for u in usage):.0f} tok · "
              f"output median {st.median(u.get('output_tokens', 0) for u in usage):.0f} tok · "
              f"reasoning {sum(u.get('reasoning_tokens', 0) or 0 for u in usage)} tok")
    names = collections.Counter(e["name"] for e in tools)
    print(f"\ntool calls: {len(tools)} · {dict(names.most_common())}")
    best, cur = (0, ""), 0
    for i, e in enumerate(tools):
        cur = cur + 1 if i and _sig(e) == _sig(tools[i - 1]) else 1
        if cur > best[0]:
            best = (cur, _sig(e))
    print(f"longest run of the SAME call: {best[0]}× {best[1][:120]}")
    moved = [e for e in tools if e["name"] in MOVES]
    ok = lambda e: bool((e.get("result") or {}).get("ok"))  # noqa: E731
    print(f"robot: {len(moved)} motion calls ({sum(ok(e) for e in moved)} ok) · "
          f"{sum(1 for e in tools if e['name'] in GRIP)} gripper calls · {sum(not ok(e) for e in tools)} failed/refused")
    for e in tools:
        if e["name"] == "close_gripper":
            r = (e.get("result") or {}).get("result") or {}
            print(f"  close at step {e['step']}: {r.get('grip_status')} (position {r.get('position')})")
    poses = [s["state"].get("pose") for s in states if isinstance(s.get("state"), dict) and s["state"].get("pose")]
    if poses:
        travel = sum(sum((a - b) ** 2 for a, b in zip(p[:3], q[:3])) ** .5 for p, q in zip(poses, poses[1:]))
        print(f"fingertip: {[round(v) for v in poses[0][:3]]} -> {[round(v) for v in poses[-1][:3]]} · "
              f"travelled {travel:.0f} mm")
    reasoning = [e for e in ev if e.get("event") == "model" and e.get("reasoning")]
    if reasoning:
        print(f"\nreasoning summaries: {len(reasoning)} (first: {reasoning[0]['reasoning'][:200]!r})")

    io = store.root / ep / "model_io.jsonl"          # local models log every prompt and raw answer
    if io.exists():
        rows = [json.loads(line) for line in io.open(encoding="utf-8")]
        print(f"\nmodel I/O ({io}): {len(rows)} requests · {sum(1 for r in rows if not r['calls'])} without a tool call")
        if turns:
            for i, r in enumerate(rows):
                user = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")
                print(f"\n--- turn {i} · {r['seconds']} s · prompt {r['prompt_tokens']} · out {r['output_tokens']}")
                print("    last user message (end):", " ".join(str(user)[-400:].split()))
                print("    answer:", " ".join(r["answer"].split())[:300])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?", help="one episode id: detailed report")
    ap.add_argument("--last", type=int)
    ap.add_argument("--driver", choices=["astra", "qwen"], help="only this driver (default: all)")
    ap.add_argument("--md", help="write the table as Markdown to this file")
    ap.add_argument("--turns", action="store_true", help="with an episode id: every local-model turn")
    args = ap.parse_args()
    if args.episode:
        detail(args.episode, args.turns)
    else:
        table(args)


if __name__ == "__main__":
    main()
