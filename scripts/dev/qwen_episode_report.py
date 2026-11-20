#!/usr/bin/env python3
"""How did the local model do in one episode? Latency, what it called, loops, what the robot did.

    .venv/bin/python scripts/dev/qwen_episode_report.py              the latest episode
    .venv/bin/python scripts/dev/qwen_episode_report.py <episode id> [--turns]   (--turns: every turn)

Reads data/episodes/<id>.jsonl (tool, state, usage, model events) and, for a local model,
data/episodes/<id>/model_io.jsonl (every prompt and raw answer).
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import _bootstrap  # noqa: F401

EP = Path("data/episodes")
MOVES = {"move_to_pose", "move_relative", "move_to_joints", "jog_joint", "go_to_position", "run_routine", "run_skill"}
GRIP = {"open_gripper", "close_gripper"}


def sig(e: dict) -> str:
    return e["name"] + json.dumps(e.get("args") or {}, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?")
    ap.add_argument("--turns", action="store_true")
    args = ap.parse_args()
    ep = args.episode or sorted(p.stem for p in EP.glob("2026*.jsonl"))[-1]
    ev = [json.loads(line) for line in (EP / f"{ep}.jsonl").open(encoding="utf-8")]
    start = next(e for e in ev if e.get("event") == "start")
    end = next((e for e in ev if e.get("event") == "end"), {})
    tools = [e for e in ev if e.get("event") == "tool" and not e.get("auto")]
    usage = [e for e in ev if e.get("event") == "usage"]
    states = [e for e in ev if e.get("event") == "state"]
    print(f"episode {ep} · model {start['meta'].get('model')} · mode {start['meta'].get('mode')} · "
          f"{(ev[-1]['t'] - ev[0]['t']) / 60:.1f} min · ended: {end.get('reason', 'no end event (stopped?)')}")

    lat = [u["latency_s"] for u in usage if u.get("latency_s") is not None]
    if lat:
        print(f"\nlatency: {len(lat)} requests · first {lat[0]:.1f} s · median {st.median(lat):.1f} s · "
              f"p90 {sorted(lat)[int(len(lat) * .9)]:.1f} s · max {max(lat):.1f} s · "
              f"prompt median {st.median(u.get('input_tokens', 0) for u in usage):.0f} tok · "
              f"output median {st.median(u.get('output_tokens', 0) for u in usage):.0f} tok")

    names = collections.Counter(e["name"] for e in tools)
    print(f"\ntool calls: {len(tools)} · {dict(names.most_common())}")
    runs, best, cur = [], (0, None), 0
    for i, e in enumerate(tools):
        cur = cur + 1 if i and sig(e) == sig(tools[i - 1]) else 1
        if cur > best[0]:
            best = (cur, sig(e))
    print(f"distinct calls: {len({sig(e) for e in tools})} · longest run of the SAME call: {best[0]}× {best[1][:120] if best[1] else ''}")
    moved = [e for e in tools if e["name"] in MOVES]
    ok = lambda e: bool((e.get("result") or {}).get("ok"))
    print(f"robot actions: {len(moved)} motion calls ({sum(ok(e) for e in moved)} ok) · "
          f"{sum(1 for e in tools if e['name'] in GRIP)} gripper calls · "
          f"{sum(1 for e in tools if not ok(e))} failed/refused calls")

    poses = [s["state"].get("pose") for s in states if isinstance(s.get("state"), dict) and s["state"].get("pose")]
    if poses:
        travel = sum(sum((a - b) ** 2 for a, b in zip(p[:3], q[:3])) ** .5 for p, q in zip(poses, poses[1:]))
        print(f"fingertip: start {[round(v) for v in poses[0][:3]]} -> end {[round(v) for v in poses[-1][:3]]} · "
              f"travelled {travel:.0f} mm over {len(poses)} logged states")
    errs = [s["state"] for s in states if isinstance(s.get("state"), dict) and s["state"].get("state_error")]
    if errs:
        print(f"!! {len(errs)} turns without a state ({errs[0].get('state_error')})")

    looks = [e for e in tools if e["name"] == "look" and ((e.get("result") or {}).get("result") or {}).get("camera") == "env"]
    def cubes(e):
        return {o["label"]: [round(v) for v in o["base_xyz_mm"]] for o in e["result"]["result"].get("objects", [])
                if o.get("base_xyz_mm") and o["label"] in ("blue", "green", "yellow")}
    if looks:
        print(f"cubes (env camera): first look {cubes(looks[0])}\n                    last look  {cubes(looks[-1])}")

    io = EP / ep / "model_io.jsonl"
    if io.exists():
        rows = [json.loads(line) for line in io.open(encoding="utf-8")]
        no_call = sum(1 for r in rows if not r["calls"])
        print(f"\nmodel I/O ({io}): {len(rows)} requests · {no_call} without a tool call")
        if args.turns:
            for i, r in enumerate(rows):
                user = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "user"), "")
                print(f"\n--- turn {i} · {r['seconds']} s · prompt {r['prompt_tokens']} · out {r['output_tokens']}")
                print("    last user message (end):", " ".join(str(user)[-400:].split()))
                print("    answer:", " ".join(r["answer"].split())[:300])


if __name__ == "__main__":
    main()
