#!/usr/bin/env python3
"""Replay old episodes to a local Qwen: at every step it sees what Astra saw before that step (the
task, Astra's earlier tool calls and the robot's real results) and must give the NEXT tool call.
Compared with what Astra actually did, and checked against the tool's real schema. No robot.

    .venv/bin/python scripts/dev/qwen_replay_eval.py                       default episodes, 2B
    .venv/bin/python scripts/dev/qwen_replay_eval.py --model mlx-community/Qwen3.5-0.8B-8bit --steps 20
Writes data/logs/qwen_replay_<model>.jsonl (one line per step) and prints the summary.

Limits (it is teacher forcing, not a live run): text only (Astra also got images), no STATE block,
results compacted as the harness does; tools that no longer exist are skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from student.context import parse_calls

EPISODES = ["20261117-165638-c1c822", "20261117-172537-5f5c16", "20261117-173052-a287ec",
            "20261117-173616-86e54e", "20261118-164411-bda29f", "20261118-171824-339d3a"]
TOL = {"mm": 10.0, "deg": 5.0}
ANGLE_KEYS = {"rx", "ry", "rz", "drx", "dry", "drz", "degrees"}
CATEGORY = {"look": "see", "find_objects": "see", "get_state": "see", "check_robot_in_view": "see",
            "move_to_pose": "move", "move_relative": "move", "move_to_joints": "move", "jog_joint": "move",
            "go_to_position": "move", "run_routine": "move", "run_skill": "move",
            "open_gripper": "grip", "close_gripper": "grip"}


def coerce(args: dict, schema: dict) -> dict:
    """Qwen writes every parameter as text: turn it into the schema's types."""
    props = (schema or {}).get("properties", {})
    out = {}
    for k, v in args.items():
        typ = (props.get(k) or {}).get("type")
        if isinstance(v, str) and typ == "boolean" and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        elif isinstance(v, str) and typ in ("number", "integer", "array", "object", "boolean"):
            try:
                v = json.loads(v)
            except ValueError:
                pass
            if typ == "integer" and isinstance(v, float) and v.is_integer():
                v = int(v)
        out[k] = v
    return out


def arg_agreement(name: str, mine: dict, astra: dict) -> float | None:
    """Share of Astra's arguments Qwen matched (numbers within tolerance, the rest equal)."""
    keys = [k for k in astra if k not in ("speed", "force", "image")]
    if not keys:
        return None
    hit = 0
    for k in keys:
        a, b = astra[k], mine.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            hit += abs(a - b) <= (TOL["deg"] if k in ANGLE_KEYS else TOL["mm"])
        elif isinstance(a, dict) and isinstance(b, dict):
            hit += a == b or (a.get("name") == b.get("name"))
        else:
            hit += a == b
    return hit / len(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--episodes", nargs="*", default=EPISODES)
    ap.add_argument("--steps", type=int, default=0, help="at most this many steps per episode (0 = all)")
    ap.add_argument("--result-chars", type=int, default=700)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen_bench import explore_tools

    from agent.context import compact_result
    from agent.explore import EXPLORE_INSTRUCTIONS
    from mlx_vlm import generate, load
    from mlx_vlm.apc import APCManager

    tools = explore_tools()
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    model, processor = load(args.model)
    apc = APCManager()        # prefix cache across requests; works with Qwen3.5's recurrent layers
    tok = processor.tokenizer
    out_path = Path("data/logs") / f"qwen_replay_{args.model.split('/')[-1]}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = out_path.open("w", encoding="utf-8")
    rows = []

    for ep in args.episodes:
        events = [json.loads(line) for line in open(f"data/episodes/{ep}.jsonl", encoding="utf-8")]
        task = next(e["task"] for e in events if e.get("event") == "start")
        msgs = [{"role": "system", "content": EXPLORE_INSTRUCTIONS}, {"role": "user", "content": task}]
        n = 0
        for e in events:
            if e.get("event") != "tool":
                continue
            name, astra_args = e["name"], e.get("args") or {}
            if name in schemas and (not args.steps or n < args.steps):
                prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
                t0 = time.perf_counter()
                r = generate(model, processor, prompt, max_tokens=160, temperature=0.0, apc_manager=apc)
                dt = time.perf_counter() - t0
                calls = parse_calls(r.text)
                mine = calls[0] if calls else None
                row = {"episode": ep, "step": e.get("step"), "astra": {"name": name, "args": astra_args},
                       "qwen_text": r.text.strip()[:400], "seconds": round(dt, 2), "prompt_tokens": r.prompt_tokens,
                       "out_tokens": r.generation_tokens, "parsed": mine is not None}
                if mine:
                    qa = coerce(mine["arguments"], schemas.get(mine["name"], {}))
                    row["qwen"] = {"name": mine["name"], "args": qa}
                    row["known_tool"] = mine["name"] in schemas
                    valid = False
                    if row["known_tool"]:
                        from actions.executor import ActionSpec
                        try:
                            ActionSpec(mine["name"], "", None, schemas[mine["name"]]).validate_args(dict(qa))
                            valid = True
                        except Exception as exc:  # noqa: BLE001
                            row["invalid"] = str(exc)[:160]
                    row["valid"] = valid
                    row["same_tool"] = mine["name"] == name
                    row["same_category"] = CATEGORY.get(mine["name"], "other") == CATEGORY.get(name, "other")
                    row["arg_match"] = arg_agreement(name, qa, astra_args) if row["same_tool"] else None
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(row)
                n += 1
                print(f"{ep[-6:]} #{n:3d} {dt:5.1f}s astra={name:16s} qwen={(mine or {}).get('name', '-'):16s} "
                      f"{'✓' if row.get('same_tool') else '✗'}", flush=True)
            # teacher forcing: what Astra really did and what the robot really answered
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": name, "arguments": astra_args}}]})
            msgs.append({"role": "tool", "content": compact_result(e.get("result") or {}, args.result_chars)})

    out.close()
    summarize(rows, out_path)


def summarize(rows: list[dict], path: Path) -> None:
    n = len(rows)
    if not n:
        print("no steps")
        return
    pct = lambda k: 100 * sum(1 for r in rows if r.get(k)) / n
    am = [r["arg_match"] for r in rows if r.get("arg_match") is not None]
    secs = sorted(r["seconds"] for r in rows)
    print(f"\n{n} steps · parsed {pct('parsed'):.0f}% · valid call {pct('valid'):.0f}% · same tool as Astra "
          f"{pct('same_tool'):.0f}% · same kind (see/move/grip) {pct('same_category'):.0f}% · args matched "
          f"{100 * sum(am) / max(len(am), 1):.0f}% (on same-tool steps) · median {secs[n // 2]:.1f} s, "
          f"p90 {secs[int(n * 0.9)]:.1f} s")
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["astra"]["name"]][0] += 1
        by[r["astra"]["name"]][1] += bool(r.get("same_tool"))
    print("per Astra tool (steps, Qwen same tool):",
          ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])))
    conf = collections.Counter((r["astra"]["name"], (r.get("qwen") or {}).get("name", "-")) for r in rows
                               if not r.get("same_tool"))
    print("most common mismatches (astra -> qwen):", ", ".join(f"{a}->{q} ×{c}" for (a, q), c in conf.most_common(8)))
    print(f"per-step log: {path}")


if __name__ == "__main__":
    main()
