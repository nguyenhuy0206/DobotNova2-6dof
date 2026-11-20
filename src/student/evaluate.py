"""Replay benchmark: the held-out episodes (student.paths.EVAL_EPISODES), step by step. At each step the
student sees exactly what it would see in training (student.context) and must give the next tool
call; compared with what Astra did and checked against the tool's real schema. No robot.

    s = evaluate(adapter=None)            frozen base
    s = evaluate(adapter=run_dir)         base + a fine-tuned adapter
    compare("base", "<run id>")

Limits: text only (Astra also got images), no STATE block; Astra's call is not the only right one.
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

from student.context import WINDOW, call_message, coerce, parse_calls, result_message, step_messages, tool_schemas
from student.paths import EVAL_EPISODES, EVALS_DIR, FROZEN_DIR

ANGLE_KEYS = {"rx", "ry", "rz", "drx", "dry", "drz", "degrees"}
CATEGORY = {"look": "see", "find_objects": "see", "get_state": "see", "check_robot_in_view": "see",
            "move_to_pose": "move", "move_relative": "move", "move_to_joints": "move", "jog_joint": "move",
            "go_to_position": "move", "run_routine": "move", "run_skill": "move",
            "open_gripper": "grip", "close_gripper": "grip"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_step(mine: dict | None, name: str, astra_args: dict, schemas: dict) -> dict:
    from actions.executor import ActionSpec
    row = {"parsed": mine is not None}
    if not mine:
        return row
    args = coerce(mine["arguments"], schemas.get(mine["name"], {}))
    row.update(qwen={"name": mine["name"], "args": args}, known_tool=mine["name"] in schemas,
               same_tool=mine["name"] == name,
               same_category=CATEGORY.get(mine["name"], "other") == CATEGORY.get(name, "other"))
    row["valid"] = False
    if row["known_tool"]:
        try:
            ActionSpec(mine["name"], "", None, schemas[mine["name"]]).validate_args(dict(args))
            row["valid"] = True
        except Exception as exc:  # noqa: BLE001
            row["invalid"] = str(exc)[:160]
    if row["same_tool"] and name == "move_to_pose" and all(_num(args.get(k)) is not None for k in ("x", "y")):
        row["xy_err_mm"] = round(((astra_args["x"] - _num(args["x"])) ** 2 + (astra_args["y"] - _num(args["y"])) ** 2) ** .5, 1)
    if row["same_tool"] and name == "run_routine":
        a, q = astra_args.get("args") or {}, args.get("args") if isinstance(args.get("args"), dict) else {}
        row["routine_same"] = args.get("name") == astra_args.get("name")
        if row["routine_same"] and _num(a.get("z")) is not None and _num(q.get("z")) is not None:
            row["z_err_mm"] = abs(a["z"] - _num(q["z"]))
    if name in ("open_gripper", "close_gripper"):
        row["gripper_inverted"] = mine["name"] in ("open_gripper", "close_gripper") and mine["name"] != name
    return row


def evaluate(adapter: Path | None = None, episodes=EVAL_EPISODES, window: int = WINDOW, max_steps: int = 0,
             label: str | None = None, base: str = str(FROZEN_DIR), log=print) -> dict:
    from agent.explore import EXPLORE_INSTRUCTIONS
    from mlx_vlm import generate, load
    from mlx_vlm.apc import APCManager

    tools = tool_schemas()
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    model, processor = load(base, adapter_path=str(adapter) if adapter else None)
    tok, apc = processor.tokenizer, APCManager()
    label = label or (Path(adapter).name if adapter else "base")
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with (EVALS_DIR / f"{label}.jsonl").open("w", encoding="utf-8") as out:
        for ep in episodes:
            events = [json.loads(line) for line in open(f"data/episodes/{ep}.jsonl", encoding="utf-8")]
            task = next(e["task"] for e in events if e.get("event") == "start")
            history, n = [], 0
            for e in events:
                if e.get("event") != "tool":
                    continue
                name, astra_args = e["name"], e.get("args") or {}
                if name in schemas and (not max_steps or n < max_steps):
                    msgs = step_messages(EXPLORE_INSTRUCTIONS, task, history, window)
                    prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=False)
                    t0 = time.perf_counter()
                    r = generate(model, processor, prompt, max_tokens=160, temperature=0.0, apc_manager=apc)
                    calls = parse_calls(r.text)
                    row = {"episode": ep, "step": e.get("step"), "astra": {"name": name, "args": astra_args},
                           "seconds": round(time.perf_counter() - t0, 2), "out_tokens": r.generation_tokens,
                           "text": r.text.strip()[:300], **score_step(calls[0] if calls else None, name, astra_args, schemas)}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(row)
                    n += 1
                history += [call_message(name, astra_args), result_message(e.get("result") or {})]
            log(f"{ep}: {sum(r.get('same_tool', False) for r in rows if r['episode'] == ep)}/{n} same tool")
    summary = summarize(rows)
    summary.update(label=label, adapter=str(adapter) if adapter else None, window=window)
    (EVALS_DIR / f"{label}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    share = lambda pred: round(100 * sum(1 for r in rows if pred(r)) / max(n, 1), 1)
    sub = lambda pred, key: [r[key] for r in rows if pred(r) and key in r]
    xy = sub(lambda r: True, "xy_err_mm")
    z = sub(lambda r: True, "z_err_mm")
    grip = [r for r in rows if "gripper_inverted" in r]
    secs = sorted(r["seconds"] for r in rows) or [0]
    return {"steps": n, "parsed_pct": share(lambda r: r.get("parsed")), "valid_pct": share(lambda r: r.get("valid")),
            "same_tool_pct": share(lambda r: r.get("same_tool")),
            "same_kind_pct": share(lambda r: r.get("same_category")),
            "move_to_pose_xy_within_10mm": f"{sum(e <= 10 for e in xy)}/{len(xy)}",
            "move_to_pose_xy_median_mm": sorted(xy)[len(xy) // 2] if xy else None,
            "routine_name_right": f"{sum(1 for r in rows if r.get('routine_same'))}/"
                                  f"{sum(1 for r in rows if r['astra']['name'] == 'run_routine' and r.get('same_tool'))}",
            "routine_z_within_10mm": f"{sum(e <= 10 for e in z)}/{len(z)}",
            "gripper_inverted": f"{sum(r['gripper_inverted'] for r in grip)}/{len(grip)}",
            "seconds_median": secs[len(secs) // 2], "seconds_p90": secs[int(len(secs) * .9)],
            "per_tool": {k: f"{v[1]}/{v[0]}" for k, v in _per_tool(rows).items()}}


def _per_tool(rows):
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["astra"]["name"]][0] += 1
        by[r["astra"]["name"]][1] += bool(r.get("same_tool"))
    return dict(sorted(by.items(), key=lambda kv: -kv[1][0]))


def compare(a: str, b: str) -> list[tuple[str, object, object]]:
    A = json.loads((EVALS_DIR / f"{a}.json").read_text(encoding="utf-8"))
    B = json.loads((EVALS_DIR / f"{b}.json").read_text(encoding="utf-8"))
    keys = [k for k in A if k not in ("per_tool", "label", "adapter")]
    return [(k, A.get(k), B.get(k)) for k in keys]
