"""Training data for the student: one example per robot step of an earlier episode.

An example is what the student would have seen at that step (student.context.step_messages) and,
as the target, the tool call the teacher (Astra) made there. The teacher shapes the data with
edits (data/student/edits.jsonl), one JSON object per line:

    {"op": "drop",    "id": "<episode>#<n>", "why": "..."}                      never train on this step
    {"op": "relabel", "id": "<episode>#<n>", "name": "...", "args": {...}, "why": "..."}   a better call
    {"op": "add", "id": "teacher-<anything>", "messages": [...], "name": "...", "args": {...}, "why": "..."}

Calls that failed or were refused stay in the history (the student must see them) but are never
targets unless the teacher relabels them. Episodes in student.paths.EVAL_EPISODES never enter.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from student.context import WINDOW, call_message, result_message, step_messages, tool_schemas
from student.paths import DATASETS_DIR, EDITS_FILE, EVAL_EPISODES


def _ok(result: dict) -> bool:
    return bool(result.get("ok")) and result.get("executed", True) is not False


def episode_examples(path: Path, system: str, tools: set[str], window: int = WINDOW) -> list[dict]:
    events = [json.loads(line) for line in path.open(encoding="utf-8")]
    start = next((e for e in events if e.get("event") == "start"), None)
    if start is None or (start.get("meta") or {}).get("mode") != "live":
        return []
    task, history, out, n = start["task"], [], [], 0
    for e in events:
        if e.get("event") != "tool":
            continue
        n += 1
        name, args, result = e["name"], e.get("args") or {}, e.get("result") or {}
        if name in tools and _ok(result):
            out.append({"id": f"{path.stem}#{n}", "episode": path.stem, "messages": step_messages(system, task, history,
                                                                                                    window),
                        "name": name, "args": args})
        history += [call_message(name, args), result_message(result)]
    return out


def load_edits(path: Path = EDITS_FILE) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_edits(examples: list[dict], edits: list[dict], tools: set[str]) -> tuple[list[dict], dict]:
    by_id = {e["id"]: e for e in examples}
    done = {"drop": 0, "relabel": 0, "add": 0, "ignored": []}
    for ed in edits:
        op, eid = ed.get("op"), ed.get("id")
        if op == "drop" and eid in by_id:
            by_id.pop(eid)
            done["drop"] += 1
        elif op == "relabel" and eid in by_id and ed.get("name") in tools:
            by_id[eid] = {**by_id[eid], "name": ed["name"], "args": ed.get("args") or {}, "edited": ed.get("why", "")}
            done["relabel"] += 1
        elif op == "add" and ed.get("messages") and ed.get("name") in tools:
            by_id[eid or f"teacher-{len(by_id)}"] = {"id": eid, "episode": "teacher", "messages": ed["messages"],
                                                      "name": ed["name"], "args": ed.get("args") or {},
                                                      "edited": ed.get("why", "")}
            done["add"] += 1
        else:
            done["ignored"].append(eid)
    return list(by_id.values()), done


def build(episodes_dir: Path = Path("data/episodes"), window: int = WINDOW, valid_share: float = 0.1,
          name: str | None = None) -> dict:
    from agent.explore import EXPLORE_INSTRUCTIONS
    schemas = tool_schemas()
    tools = {t["function"]["name"] for t in schemas}
    examples = []
    for path in sorted(episodes_dir.glob("2026*.jsonl")):
        if path.stem not in EVAL_EPISODES:
            examples += episode_examples(path, EXPLORE_INSTRUCTIONS, tools, window)
    examples, applied = apply_edits(examples, load_edits(), tools)
    # split by episode, so no step of a validation episode is trained on
    is_valid = lambda ep: int(hashlib.md5(ep.encode()).hexdigest(), 16) % 1000 < valid_share * 1000
    train = [e for e in examples if not is_valid(e["episode"])]
    valid = [e for e in examples if is_valid(e["episode"])]
    did = name or time.strftime("%Y%m%d-%H%M%S")
    out = DATASETS_DIR / did
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("valid", valid)):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "tools.json").write_text(json.dumps(schemas, indent=1) + "\n", encoding="utf-8")
    meta = {"id": did, "window": window, "train": len(train), "valid": len(valid),
            "episodes": len({e["episode"] for e in examples}), "edits": applied, "held_out": list(EVAL_EPISODES),
            "targets": _count(e["name"] for e in train)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def _count(names) -> dict:
    from collections import Counter
    return dict(Counter(names).most_common())


def load(did: str) -> tuple[list[dict], list[dict], list[dict]]:
    d = DATASETS_DIR / did
    read = lambda f: [json.loads(line) for line in (d / f).open(encoding="utf-8")]
    return read("train.jsonl"), read("valid.jsonl"), json.loads((d / "tools.json").read_text(encoding="utf-8"))
