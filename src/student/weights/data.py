"""Decision items for the weight lab: one step of an earlier Astra episode, what the student would see
there (student.context.step_messages) and, as the label, the tool Astra called next.

    {"id": "<episode>#<n>", "tags": ["after_look"], "messages": [...], "label": {"tool": "move_to_pose"}}

tags:
  after_look   the previous call was a `look` that returned objects: the student's failure mode is to
               look again here (episode 20261119-131530: 17 looks in a row)
  other        any other step

dev  (the teacher sees it, labels included): Astra episodes outside student.paths.EVAL_EPISODES
test (runner only, never shown to the teacher): the held-out EVAL_EPISODES
The tool list is the same for every item: data/tools.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from student.context import WINDOW, call_message, result_message, step_messages, tool_schemas
from student.paths import EVAL_EPISODES

LAB = Path(__file__).resolve().parents[3] / "data" / "student" / "weightlab"
DATA = LAB / "data"            # the teacher may read this
RUNNER = LAB / "runner"        # the teacher must not


def _found_objects(result: dict) -> bool:
    r = result.get("result") if isinstance(result.get("result"), dict) else {}
    return bool(result.get("ok")) and bool(r.get("objects"))


def _items(path: Path, system: str, tools: set[str]) -> list[dict]:
    events = [json.loads(line) for line in path.open(encoding="utf-8")]
    start = next((e for e in events if e.get("event") == "start"), None)
    if start is None or (start.get("meta") or {}).get("mode") != "live" or \
            str((start.get("meta") or {}).get("model", "")).startswith("qwen"):
        return []
    task, history, out, prev, n = start["task"], [], [], None, 0
    for e in events:
        if e.get("event") != "tool":
            continue
        n += 1
        name, args, result = e["name"], e.get("args") or {}, e.get("result") or {}
        if name in tools and result.get("executed", True) is not False:
            after_look = prev is not None and prev["name"] == "look" and _found_objects(prev.get("result") or {})
            out.append({"id": f"{path.stem}#{n}", "tags": ["after_look" if after_look else "other"],
                        "messages": step_messages(system, task, history, WINDOW), "label": {"tool": name}})
        history += [call_message(name, args), result_message(result)]
        prev = e
    return out


def build(episodes_dir: Path = Path("data/episodes"), dev_size: int = 60, test_size: int = 60, seed: int = 0) -> dict:
    """dev: after_look steps where Astra acted (not looked) + after_look steps where it looked again +
    other steps, so an edit that just bans `look` is caught."""
    from agent.explore import EXPLORE_INSTRUCTIONS
    schemas = tool_schemas()
    tools = {t["function"]["name"] for t in schemas}
    pool = {"dev": [], "test": []}
    for path in sorted(episodes_dir.glob("2026*.jsonl")):
        pool["test" if path.stem in EVAL_EPISODES else "dev"] += _items(path, EXPLORE_INSTRUCTIONS, tools)
    rng = random.Random(seed)
    picked = {}
    for split, size in (("dev", dev_size), ("test", test_size)):
        items = pool[split]
        act = [i for i in items if "after_look" in i["tags"] and i["label"]["tool"] != "look"]
        relook = [i for i in items if "after_look" in i["tags"] and i["label"]["tool"] == "look"]
        other = [i for i in items if "other" in i["tags"]]
        for group in (act, relook, other):
            rng.shuffle(group)
        k_act, k_relook = size * 6 // 10, size * 15 // 100
        chosen = act[:k_act] + relook[:k_relook]
        chosen += other[:size - len(chosen)]
        rng.shuffle(chosen)
        picked[split] = chosen
    DATA.mkdir(parents=True, exist_ok=True)
    RUNNER.mkdir(parents=True, exist_ok=True)
    (DATA / "tools.json").write_text(json.dumps(schemas) + "\n", encoding="utf-8")
    for split, folder in (("dev", DATA), ("test", RUNNER)):
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in picked[split]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    count = lambda items, pred: sum(1 for i in items if pred(i))
    return {split: {"items": len(items),
                    "after_look_act": count(items, lambda i: "after_look" in i["tags"] and i["label"]["tool"] != "look"),
                    "after_look_relook": count(items, lambda i: "after_look" in i["tags"] and i["label"]["tool"] == "look"),
                    "other": count(items, lambda i: "other" in i["tags"])}
            for split, items in picked.items()}


def load_items(split: str) -> list[dict]:
    folder = DATA if split == "dev" else RUNNER
    return [json.loads(line) for line in (folder / f"{split}.jsonl").open(encoding="utf-8")]


def load_tools() -> list[dict]:
    return json.loads((DATA / "tools.json").read_text(encoding="utf-8"))
