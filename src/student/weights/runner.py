"""The runner (operator side): builds the edited student, spends the forward budget, scores.

    forward({"type": "eval" | "activations", "model": "original" | "edited" | "both", "ids": [...]})
        the teacher's only way to run the model (dev items only, at most 128 per call)
    test()  the held-out test set, original vs edited: never offered to the teacher
    reset() drop every edit and refill the budget

Scoring an item: the student answers once (greedy); its first tool call is compared with the label
(the tool Astra called there). `look_again` = share of after_look items where it calls `look` again.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from student.context import parse_calls
from student.paths import FROZEN_DIR
from student.weights import kernel
from student.weights.data import RUNNER, load_items, load_tools

BUDGET_FILE = RUNNER / "budget.json"
LOG_FILE = RUNNER / "forwards.jsonl"


# ------------------------------------------------------------------------------------------- budget
def budget() -> dict:
    return json.loads(BUDGET_FILE.read_text()) if BUDGET_FILE.exists() else {"total": 40, "used": 0}


def set_budget(total: int) -> None:
    RUNNER.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(json.dumps({"total": total, "used": 0}))


def _spend() -> int:
    b = budget()
    if b["used"] >= b["total"]:
        raise RuntimeError(f"forward-pass budget exhausted ({b['total']})")
    b["used"] += 1
    RUNNER.mkdir(parents=True, exist_ok=True)
    BUDGET_FILE.write_text(json.dumps(b))
    return b["used"]


# ------------------------------------------------------------------------------------------- model
def _set_tensor(model, name: str, arr: np.ndarray) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    path, attr = name.rsplit(".", 1)
    parent_path, leaf = path.rsplit(".", 1)
    parent = model
    for part in parent_path.split("."):
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    module = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
    if attr == "weight" and isinstance(module, nn.QuantizedLinear):
        lin = nn.Linear(arr.shape[1], arr.shape[0], bias=False)
        lin.weight = mx.array(arr, dtype=mx.float16)            # full precision: small edits survive
        setattr(parent, leaf, lin)
    else:
        old = getattr(module, attr)
        module.update({attr: mx.array(arr).astype(old.dtype)})


def build_model(which: str):
    from mlx_vlm import load
    model, processor = load(str(FROZEN_DIR))
    if which == "edited":
        frac, changed, total = kernel._changed_fraction()
        if frac > kernel.MAX_CHANGED_FRACTION:
            raise PermissionError(f"the edit changes {frac:.3%} of the language-model weights "
                                  f"(limit {kernel.MAX_CHANGED_FRACTION:.1%})")
        for name in kernel.edited():
            _set_tensor(model, name, kernel.load(name))
    return model, processor


def edit_sha() -> str:
    h = hashlib.sha256()
    for n in kernel.edited():
        h.update(n.encode())
        h.update((kernel.EDITED / f"{n}.npy").read_bytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------- scoring
def _answer(model, processor, tools, item, apc) -> tuple[str | None, str]:
    from mlx_vlm import generate
    prompt = processor.tokenizer.apply_chat_template(item["messages"], tools=tools, add_generation_prompt=True,
                                                     tokenize=False, enable_thinking=False)
    r = generate(model, processor, prompt, max_tokens=64, temperature=0.0, apc_manager=apc)
    calls = parse_calls(r.text + ("" if "</tool_call>" in r.text else "</function></tool_call>"))
    return (calls[0]["name"] if calls else None), r.text


def _eval(which: str, items: list[dict]) -> list[dict]:
    from mlx_vlm.apc import APCManager
    model, processor = build_model(which)
    tools, apc, rows = load_tools(), APCManager(), []
    for it in items:
        tool, text = _answer(model, processor, tools, it, apc)
        rows.append({"id": it["id"], "tags": it["tags"], "label": it["label"]["tool"], "answer": tool,
                     "correct": int(tool == it["label"]["tool"]),
                     "look_again": int("after_look" in it["tags"] and tool == "look"), "text": text[:200]})
    return rows


def _activations(which: str, items: list[dict]) -> np.ndarray:
    """Residual stream at the last prompt token after every layer: (items, layers, hidden)."""
    import mlx.core as mx
    model, processor = build_model(which)
    tools, lm = load_tools(), model.language_model
    layers = list(range(len(lm.model.layers)))
    out = []
    for it in items:
        prompt = processor.tokenizer.apply_chat_template(it["messages"], tools=tools, add_generation_prompt=True,
                                                         tokenize=False, enable_thinking=False)
        ids = mx.array([processor.tokenizer.encode(prompt, add_special_tokens=False)])
        hs = lm(ids, capture_layer_ids=layers, skip_logits=True).hidden_states
        out.append(np.stack([np.array(h[0, -1].astype(mx.float32)) for h in hs]))
    return np.stack(out)


def _summary(results: dict[str, list[dict]]) -> dict:
    s = {}
    for w, rows in results.items():
        al = [r for r in rows if "after_look" in r["tags"]]
        s[f"acc_{w}"] = round(sum(r["correct"] for r in rows) / max(len(rows), 1), 4)
        s[f"look_again_{w}"] = round(sum(r["look_again"] for r in al) / max(len(al), 1), 4)
        s[f"no_tool_call_{w}"] = sum(r["answer"] is None for r in rows)
    if len(results) == 2:
        a, b = (np.array([r["correct"] for r in results[k]]) for k in ("original", "edited"))
        d = b - a
        s.update(diff=round(float(d.mean()), 4),
                 se=round(float(d.std(ddof=1) / math.sqrt(len(d))), 4) if len(d) > 1 else None,
                 rescued=int((d == 1).sum()), broken=int((d == -1).sum()))
    return s


def forward(request: dict) -> dict:
    """The teacher's tool. request: {"type": "eval"|"activations", "model": "original"|"edited"|"both", "ids": [...]}"""
    kind, which = request.get("type", "eval"), request.get("model", "both")
    if kind not in ("eval", "activations") or which not in ("original", "edited", "both"):
        raise ValueError("type: eval | activations; model: original | edited | both")
    dev = load_items("dev")
    want = set(request.get("ids") or [i["id"] for i in dev])
    items = [i for i in dev if i["id"] in want][:128]
    if not items:
        raise ValueError("no dev item has those ids")
    k = _spend()
    out_dir = kernel.WORK / f"fwd_{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    whichs = ["original", "edited"] if which == "both" else [which]
    summary = {"pass": k, "type": kind, "n": len(items), "edit_sha": edit_sha(), "edited": kernel.edited()}
    if kind == "eval":
        results = {w: _eval(w, items) for w in whichs}
        for w, rows in results.items():
            with (out_dir / f"answers_{w}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        summary.update(_summary(results))
    else:
        for w in whichs:
            np.save(out_dir / f"acts_{w}.npy", _activations(w, items))
        (out_dir / "ids.json").write_text(json.dumps([i["id"] for i in items]))
        summary["files"] = sorted(p.name for p in out_dir.iterdir())
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["budget_left"] = budget()["total"] - budget()["used"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**summary, "request": {**request, "ids": len(items)}}) + "\n")
    return summary


def test() -> dict:
    """Held-out: original vs edited on the test set (runner only, no budget)."""
    items = load_items("test")
    results = {w: _eval(w, items) for w in ("original", "edited")}
    s = {"n": len(items), "edit_sha": edit_sha(), "edited": kernel.edited(), **_summary(results)}
    with (RUNNER / "tests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **s}) + "\n")
    return s


def reset(total: int = 40) -> None:
    for n in kernel.edited():
        kernel.drop_edit(n)
    set_budget(total)
