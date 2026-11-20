"""The teacher session: Astra (API) edits the student's weights through the kernel API.

Astra gets three tools and nothing else:
  python(code)     runs in a separate process with the kernel API + numpy imported, cwd = the work
                   folder; no API keys in its environment; state persists only through files in work/
  forward(request) the runner's forward pass (fixed budget); Astra never runs the model itself
  finish(report)   ends the session
It does not control the robot or the student. After the session the operator runs the held-out
test (runner.test), which Astra never sees.

Not an OS sandbox (one user, one machine): the separation is by convention + the kernel's checks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from student.weights import kernel, runner
from student.weights.data import DATA, RUNNER

SRC = str(Path(__file__).resolve().parents[2])

PRELUDE = f"""
import sys; sys.path.insert(0, {SRC!r})
import json
import numpy as np
from student.weights.kernel import (NAMES, info, load, load_original, save_edit, drop_edit, edited,
                                    ablate_direction, MAX_CHANGED_FRACTION)
DATA = {str(DATA)!r}
"""

def _model_line() -> str:
    t = kernel.text_config()
    return (f"{kernel.FROZEN_DIR.name.removesuffix('-frozen')}, {t.get('num_hidden_layers')} layers, "
            f"hidden size {t.get('hidden_size')}")


INSTRUCTIONS = f"""You are the teacher of a small language model, the student ({_model_line()}). The student drives a robot arm by calling tools. Its problem: after a `look` that
already returned the objects (their robot coordinates), it calls `look` again and again instead of
acting; in a live run it looked 17 times in a row and did nothing else.

Your job: change the student's WEIGHTS so that at such steps it takes the next action, measured on the
dev items, without breaking its other choices. You never run the model yourself and you do not control
the robot.

Data ({DATA}): dev.jsonl, one decision step per line:
  {{"id", "tags": ["after_look" | "other"], "messages": [chat the student sees], "label": {{"tool": the tool
  the expert called next}}}}; tools.json: the tool list the student gets with every item.

Tools:
  python(code): runs Python in a fresh process (cwd = {kernel.WORK}); already imported: numpy as np, json,
    and the kernel API:
      NAMES                      every tensor name (e.g. language_model.model.layers.7.mlp.down_proj.weight)
      info(name)                 {{shape, quantized, in_scope, edited}}
      load(name) / load_original(name)  float32 numpy array, shape (out, in) for linear layers
      save_edit(name, array)     store an edited tensor (same shape)
      drop_edit(name), edited()
      ablate_direction(W, u, alpha=1.0, cols=None)   W - alpha * u (u^T W), u in W's output space
    Keep notes and intermediate results in files in the work folder (notes.md, *.npy): nothing else survives
    between calls. Printed output is cut at 6000 characters.
  forward(request): {{"type": "eval" | "activations", "model": "original" | "edited" | "both", "ids": [dev ids]}}
    eval: the student answers each item; returns accuracy (same tool as the label), look_again (share of
    after_look items where it calls look again), and for "both": diff, se, rescued, broken. Per-item answers
    are written to fwd_<k>/answers_<model>.jsonl.
    activations: the residual stream at the last prompt token after every layer, saved as
    fwd_<k>/acts_<model>.npy with shape (items, layers, hidden) and fwd_<k>/ids.json.
    At most 128 items per call. The budget of forward calls is fixed and counted.
  finish(report): end the session with what you changed, why, and the measured effect.

Rules (enforced): the token embedding (also the output head), the final norm and the vision tower are out
of scope; no whole-tensor scaling; the edited model may change at most {kernel.MAX_CHANGED_FRACTION:.1%} of the
language model's weights. There is a held-out test set you will not see: an edit that only fits the dev
items will not count.
"""

TOOLS = [
    {"type": "function", "name": "python", "description": "Run Python with the kernel API (see instructions).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "forward", "description": "The runner's forward pass (budgeted).",
     "parameters": {"type": "object", "properties": {
         "type": {"type": "string", "enum": ["eval", "activations"]},
         "model": {"type": "string", "enum": ["original", "edited", "both"]},
         "ids": {"type": "array", "items": {"type": "string"}}}, "required": ["type", "model"]}},
    {"type": "function", "name": "finish", "description": "End the session with a report.",
     "parameters": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}},
]


def run_python(code: str, timeout: int = 900) -> str:
    kernel.WORK.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "TMPDIR")}
    try:
        p = subprocess.run([sys.executable, "-I", "-c", PRELUDE + "\n" + code], cwd=kernel.WORK, env=env,
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + ("\n[stderr]\n" + p.stderr[-3000:] if p.returncode else "")).strip()
    except subprocess.TimeoutExpired:
        out = f"timed out after {timeout} s"
    return out[:6000] or "(no output)"


def session(max_steps: int = 80, log=print) -> dict:
    from agent.astra_client import AstraClient
    from common.config import load_agent_config

    cfg = load_agent_config()
    cfg.max_output_tokens = max(cfg.max_output_tokens, 4000)
    client = AstraClient(cfg)
    RUNNER.mkdir(parents=True, exist_ok=True)
    logf = (RUNNER / f"teacher_{time.strftime('%Y%m%d-%H%M%S')}.jsonl").open("w", encoding="utf-8")
    b = runner.budget()
    first = (f"Begin. Forward budget: {b['total'] - b['used']} calls left. Edits already saved: {kernel.edited() or 'none'}. "
             f"Work folder: {kernel.WORK}.")
    response = client.create(first, tools=TOOLS, instructions=INSTRUCTIONS)
    report = None
    for step in range(max_steps):
        calls = [i for i in response.output if getattr(i, "type", None) == "function_call"]
        logf.write(json.dumps({"step": step, "text": response.output_text, "calls": len(calls)}) + "\n")
        if response.output_text:
            log(f"[astra] {response.output_text[:400]}")
        if not calls:
            report = response.output_text
            break
        outputs = []
        for c in calls:
            try:
                args = json.loads(c.arguments or "{}")
            except ValueError:
                args = {}
            if c.name == "python":
                result = run_python(args.get("code", ""))
            elif c.name == "forward":
                try:
                    result = json.dumps(runner.forward(args))
                except Exception as exc:  # noqa: BLE001 — the teacher reads the error
                    result = f"error: {exc}"
            elif c.name == "finish":
                report = args.get("report", "")
                result = "session ended"
            else:
                result = f"unknown tool {c.name}"
            log(f"[{c.name}] {json.dumps(args)[:200]} -> {result[:300]}")
            logf.write(json.dumps({"step": step, "tool": c.name, "args": args, "result": result[:6000]}) + "\n")
            logf.flush()
            outputs.append({"type": "function_call_output", "call_id": c.call_id, "output": result})
        if report is not None:
            break
        response = client.create(outputs, tools=TOOLS, previous_response_id=response.id, instructions=INSTRUCTIONS)
    logf.close()
    return {"report": report, "edited": kernel.edited(), "budget": runner.budget(), "edit_sha": runner.edit_sha()}
