"""One attempt = one episode in program mode: what ran, what came out, and the report the next episode starts with.

ENPIRE/ASPIRE loop (23 Nov): the program runs locally, an outcome is established outside the agent, and the next
revision starts from that outcome plus the program's own trace. The outcome comes from the judge ledger when it
reached a verdict; while the verifier is not qualified yet (inconclusive), the operator's label of the attempt is
used and says so. Labels live in data/judge/ (no agent tool reads it) and also qualify the verifier later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from common.paths import DATA_DIR

LABELS_FILE = DATA_DIR / "judge" / "labels.jsonl"
LABELS = {"d": "success", "k": "failure", "?": "unclear"}


def record_label(episode: str, label: str, note: str = "", path: Path | None = None) -> dict:
    """The operator's ground truth for one attempt (success / failure / unclear + why)."""
    if label not in LABELS.values():
        raise ValueError(f"label must be one of {sorted(LABELS.values())}")
    path = Path(path or LABELS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"episode": episode, "label": label, "note": note.strip()[:300], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def labels(path: Path | None = None) -> dict:
    """episode -> the latest operator label."""
    path = Path(path or LABELS_FILE)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["episode"]: r for r in rows}


def outcome(episode: str, ledger, label_rows: dict) -> dict:
    """{status, source, detail}: the judge's verdict when it reached one, else the operator's label."""
    v = ledger.outcome(episode)
    if v.get("status") in ("passed", "failed"):
        return {"status": v["status"], "source": "verifier", "detail": ""}
    lab = label_rows.get(episode)
    if lab:
        status = {"success": "passed", "failure": "failed"}.get(lab["label"], "inconclusive")
        return {"status": status, "source": "operator", "detail": lab.get("note", "")}
    return {"status": "inconclusive", "source": "none", "detail": "no verdict and no operator label"}


def program_runs(log: list) -> list[dict]:
    """Every run_routine of the episode: routine, version, args, ok, detail, first failing trace entry, length."""
    runs = []
    for e in log:
        if e.get("event") != "tool" or e.get("name") != "run_routine":
            continue
        r = (e.get("result") or {})
        res = r.get("result") if isinstance(r.get("result"), dict) else {}
        from agent.events import flat_trace
        trace = flat_trace(res.get("trace") or [])
        bad = next((t for t in trace if not t.get("ok") and "trace" not in t), None) or \
            next((t for t in trace if not t.get("ok")), None)
        runs.append({"routine": res.get("routine") or (e.get("args") or {}).get("name"), "version": res.get("version"),
                     "args": (e.get("args") or {}).get("args"), "ok": bool(res.get("ok")) if r.get("ok") else False,
                     "detail": str(res.get("detail") or r.get("error") or "")[:240], "steps": len(trace),
                     "first_failed_step": bad, "last_steps": trace[-3:]})
    return runs


def report(store, ledger, episode: str, label_rows: dict | None = None) -> str:
    """The text the next episode starts with (facts only: no advice)."""
    log = store.load(episode)
    o = outcome(episode, ledger, labels() if label_rows is None else label_rows)
    lines = [f"PREVIOUS ATTEMPT (episode {episode}): outcome {o['status'].upper()} (source: {o['source']})"
             + (f" — {o['detail']}" if o["detail"] else "")]
    end = next((e for e in reversed(log) if e.get("event") == "end"), None)
    if end:
        lines.append(f"It ended: {str(end.get('reason'))[:200]}")
    check = next((e for e in reversed(log) if e.get("event") == "self_check"), None)
    if check:
        from agent.explore.checking import autonomous_feedback
        lines.append(autonomous_feedback(check))
    runs = program_runs(log)
    acts = [e for e in log if e.get("event") == "tool" and not e.get("auto") and e.get("name") not in (
        "look", "get_state", "list_routines", "read_routine", "query_observations", "read_history")]
    if not runs and acts:
        # 23 Nov threading task: the agent moves step by step; the report lists what it did, not programs
        lines.append(f"{len(acts)} actions; the last {min(5, len(acts))}:")
        for e in acts[-5:]:
            r = e.get("result") or {}
            pose = (r.get("result") or {}).get("pose") if isinstance(r.get("result"), dict) else None
            lines.append(f"- {e['name']}({json.dumps(e.get('args') or {}, ensure_ascii=False)[:160]}) -> "
                         + ("ok" + (f" tip {[round(v, 1) for v in pose[:3]]}" if pose else "") if r.get("ok")
                            else "ERROR " + str(r.get("error", ""))[:160]))
    if not runs and not acts:
        lines.append("No action was run in that episode.")
    elif len(runs) > 3:
        lines.append(f"{len(runs)} program runs; the last 3:")
    for r in runs[-3:]:
        lines.append(f"- ran {r['routine']} v{r['version']} args={json.dumps(r['args'], ensure_ascii=False)[:200]}: "
                     f"{'ok' if r['ok'] else 'FAILED'} after {r['steps']} steps"
                     + (f" — {r['detail']}" if r['detail'] else ""))
        if r["first_failed_step"]:
            lines.append(f"  first failed step: {json.dumps(r['first_failed_step'], ensure_ascii=False)[:400]}")
        elif r["last_steps"]:
            lines.append(f"  last steps: {json.dumps(r['last_steps'], ensure_ascii=False)[:400]}")
    # 24 Nov: what the agent learned in that attempt (e.g. which way the picture moves for a move) was lost:
    # its own final report goes with the next attempt (a claim, not verified); the self-check is above
    final = next((e.get("text") for e in reversed(log) if e.get("event") == "model" and e.get("text")), "")
    if final:
        lines.append("Your final report then (your own words, not verified): " + " ".join(final.split())[:1500])
    now = _current_versions({r["routine"] for r in runs[-3:]})
    changed = [f"{n} v{v}" for n, v in now.items() if v not in {r["version"] for r in runs if r["routine"] == n}]
    if changed:
        lines.append("Changed since (the review rewrote them): now " + ", ".join(changed) + ".")
    return "\n".join(lines)


def _current_versions(names: set) -> dict:
    try:
        from actions.astra_code import AstraCode
        items = AstraCode.default().items
    except Exception:  # noqa: BLE001 — no code directory: nothing to compare
        return {}
    return {n: items[n]["version"] for n in names if n in items}


def last_astra_episode(store) -> str | None:
    eps = store.list(driver="astra")
    return eps[-1] if eps else None
