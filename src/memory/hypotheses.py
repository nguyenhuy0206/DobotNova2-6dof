"""LEGACY (until 2026-11-17): free-text hypotheses with status, scope and evidence.

Replaced by memory/knowledge.py (observations -> concepts -> hypotheses with predictions).
data/memory/hypotheses.json is kept unchanged and is read through read_history(kind=old_hypotheses).

Raw experience retrieved into the prompt (free-text notes, trial logs) misleads more
than it helps: it carries the conditions it was written under without saying so.
Here every piece of knowledge is a claim that must earn its status:

    {"id": "K7", "claim": "...", "scope": {"skill": "pick_block_wrist", "rz": "81.9", ...},
     "evidence": [{"t", "episode", "result": "supports"|"contradicts", "observation", "source"}],
     "status": open | supported | contested | refuted | archived}

Status is COMPUTED from the evidence (never set by hand), except archived:
    supported   >= 2 supporting, 0 contradicting
    refuted     contradicting > supporting and (>= 2 contradicting, or 0 supporting)
    contested   both kinds of evidence
    open        otherwise (e.g. one observation)
Only the digest (claim + status + counts + scope) goes into the prompt; archived ones
never do. Nothing is deleted: archived hypotheses keep their evidence and reason.
data/memory/hypotheses.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from common.errors import ValidationError
from common.paths import MEMORY_DIR

HYPOTHESES_FILE = MEMORY_DIR / "hypotheses.json"
RESULTS = ("supports", "contradicts")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def status_of(h: dict) -> str:
    if h.get("archived"):
        return "archived"
    sup = sum(1 for e in h.get("evidence", []) if e.get("result") == "supports")
    con = sum(1 for e in h.get("evidence", []) if e.get("result") == "contradicts")
    if con > sup and (con >= 2 or sup == 0):
        return "refuted"
    if sup >= 2 and con == 0:
        return "supported"
    if sup and con:
        return "contested"
    return "open"


class HypothesisStore:
    prompt_header = ("Your hypotheses from earlier experiments (status is computed from recorded evidence; scope = "
                     "the conditions they were observed under — they may not hold outside it):")

    def __init__(self, path: Path = HYPOTHESES_FILE):
        self.path = path
        self.episode: str | None = None          # set by the harness: evidence is tagged with it
        self.items: list[dict] = []
        if path.exists():
            self.items = json.loads(path.read_text(encoding="utf-8")).get("hypotheses", [])

    # ---------------------------------------------------------------- storage
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"hypotheses": self.items}, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")

    def _get(self, hid: str) -> dict:
        h = next((x for x in self.items if x["id"] == hid), None)
        if h is None:
            raise ValidationError(f"no hypothesis {hid!r}")
        return h

    def _view(self, h: dict, evidence: int = 3) -> dict:
        sup = sum(1 for e in h["evidence"] if e["result"] == "supports")
        out = {"id": h["id"], "status": status_of(h), "claim": h["claim"], "scope": h["scope"],
               "supports": sup, "contradicts": len(h["evidence"]) - sup, "updated": h["updated"],
               "evidence": h["evidence"][-evidence:]}
        if h.get("archived"):
            out["archived"] = h["archived"]
        return out

    # ---------------------------------------------------------------- tools
    def propose(self, claim: str, scope: dict | None = None, observation: str = "",
                result: str = "supports", source: str = "astra") -> dict:
        """New hypothesis (or the existing one with the same claim + scope). With an
        observation, that is its first piece of evidence."""
        claim = str(claim).strip()
        if len(claim) < 10:
            raise ValidationError("claim: one testable sentence")
        scope = {str(k)[:30]: str(v)[:80] for k, v in (scope or {}).items()}
        if len(scope) > 8:
            raise ValidationError("scope: at most 8 conditions")
        h = next((x for x in self.items if _norm(x["claim"]) == _norm(claim) and x["scope"] == scope
                  and not x.get("archived")), None)
        now = time.strftime("%Y-%m-%d %H:%M")
        if h is None:
            h = {"id": f"K{len(self.items) + 1}", "claim": claim[:300], "scope": scope, "evidence": [],
                 "created": now, "updated": now, "source": source}
            self.items.append(h)
        if observation:
            return self.record(h["id"], result, observation, source=source)
        self._save()
        return self._view(h)

    def record(self, hypothesis_id: str, result: str, observation: str, source: str = "astra") -> dict:
        """Add one piece of evidence; the status follows from all the evidence."""
        if result not in RESULTS:
            raise ValidationError(f"result must be one of {RESULTS}")
        if len(str(observation).strip()) < 5:
            raise ValidationError("observation: what was measured / seen (numbers)")
        h = self._get(hypothesis_id)
        if h.get("archived"):
            raise ValidationError(f"{hypothesis_id} is archived ({h['archived']['reason']})")
        before = status_of(h)
        h["evidence"].append({"t": time.strftime("%Y-%m-%d %H:%M"), "episode": self.episode, "result": result,
                              "observation": str(observation).strip()[:240], "source": source})
        h["updated"] = h["evidence"][-1]["t"]
        self._save()
        out = self._view(h)
        if out["status"] != before:
            out["status_changed"] = f"{before} -> {out['status']}"
        return out

    def generalize(self, from_ids: list, claim: str, scope: dict | None = None) -> dict:
        """A broader hypothesis built from narrower ones: it inherits ALL their evidence
        (supporting and contradicting, tagged with where it came from), so its status is
        earned over every scope it covers. The narrower ones stay, linked to it."""
        if not isinstance(from_ids, list) or len(from_ids) < 2:
            raise ValidationError("from_ids: at least 2 hypotheses to generalize from")
        kids = [self._get(i) for i in from_ids]
        if any(k.get("archived") for k in kids):
            raise ValidationError("cannot generalize from archived hypotheses")
        h = self.propose(claim, scope)
        rec = self._get(h["id"])
        if rec["id"] in from_ids:
            raise ValidationError("the general claim must differ from the ones it generalizes")
        for k in kids:
            for e in k["evidence"]:
                rec["evidence"].append({**e, "source": f"{e.get('source', '')} via {k['id']}".strip()})
            k["generalized_into"] = rec["id"]
        rec["generalizes"] = sorted(set(rec.get("generalizes", [])) | {k["id"] for k in kids})
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M")
        self._save()
        return {**self._view(rec), "generalizes": rec["generalizes"]}

    def archive(self, hypothesis_id: str, reason: str) -> dict:
        """Take a hypothesis out of the prompt (kept with its evidence)."""
        if len(str(reason).strip()) < 5:
            raise ValidationError("reason: why it no longer applies")
        h = self._get(hypothesis_id)
        h["archived"] = {"t": time.strftime("%Y-%m-%d %H:%M"), "reason": str(reason).strip()[:200]}
        self._save()
        return self._view(h)

    def list(self, status: str | None = None, query: str | None = None, limit: int = 30) -> dict:
        rows = [h for h in self.items if status in (None, "", status_of(h))
                and (not query or _norm(query) in _norm(h["claim"] + " " + json.dumps(h["scope"])))]
        return {"hypotheses": [self._view(h) for h in rows[-limit:]], "total": len(rows),
                "by_status": self.counts()}

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for h in self.items:
            out[status_of(h)] = out.get(status_of(h), 0) + 1
        return out

    # ---------------------------------------------------------------- prompt
    def digest(self, max_chars: int = 2400, refuted: int = 4) -> str:
        """Prompt view: supported / contested / open (newest first), then a few refuted ones
        as 'ruled out'. Archived never. Claims + counts + scope only — no raw observations."""
        def line(h):
            v = self._view(h, 0)
            scope = ", ".join(f"{k}={val}" for k, val in v["scope"].items())
            return (f"- {v['id']} [{v['status']} {v['supports']}✓ {v['contradicts']}✗ · {v['updated'][:10]}"
                    + (f" · {scope}" if scope else "") + f"] {v['claim'][:220]}")
        order = {"supported": 0, "contested": 1, "open": 2}
        live = sorted((h for h in self.items if status_of(h) in order), key=lambda h: h["updated"], reverse=True)
        live = [line(h) for h in sorted(live, key=lambda h: order[status_of(h)])]    # stable: newest first per status
        ruled = [line(h) for h in self.items if status_of(h) == "refuted"][-refuted:]
        tail = (["Ruled out:"] + ruled) if ruled else []
        budget = max_chars - sum(len(x) + 1 for x in tail)
        while live and sum(len(x) + 1 for x in live) > budget:
            live.pop()                                                # drop the oldest / weakest first
        return "\n".join(live + tail)

    def recent_text(self, limit: int = 30) -> str:
        return self.digest()

    # ---------------------------------------------------------------- upkeep
    def archive_out_of_scope(self, known: set[str]) -> list[str]:
        """Archive hypotheses whose scope names a skill/routine/check that no longer exists."""
        done = []
        for h in self.items:
            if h.get("archived"):
                continue
            gone = [h["scope"][k] for k in ("skill", "routine", "check") if h["scope"].get(k)
                    and h["scope"][k] not in known]
            if gone:
                h["archived"] = {"t": time.strftime("%Y-%m-%d %H:%M"),
                                 "reason": f"scope names {', '.join(gone)}, which no longer exists"}
                done.append(h["id"])
        if done:
            self._save()
        return done


# note topics that are raw experience or plans, not testable claims -> archived on migration
ARCHIVE_TOPICS = {
    "grasp_trials": "raw trial log (experience, not a hypothesis)",
    "grasp_trials_audit": "raw trial log (experience, not a hypothesis)",
    "grasp_analysis": "episode summary table (experience, not a hypothesis)",
    "grasp_recovery": "episode summary (experience, not a hypothesis)",
    "grasp_best": "episode summary / plan for the next attempt, not a testable claim",
    "calibration": "describes one env calibration that was replaced since; verify_calibration measures the current one",
}


def migrate_notes(notes: list[dict], store: HypothesisStore, removed_names: set[str]) -> dict:
    """One-off: raw notes -> hypotheses. Raw experience / plans (ARCHIVE_TOPICS) and notes
    about removed skills/routines are ARCHIVED (kept with the note as evidence, never shown);
    a 'causes' note (a diagnosis test confirmed it) becomes a claim with one supporting
    observation; other measured notes (wrist_camera, geometry …) open claims scoped to their date."""
    stats = {"open": 0, "archived": 0}
    for n in notes:
        text, topic, t = str(n.get("text", "")).strip(), n.get("topic", "general"), n.get("t", "")
        if len(text) < 10:
            continue
        scope = {"topic": topic}
        m = re.match(r"^(\S+) (\S+): cause = (.+?)(?: \(evidence: (.+?)\))?(?:; fix = (.+))?$", text, re.S)
        if topic == "causes" and m:
            claim = f"{m.group(1)} {m.group(2)} is caused by: {m.group(3)}"
            scope = {"skill": m.group(1), "failure": m.group(2)}
            obs = (m.group(4) or "confirmed by a diagnosis test") + (f"; fix: {m.group(5)}" if m.group(5) else "")
        else:
            claim, obs = text, "written as a note (migrated; no structured evidence)"
        h = store.propose(claim[:300], scope)
        h_rec = store._get(h["id"])
        h_rec["evidence"].append({"t": t, "episode": None, "result": "supports", "observation": obs[:240],
                                  "source": f"note:{topic}"})
        h_rec["updated"] = t or h_rec["updated"]
        mentioned = sorted(name for name in removed_names if name in text)
        if topic in ARCHIVE_TOPICS:
            h_rec["archived"] = {"t": time.strftime("%Y-%m-%d %H:%M"), "reason": ARCHIVE_TOPICS[topic]}
        elif mentioned:
            h_rec["archived"] = {"t": time.strftime("%Y-%m-%d %H:%M"),
                                 "reason": f"refers to removed {', '.join(mentioned)}"}
        stats["archived" if h_rec.get("archived") else "open"] += 1
    store._save()
    return stats
