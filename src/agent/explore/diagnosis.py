"""Failure diagnosis: every failure gets hypotheses, every hypothesis gets a test.

    failure F1 (a skill or a robot action failed)
      -> diagnose(F1, why, hypotheses=[…])                             H1, H2, …
         with a knowledge store each hypothesis IS a knowledge hypothesis (concept + mechanism +
         prediction when -> expect, or the id of an existing one), so the code checks it against
         every recorded observation, past and future; without one: {cause, test, expect} text
      -> Astra runs the tests (look, a small move, a changed param …)
      -> conclude(H1, observed, verdict, fix)      confirmed / rejected / inconclusive

Rules enforced here (not only asked for in the prompt):
  * a single failure is just data: Astra may retry with a change right away. Only when
    the SAME failure (same action + code) repeats gate_after_repeats times without a
    diagnosis are run_skill / save_skill refused until it is diagnosed (0 = never);
  * the open failures / untested hypotheses are shown in STATE, so they are not lost
    in a long run;
  * what lasts is the knowledge hypothesis (memory/knowledge.py, deduplicated by concept +
    prediction / mechanism); the failure records, verdicts and fixes live in this episode only.
The whole investigation is in the episode log (diagnose / conclude tool events) and
in summary() for the final report and the self-review.
"""
from __future__ import annotations

import json
import re

from common.errors import ValidationError

# actions whose failure needs an explanation (read-only looks and memory tools do not)
WATCHED = {"run_skill", "run_routine", "move_to_pose", "move_relative", "move_to_joints", "jog_joint",
           "go_to_position", "close_gripper"}
GATED = {"run_skill", "save_skill"}
VERDICTS = ("confirmed", "rejected", "inconclusive")
# refusals that are a known state, not a failure to investigate (the result says what to do)
NOT_FAILURES: set[str] = set()


class Investigation:
    def __init__(self, max_open: int = 6, gate_after_repeats: int = 2, knowledge=None):
        self.knowledge = knowledge          # memory.knowledge.KnowledgeStore, or None (free-text hypotheses)
        self.max_open = max_open
        self.gate_after_repeats = gate_after_repeats
        self.start()

    def start(self, episode: str | None = None) -> None:
        self.episode = episode
        self.failures: list[dict] = []
        self.hypotheses: list[dict] = []

    # ------------------------------------------------------------ harness hooks
    def before(self, name: str, args: dict) -> str | None:
        """Refusal text when a retry would skip the diagnosis, else None."""
        if name not in GATED or self.gate_after_repeats <= 0:
            return None
        groups: dict = {}
        for f in self.failures:
            if not f["diagnosed"]:
                groups.setdefault((f["what"], f["code"]), []).append(f["id"])
        repeated = [(k, ids) for k, ids in groups.items() if len(ids) >= self.gate_after_repeats]
        if not repeated:
            return None
        (what, code), ids = repeated[0]
        return (f"not executed: {what} failed with {code} {len(ids)}× ({', '.join(ids)}) without a diagnosis. "
                f"Call diagnose first (same turn is fine, before {name}): why it keeps failing + 1–3 hypotheses, each "
                "with a cheap test and the expected result — or change the procedure itself (write_routine / "
                "save_skill a new version).")

    def after(self, name: str, args: dict, result: dict) -> dict | None:
        """Register a failure; returns the record added to the tool result (or None)."""
        if name not in WATCHED:
            return None
        res = result.get("result") if isinstance(result.get("result"), dict) else {}
        missed = name == "close_gripper" and result.get("ok") and res.get("holding") is False
        dropped = res.get("failure") == "gripper_reported_drop"   # set by the harness (held before, dropped after)
        failed = (not result.get("ok")) or (name in ("run_skill", "run_routine") and res.get("ok") is False) \
            or missed or dropped
        if not failed:
            return None
        code = "grasp_missed" if missed else (
            res.get("failure") or "error")
        if code in NOT_FAILURES:
            return None
        detail = res.get("detail") or result.get("error") or res.get("advice") or (
            f"closed to position {res.get('position')} without holding anything" if missed else "")
        f = {"id": f"F{len(self.failures) + 1}", "action": name,
             "what": (args.get("name") if name in ("run_skill", "run_routine") else name), "code": code,
             "detail": str(detail)[:200], "diagnosed": False}
        self.failures.append(f)
        same = sum(1 for x in self.failures if not x["diagnosed"] and (x["what"], x["code"]) == (f["what"], code))
        nxt = ("change ONE thing and retry" if self.gate_after_repeats <= 0 or same < self.gate_after_repeats else
               "same failure repeated: diagnose it (why + hypotheses with tests) before the next run_skill")
        return {"failure_id": f["id"], "next": nxt}

    # ------------------------------------------------------------ tools
    def diagnose(self, why: str, hypotheses: list, failure_id: str | None = None) -> dict:
        fail = self._failure(failure_id)
        # one diagnosis covers the earlier undiagnosed failures of the same kind
        for x in self.failures:
            if not x["diagnosed"] and (x["what"], x["code"]) == (fail["what"], fail["code"]) and x is not fail:
                x["diagnosed"], x["why"] = True, f"see {fail['id']}"
        if not why or len(why.strip()) < 10:
            raise ValidationError("why: say in one sentence what went wrong, from the evidence")
        if not isinstance(hypotheses, list) or not 1 <= len(hypotheses) <= 3:
            raise ValidationError("give 1–3 hypotheses (2–3 when the cause is not obvious)")
        out = []
        if self.knowledge is not None:
            linked = [self._link(h) for h in hypotheses]           # validate all before recording any
        for i, h in enumerate(hypotheses):
            if self.knowledge is not None:
                k = linked[i]
                rec = {"id": f"H{len(self.hypotheses) + 1}", "failure": fail["id"], "knowledge": k["id"],
                       "cause": k["mechanism"][:200], "test": str(h["test"])[:200],
                       "expect": json.dumps(k["expect"])[:200], "verdict": None, "observed": None, "fix": None}
                self.hypotheses.append(rec)
                out.append({"id": rec["id"], "knowledge": _brief(k), "test": rec["test"]})
                continue
            if not isinstance(h, dict) or not all(str(h.get(k, "")).strip() for k in ("cause", "test", "expect")):
                raise ValidationError("each hypothesis needs cause, test (a cheap check you will run) and expect "
                                      "(what that test shows if the cause is true)")
            rec = {"id": f"H{len(self.hypotheses) + 1}", "failure": fail["id"], "cause": str(h["cause"])[:200],
                   "test": str(h["test"])[:200], "expect": str(h["expect"])[:200], "verdict": None,
                   "observed": None, "fix": None}
            self.hypotheses.append(rec)
            out.append({k: rec[k] for k in ("id", "cause", "test")})
        fail["diagnosed"], fail["why"] = True, why.strip()[:300]
        return {"failure": fail["id"], "hypotheses": out,
                "next": "run the cheapest decisive test first; then conclude(id, observed, verdict, fix)"}

    def conclude(self, hypothesis_id: str, observed: str, verdict: str, fix: str = "") -> dict:
        m = re.match(r"\s*(H\d+)\b", str(hypothesis_id))          # STATE shows "H2(F1) cause…"
        hypothesis_id = m.group(1) if m else hypothesis_id
        h = next((x for x in self.hypotheses if x["id"] == hypothesis_id), None)
        if h is None:
            raise ValidationError(f"no hypothesis {hypothesis_id!r} (open: {[x['id'] for x in self._untested()]})")
        if verdict not in VERDICTS:
            raise ValidationError(f"verdict must be one of {VERDICTS}")
        if not observed or len(observed.strip()) < 5:
            raise ValidationError("observed: what the test actually showed (numbers / what the camera saw)")
        h.update(verdict=verdict, observed=observed.strip()[:300], fix=(fix or "").strip()[:200] or None)
        out = {"hypothesis": h["id"], "verdict": verdict}
        if h.get("knowledge") and self.knowledge is not None:
            try:                    # the lasting status is computed from observations, not from this verdict
                _, kh = self.knowledge._hyp(h["knowledge"])
                out["knowledge"] = _brief(self.knowledge._hview(kh, ids=0))
            except ValidationError:
                pass
        if verdict == "confirmed":
            out["next"] = "apply the fix, then verify it worked (the same test, or the retried action)"
        if verdict != "confirmed":
            left = [x["id"] for x in self._untested() if x["failure"] == h["failure"]]
            out["next"] = (f"test the next hypothesis: {', '.join(left)}" if left else
                           "all hypotheses tested without a confirmed cause: propose new ones with diagnose")
        return out

    def _link(self, h) -> dict:
        """A diagnose hypothesis -> the knowledge hypothesis it is (existing, or proposed now)."""
        if not isinstance(h, dict) or not str(h.get("test", "")).strip():
            raise ValidationError("each hypothesis needs test: the action you will run whose recorded observation "
                                  "decides it")
        k = self.knowledge
        if h.get("hypothesis_id"):
            c, kh = k._hyp(str(h["hypothesis_id"]))
            if kh.get("archived"):
                raise ValidationError(f"{kh['id']} is archived")
            return k._hview(kh, ids=0)
        if not all(h.get(f) for f in ("mechanism", "when", "expect")) or not (h.get("concept_id") or h.get("concept")):
            raise ValidationError("each hypothesis: hypothesis_id of an existing one, or concept_id (or concept = a new "
                                  "quantity) + mechanism + when + expect (a prediction the observations can check)")
        cid = h.get("concept_id")
        if not cid:
            cid = k.define_concept(str(h["concept"]))["id"]       # refused if an existing concept covers it
        return k.propose(cid, h["mechanism"], h["when"], h["expect"])

    # ------------------------------------------------------------ views
    def _failure(self, failure_id: str | None) -> dict:
        if failure_id:
            m = re.match(r"\s*(F\d+)\b", str(failure_id))      # STATE shows "F3 go_to_position:error"
            failure_id = m.group(1) if m else failure_id
            f = next((x for x in self.failures if x["id"] == failure_id), None)
            if f is None:
                raise ValidationError(f"no failure {failure_id!r} (known: {[x['id'] for x in self.failures]})")
            return f
        open_ = [f for f in self.failures if not f["diagnosed"]]
        if open_:
            return open_[-1]
        if self.failures:
            return self.failures[-1]
        raise ValidationError("no failure recorded in this episode to diagnose")

    def _untested(self) -> list[dict]:
        return [h for h in self.hypotheses if h["verdict"] is None]

    def state(self) -> dict | None:
        """Compact view for STATE (only while something is open)."""
        open_ = [f"{f['id']} {f['what']}:{f['code']}" for f in self.failures if not f["diagnosed"]][-self.max_open:]
        untested = [f"{h['id']}({h['failure']}) {h['cause'][:70]}" for h in self._untested()][-self.max_open:]
        if not open_ and not untested:
            return None
        return {"undiagnosed_failures": open_, "untested_hypotheses": untested}

    def summary(self) -> dict:
        per = []
        for f in self.failures:
            hs = [h for h in self.hypotheses if h["failure"] == f["id"]]
            confirmed = [h for h in hs if h["verdict"] == "confirmed"]
            per.append({"failure": f["id"], "what": f["what"], "code": f["code"], "why": f.get("why"),
                        "cause": confirmed[0]["cause"] if confirmed else None,
                        "fix": confirmed[0]["fix"] if confirmed else None,
                        "hypotheses": [f"{h['id']} {h['verdict'] or 'untested'}: {h['cause'][:80]}" for h in hs]})
        return {"failures": len(self.failures), "diagnosed": sum(f["diagnosed"] for f in self.failures),
                "confirmed_causes": sum(1 for h in self.hypotheses if h["verdict"] == "confirmed"),
                "untested": len(self._untested()), "per_failure": per}

    def report_text(self) -> str:
        s = self.summary()
        if not s["failures"]:
            return ""
        lines = [f"{s['failures']} failures · {s['diagnosed']} diagnosed · {s['confirmed_causes']} causes confirmed "
                 f"· {s['untested']} hypotheses untested"]
        for p in s["per_failure"]:
            lines.append(f"  {p['failure']} {p['what']} {p['code']}: "
                         + (f"cause = {p['cause']}" + (f" → fix: {p['fix']}" if p["fix"] else "") if p["cause"]
                            else ("why: " + p["why"] if p["why"] else "NOT diagnosed")))
        return "\n".join(lines)


def _brief(v: dict) -> dict:
    return {k: v.get(k) for k in ("id", "status", "supports", "contradicts", "predicted", "when", "expect")
            if k in v} | ({"existing": True} if v.get("existing") else {})


def specs(inv: Investigation):
    from actions.executor import ActionSpec
    from actions.registry import _obj

    if inv.knowledge is not None:
        cond = {"type": "object", "description": "variable -> value, [lo, hi] or [one, of, these] "
                                                 "(names as in query_observations)"}
        hyp = {"type": "object", "properties": {
            "hypothesis_id": {"type": "string", "description": "an existing knowledge hypothesis (C#.n)"},
            "concept_id": {"type": "string"}, "concept": {"type": "string", "description": "a new quantity"},
            "mechanism": {"type": "string"}, "when": cond, "expect": cond,
            "test": {"type": "string", "description": "the action you will run whose observation decides it"}},
            "required": ["test"]}
        doc = ("After a failure (result shows failure_id): say WHY it failed from the evidence and give 1–3 "
               "hypotheses about the cause. Each one is a knowledge hypothesis: the id of an existing one, or "
               "concept_id (or concept = a new quantity) + mechanism + a prediction (when -> expect) — it is "
               "stored and checked against every recorded observation, past and future (the result shows how it "
               "fits the past ones) — plus the test you will run. Required only when the same failure repeats; "
               "may be sent in the same turn as the first test.")
    else:
        hyp = {"type": "object", "properties": {"cause": {"type": "string"}, "test": {"type": "string"},
                                                "expect": {"type": "string"}},
               "required": ["cause", "test", "expect"]}
        doc = ("After a failure (result shows failure_id): say WHY it failed from the evidence and give 1–3 "
               "hypotheses about the cause, each with a cheap TEST you will run (look, a "
               "small move, one changed parameter …) and what you EXPECT to see if it is true. Required only "
               "when the same failure repeats; may be sent in the same turn as the first test.")
    return [
        ActionSpec("diagnose", doc,
                   inv.diagnose, _obj({"why": {"type": "string"},
                                       "hypotheses": {"type": "array", "items": hyp, "minItems": 1, "maxItems": 3},
                                       "failure_id": {"type": "string"}}, ["why", "hypotheses"]),
                   read_only=True),
        ActionSpec("conclude",
                   "Record a hypothesis test: what you observed and the verdict (confirmed / rejected / "
                   "inconclusive), plus the fix for a confirmed cause. Kept for this episode and its review; a "
                   "knowledge hypothesis' lasting status is computed from the observations (shown in the result).",
                   inv.conclude, _obj({"hypothesis_id": {"type": "string"}, "observed": {"type": "string"},
                                       "verdict": {"type": "string", "enum": list(VERDICTS)},
                                       "fix": {"type": "string"}}, ["hypothesis_id", "observed", "verdict"]),
                   read_only=True),
    ]
