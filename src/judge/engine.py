"""Sequence evaluation and provenance; only finalized ledger outcomes earn credit."""
from __future__ import annotations

import uuid

from judge.ledger import JudgeLedger
from judge.predicates import evidence_gate, grasp_aligned, held_after_lift, placed_on_table, polygon_relation
from judge.records import Evidence, PredicateResult, Verdict
from judge.spec import TaskSpec, digest, verifier_hash


class JudgeSession:
    def __init__(self, episode_id: str, spec: TaskSpec, ledger: JudgeLedger, started_ns: int):
        self.episode_id, self.spec, self.ledger, self.started_ns = episode_id, spec, ledger, started_ns
        self.attempt = 0
        self.baseline: Evidence | None = None
        self.initial: Evidence | None = None
        self.held: PredicateResult | None = None
        self.held_ns: int | None = None
        self.released_ns: int | None = None
        self.evidence: dict[str, Evidence] = {}
        self.invariant_failure = False
        self.skill_dependency_hash: str | None = None
        self.version_hash = verifier_hash()

    def start_attempt(self, baseline: Evidence | None):
        self.attempt += 1
        self.baseline, self.held, self.held_ns, self.released_ns = baseline, None, None, None

    def evaluate(self, event: str, now_ns: int, evidence: tuple[Evidence, ...],
                 claim: dict | None = None) -> Verdict:
        self.evidence.update((e.evidence_id, e) for e in evidence)
        if self.baseline:
            self.evidence[self.baseline.evidence_id] = self.baseline
        robot = next((e for e in reversed(evidence) if e.source == "robot"), None)
        wrist = next((e for e in reversed(evidence) if e.source == "wrist"), None)
        views = tuple(e for e in evidence if e.source == "env")
        if (robot and robot.observation.dropped is True
                and evidence_gate((robot,), self.spec, now_ns) is None):
            self.invariant_failure = True
        if (self.held is not None and self.released_ns is None and robot
                and robot.observation.holding is False and evidence_gate((robot,), self.spec, now_ns) is None):
            self.invariant_failure = True
        if event == "episode_start" and views:
            self.initial = views[0]
        if event == "after_open" and self.released_ns is None:
            self.released_ns = now_ns
        aligned = grasp_aligned(wrist, self.spec, now_ns)
        held = held_after_lift(self.baseline, robot, wrist, self.spec, now_ns)
        if held.value == "true" and self.released_ns is None:
            self.held, self.held_ns = held, now_ns
        placed = placed_on_table(robot, views, self.spec, now_ns)
        results = (aligned, self.held or held, placed)
        reasons = []
        initial_reason = (evidence_gate((self.initial,), self.spec, self.initial.captured_ns or now_ns)
                          if self.initial else "missing_initial_evidence")
        if initial_reason:
            reasons.append(initial_reason)
        elif self.initial.observation.visibility != "full":
            reasons.append("initial_visibility_unverified")
        elif polygon_relation(self.initial, self.spec) != "false":
            reasons.append("initial_outside_destination_unverified")
        if not {"held_after_lift", "placed_on_table"} <= set(self.spec.validated_predicates):
            reasons.append("predicates_not_validated")
        if now_ns < self.started_ns or now_ns - self.started_ns > self.spec.deadline_s * 1e9:
            reasons.append("deadline_exceeded")
        if (self.held_ns is None or self.released_ns is None or self.released_ns <= self.held_ns
                or not views or views[0].captured_ns is None or views[0].captured_ns <= self.released_ns):
            reasons.append("required_sequence_unverified")
        provisional = any(e.legacy for e in self.evidence.values())
        if provisional:
            reasons.append("legacy_provisional")
        if self.invariant_failure:
            reasons.append("drop_invariant_violated")
        satisfied = not reasons and results[1].value == "true" and placed.value == "true"
        final = event == "episode_end"
        status = "passed" if satisfied else "failed" if self.invariant_failure or (
            final and placed.value == "false" and "deadline_exceeded" not in reasons
            and not provisional) else "inconclusive"
        if not satisfied:
            reasons.extend(r.reason for r in results[1:] if r.value != "true")
        refs = tuple(self.evidence.values())
        verdict = Verdict(uuid.uuid4().hex, self.episode_id, f"{self.episode_id}:{self.attempt}", event,
                          self.spec.task_hash, self.version_hash, self.spec.calibration_hash,
                          self.skill_dependency_hash, digest([e.__dict__ | {"observation": e.observation.__dict__}
                                                            for e in refs]),
                          results, refs, status, bool(final and satisfied and not provisional),
                          tuple(dict.fromkeys(reasons)), provisional)
        self.ledger.append(verdict, claim)
        return verdict
