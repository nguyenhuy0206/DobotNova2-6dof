"""Private harness integration, including nested primitives and exceptional exits."""
from __future__ import annotations

import time

from judge.engine import JudgeSession
from judge.ledger import JudgeLedger
from judge.predicates import grasp_aligned
from judge.records import public_feedback
from judge.spec import load_spec


class HarnessJudge:
    def __init__(self, capture=None, ledger=None, spec=None):
        self.spec = spec or load_spec()
        self.ledger = ledger or JudgeLedger()
        self.capture = capture
        self.session = None
        self.views = []

    def _acquire(self):
        return self.capture.acquire() if self.capture else ()

    def _evaluate(self, event):
        evidence = self._acquire()
        now = time.monotonic_ns()
        if self.session.released_ns is not None:
            self.views.extend(e for e in evidence if e.source == "env")
            self.views = [e for e in self.views if e.captured_ns is not None
                          and now - e.captured_ns <= self.spec.max_age_ns]
            evidence = tuple(e for e in evidence if e.source != "env") + tuple(self.views)
        self.session.evaluate(event, now, evidence)
        return evidence

    def start(self, episode):
        self.views = []
        self.session = JudgeSession(episode, self.spec, self.ledger, time.monotonic_ns())
        self._evaluate("episode_start")

    def before(self, name, args):
        if name not in {"close_gripper", "set_gripper"}:
            return {}
        evidence = self._acquire()
        robot = next((e for e in evidence if e.source == "robot"), None)
        wrist = next((e for e in evidence if e.source == "wrist"), None)
        self.session.start_attempt(robot)
        self.views = []
        self.session.evaluate("before_close", time.monotonic_ns(), evidence)
        return public_feedback(grasp_aligned(wrist, self.spec, time.monotonic_ns()))

    def after(self, name, args):
        event = "after_open" if name == "open_gripper" else "after_close" if name == "close_gripper" else "after_action"
        if event == "after_open":
            self.session.released_ns = time.monotonic_ns()
            self.views = []
        evidence = self._evaluate(event)
        if name == "look" and args.get("camera") == "wrist":
            wrist = next((e for e in evidence if e.source == "wrist"), None)
            return public_feedback(grasp_aligned(wrist, self.spec, time.monotonic_ns()))

    def finish(self):
        self._evaluate("episode_end")
        return self.ledger.metrics(self.session.episode_id)
