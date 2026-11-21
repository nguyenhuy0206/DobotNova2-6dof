"""Append-only private verdict storage; claims cannot occupy verdict fields."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from common.paths import DATA_DIR
from judge.records import Verdict

LEDGER_FILE = DATA_DIR / "judge" / "ledger.jsonl"


class JudgeLedger:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or LEDGER_FILE)

    def append(self, verdict: Verdict, claim: dict | None = None) -> None:
        row = verdict.private_dict()
        if verdict.success_credit and (verdict.status != "passed" or verdict.provisional or verdict.event != "episode_end"):
            raise ValueError("only a final non-provisional pass earns credit")
        if claim is not None:
            row["claim"] = claim
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lock read/check/append together; incomplete or duplicate entries fail closed.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            content = f.read()
            if content and not content.endswith("\n"):
                raise ValueError("incomplete ledger entry")
            rows = [json.loads(line) for line in content.splitlines()]
            if any(r["evaluation_id"] == verdict.evaluation_id for r in rows):
                raise ValueError("evaluation already recorded")
            if any(r["episode_id"] == verdict.episode_id and r["event"] == "episode_end" for r in rows):
                raise ValueError("episode already finalized")
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return [json.loads(line) for line in f]

    def outcome(self, episode_id: str) -> dict:
        rows = [r for r in self.rows() if r["episode_id"] == episode_id and r["event"] == "episode_end"]
        if len(rows) != 1:
            return {"status": "inconclusive", "success_credit": False, "evaluation_id": None}
        r = rows[0]
        return {"status": r["status"], "success_credit": bool(r["success_credit"] and not r["provisional"]),
                "evaluation_id": r["evaluation_id"]}

    def metrics(self, episode_id: str) -> dict:
        attempts = sum(r["event"] == "before_close" for r in self.rows() if r["episode_id"] == episode_id)
        return {**self.outcome(episode_id), "grasp_attempts": attempts}
