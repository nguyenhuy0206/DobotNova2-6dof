"""Astra's lab notebook: short lessons that carry over between runs.

data/memory/astra_notes.jsonl, one {"t", "topic", "text"} per line. The newest
notes are put into the instructions of the next run, so what Astra learned
("+dx in base moves the gripper LEFT in the env image", "box top is at z≈…")
isn't rediscovered every time. Skills (memory/skills.py) hold *procedures*;
notes hold *facts*.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from common.errors import ValidationError
from common.paths import MEMORY_DIR

NOTES_FILE = MEMORY_DIR / "astra_notes.jsonl"


class NotesStore:
    def __init__(self, path: Path = NOTES_FILE, max_len: int = 800):
        self.path, self.max_len = path, max_len

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def add(self, text: str, topic: str = "general") -> dict:
        text = str(text).strip()
        if len(text) < 3:
            raise ValidationError("note is empty")
        if len(text) > self.max_len:                     # keep it, but short: notes go into every prompt
            text = text[:self.max_len - 1].rstrip() + "…"
        note = {"t": time.strftime("%Y-%m-%d %H:%M"), "topic": str(topic)[:40] or "general", "text": text}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
        return {"saved": note, "total_notes": len(self.all())}

    def read(self, topic: str | None = None, limit: int = 30) -> dict:
        notes = [n for n in self.all() if topic in (None, "", n.get("topic"))]
        return {"notes": notes[-limit:], "total": len(notes)}

    def digest(self, max_chars: int = 1800, trials: int = 3) -> str:
        """Compact notes for the bounded context: the newest note of every topic, plus the
        last ``trials`` grasp_trials lines (oldest dropped first if over max_chars).
        Nothing is deleted from the file."""
        notes = self.all()
        latest: dict[str, dict] = {}
        for n in notes:
            if n.get("topic") != "grasp_trials":
                latest[n.get("topic", "general")] = n
        picked = sorted(latest.values(), key=lambda n: n["t"])
        picked += [n for n in notes if n.get("topic") == "grasp_trials"][-trials:]
        lines = [f"- [{n['t']} · {n['topic']}] {n['text'][:400]}" for n in picked]
        while lines and sum(len(x) + 1 for x in lines) > max_chars:
            lines.pop(0)
        return "\n".join(lines)

    def recent_text(self, limit: int = 30) -> str:
        return "\n".join(f"- [{n['t']} · {n['topic']}] {n['text']}" for n in self.all()[-limit:])
