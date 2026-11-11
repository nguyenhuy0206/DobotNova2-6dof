"""Skill library: reusable, named action sequences the agent can recall.

Retrieval is plain keyword overlap for now. Swap ``search`` for an embedding
index later without changing callers.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.paths import MEMORY_DIR

SKILLS_FILE = MEMORY_DIR / "skills.json"


@dataclass
class Skill:
    name: str
    description: str
    steps: list[dict] = field(default_factory=list)   # [{"tool": ..., "args": {...}}]
    tags: list[str] = field(default_factory=list)
    successes: int = 0
    failures: int = 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SkillLibrary:
    def __init__(self, path: Path = SKILLS_FILE):
        self.path = path

    def all(self) -> list[Skill]:
        if not self.path.exists():
            return []
        return [Skill(**s) for s in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self, skill: Skill) -> None:
        skills = [s for s in self.all() if s.name != skill.name] + [skill]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(s) for s in skills], ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    def search(self, query: str, k: int = 3) -> list[Skill]:
        q = _tokens(query)
        scored = [(len(q & _tokens(" ".join([s.name, s.description, *s.tags]))), s) for s in self.all()]
        return [s for score, s in sorted(scored, key=lambda x: -x[0]) if score > 0][:k]
