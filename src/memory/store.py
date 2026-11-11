"""JSON/JSONL stores under data/."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from common.paths import EPISODES_DIR, POSITIONS_FILE
from common.types import SavedPosition


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class PositionStore:
    """Named joint positions taught from the UI (was positions.json)."""

    def __init__(self, path: Path = POSITIONS_FILE):
        self.path = path

    def all(self) -> list[SavedPosition]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []

    def get(self, position_id: str) -> SavedPosition | None:
        return next((p for p in self.all() if p.get("id") == position_id), None)

    def find_by_name(self, name: str) -> SavedPosition | None:
        return next((p for p in self.all() if p.get("name") == name), None)

    def add(self, name: str, joints_deg: list[float], pose: list[float]) -> list[SavedPosition]:
        positions = self.all()
        positions.append({"id": str(int(time.time() * 1000)), "name": name,
                          "joints_deg": joints_deg, "pose": pose})
        _atomic_write_json(self.path, positions)
        return positions

    def delete(self, position_id: str) -> list[SavedPosition]:
        positions = [p for p in self.all() if p.get("id") != position_id]
        _atomic_write_json(self.path, positions)
        return positions


class EpisodeStore:
    """One JSONL file per episode: every agent/tool step is appended as a line.

    Episodes are the raw experience used later for skill extraction or
    dataset/VLA training.
    """

    def __init__(self, root: Path = EPISODES_DIR):
        self.root = root

    def start(self, task: str, meta: dict | None = None) -> str:
        episode_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.append(episode_id, {"event": "start", "task": task, "meta": meta or {}})
        return episode_id

    def append(self, episode_id: str, record: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"t": time.time(), **record}
        with (self.root / f"{episode_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def load(self, episode_id: str) -> list[dict]:
        path = self.root / f"{episode_id}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))
