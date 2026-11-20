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


def driver_of(meta: dict | None) -> str:
    """Who drove an episode, from its start meta: "astra" (the API model) or "qwen" (the local student).
    Episodes before 19 Nov 17:40 carry no "driver"; theirs is read from the model name."""
    meta = meta or {}
    if meta.get("driver"):
        return str(meta["driver"])
    return "qwen" if str(meta.get("model", "")).startswith("qwen-local") else "astra"


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

    def save_image(self, episode_id: str, name: str, jpeg_b64: str) -> Path:
        """Store a camera image the agent saw, in data/episodes/<id>/."""
        import base64
        folder = self.root / episode_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_bytes(base64.b64decode(jpeg_b64))
        return path

    def load(self, episode_id: str) -> list[dict]:
        path = self.root / f"{episode_id}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def driver(self, episode_id: str) -> str:
        path = self.root / f"{episode_id}.jsonl"
        try:
            with path.open(encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
        except (OSError, ValueError):
            return "astra"
        return driver_of(first.get("meta"))

    def list(self, driver: str | None = "astra") -> list[str]:
        """Episode ids (YYYYMMDD-HHMMSS-xxxxxx), oldest first; index.jsonl etc. are not episodes.
        driver: only the episodes that driver ran (default: Astra's: Qwen's runs are not Astra's
        experience); None = every episode."""
        ids = sorted(p.stem for p in self.root.glob("*.jsonl") if p.stem[:8].isdigit())
        return ids if driver is None else [i for i in ids if self.driver(i) == driver]
