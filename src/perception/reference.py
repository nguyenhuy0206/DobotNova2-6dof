"""Operator reference pictures for a task outcome, and comparison of a fresh picture with them.

24 Nov (operator, threading): the operator saves wrist pictures of the finished state ("success") and of
the unfinished state ("fail") at the aligned start pose; the agent compares the picture it has now with
them. The numbers are a similarity cue, not a verdict: the composite picture lets the agent look itself.

    data/references/<task>/<label>/<timestamp>.jpg      label: success | fail
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from common.paths import DATA_DIR

REFERENCES_DIR = DATA_DIR / "references"
LABELS = ("success", "fail")
SIZE = (160, 120)                 # comparison resolution: small, blurred -> tolerant of noise


def save_reference(image_bgr: np.ndarray, task: str, label: str, root: Path = REFERENCES_DIR) -> Path:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    folder = Path(root) / task / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    cv2.imwrite(str(path), image_bgr)
    return path


def load_references(task: str, root: Path = REFERENCES_DIR) -> dict[str, list[tuple[Path, np.ndarray]]]:
    out = {}
    for label in LABELS:
        files = sorted((Path(root) / task / label).glob("*.jpg"))
        out[label] = [(f, img) for f in files if (img := cv2.imread(str(f))) is not None]
    return out


def _prep(image_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Normalised cross-correlation of the two pictures (small, blurred, grey): 1 = identical, ~0 unrelated."""
    a, b = _prep(a_bgr), _prep(b_bgr)
    return float((a * b).mean())


def compare(now_bgr: np.ndarray, refs: dict[str, list[tuple[Path, np.ndarray]]]) -> dict:
    """Best similarity per label, which label is closer, and a side-by-side picture (now | best success |
    best fail), each 320 px wide, nothing drawn on them."""
    missing = [label for label in LABELS if not refs.get(label)]
    if missing:
        raise ValueError(f"no reference pictures for {missing}: the operator saves them with "
                         "scripts/dev/capture_reference.py")
    best = {}
    for label in LABELS:
        scored = [(similarity(now_bgr, img), path, img) for path, img in refs[label]]
        best[label] = max(scored, key=lambda s: s[0])
    tiles = [now_bgr] + [best[label][2] for label in LABELS]
    w = 320
    tiles = [cv2.resize(t, (w, int(t.shape[0] * w / t.shape[1]))) for t in tiles]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    scores = {label: round(best[label][0], 3) for label in LABELS}
    closer = max(scores, key=scores.get)
    return {"scores": scores, "closer_to": closer, "margin": round(abs(scores["success"] - scores["fail"]), 3),
            "references": {label: best[label][1].name for label in LABELS},
            "n_references": {label: len(refs[label]) for label in LABELS},
            "picture": np.hstack(tiles)}
