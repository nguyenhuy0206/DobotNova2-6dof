"""Load only operator configuration, outside the agent tool namespace."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPEC = Path(__file__).parent / "config" / "pick_place_cube_v0.json"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verifier_hash() -> str:
    root = Path(__file__).parent
    return digest([(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(root.rglob("*.py"))])


@dataclass(frozen=True)
class TaskSpec:
    task: str
    object_id: str
    object_color: str | None
    destination: tuple[tuple[float, float], ...]
    finger_reference: tuple[float, float, float] | None
    calibration_hash: str | None
    validated_predicates: tuple[str, ...]
    deadline_s: float
    max_age_ns: int
    max_skew_ns: int
    min_lift_mm: float
    alignment_px: float
    alignment_deg: float
    table_tolerance_mm: float
    persistence_ns: int
    task_hash: str


def load_spec(path: Path = DEFAULT_SPEC) -> TaskSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"task", "object_id", "object_color", "destination", "finger_reference", "calibration_hash",
                "validated_predicates", "deadline_s", "max_age_ns", "max_skew_ns", "min_lift_mm",
                "alignment_px", "alignment_deg", "table_tolerance_mm", "persistence_ns",
                "initial", "required_sequence", "unknown"}
    if set(raw) != expected or raw["task"] != "pick_place_cube_v0":
        raise ValueError("unsupported task spec fields")
    if (raw["initial"] != ["object_identifiable", "object_outside_destination"]
            or raw["required_sequence"] != ["held_after_lift", "placed_on_table"]
            or raw["unknown"] != "no_credit" or not raw["object_id"]):
        raise ValueError("unsupported task semantics")
    names = {"grasp_aligned", "held_after_lift", "placed_on_table"}
    if not set(raw["validated_predicates"]) <= names:
        raise ValueError("unknown predicate")
    numeric = ("deadline_s", "max_age_ns", "max_skew_ns", "min_lift_mm", "alignment_px",
               "alignment_deg", "table_tolerance_mm", "persistence_ns")
    if any(type(raw[k]) not in (int, float) or not math.isfinite(raw[k]) or raw[k] <= 0 for k in numeric):
        raise ValueError("limits must be finite and positive")
    polygon = tuple(tuple(p) for p in raw["destination"])
    if polygon:
        if len(polygon) < 3 or any(len(p) != 2 or any(not math.isfinite(v) for v in p) for p in polygon):
            raise ValueError("invalid destination polygon")
        # v0 accepts strictly convex operator polygons; corners/edges are uncertainty boundaries.
        turns = [(polygon[(i + 1) % len(polygon)][0] - p[0]) *
                 (polygon[(i + 2) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][1]) -
                 (polygon[(i + 1) % len(polygon)][1] - p[1]) *
                 (polygon[(i + 2) % len(polygon)][0] - polygon[(i + 1) % len(polygon)][0])
                 for i, p in enumerate(polygon)]
        if not (all(v > 0 for v in turns) or all(v < 0 for v in turns)):
            raise ValueError("destination must be strictly convex")
    ref = raw["finger_reference"]
    if ref is not None and (len(ref) != 3 or any(not math.isfinite(v) for v in ref)):
        raise ValueError("invalid finger reference")
    values = {k: raw[k] for k in expected - {"initial", "required_sequence", "unknown"}}
    values.update(destination=polygon, finger_reference=tuple(ref) if ref is not None else None,
                  validated_predicates=tuple(raw["validated_predicates"]), task_hash=digest(raw))
    return TaskSpec(**values)
