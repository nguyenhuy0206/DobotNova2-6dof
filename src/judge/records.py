"""Immutable sensor records preserve provenance before state is combined."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

Truth = Literal["true", "false", "unknown"]
Scalar = float | int | str | bool | None


@dataclass(frozen=True)
class Observation:
    pixel: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    edge_deg: float | None = None
    edge_uncertainty_deg: float | None = None
    visibility: Literal["full", "partial", "occluded", "unknown"] = "unknown"
    uncertainty_px: float | None = None
    between_fingers: bool | None = None
    z_mm: float | None = None
    uncertainty_mm: float | None = None
    holding: bool | None = None
    opened: bool | None = None
    retracted: bool | None = None
    opened_since_ns: int | None = None
    retracted_since_ns: int | None = None
    # Signed separation of the cube bottom from the operator's table plane.
    support_gap_mm: float | None = None
    dropped: bool | None = None

    def __post_init__(self):
        for key in ("pixel", "bbox"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, tuple(value))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: Literal["wrist", "env", "robot"]
    captured_ns: int | None
    sequence_id: int | None
    stream_id: str | None
    file: str | None
    sha256: str | None
    object_id: str | None
    identity_verified: bool = False
    calibration_hash: str | None = None
    calibration_applicable: bool = False
    legacy: bool = False
    observation: Observation = Observation()


@dataclass(frozen=True)
class PredicateResult:
    predicate: str
    value: Truth
    measurements: tuple[tuple[str, Scalar], ...]
    evidence_ids: tuple[str, ...]
    reason: str
    advisory: bool = False


def public_feedback(result: PredicateResult) -> dict:
    """Only wrist measurements cross the boundary; no verdicts or configuration."""
    if result.predicate != "grasp_aligned":
        return {}
    allowed = {"du_px", "dv_px", "edge_deg", "visibility"}
    out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
           for k, v in result.measurements if k in allowed}
    # no finger reference / no cube: nothing measured, so nothing shown (not a row of nulls on every close)
    return out if out.get("du_px") is not None and out.get("dv_px") is not None else {}


@dataclass(frozen=True)
class Verdict:
    evaluation_id: str
    episode_id: str
    physical_attempt_id: str
    event: str
    task_hash: str
    verifier_hash: str
    calibration_hash: str | None
    skill_dependency_hash: str | None
    input_manifest_hash: str
    predicate_results: tuple[PredicateResult, ...]
    evidence_references: tuple[Evidence, ...]
    status: Literal["passed", "failed", "inconclusive"]
    success_credit: bool
    reason_codes: tuple[str, ...]
    provisional: bool = False

    def private_dict(self) -> dict:
        return asdict(self)
