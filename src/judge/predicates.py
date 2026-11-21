"""Conservative world predicates; missing prerequisites never satisfy a goal."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from judge.records import Evidence, PredicateResult
from judge.spec import TaskSpec


def evidence_gate(evidence: tuple[Evidence, ...], spec: TaskSpec, now_ns: int,
                  synchronized: bool = True) -> str | None:
    if not evidence:
        return "missing_evidence"
    seen, streams, times = set(), {}, []
    for e in evidence:
        if e.legacy:
            return "legacy_evidence"
        if e.evidence_id in seen:
            return "duplicate_evidence"
        seen.add(e.evidence_id)
        if e.object_id != spec.object_id or not e.identity_verified:
            return "identity_unverified"
        if not spec.calibration_hash or not e.calibration_applicable or e.calibration_hash != spec.calibration_hash:
            return "calibration_inapplicable"
        if (type(e.captured_ns) is not int or type(e.sequence_id) is not int or e.sequence_id < 0
                or not e.stream_id):
            return "missing_capture_metadata"
        if not 0 <= now_ns - e.captured_ns <= spec.max_age_ns:
            return "stale_or_future_evidence"
        previous = streams.get((e.source, e.stream_id))
        if previous and (e.sequence_id <= previous.sequence_id or e.captured_ns <= previous.captured_ns):
            return "evidence_out_of_order"
        streams[e.source, e.stream_id] = e
        if not e.file or not e.sha256:
            return "missing_reference"
        path = Path(e.file)
        # Evidence is never reconstructed from Astra's memory.
        if any(p in {"memory", "astra_skills"} for p in path.resolve().parts):
            return "untrusted_reference"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != e.sha256:
            return "reference_mismatch"
        times.append(e.captured_ns)
    if synchronized and max(times) - min(times) > spec.max_skew_ns:
        return "evidence_unsynchronized"
    return None


def _result(name, value, evidence, reason, **measurements):
    return PredicateResult(name, value, tuple(measurements.items()), tuple(e.evidence_id for e in evidence),
                           reason, advisory=name == "grasp_aligned")


def _finite(*values):
    return all(type(v) in (int, float) and math.isfinite(v) for v in values)


def grasp_aligned(wrist: Evidence | None, spec: TaskSpec, now_ns: int) -> PredicateResult:
    ev = (wrist,) if wrist else ()
    o = wrist.observation if wrist else None
    m = {"du_px": None, "dv_px": None, "edge_deg": o.edge_deg if o else None,
         "visibility": o.visibility if o else "unknown"}
    if o and o.pixel and spec.finger_reference:
        u, v, angle = spec.finger_reference
        m.update(du_px=o.pixel[0] - u, dv_px=o.pixel[1] - v,
                 edge_deg=((o.edge_deg - angle + 45) % 90 - 45) if o.edge_deg is not None else None)
    reason = evidence_gate(ev, spec, now_ns)
    if reason:
        return _result("grasp_aligned", "unknown", ev, reason, **m)
    if wrist.source != "wrist" or not spec.finger_reference:
        return _result("grasp_aligned", "unknown", ev, "missing_finger_reference", **m)
    if o.visibility != "full" or not _finite(m["du_px"], m["dv_px"], m["edge_deg"], o.uncertainty_px,
                                            o.edge_uncertainty_deg):
        return _result("grasp_aligned", "unknown", ev, "visibility_or_uncertainty_missing", **m)
    if o.uncertainty_px < 0 or o.edge_uncertainty_deg < 0:
        return _result("grasp_aligned", "unknown", ev, "invalid_uncertainty", **m)
    inside = (abs(m["du_px"]) + o.uncertainty_px < spec.alignment_px
              and abs(m["dv_px"]) + o.uncertainty_px < spec.alignment_px
              and abs(m["edge_deg"]) + o.edge_uncertainty_deg < spec.alignment_deg)
    outside = (abs(m["du_px"]) - o.uncertainty_px > spec.alignment_px
               or abs(m["dv_px"]) - o.uncertainty_px > spec.alignment_px
               or abs(m["edge_deg"]) - o.edge_uncertainty_deg > spec.alignment_deg)
    value = "true" if inside else "false" if outside else "unknown"
    return _result("grasp_aligned", value, ev, "advisory_only", **m)


def held_after_lift(baseline: Evidence | None, robot: Evidence | None, wrist: Evidence | None,
                    spec: TaskSpec, now_ns: int) -> PredicateResult:
    ev = tuple(e for e in (baseline, robot, wrist) if e is not None)
    m = {"lift_mm": None, "holding": robot.observation.holding if robot else None,
         "between_fingers": wrist.observation.between_fingers if wrist else None}
    if baseline and robot and _finite(baseline.observation.z_mm, robot.observation.z_mm):
        m["lift_mm"] = robot.observation.z_mm - baseline.observation.z_mm
    if not baseline or not robot or not wrist:
        return _result("held_after_lift", "unknown", ev, "missing_evidence", **m)
    reason = (evidence_gate((baseline,), spec, baseline.captured_ns or now_ns)
              or evidence_gate((robot, wrist), spec, now_ns))
    if reason:
        return _result("held_after_lift", "unknown", ev, reason, **m)
    if (baseline.source != "robot" or robot.source != "robot" or wrist.source != "wrist"
            or baseline.stream_id != robot.stream_id or baseline.sequence_id >= robot.sequence_id
            or baseline.captured_ns >= min(robot.captured_ns, wrist.captured_ns)):
        return _result("held_after_lift", "unknown", ev, "invalid_lift_order", **m)
    o, b, w = robot.observation, baseline.observation, wrist.observation
    if w.visibility != "full":
        return _result("held_after_lift", "unknown", ev, "occluded", **m)
    if o.dropped is True or o.holding is False or w.between_fingers is False:
        return _result("held_after_lift", "false", ev, "not_held", **m)
    if not _finite(m["lift_mm"], o.uncertainty_mm, b.uncertainty_mm) or o.uncertainty_mm < 0 or b.uncertainty_mm < 0:
        return _result("held_after_lift", "unknown", ev, "missing_lift_uncertainty", **m)
    lo = m["lift_mm"] - o.uncertainty_mm - b.uncertainty_mm
    hi = m["lift_mm"] + o.uncertainty_mm + b.uncertainty_mm
    if hi < spec.min_lift_mm:
        return _result("held_after_lift", "false", ev, "not_lifted", **m)
    value = "true" if lo > spec.min_lift_mm and o.holding is True and w.between_fingers is True else "unknown"
    return _result("held_after_lift", value, ev, "held" if value == "true" else "ambiguous_lift", **m)


def polygon_relation(e: Evidence, spec: TaskSpec) -> str:
    """Classify the entire uncertainty box against a convex image polygon."""
    o, polygon = e.observation, spec.destination
    if not polygon or o.bbox is None or not _finite(*o.bbox, o.uncertainty_px) or o.uncertainty_px < 0:
        return "unknown"
    x, y, w, h = o.bbox
    if w <= 0 or h <= 0:
        return "unknown"
    r = o.uncertainty_px
    corners = [(x - r, y - r), (x + w + r, y - r), (x + w + r, y + h + r), (x - r, y + h + r)]
    # Separating axis test also catches a box spanning a polygon or a concavity boundary.
    axes = [(1, 0), (0, 1)] + [(b[1] - a[1], a[0] - b[0]) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    for ax, ay in axes:
        ps, cs = [ax * x + ay * y for x, y in polygon], [ax * x + ay * y for x, y in corners]
        if max(cs) < min(ps) or min(cs) > max(ps):
            return "false"
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1]))
    sign = 1 if area > 0 else -1
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if any(sign * ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) <= 0 for x, y in corners):
            return "unknown"
    return "true"


def placed_on_table(robot: Evidence | None, views: tuple[Evidence, ...], spec: TaskSpec,
                    now_ns: int) -> PredicateResult:
    ev = ((robot,) if robot else ()) + views
    m = {"views": len(views), "support_gap_mm": views[-1].observation.support_gap_mm if views else None}
    if robot is None or len(views) < 2:
        return _result("placed_on_table", "unknown", ev, "missing_persistence_evidence", **m)
    reason = evidence_gate(views, spec, now_ns, synchronized=False) or evidence_gate((robot, views[-1]), spec, now_ns)
    if reason:
        return _result("placed_on_table", "unknown", ev, reason, **m)
    if robot.source != "robot" or any(e.source != "env" for e in views) or len({e.stream_id for e in views}) != 1:
        return _result("placed_on_table", "unknown", ev, "wrong_source", **m)
    if any(e.observation.visibility != "full" for e in views):
        return _result("placed_on_table", "unknown", ev, "occluded", **m)
    r = robot.observation
    if r.dropped is True:
        return _result("placed_on_table", "false", ev, "dropped", **m)
    if r.opened is not True or r.retracted is not True:
        return _result("placed_on_table", "unknown", ev, "release_or_retraction_unverified", **m)
    opened_ns = r.opened_since_ns if r.opened_since_ns is not None else robot.captured_ns
    retracted_ns = r.retracted_since_ns if r.retracted_since_ns is not None else robot.captured_ns
    if max(opened_ns, retracted_ns) > views[0].captured_ns:
        return _result("placed_on_table", "unknown", ev, "persistence_before_retraction_unverified", **m)
    relations = [polygon_relation(e, spec) for e in views]
    if "false" in relations:
        return _result("placed_on_table", "false", ev, "outside_destination", **m)
    if "unknown" in relations:
        return _result("placed_on_table", "unknown", ev, "destination_boundary_or_missing", **m)
    for e in views:
        o = e.observation
        if not _finite(o.support_gap_mm, o.uncertainty_mm) or o.uncertainty_mm < 0:
            return _result("placed_on_table", "unknown", ev, "table_support_unverified", **m)
        lo, hi = o.support_gap_mm - o.uncertainty_mm, o.support_gap_mm + o.uncertainty_mm
        if lo > spec.table_tolerance_mm or hi < -spec.table_tolerance_mm:
            return _result("placed_on_table", "false", ev, "not_supported_on_table", **m)
        if lo <= -spec.table_tolerance_mm or hi >= spec.table_tolerance_mm:
            return _result("placed_on_table", "unknown", ev, "support_boundary", **m)
    if views[-1].captured_ns - views[0].captured_ns < spec.persistence_ns:
        return _result("placed_on_table", "unknown", ev, "insufficient_persistence", **m)
    return _result("placed_on_table", "true", ev, "placed", **m)
