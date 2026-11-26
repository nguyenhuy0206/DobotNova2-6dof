"""Independent verifier logic and harness wiring, with files and synthetic sensors only."""
import hashlib
import json
import runpy
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from actions.executor import ActionExecutor, ActionSpec
from agent.harness import AgentHarness
from agent.explore.learning import OutcomeRecorder, history
from common.config import AgentConfig, PerceptionConfig
from judge.engine import JudgeSession
from judge.harness import HarnessJudge
from judge.ledger import JudgeLedger
from judge.predicates import evidence_gate, grasp_aligned, held_after_lift, placed_on_table
from judge.records import Evidence, Observation, public_feedback
from judge.spec import DEFAULT_SPEC, load_spec
from memory.skills import Skill, SkillLibrary
from memory.store import EpisodeStore

S = 1_000_000_000


@pytest.fixture
def spec(tmp_path):
    raw = json.loads(DEFAULT_SPEC.read_text())
    raw.update(object_color="blue", destination=[[0, 0], [100, 0], [100, 100], [0, 100]],
               finger_reference=[50, 50, 0], calibration_hash="calibration",
               validated_predicates=["held_after_lift", "placed_on_table"])
    path = tmp_path / "task.json"
    path.write_text(json.dumps(raw))
    return load_spec(path)


@pytest.fixture
def evidence(tmp_path, spec):
    count = 0

    def make(source="wrist", ns=3*S, seq=None, **measurements):
        nonlocal count
        count += 1
        data = json.dumps({"count": count, "source": source}).encode()
        path = tmp_path / f"raw-{count}.json"
        path.write_bytes(data)
        defaults = dict(pixel=(50, 50), bbox=(40, 40, 20, 20), edge_deg=0, edge_uncertainty_deg=1,
                        visibility="full", uncertainty_px=1, uncertainty_mm=1, z_mm=50,
                        between_fingers=True, holding=True, opened=True, retracted=True,
                        support_gap_mm=0, opened_since_ns=2*S, retracted_since_ns=2*S)
        defaults.update(measurements)
        return Evidence(str(count), source, ns, seq if seq is not None else count, source + "-stream",
                        str(path), hashlib.sha256(data).hexdigest(), spec.object_id, True,
                        spec.calibration_hash, True, observation=Observation(**defaults))
    return make


def test_evidence_is_immutable(evidence):
    e = evidence(pixel=[2, 3])
    with pytest.raises(FrozenInstanceError):
        e.sequence_id = 3
    with pytest.raises(FrozenInstanceError):
        e.observation.pixel = (3, 4)
    assert e.observation.pixel == (2, 3)


@pytest.mark.parametrize("changes,reason", [
    ({"legacy": True}, "legacy_evidence"),
    ({"object_id": "other"}, "identity_unverified"),
    ({"identity_verified": False}, "identity_unverified"),
    ({"calibration_hash": "old"}, "calibration_inapplicable"),
    ({"calibration_applicable": False}, "calibration_inapplicable"),
    ({"captured_ns": None}, "missing_capture_metadata"),
    ({"sequence_id": None}, "missing_capture_metadata"),
    ({"stream_id": None}, "missing_capture_metadata"),
    ({"captured_ns": 0}, "stale_or_future_evidence"),
    ({"captured_ns": 4*S}, "stale_or_future_evidence"),
    ({"file": None}, "missing_reference"),
    ({"sha256": "wrong"}, "reference_mismatch"),
])
def test_gate_missing_prerequisites_are_unknown(spec, evidence, changes, reason):
    e = replace(evidence(), **changes)
    assert evidence_gate((e,), spec, 3*S) == reason
    result = grasp_aligned(e, spec, 3*S)
    assert result.value == "unknown" and result.reason == reason


def test_gate_order_sync_and_file_integrity(spec, evidence):
    a, b = evidence(ns=2*S, seq=2), evidence(ns=3*S, seq=1)
    assert evidence_gate((a, b), spec, 3*S) == "evidence_out_of_order"
    assert evidence_gate((b, b), spec, 3*S) == "duplicate_evidence"
    assert evidence_gate((evidence("robot", ns=2*S), evidence()), spec, 3*S) == "evidence_unsynchronized"
    e = evidence()
    Path(e.file).write_text("tampered")
    assert evidence_gate((e,), spec, 3*S) == "reference_mismatch"
    assert evidence_gate((), spec, 3*S) == "missing_evidence"


@pytest.mark.parametrize("pixel,angle,visibility,value", [
    ((50, 50), 0, "full", "true"), ((70, 50), 0, "full", "false"),
    ((59, 50), 0, "full", "unknown"), ((50, 50), 20, "full", "false"),
    ((50, 50), 0, "partial", "unknown"), ((50, 50), 0, "occluded", "unknown"),
])
def test_alignment_is_advisory(spec, evidence, pixel, angle, visibility, value):
    result = grasp_aligned(evidence(pixel=pixel, edge_deg=angle, visibility=visibility), spec, 3*S)
    assert result.value == value and result.advisory
    feedback = public_feedback(result)
    assert set(feedback) == {"du_px", "dv_px", "edge_deg", "visibility"}
    assert feedback["du_px"] == pixel[0] - 50
    assert public_feedback(replace(result, predicate="held_after_lift")) == {}


@pytest.mark.parametrize("changes,value", [
    ({}, "true"), ({"holding": False}, "false"), ({"dropped": True}, "false"),
    ({"z_mm": 10}, "false"), ({"z_mm": 30}, "unknown"), ({"uncertainty_mm": None}, "unknown"),
])
def test_held_requires_real_lift_and_gripper(spec, evidence, changes, value):
    baseline = evidence("robot", ns=S, seq=0, z_mm=0)
    robot = evidence("robot", **changes)
    assert held_after_lift(baseline, robot, evidence(), spec, 3*S).value == value


@pytest.mark.parametrize("changes,value", [
    ({"between_fingers": False}, "false"), ({"between_fingers": None}, "unknown"),
    ({"visibility": "occluded"}, "unknown"),
])
def test_held_needs_visible_cube_between_fingers(spec, evidence, changes, value):
    assert held_after_lift(evidence("robot", ns=S, seq=0, z_mm=0), evidence("robot"),
                           evidence(**changes), spec, 3*S).value == value


@pytest.mark.parametrize("kind", ["wrong_object", "stale", "missing"])
def test_held_hard_negatives_abstain(spec, evidence, kind):
    wrist = evidence()
    if kind == "wrong_object":
        wrist = replace(wrist, object_id="other")
    elif kind == "stale":
        wrist = replace(wrist, captured_ns=0)
    else:
        wrist = None
    assert held_after_lift(evidence("robot", ns=S, seq=0, z_mm=0), evidence("robot"),
                           wrist, spec, 3*S).value == "unknown"


def place_inputs(evidence, **changes):
    return evidence("robot"), (evidence("env", ns=2*S, **changes), evidence("env", ns=3*S, **changes))


@pytest.mark.parametrize("changes,value", [
    ({}, "true"), ({"support_gap_mm": 20}, "false"),  # hovering above the destination
    ({"bbox": (120, 40, 20, 20)}, "false"),
    ({"bbox": (0, 40, 20, 20)}, "unknown"),
    ({"support_gap_mm": None}, "unknown"),
    ({"support_gap_mm": 2}, "unknown"),
    ({"visibility": "occluded"}, "unknown"),
    ({"uncertainty_mm": -1}, "unknown"),
])
def test_placement_requires_support_full_containment_and_persistence(spec, evidence, changes, value):
    robot, views = place_inputs(evidence, **changes)
    assert placed_on_table(robot, views, spec, 3*S).value == value


@pytest.mark.parametrize("change,value", [({"opened": False}, "unknown"), ({"retracted": False}, "unknown"),
                                          ({"dropped": True}, "false")])
def test_placement_release_retraction_and_drop(spec, evidence, change, value):
    robot, views = place_inputs(evidence)
    robot = replace(robot, observation=replace(robot.observation, **change))
    assert placed_on_table(robot, views, spec, 3*S).value == value


@pytest.mark.parametrize("kind", ["wrong_object", "stale", "repeated", "one_view", "too_brief"])
def test_placement_hard_negatives(spec, evidence, kind):
    robot, views = place_inputs(evidence)
    if kind == "wrong_object":
        views = (views[0], replace(views[1], object_id="other"))
    elif kind == "stale":
        views = (replace(views[0], captured_ns=0), views[1])
    elif kind == "repeated":
        views = (views[0], replace(views[0], evidence_id="repeat", captured_ns=3*S))
    elif kind == "one_view":
        views = views[:1]
    else:
        views = (replace(views[0], captured_ns=3*S-1), views[1])
    assert placed_on_table(robot, views, spec, 3*S).value == "unknown"


def completed_session(spec, evidence, ledger, episode="episode"):
    session = JudgeSession(episode, spec, ledger, 0)
    session.evaluate("episode_start", S, (evidence("env", ns=S, bbox=(150, 40, 20, 20)),))
    session.start_attempt(evidence("robot", ns=S, seq=0, z_mm=0))
    session.evaluate("after_lift", 2*S, (evidence("robot", ns=2*S), evidence(ns=2*S)))
    session.evaluate("after_open", 2*S+1, ())
    robot = evidence("robot", ns=4*S)
    views = (evidence("env", ns=3*S), evidence("env", ns=4*S))
    return session, (robot, *views)


def test_task_sequence_and_ledger_only_credit(spec, evidence, tmp_path):
    ledger = JudgeLedger(tmp_path / "ledger.jsonl")
    session, final_evidence = completed_session(spec, evidence, ledger)
    before = ledger.path.read_bytes()
    final = session.evaluate("episode_end", 4*S, final_evidence, claim={"status": "failed", "success_credit": False})
    assert final.status == "passed" and final.success_credit
    assert ledger.path.read_bytes().startswith(before)
    assert ledger.outcome("episode")["success_credit"] is True
    row = ledger.rows()[-1]
    assert row["claim"]["status"] == "failed" and row["status"] == "passed"
    assert all(row[k] for k in ("task_hash", "verifier_hash", "input_manifest_hash", "calibration_hash"))
    with pytest.raises(ValueError, match="already"):
        ledger.append(final)
    with pytest.raises(ValueError, match="finalized"):
        ledger.append(replace(final, evaluation_id="different"))


@pytest.mark.parametrize("kind", ["advisory", "no_lift", "no_initial", "deadline", "legacy", "unvalidated", "new_attempt"])
def test_unknown_and_advisory_never_earn_credit(spec, evidence, tmp_path, kind):
    ledger = JudgeLedger(tmp_path / "ledger.jsonl")
    session, ev = completed_session(spec, evidence, ledger)
    if kind in {"advisory", "no_lift"}:
        session.held = session.held_ns = None
    if kind == "no_initial":
        session.initial = None
    if kind == "unvalidated":
        session.spec = replace(spec, validated_predicates=())
    if kind == "new_attempt":
        session.start_attempt(evidence("robot", z_mm=50))
    if kind == "legacy":
        ev = (replace(ev[0], legacy=True), *ev[1:])
    result = session.evaluate("episode_end", 100*S if kind == "deadline" else 4*S, ev)
    assert result.success_credit is False
    assert ledger.outcome("episode")["success_credit"] is False


def test_drop_is_latched_through_occlusion(spec, evidence, tmp_path):
    session, ev = completed_session(spec, evidence, JudgeLedger(tmp_path / "ledger.jsonl"))
    session.evaluate("after_action", 3*S, (evidence("robot", dropped=True),))
    result = session.evaluate("episode_end", 4*S, (ev[0], *[replace(e, observation=replace(e.observation, visibility="occluded")) for e in ev[1:]]))
    assert result.status == "failed" and not result.success_credit
    assert "drop_invariant_violated" in result.reason_codes


def test_partial_ledger_is_not_repaired_or_overwritten(spec, tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"evaluation_id": "partial"}')
    session = JudgeSession("episode", spec, JudgeLedger(path), 0)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        session.evaluate("episode_end", S, ())
    assert path.read_bytes() == before


def test_spec_strict_and_default_unconfigured(tmp_path):
    default = load_spec()
    # 23 Nov: the operator drew the destination (white box top); credit stays off until the verifier is qualified
    assert not default.calibration_hash and not default.validated_predicates
    raw = json.loads(DEFAULT_SPEC.read_text())
    raw["required_sequence"] = []
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_spec(path)
    raw = json.loads(DEFAULT_SPEC.read_text())
    raw["destination"] = [[0, 0], [10, 10], [0, 10], [10, 0]]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_spec(path)


def test_claims_cannot_change_credit_or_read_private_files(spec, evidence, tmp_path, monkeypatch):
    ledger = JudgeLedger(tmp_path / "judge" / "ledger.jsonl")
    session, ev = completed_session(spec, evidence, ledger)
    session.evaluate("episode_end", 4*S, ev)
    store = EpisodeStore(tmp_path / "ep")
    recorder = OutcomeRecorder(store, tmp_path / "index.jsonl", ledger)
    recorder.episode_id = "episode"
    # Binding the review is a harness operation. Tools themselves never read the ledger.
    monkeypatch.setattr(ledger, "outcome", lambda *args: pytest.fail("Astra tool read private ledger"))
    assert recorder.record(False)["recorded"]["claim"]["success"] is False
    recorder.record(True, 1000)
    assert recorder.last["success"] is True and recorder.last["claim"]["blocks_moved"] == 1000
    assert history(tmp_path / "index.jsonl", claims_only=True)[0]["claim"]["success"] is True
    other = OutcomeRecorder(store, tmp_path / "other.jsonl", JudgeLedger(tmp_path / "empty.jsonl"))
    other.episode_id = "unjudged"
    other.record(True, 100)
    assert other.last["success"] is False


def test_operator_promotion_requires_matching_ledger_and_rollback_loses_trust(spec, evidence, tmp_path):
    ledger = JudgeLedger(tmp_path / "ledger.jsonl")
    library = SkillLibrary(tmp_path / "skills.json")
    library.revise(Skill("pick", "pick", steps=[{"tool": "close_gripper"}]))
    library.add_evidence("pick", True, True)
    assert library.get("pick").validation_status == "candidate"
    assert not library.promote("pick", 1, ledger=ledger)["promoted"]
    session, ev = completed_session(spec, evidence, ledger)
    session.skill_dependency_hash = library.dependency_hash("pick")
    final = session.evaluate("episode_end", 4*S, ev)
    assert not library.promote("pick", evaluation_ids=[final.evaluation_id] * 2, ledger=ledger)["promoted"]
    second, ev = completed_session(spec, evidence, ledger, episode="second")
    second.skill_dependency_hash = library.dependency_hash("pick")
    other = second.evaluate("episode_end", 4*S, ev)
    citations = [final.evaluation_id, other.evaluation_id]
    assert library.promote("pick", evaluation_ids=citations, ledger=ledger)["promoted"]
    library.revise(Skill("pick", "changed", steps=[{"tool": "open_gripper"}]))
    assert not library.promote("pick", evaluation_ids=citations, ledger=ledger)["promoted"]
    assert library.rollback("pick", 1)["validation_status"] == "candidate"


def test_nested_execution_error_and_public_feedback_before_close(spec, evidence, tmp_path):
    ledger = JudgeLedger(tmp_path / "ledger.jsonl")
    captured = []
    class Capture:
        def acquire(self):
            captured.append("capture")
            now = time.monotonic_ns()
            return evidence("robot", ns=now), evidence(ns=now)
    judge = HarnessJudge(Capture(), ledger, spec)
    def close():
        captured.append("close")
        raise RuntimeError("handler crashed")
    ex = ActionExecutor([ActionSpec("look", "look", lambda camera: {"objects": []},
                                   {"properties": {"camera": {"type": "string"}}}, read_only=True),
                         ActionSpec("close_gripper", "close", close, moves_robot=True),
                         ActionSpec("run_routine", "nested", lambda: ex.execute("close_gripper"), delegates=True)], "live")
    script = iter([NS(id="1", output=[NS(type="function_call", name="look", arguments='{"camera":"wrist"}', call_id="c")], output_text=""),
                   NS(id="2", output=[NS(type="function_call", name="run_routine", arguments="{}", call_id="d")], output_text="")])
    inputs = []
    def create(input, **kw):
        inputs.append(input)
        return next(script)
    harness = AgentHarness(NS(create=create), ex, AgentConfig(), EpisodeStore(tmp_path / "ep"),
                           SkillLibrary(tmp_path / "skills.json"), judge=judge)
    with pytest.raises(RuntimeError, match="crashed"):
        harness.run("task")
    feedback = json.loads(inputs[1][0]["output"])["grasp_measurements"]
    assert feedback == {"du_px": 0, "dv_px": 0, "edge_deg": 0, "visibility": "full"}
    assert captured.index("close") > 1
    assert ledger.rows()[-1]["event"] == "episode_end"
    assert "after_close" in [r["event"] for r in ledger.rows()]
    assert harness.last_metrics["success_credit"] is False
    assert ex.lifecycle is None
    assert "predicate_results" not in json.dumps(inputs)


def test_images_do_not_overwrite_within_a_turn(tmp_path):
    import base64
    store = EpisodeStore(tmp_path / "ep")
    h = AgentHarness(None, ActionExecutor([]), AgentConfig(), store, SkillLibrary(tmp_path / "s.json"))
    episode = store.start("task")
    results = [{"result": {"image_jpeg_b64": base64.b64encode(data).decode()}} for data in (b"one", b"two")]
    for result in results:
        h._take_image(episode, 1, "look", result)
    paths = [Path(r["result"]["image_file"]) for r in results]
    assert paths[0] != paths[1] and [p.read_bytes() for p in paths] == [b"one", b"two"]


def test_scene_propagates_original_capture_metadata():
    from drivers.camera import Frame
    from perception.scene import Scene
    frame = Frame(np.zeros((100, 100, 3), np.uint8), captured_ns=123, sequence_id=4, stream_id="source")
    scene = Scene(PerceptionConfig(), {"wrist": NS(read=lambda: frame)})
    for result in (scene.look("wrist"), scene.find_report("blue", "wrist")):
        assert (result["captured_ns"], result["sequence_id"], result["stream_id"]) == (123, 4, "source")


def test_offline_replay_ignores_forged_measurements_and_legacy_never_credits(tmp_path):
    import cv2
    replay = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/dev/judge_replay.py"))["replay"]
    ep = "20261119-175951-9b3722"
    source = tmp_path / "episodes"
    source.mkdir()
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    log = source / f"{ep}.jsonl"
    log.write_text(json.dumps({"event": "tool", "name": "look", "result": {"result": {
        "camera": "wrist", "image_file": str(image), "objects": [{"pixel": [50, 50], "holding": True}]}}}) + "\n")
    before = log.read_bytes()
    result = replay(ep, source, tmp_path / "replay")
    assert result["legacy_frames"] == 1 and not result["success_credit"] and result["provisional"]
    assert result["predicate_counts"]["grasp_aligned"] == {"unknown": 1}
    assert log.read_bytes() == before


def test_capture_preserves_raw_data_and_requires_operator_profile(spec, tmp_path):
    from drivers.camera import Frame, Intrinsics
    from judge.capture import EvidenceCapture
    from judge.spec import digest

    pixels = np.zeros((100, 100, 3), np.uint8)
    pixels[30:71, 30:71] = (220, 0, 0)
    now = time.monotonic_ns()
    frames = {name: Frame(pixels, np.ones((100, 100), np.float32), Intrinsics(100, 100, 100, 100, 50, 50),
                          captured_ns=now, sequence_id=7, stream_id=name + "-stream") for name in ("env", "wrist")}
    profile = {"validated": True, "object_id": spec.object_id, "object_color": "blue",
               "single_object_scope_confirmed": True,
               "stream_ids": {name: name + "-stream" for name in frames},
               "image_sizes": {name: [100, 100] for name in frames},
               "bbox_size_bounds": {name: [[40, 42], [40, 42]] for name in frames},
               "minimum_fill_ratio": 0.9,
               "wrist_pose_bounds": [[-200, 200]] * 6, "robot_uncertainty_mm": 0.1,
               "pixel_uncertainty_px": 1, "edge_uncertainty_deg": 1,
               "finger_polygon": [[20, 20], [80, 20], [80, 80], [20, 80]],
               "camera_to_base_mm": np.eye(4).tolist(), "cube_height_mm": 40,
               "table_z_mm": 960, "support_uncertainty_mm": 0.5,
               "open_position": 1000, "open_position_tolerance": 20, "retracted_z_min_mm": 20}
    cfg = replace(spec, calibration_hash=digest(profile))
    capture = EvidenceCapture(NS(observe=lambda name: frames[name]),
                              lambda: {"pose": [0, 0, 50, 0, 0, 0],
                                       "gripper": {"position": 800, "grip_status": "object caught"}},
                              tmp_path / "raw", cfg, profile)
    robot, wrist, env = capture.acquire()
    assert wrist.captured_ns == now and wrist.sequence_id == 7 and wrist.stream_id == "wrist-stream"
    assert wrist.identity_verified and wrist.observation.between_fingers is True
    assert env.observation.support_gap_mm == 0
    assert evidence_gate((wrist,), cfg, time.monotonic_ns()) is None
    with np.load(env.file) as raw:
        assert np.array_equal(raw["color"], pixels) and np.all(raw["depth_m"] == 1)
    frames["wrist"].stream_id = "restarted"
    assert capture.camera("wrist").calibration_applicable is False
    unconfigured = EvidenceCapture(capture.scene, capture.state_fn, tmp_path / "unconfigured", spec, {})
    acquired = unconfigured.acquire()
    assert all(not e.calibration_applicable for e in acquired)
    # unconfigured: robot state only; no colour/depth files that could never earn credit
    assert [e.source for e in acquired] == ["robot"]
    assert not list((tmp_path / "unconfigured").glob("*.npz"))


def test_dry_run_and_rejection_never_trigger_physical_judge_events(spec, tmp_path):
    events = []
    hook = NS(before=lambda *a: events.append("before"), after=lambda *a: events.append("after"))
    for mode in ("dry_run", "confirm"):
        executor = ActionExecutor([ActionSpec("close_gripper", "close", lambda: pytest.fail("hardware handler"),
                                             moves_robot=True)], mode=mode)
        executor.lifecycle = hook
        assert executor.execute("close_gripper")["executed"] is False
    assert events == []


def test_history_tools_read_claims_without_opening_ledger(tmp_path, monkeypatch):
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"episode": "legacy", "success": True, "blocks_moved": 9}) + "\n")
    ledger = NS(rows=lambda: pytest.fail("private read"), outcome=lambda ep: pytest.fail("private read"))
    rows = history(index, ledger=ledger, claims_only=True)
    assert rows[0]["claim"]["success"] is True and "success" not in rows[0]
    assert history(index, ledger=JudgeLedger(tmp_path / "absent.jsonl"))[0]["success"] is False


def test_missing_initial_and_timeout_remain_inconclusive(spec, evidence, tmp_path):
    session, ev = completed_session(spec, evidence, JudgeLedger(tmp_path / "ledger.jsonl"))
    result = session.evaluate("episode_end", 100*S, ev)
    assert result.status == "inconclusive" and result.success_credit is False


def test_camera_transports_preserve_capture_identity_without_hardware(monkeypatch):
    import io
    import cv2
    from drivers.camera import Frame, Intrinsics, RealSenseRemoteCamera
    from drivers.realsense_helper import pack_frame
    from perception.scene import UIFrameSource

    color = np.zeros((10, 10, 3), np.uint8)
    frame = Frame(color, np.ones((10, 10), np.float32), Intrinsics(10, 10, 100, 100, 5, 5),
                  captured_ns=123, sequence_id=9, stream_id="producer")
    payload = pack_frame(frame)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(payload))
    remote = RealSenseRemoteCamera.__new__(RealSenseRemoteCamera)
    remote.url, remote.timeout = "http://synthetic.test", 1
    for result in (remote.read(), remote.read()):
        assert (result.captured_ns, result.sequence_id, result.stream_id) == (123, 9, "producer")
    jpeg = cv2.imencode(".jpg", color)[1].tobytes()
    def response(*args, **kwargs):
        stream = io.BytesIO(jpeg)
        stream.headers = {"X-Capture-Monotonic-Ns": "456", "X-Frame-Sequence": "12", "X-Frame-Stream": "wrist"}
        return stream
    monkeypatch.setattr("urllib.request.urlopen", response)
    ui_source = UIFrameSource("http://synthetic.test", "wrist")
    assert ui_source.read().captured_ns == 456
    def legacy(*args, **kwargs):
        stream = io.BytesIO(jpeg)
        stream.headers = {}
        return stream
    monkeypatch.setattr("urllib.request.urlopen", legacy)
    assert ui_source.read().captured_ns is None
