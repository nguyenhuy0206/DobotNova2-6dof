"""Knowledge as hypotheses: computed status, scope, evidence; archived never reach the prompt."""
import pytest

from common.errors import ValidationError
from memory.hypotheses import HypothesisStore, migrate_notes, status_of


def store(tmp_path):
    return HypothesisStore(tmp_path / "h.json")


def test_status_is_computed_from_evidence(tmp_path):
    s = store(tmp_path)
    h = s.propose("target_v 70 puts the fingers over the cube at rz 81.9", {"rz": "81.9", "routine": "wrist_center"},
                  "blue touch hit the cube top at 73.4")
    assert h["status"] == "open"
    assert s.record(h["id"], "supports", "green touch hit at 79.7")["status"] == "supported"
    assert s.record(h["id"], "contradicts", "blue at rz 0 hit the box")["status"] == "contested"
    two_two = s.record(h["id"], "contradicts", "yellow at rz 0 hit the box")      # 2 for, 2 against
    assert two_two["status"] == "contested" and "status_changed" not in two_two
    assert s.record(h["id"], "contradicts", "again at rz 0")["status_changed"] == "contested -> refuted"
    one = s.propose("box top is at z 60", observation="contact at 59.5", result="contradicts")
    assert one["status"] == "refuted"                              # no support, one against
    with pytest.raises(ValidationError):
        s.record(h["id"], "maybe", "x" * 10)


def test_same_claim_and_scope_is_one_hypothesis(tmp_path):
    s = store(tmp_path)
    a = s.propose("rz 30.9 lets wrist centring converge", {"color": "blue"})
    b = s.propose("RZ 30.9 lets wrist  centring converge", {"color": "blue"})
    c = s.propose("rz 30.9 lets wrist centring converge", {"color": "green"})
    assert a["id"] == b["id"] != c["id"]


def test_digest_has_status_and_scope_not_raw_observations_and_never_archived(tmp_path):
    s = store(tmp_path)
    k = s.propose("cube tops are 8 mm lower than the env camera says", {"color": "blue"}, "measured 73.4 vs 81.8")
    gone = s.propose("measure_block_height needs target 70", {"skill": "measure_block_height"}, "6/6 hits")
    s.archive(gone["id"], "skill removed")
    d = s.digest()
    assert f"{k['id']} [open 1✓ 0✗" in d and "color=blue" in d
    assert "measured 73.4" not in d                                 # evidence stays out of the prompt
    assert gone["id"] not in d
    with pytest.raises(ValidationError, match="archived"):
        s.record(gone["id"], "supports", "still true")
    assert HypothesisStore(tmp_path / "h.json").list(status="archived")["total"] == 1   # persisted, not deleted


def test_out_of_scope_hypotheses_are_archived(tmp_path):
    s = store(tmp_path)
    keep = s.propose("approach skips when already above the cube", {"routine": "approach"})
    drop = s.propose("measuring needs a slow probe", {"skill": "measure_block_height"})
    assert s.archive_out_of_scope({"approach", "pick_block_wrist"}) == [drop["id"]]
    assert status_of(s._get(keep["id"])) == "open" and "no longer exists" in s._get(drop["id"])["archived"]["reason"]


def test_migration_archives_raw_experience_and_removed_skills(tmp_path):
    notes = [
        {"t": "2026-11-16 14:41", "topic": "causes", "text": "pick_block_wrist not_centred: cause = Probe touched "
         "the box. (evidence: blue moved 64 mm)"},
        {"t": "2026-11-16 16:10", "topic": "grasp_trials", "text": "blue#1 rz=50.9 depth=6 -> grip=800"},
        {"t": "2026-11-16 16:10", "topic": "grasp_best", "text": "First next attempt: fresh env localisation"},
        {"t": "2026-11-16 17:06", "topic": "causes", "text": "measure_block_height contact: cause = target too high"},
        {"t": "2026-11-12 18:14", "topic": "wrist_camera", "text": "tool +dy10 moves green right ~56px"},
    ]
    s = store(tmp_path)
    assert migrate_notes(notes, s, {"measure_block_height"}) == {"open": 2, "archived": 3}
    d = s.digest()
    assert "Probe touched the box" in d and "~56px" in d
    assert "grip=800" not in d and "First next attempt" not in d and "measure_block_height" not in d
    assert s.list(status="archived")["total"] == 3


def test_generalize_inherits_all_evidence_and_earns_its_status(tmp_path):
    s = store(tmp_path)
    blue = s.propose("env camera reads the blue cube top too high", {"color": "blue"}, "73.4 measured vs 81.8")
    s.record(blue["id"], "supports", "73.1 vs 81.8")
    green = s.propose("env camera reads the green cube top too high", {"color": "green"}, "79.7 vs 82.0")
    g = s.generalize([blue["id"], green["id"]], "the env camera reads cube tops too high on the box",
                     {"place": "white box"})
    assert g["supports"] == 3 and g["status"] == "supported" and g["generalizes"] == [blue["id"], green["id"]]
    assert s._get(blue["id"])["generalized_into"] == g["id"]
    yellow = s.propose("env camera reads the yellow top too high", {"color": "yellow"}, "84 vs 80.7", "contradicts")
    g2 = s.generalize([g["id"], yellow["id"]], "the env camera reads ALL cube tops too high", {"place": "white box"})
    assert g2["status"] == "contested"                              # the counter-example comes along
    with pytest.raises(ValidationError):
        s.generalize([blue["id"]], "too narrow to generalize from one", {})
