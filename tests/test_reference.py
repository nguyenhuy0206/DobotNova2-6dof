"""Operator reference pictures and compare_reference (24 Nov, threading)."""
import numpy as np
import pytest

from perception.reference import compare, load_references, save_reference


def scene_picture(shift: int) -> np.ndarray:
    img = np.full((480, 640, 3), 80, np.uint8)
    img[200:280, 100 + shift:160 + shift] = 230          # a bright bar that moves
    img[100:120, 400:600] = 30
    return img


def test_the_closer_reference_wins_and_the_picture_has_three_tiles(tmp_path):
    save_reference(scene_picture(200), "threading", "success", root=tmp_path)
    save_reference(scene_picture(0), "threading", "fail", root=tmp_path)
    refs = load_references("threading", root=tmp_path)
    out = compare(scene_picture(195), refs)
    assert out["closer_to"] == "success" and out["scores"]["success"] > out["scores"]["fail"]
    assert out["picture"].shape[1] == 3 * (320 + 4)
    out = compare(scene_picture(5), refs)
    assert out["closer_to"] == "fail"


def test_missing_references_say_how_to_make_them(tmp_path):
    save_reference(scene_picture(0), "threading", "fail", root=tmp_path)
    with pytest.raises(ValueError, match="capture_reference"):
        compare(scene_picture(0), load_references("threading", root=tmp_path))


def test_compare_reference_tool_reports_and_labels_its_picture(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import perception.reference as ref
    from agent.explore.tools import reference_spec
    monkeypatch.setattr(ref, "REFERENCES_DIR", tmp_path)
    save_reference(scene_picture(200), "threading", "success", root=tmp_path)
    save_reference(scene_picture(0), "threading", "fail", root=tmp_path)
    monkeypatch.setattr(ref, "load_references", lambda task, root=tmp_path: load_references(task, root=tmp_path))
    scene = NS(sources={"wrist": None}, observe=lambda cam: NS(color=scene_picture(198)),
               encode_image=lambda img, objs: "b64")
    out = reference_spec(scene, "threading").handler()
    assert out["closer_to"] == "success" and out["image_jpeg_b64"] == "b64"
    assert "LEFT = the wrist camera now" in out["picture_label"]
