import numpy as np

from common.poses import pose_to_matrix
from perception.transforms import HandEye, deproject_pixel


def test_deproject_center():
    assert np.allclose(deproject_pixel(320, 240, 0.5, 600, 600, 320, 240), [0, 0, 0.5])


def test_pose_identity():
    T = pose_to_matrix([100, 200, 300, 0, 0, 0])
    assert np.allclose(T[:3, :3], np.eye(3)) and np.allclose(T[:3, 3], [0.1, 0.2, 0.3])


def test_eye_to_hand_identity():
    he = HandEye(np.eye(4), eye_in_hand=False)
    assert np.allclose(he.camera_to_base(np.array([0.1, 0.0, 0.2])), [100, 0, 200])


def test_hand_eye_roundtrip(tmp_path):
    T = np.eye(4); T[0, 3] = 0.05
    HandEye(T, True).save(tmp_path / "he.json")
    assert np.allclose(HandEye.load(tmp_path / "he.json").T, T)
