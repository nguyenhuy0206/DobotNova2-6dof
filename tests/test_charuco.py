"""ChArUco calibration: detection on a rendered board, and the solvers against a known ground truth."""
import cv2
import numpy as np
import pytest

from perception import charuco as ch
from perception.transforms import pose_to_matrix

SPEC = ch.BoardSpec()                       # the printed board: 7x5, 20.85 mm squares, 15 mm markers
K = np.array([[520.0, 0, 322], [0, 518.0, 238], [0, 0, 1]])
DIST = np.array([-0.08, 0.05, 0.001, -0.0005, 0.0])
SIZE = (640, 480)

# ground truth: wrist camera ~57 mm off the tool axis, looking along it; board flat on the table
T_FLANGE_CAM = pose_to_matrix([57, 5, 40, 0, 0, 90])
T_BASE_BOARD = pose_to_matrix([-330, -360, 5, 0, 0, 30])
T_BASE_ENV = pose_to_matrix([200, -620, 135, -105, 0, 20])


def _det(T_cam_board, noise_px=0.15, rng=None):
    ids = np.arange((SPEC.squares[0] - 1) * (SPEC.squares[1] - 1))
    obj = ch.object_points(SPEC, ids)
    r = cv2.Rodrigues(T_cam_board[:3, :3])[0]
    px, _ = cv2.projectPoints(obj, r, T_cam_board[:3, 3], K, DIST)
    px = px.reshape(-1, 2) + (rng.normal(0, noise_px, (len(ids), 2)) if rng is not None else 0)
    ok = (px[:, 0] > 0) & (px[:, 0] < SIZE[0]) & (px[:, 1] > 0) & (px[:, 1] < SIZE[1])
    return ch.Detection(px[ok].astype(np.float32), ids[ok], 12, SIZE)


def _views(n=15, seed=1):
    rng = np.random.default_rng(seed)
    centre = T_BASE_BOARD @ np.r_[SPEC.center_m, 1]
    out = []
    for _ in range(n):
        tilt = rng.uniform(-18, 18, 2)
        twist = rng.uniform(-40, 40)
        height = rng.uniform(0.20, 0.32)
        # camera above the board centre looking down, tilted; flange from the camera
        T_base_cam = pose_to_matrix([0, 0, 0, 180 + tilt[0], tilt[1], twist])
        T_base_cam[:3, 3] = centre[:3] - T_base_cam[:3, 2] * height + rng.normal(0, 0.02, 3)
        T_base_flange = T_base_cam @ ch.inv(T_FLANGE_CAM)
        out.append((T_base_flange, _det(ch.inv(T_base_cam) @ T_BASE_BOARD, rng=rng)))
    return out


def test_detects_the_printed_board_in_a_rendered_oblique_view():
    img = SPEC.board().generateImage((7 * 60, 5 * 60), marginSize=30)
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[200, 250], [460, 250], [560, 400], [90, 400]])        # seen from low and far
    canvas = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), SIZE, borderValue=200)
    det = ch.detect(canvas, SPEC)
    assert det is not None and len(det) >= 18


def test_intrinsics_hand_eye_and_env_chain_recover_the_ground_truth():
    views = _views()
    dets = [d for _, d in views]
    K_est, dist_est, rms = ch.calibrate_intrinsics(dets, SPEC)
    assert rms < 0.5
    assert K_est[0, 0] == pytest.approx(K[0, 0], rel=0.02)

    res = ch.hand_eye([T for T, _ in views], dets, SPEC, K_est, dist_est)
    assert res.reproj_px < 0.5
    assert np.linalg.norm(res.T_flange_cam[:3, 3] - T_FLANGE_CAM[:3, 3]) * 1000 < 2.0
    assert ch.rot_deg(res.T_flange_cam[:3, :3].T @ T_FLANGE_CAM[:3, :3]) < 0.5
    assert np.linalg.norm(res.T_base_board[:3, 3] - T_BASE_BOARD[:3, 3]) * 1000 < 2.0

    # env camera sees the same board: chain through it
    K_env = np.array([[386.0, 0, 320.5], [0, 385.0, 244.0], [0, 0, 1]])
    T_env_board = ch.inv(T_BASE_ENV) @ T_BASE_BOARD
    obj = ch.object_points(SPEC, np.arange(24))
    px, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_env_board[:3, :3])[0], T_env_board[:3, 3], K_env, None)
    env_pose, err = ch.board_pose(ch.Detection(px.reshape(-1, 2).astype(np.float32), np.arange(24), 12, SIZE),
                                  SPEC, K_env)
    assert err < 0.1
    T = ch.env_from_board(res.T_base_board, env_pose)
    assert np.linalg.norm(T[:3, 3] - T_BASE_ENV[:3, 3]) * 1000 < 3.0


def test_average_poses_is_a_rotation():
    Ts = [pose_to_matrix([1, 2, 3, 0, 0, a]) for a in (9, 10, 11)]
    T = ch.average_poses(Ts)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))
    assert ch.rot_deg(T[:3, :3].T @ Ts[1][:3, :3]) < 1e-6


# --- the procedure (offline part) ---------------------------------------------------------------

from perception import charuco_calibration as cc  # noqa: E402
from perception.transforms import matrix_to_rpy_deg  # noqa: E402


def _session(env_pose=T_BASE_ENV):
    views = [{"label": str(i), "image": f"w{i}.jpg", "detection": d.to_dict(),
              "flange_pose": [*(T[:3, 3] * 1000).tolist(), *matrix_to_rpy_deg(T[:3, :3])]}
             for i, (T, d) in enumerate(_views())]
    K_env = np.array([[386.0, 0, 320.5], [0, 385.0, 244.0], [0, 0, 1]])
    T_env_board = ch.inv(env_pose) @ T_BASE_BOARD
    px, _ = cv2.projectPoints(ch.object_points(SPEC, np.arange(24)), cv2.Rodrigues(T_env_board[:3, :3])[0],
                              T_env_board[:3, 3], K_env, None)
    det = ch.Detection(px.reshape(-1, 2).astype(np.float32), np.arange(24), 12, SIZE)
    return {"wrist": {"views": views}, "env": {"K": K_env.tolist(), "frames": [{"image": "e.jpg", "detection": det.to_dict()}]}}


def _cfg(**measured):
    return cc.CalibrationConfig(board=SPEC, measured=measured)


def test_solve_accepts_a_consistent_session_and_checks_the_tape_measure():
    cam = T_BASE_ENV[:3, 3] * 1000
    rep = cc.solve(_session(), _cfg(env_cam_to_base_mm=float(np.hypot(cam[0], cam[1])) + 15,
                                    env_cam_height_mm=cam[2] - 5 - 10), table_z_mm=5)    # board lies at z 5
    assert rep["accepted"], rep["problems"]
    assert np.allclose(rep["env"]["camera_in_base_mm"], cam, atol=3)
    assert [c["ok"] for c in rep["checks"]] == [True, True]


def test_solve_refuses_when_the_tape_measure_disagrees():
    rep = cc.solve(_session(), _cfg(env_cam_to_base_mm=900), table_z_mm=5)
    assert not rep["accepted"]
    assert any("D455 to robot base" in p for p in rep["problems"])


def test_solve_refuses_when_the_d455_cannot_see_the_board():
    data = _session()
    data["env"]["frames"] = [{"image": "e.jpg", "detection": None}]
    rep = cc.solve(data, _cfg())
    assert not rep["accepted"] and any("D455 does not see the board" in p for p in rep["problems"])
    assert rep["wrist_accepted"] and not rep["env_accepted"]     # the wrist part stands on its own
    assert rep["_solution"]["T_base_env"] is None


def test_view_poses_pivot_about_the_board():
    start = [-300.0, -350.0, 300.0, 180.0, 0.0, 30.0]
    poses = cc.view_poses(start, 0.25)
    pivot = np.array(start[:3]) + pose_to_matrix(start)[:3, 2] * 250
    for (_, p), (_, _, _, dd) in zip(poses, cc.VIEWS):
        axis = pose_to_matrix(p)[:3, 2]
        assert np.allclose(np.array(p[:3]) + axis * (250 + dd), pivot, atol=1e-6)
    assert poses[0][1] == pytest.approx(start)


def test_save_writes_only_the_accepted_parts(tmp_path):
    data = _session()
    data["env"]["frames"] = [{"image": "e.jpg", "detection": None}]
    rep = cc.solve(data, _cfg())
    env_file = tmp_path / "hand_eye_env.json"
    env_file.write_text('{"T": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "eye_in_hand": false}')
    cc.save(rep, env_file, "x", result_file=tmp_path / "r.json", wrist_file=tmp_path / "w.json")
    assert env_file.exists() and not (tmp_path / "hand_eye_env.x.old.json").exists()   # env untouched
    assert (tmp_path / "w.json").exists() and "T_flange_wrist" in (tmp_path / "r.json").read_text()


def test_solve_refuses_a_board_that_is_not_on_the_table():
    rep = cc.solve(_session(), _cfg(), table_z_mm=42)                     # board really at z 5
    assert not rep["wrist_accepted"] and any("off the table" in p for p in rep["problems"])


def test_camera_orbit_keeps_the_board_centre_at_the_same_pixel():
    X = pose_to_matrix([-108, -11, 51, 0, 60, 0])                   # camera at an angle to the tool, like the rig
    T_bf0 = pose_to_matrix([-280, -500, 330, 180, 0, 90])
    P = np.array([-0.256, -0.517, 0.005])
    T_bc0 = T_bf0 @ X
    v0 = T_bc0[:3, :3].T @ (P - T_bc0[:3, 3])
    for (label, T_bf), (_, _, _, dd) in zip(cc.camera_view_poses(T_bf0, P, X), cc.VIEWS):
        T_bc = T_bf @ X
        v = T_bc[:3, :3].T @ (P - T_bc[:3, 3])
        assert np.allclose(v / np.linalg.norm(v), v0 / np.linalg.norm(v0), atol=1e-9), label   # same pixel
        assert np.linalg.norm(v) == pytest.approx(np.linalg.norm(v0) + dd / 1000)
