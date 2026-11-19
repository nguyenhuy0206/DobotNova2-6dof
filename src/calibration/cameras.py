"""Both cameras calibrated from one ChArUco board, by the robot itself (no model in the loop).

    1. env    the D455 looks at the board (several frames)               -> T_env_board
    2. wrist  the robot pivots the wrist camera around the board centre   -> views (flange pose, corners)
    3. solve  wrist intrinsics; hand-eye -> T_flange_wrist, T_base_board; T_base_env through the board
    4. check  quality gates + the tape-measured cell (config/perception.yaml calibration.measured)
    5. save   only when every gate passes; the old files are kept as *.<stamp>.old.json

Everything captured is written to data/calibration/<stamp>/ (images + views.json) so a run can be
solved again offline (solve_session) after a change to the solver or the board numbers.

The robot uses the normal primitives: speed cap, table keep-out, IK reachability and the
self-collision guard all apply; a pose that cannot be reached is skipped before moving.
The board may lie flat or lean toward the D455: nothing below assumes it is flat.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from calibration.board import BoardSpec, Detection, board_pose, calibrate_intrinsics, detect
from calibration.files import CALIBRATION_FILE, SESSIONS_DIR, WRIST_FILE, backup
from calibration.geometry import average_poses, inv
from calibration.hand_eye import HandEyeResult, depth_check_mm, env_from_board, hand_eye
from common.errors import RobotError, SafetyError, ValidationError
from perception.transforms import HandEye, matrix_to_rpy_deg, pose_to_matrix


# wrist views: rotation of the tool about its own X, Y, Z (deg) and change of distance to the
# board (mm, + = farther), all about the pivot (the point on the tool axis at the board).
VIEWS = [(0, 0, 0, 0), (12, 0, 0, 0), (-12, 0, 0, 0), (0, 12, 0, 0), (0, -12, 0, 0),
         (0, 0, 30, 0), (0, 0, -30, 0), (10, 10, 20, 40), (-10, -10, -20, 40), (10, -10, -30, 60),
         (-10, 10, 30, 60), (15, 0, 40, 20), (-15, 0, -40, 20), (0, 15, -40, 30), (0, -15, 40, 30),
         (8, 8, 0, 80), (-8, -8, 0, 80), (18, 0, 0, 0), (0, 18, 0, 0), (-18, 0, 15, 10),
         # milder ones for a start near the edge of the reach
         (6, 0, 20, 50), (-6, 0, -20, 50), (0, 6, -20, 50), (0, -6, 20, 50), (0, 0, 15, 30),
         (0, 0, -15, 30), (5, 5, 45, 60), (-5, -5, -45, 60)]


@dataclass
class Gates:
    min_wrist_views: int = 8
    max_intrinsics_px: float = 1.0
    max_hand_eye_px: float = 2.5        # small board + strongly distorted webcam: ~1.7 px on 18 Nov
    max_board_off_table_mm: float = 10.0   # a flat board's centre must lie on the taught table
    max_board_tilt_deg: float = 5.0
    min_env_corners: int = 8
    max_env_px: float = 1.5
    max_env_depth_mm: float = 25.0       # board distance: image pose vs depth image
    tolerance_mm: float = 40.0           # vs the tape-measured cell


@dataclass
class CalibrationConfig:
    board: BoardSpec = field(default_factory=BoardSpec)
    measured: dict = field(default_factory=dict)
    gates: Gates = field(default_factory=Gates)
    wrist_camera: str = "wrist"
    env_camera: str = "env"

    @classmethod
    def from_dict(cls, d: dict | None) -> "CalibrationConfig":
        d = d or {}
        g = Gates(**{k: v for k, v in (d.get("gates") or {}).items() if k in Gates.__dataclass_fields__})
        m = dict(d.get("measured") or {})
        if "tolerance_mm" in m:
            g.tolerance_mm = float(m.pop("tolerance_mm"))
        return cls(BoardSpec.from_dict(d.get("board") or {}), m, g,
                   d.get("wrist_camera", "wrist"), d.get("env_camera", "env"))


def camera_matrix_of(intr) -> np.ndarray:
    """3x3 camera matrix from a drivers.camera.Intrinsics."""
    return np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], float)


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------------------------
# capture (robot + cameras)
# --------------------------------------------------------------------------------------------

def capture_env(scene, cfg: CalibrationConfig, session: Path, frames: int = 6, log=print) -> dict:
    """D455 frames of the board: detections + colour intrinsics (+ depth for the cross-check)."""
    out = {"frames": []}
    for i in range(frames):
        f = scene.observe(cfg.env_camera)
        cv2.imwrite(str(session / f"env_{i:02d}.jpg"), f.color)
        if f.depth_m is not None:
            np.save(session / f"env_{i:02d}_depth.npy", (f.depth_m * 1000).astype(np.uint16))
        if f.intrinsics is not None:
            out["K"] = camera_matrix_of(f.intrinsics).tolist()
        det = detect(f.color, cfg.board, scales=(1, 2, 3))
        out["frames"].append({"image": f"env_{i:02d}.jpg", "detection": det.to_dict() if det else None})
        time.sleep(0.15)
    n = [len(fr["detection"]["ids"]) if fr["detection"] else 0 for fr in out["frames"]]
    log(f"env: board corners per frame {n}")
    return out


def _pivot_distance_m(det: Detection, cfg: CalibrationConfig) -> float:
    """Rough camera->board distance from the first wrist view (guessed focal length: the
    pivot only has to keep the board in the picture)."""
    w, h = det.image_size
    K = np.array([[0.9 * w, 0, w / 2], [0, 0.9 * w, h / 2], [0, 0, 1]])
    T, _ = board_pose(det, cfg.board, K)
    return float(np.clip(np.linalg.norm(T[:3, :3] @ cfg.board.center_m + T[:3, 3]), 0.12, 0.5))


def view_poses(start_tip: list[float], pivot_m: float, views=VIEWS) -> list[tuple[str, list[float]]]:
    """Fingertip poses that pivot the tool about the point `pivot_m` along its axis."""
    T0 = pose_to_matrix(start_tip)
    R0, tip0 = T0[:3, :3], np.array(start_tip[:3], float)
    pivot = tip0 + R0[:, 2] * pivot_m * 1000
    out = []
    for tx, ty, tz, dd in views:
        R = R0 @ pose_to_matrix([0, 0, 0, tx, ty, tz])[:3, :3]
        tip = pivot - R[:, 2] * (pivot_m * 1000 + dd)
        out.append((f"tilt ({tx},{ty}) twist {tz} dist {dd:+d}", [*map(float, tip), *matrix_to_rpy_deg(R)]))
    return out


@dataclass
class Prior:
    """An earlier wrist solution, used only to AIM the camera (never in the new solve)."""
    X: np.ndarray                 # T_flange_cam
    K: np.ndarray
    dist: np.ndarray
    T_base_board: np.ndarray
    T_cam_board_ref: np.ndarray   # a view that saw the board well (the earlier start)


def prior_from_session(session: Path, cfg: CalibrationConfig) -> Prior:
    data = json.loads((session / "views.json").read_text(encoding="utf-8"))
    views = data["wrist"]["views"]
    dets = [Detection.from_dict(v["detection"]) for v in views]
    K, dist, _ = calibrate_intrinsics(dets, cfg.board)
    he = hand_eye([pose_to_matrix(v["flange_pose"]) for v in views], dets, cfg.board, K, dist)
    best = max(range(len(dets)), key=lambda i: len(dets[i]))
    return Prior(he.T_flange_cam, K, dist, he.T_base_board, board_pose(dets[best], cfg.board, K, dist)[0])


def _pose(T: np.ndarray) -> list[float]:
    """4x4 (metres) -> [x, y, z mm, rx, ry, rz deg]."""
    return [*map(float, T[:3, 3] * 1000), *matrix_to_rpy_deg(T[:3, :3])]


def camera_view_poses(T_base_flange0: np.ndarray, P_base: np.ndarray, X: np.ndarray,
                      views=VIEWS) -> list[tuple[str, np.ndarray]]:
    """Flange poses that orbit the CAMERA about the board centre P: (tx, ty, tz) rotate the camera
    about its own axes (tz = about the optical axis), dd changes the distance (mm); P stays at the
    same pixel, so the board stays in the picture whatever the camera's angle to the tool."""
    T_bc0 = T_base_flange0 @ X
    Rc0, c0 = T_bc0[:3, :3], T_bc0[:3, 3]
    v = Rc0.T @ (P_base - c0)
    d0 = float(np.linalg.norm(v))
    Xi = inv(X)
    out = []
    for tx, ty, tz, dd in views:
        Rc = Rc0 @ pose_to_matrix([0, 0, 0, tx, ty, tz])[:3, :3]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rc, P_base - Rc @ (v * (d0 + dd / 1000) / d0)
        out.append((f"camera tilt ({tx},{ty}) twist {tz} dist {dd:+d}", T @ Xi))
    return out


def aim_start(prims, prior: Prior, cfg: CalibrationConfig, step_deg: float = 10.0) -> list[float] | None:
    """A fingertip pose where the wrist camera sees the board as in the prior's reference view,
    turned about the vertical through the board centre to the reachable pose closest to the base."""
    P = (prior.T_base_board @ np.r_[cfg.board.center_m, 1])[:3]
    T_bc_ref = prior.T_base_board @ inv(prior.T_cam_board_ref)
    best = None
    for psi in np.arange(0, 360, step_deg):
        Rz = np.eye(4)
        Rz[:3, :3] = pose_to_matrix([0, 0, 0, 0, 0, float(psi)])[:3, :3]
        Rz[:3, 3] = P - Rz[:3, :3] @ P                       # rotation about the vertical through P
        flange = _pose(Rz @ T_bc_ref @ inv(prior.X))
        tip = prims.tip_from_flange(flange)
        try:
            prims._check_points(tip, flange[:3])
            prims._ik(flange)
        except (ValidationError, SafetyError, RobotError):
            continue
        r = float(np.hypot(*flange[:2]))
        if best is None or r < best[0]:
            best = (r, tip)
    return None if best is None else best[1]


FRESH_SHIFT_PX = 3.0      # corners moved at least this much: the frame is from after the move
STILL_SHIFT_PX = 0.8      # two frames in a row within this: the camera has stopped moving


def corner_shift_px(a: Detection, b: Detection) -> float:
    """Mean pixel shift of the corners two detections share (inf when they share fewer than 4)."""
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 4:
        return float("inf")
    return float(np.linalg.norm(a.corners[ia] - b.corners[ib], axis=1).mean())


def capture_wrist(prims, scene, cfg: CalibrationConfig, session: Path, speed: int = 10, settle_s: float = 1.0,
                  max_views: int = 16, max_drop_mm: float = 20.0, prior: Prior | None = None,
                  log=print) -> dict:
    """Pivot the wrist camera around the board; record (flange pose, corners) at each pose."""
    def grab(i: int, before: Detection | None = None, timeout_s: float = 5.0):
        """A FRESH and STILL frame: the wrist stream can lag (18 Nov: a view after a 30° twist was
        the start image again). After a move, wait until the corners differ from `before`, then
        accept two frames in a row that agree. None = no board, or no fresh still frame."""
        deadline, last = time.monotonic() + timeout_s, None
        while time.monotonic() < deadline:
            img = scene.observe(cfg.wrist_camera).color
            det = detect(img, cfg.board)
            if det is None:
                last = None
            elif before is not None and corner_shift_px(det, before) < FRESH_SHIFT_PX:
                last = None                             # still the picture from before the move
            elif last is not None and corner_shift_px(det, last[0]) < STILL_SHIFT_PX:
                cv2.imwrite(str(session / f"wrist_{i:02d}.jpg"), img)
                return det, f"wrist_{i:02d}.jpg"
            else:
                last = (det, img)
            time.sleep(0.25)
        return None, None

    state = prims.get_state()
    det0, image0 = grab(0)
    if det0 is None and prior is not None:
        aim = aim_start(prims, prior, cfg)
        if aim is not None:
            log(f"wrist: board not in view; aiming the camera at it from {[round(v) for v in aim[:3]]}")
            prims.move_to_pose(*aim, speed=speed)
            time.sleep(settle_s)
            state = prims.get_state()
            det0, image0 = grab(0)
    start = state["pose"]
    if det0 is None:
        raise ValidationError("the wrist camera does not see the ChArUco board from the current pose: move the "
                              "gripper above the board (wrist camera looking at it, ~20–30 cm away) and run again")
    pivot = _pivot_distance_m(det0, cfg)
    # never much lower than the start; the primitives' table keep-out still applies
    floor = start[2] - max_drop_mm
    if prior is not None:                       # orbit the camera about the board centre it sees now
        T_bf0 = pose_to_matrix(state["flange_pose"])
        T_cb0, _ = board_pose(det0, cfg.board, prior.K, prior.dist)
        P = (T_bf0 @ prior.X @ T_cb0 @ np.r_[cfg.board.center_m, 1])[:3]
        poses = [(label, prims.tip_from_flange(_pose(T))) for label, T in camera_view_poses(T_bf0, P, prior.X)]
        floor = start[2] - 3 * max_drop_mm      # orbiting tilts the camera, so the tip moves more
        log(f"wrist: board centre ~{[round(v) for v in P * 1000]} mm; orbiting the camera around it")
    else:
        poses = view_poses(start, pivot)
        log(f"wrist: board ~{pivot * 1000:.0f} mm from the camera; pivoting around it")

    # the start view is already in hand: never send a zero-length move (18 Nov: alarms on one)
    views = [{"label": "start", "image": image0, "flange_pose": state["flange_pose"], "detection": det0.to_dict()}]
    skipped = []
    log(f"   0 start: ✓ {len(det0)} corners")
    for i, (label, pose) in enumerate(poses[1:], 1):
        if len(views) >= max_views:
            break
        if pose[2] < floor:
            skipped.append({"pose": label, "why": f"fingertip below {floor:.0f} mm (start {start[2]:.0f})"})
            log(f"  {i:2d} {label}: skipped — fingertip would go below {floor:.0f} mm")
            continue
        try:
            prims._ik(prims.flange_from_tip(pose))      # unreachable: skip without moving
            prims.move_to_pose(*pose, linear=True, speed=speed)
        except ValidationError as exc:
            skipped.append({"pose": label, "why": "unreachable" if "reachable" in str(exc) else str(exc)[:120]})
            log(f"  {i:2d} {label}: skipped — {skipped[-1]['why']}")
            continue
        except (SafetyError, RobotError) as exc:
            if prims.get_state()["error"]:
                raise                                   # alarm: stop everything
            skipped.append({"pose": label, "why": str(exc)[:120]})
            continue
        time.sleep(settle_s)
        det, image = grab(i, before=Detection.from_dict(views[-1]["detection"]))
        flange = prims.get_state()["flange_pose"]
        if det is None:
            skipped.append({"pose": label, "why": "board not seen (or no fresh, still frame)"})
            log(f"  {i:2d} {label}: board not seen (or no fresh, still frame)")
            continue
        views.append({"label": label, "image": image, "flange_pose": flange, "detection": det.to_dict()})
        log(f"  {i:2d} {label}: ✓ {len(det)} corners")
    now = prims.get_state()["pose"]
    if max(abs(a - b) for a, b in zip(now[:3], start[:3])) > 0.5 or \
            max(abs((a - b + 180) % 360 - 180) for a, b in zip(now[3:], start[3:])) > 0.2:
        log("wrist: back to the start pose")
        prims.move_to_pose(*start, linear=True, speed=speed)
    return {"views": views, "skipped": skipped, "start_tip": start, "pivot_mm": round(pivot * 1000, 1)}


# --------------------------------------------------------------------------------------------
# solve + checks (no hardware: also used offline on a saved session)
# --------------------------------------------------------------------------------------------

def solve(data: dict, cfg: CalibrationConfig, table_z_mm: float | None = None, session: Path | None = None) -> dict:
    g, spec = cfg.gates, cfg.board
    rep: dict = {"board": {"squares": list(spec.squares), "square_mm": spec.square_mm, "marker_mm": spec.marker_mm,
                           "dictionary": spec.dictionary}, "problems": [], "warnings": []}
    bad = {"wrist": [], "env": []}

    def problem(part: str, text: str) -> None:
        bad[part].append(text)
        rep["problems"].append(f"{part}: {text}")

    # wrist
    views = data["wrist"]["views"]
    dets = [Detection.from_dict(v["detection"]) for v in views]
    rep["wrist_views"] = len(views)
    wrist = None
    if len(views) < g.min_wrist_views:
        problem("wrist", f"only {len(views)} wrist views with the board (need {g.min_wrist_views})")
    else:
        K, dist, rms = calibrate_intrinsics(dets, spec)
        T_bf = [pose_to_matrix(v["flange_pose"]) for v in views]
        he = hand_eye(T_bf, dets, spec, K, dist)
        wrist = {"K": K, "dist": dist, "intrinsics_px": rms, "he": he}
        off = he.T_flange_cam[:3, 3] * 1000
        board = (he.T_base_board[:3, :3] @ spec.center_m + he.T_base_board[:3, 3]) * 1000
        view_deg = float(np.degrees(np.arccos(np.clip(he.T_flange_cam[2, 2], -1, 1))))
        rep["wrist"] = {"intrinsics_rms_px": round(rms, 3), "fx_fy": [round(K[0, 0], 1), round(K[1, 1], 1)],
                        "hand_eye_rms_px": round(he.reproj_px, 3),
                        "worst_view_px": round(max(he.per_view_px), 3),
                        "camera_in_flange_mm": off.round(1).tolist(),
                        "camera_off_tool_axis_mm": round(float(np.hypot(off[0], off[1])), 1),
                        "camera_view_vs_tool_axis_deg": round(view_deg, 1),
                        "board_centre_in_base_mm": board.round(1).tolist(),
                        "board_normal_vs_vertical_deg": round(float(np.degrees(np.arccos(
                            abs(np.clip(he.T_base_board[2, 2], -1, 1))))), 1)}
        if rms > g.max_intrinsics_px:
            problem("wrist", f"intrinsics RMS {rms:.2f} px > {g.max_intrinsics_px}")
        # physical check, independent of the solver: the board lies flat on the table
        if table_z_mm is not None and data.get("board_on_table", True):
            off = float(board[2] - table_z_mm)
            rep["wrist"]["board_above_table_mm"] = round(off, 1)
            if abs(off) > g.max_board_off_table_mm:
                problem("wrist", f"the board comes out {off:+.0f} mm off the table (limit ±{g.max_board_off_table_mm:g})")
            if rep["wrist"]["board_normal_vs_vertical_deg"] > g.max_board_tilt_deg:
                problem("wrist", f"the board comes out tilted {rep['wrist']['board_normal_vs_vertical_deg']}° "
                                 f"(limit {g.max_board_tilt_deg:g}°)")
        if he.reproj_px > g.max_hand_eye_px:
            problem("wrist", f"hand-eye RMS {he.reproj_px:.2f} px > {g.max_hand_eye_px} "
                                   "(robot pose and image disagree: board moved, wrong board size, or bad views)")

    # env
    env = data.get("env") or {}
    env_dets = [Detection.from_dict(f["detection"]) for f in env.get("frames", []) if f.get("detection")]
    env_dets = [d for d in env_dets if len(d) >= g.min_env_corners]
    T_base_env = None
    if not env_dets or "K" not in env:
        n = max([len(f["detection"]["ids"]) for f in env.get("frames", []) if f.get("detection")], default=0)
        problem("env", f"the D455 does not see the board well enough (best frame {n} corners, need "
                               f"{g.min_env_corners}): lean the board toward the D455 or bring it closer")
    else:
        K_env = np.array(env["K"])
        poses = [board_pose(d, spec, K_env) for d in env_dets]
        T_env_board = average_poses([T for T, _ in poses])
        px = float(np.median([e for _, e in poses]))
        spread = max(np.linalg.norm(T[:3, 3] - T_env_board[:3, 3]) for T, _ in poses) * 1000
        rep["env"] = {"frames_used": len(env_dets), "corners": int(np.median([len(d) for d in env_dets])),
                      "reproj_px": round(px, 3), "frame_spread_mm": round(float(spread), 1),
                      "board_distance_mm": round(float(np.linalg.norm(T_env_board[:3, :3] @ spec.center_m
                                                                      + T_env_board[:3, 3])) * 1000, 1)}
        if px > g.max_env_px:
            problem("env", f"board pose reprojection {px:.2f} px > {g.max_env_px}")
        if session is not None:
            depth_file = session / env["frames"][0]["image"].replace(".jpg", "_depth.npy")
            if depth_file.exists():
                dmm = depth_check_mm(env_dets[0], spec, poses[0][0], np.load(depth_file) / 1000.0)
                rep["env"]["depth_minus_board_mm"] = dmm
                if dmm is not None and abs(dmm) > g.max_env_depth_mm:
                    problem("env", f"D455 depth and board pose disagree by {dmm:.0f} mm "
                                           "(wrong square size, or the depth is not aligned to colour)")
        if wrist is not None:
            T_base_env = env_from_board(wrist["he"].T_base_board, T_env_board)
            rep["env"]["camera_in_base_mm"] = (T_base_env[:3, 3] * 1000).round(1).tolist()

    rep["checks"] = measured_checks(cfg, T_base_env, wrist["he"] if wrist else None, table_z_mm)
    for c in rep["checks"]:
        if not c["ok"]:
            text = f"{c['name']}: solved {c['solved_mm']} mm vs measured {c['measured_mm']} mm"
            rep["warnings"].append(text) if c.get("soft") else problem("env", text)
    if T_base_env is None and not bad["env"]:
        problem("env", "needs the wrist calibration (the board's place in the base frame)")
    # each camera stands on its own: the wrist can be accepted while the D455 is not
    rep["wrist_accepted"] = wrist is not None and not bad["wrist"]
    rep["env_accepted"] = rep["wrist_accepted"] and T_base_env is not None and not bad["env"]
    rep["accepted"] = rep["wrist_accepted"] and rep["env_accepted"]
    rep["_solution"] = {"T_base_env": T_base_env if rep["env_accepted"] else None,
                        "wrist": wrist if rep["wrist_accepted"] else None}
    return rep


def measured_checks(cfg: CalibrationConfig, T_base_env, he: HandEyeResult | None, table_z_mm) -> list[dict]:
    """Solved geometry vs the tape measure. Soft checks only warn (rough measurement)."""
    m, tol, out = cfg.measured, cfg.gates.tolerance_mm, []

    def add(name, solved, key, soft=False):
        if key in m and solved is not None:
            out.append({"name": name, "solved_mm": round(float(solved), 1), "measured_mm": float(m[key]),
                        "ok": bool(abs(solved - float(m[key])) <= tol), **({"soft": True} if soft else {})})

    if T_base_env is not None:
        cam = T_base_env[:3, 3] * 1000
        add("D455 to robot base (horizontal)", np.hypot(cam[0], cam[1]), "env_cam_to_base_mm")
        if table_z_mm is not None:
            add("D455 height above the table", cam[2] - table_z_mm, "env_cam_height_mm")
        if he is not None:
            board = (he.T_base_board[:3, :3] @ cfg.board.center_m + he.T_base_board[:3, 3]) * 1000
            add("D455 to board centre", np.linalg.norm(board - cam), "env_cam_to_board_mm", soft=True)
    if he is not None:
        off = he.T_flange_cam[:3, 3] * 1000
        add("wrist camera off the tool axis", np.hypot(off[0], off[1]), "wrist_cam_offset_mm", soft=True)
        if "base_to_board_mm" in m:
            board = (he.T_base_board[:3, :3] @ cfg.board.center_m + he.T_base_board[:3, 3]) * 1000
            add("robot base to board centre (horizontal)", np.hypot(board[0], board[1]), "base_to_board_mm", soft=True)
    return out


# --------------------------------------------------------------------------------------------
# save
# --------------------------------------------------------------------------------------------

def save(rep: dict, env_path: Path, stamp: str, scene=None, result_file: Path = CALIBRATION_FILE,
         wrist_file: Path = WRIST_FILE) -> None:
    """Write the accepted calibration (env HandEye file the Scene reads, wrist HandEye, full result)."""
    sol = rep["_solution"]
    if sol["T_base_env"] is not None:
        backup(env_path, stamp)
        HandEye(sol["T_base_env"], eye_in_hand=False).save(env_path)
        if scene is not None:
            scene.env_calibration = HandEye(sol["T_base_env"], eye_in_hand=False)
    w = sol["wrist"]
    if w is not None:
        backup(wrist_file, stamp)
        HandEye(w["he"].T_flange_cam, eye_in_hand=True).save(wrist_file)
    result = {k: v for k, v in rep.items() if not k.startswith("_")}
    result["created"] = stamp
    if w is not None:
        result["wrist_intrinsics"] = {"K": w["K"].tolist(), "dist": w["dist"].tolist()}
        result["T_flange_wrist"] = w["he"].T_flange_cam.tolist()
        result["T_base_board"] = w["he"].T_base_board.tolist()
    if sol["T_base_env"] is not None:
        result["T_base_env"] = sol["T_base_env"].tolist()
    backup(result_file, stamp)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def public(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


# --------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------

def run(prims, scene, cfg: CalibrationConfig, env_path: Path, speed: int = 10, max_views: int = 16,
        apply: bool = True, prior: Prior | None = None, log: Callable[[str], None] = print) -> dict:
    stamp = _stamp()
    session = SESSIONS_DIR / stamp
    session.mkdir(parents=True, exist_ok=True)
    data = {"stamp": stamp, "board": dict(cfg.board.__dict__), "measured": cfg.measured,
            "table_z_mm": prims.table_z_mm}
    data["env"] = capture_env(scene, cfg, session, log=log)
    data["wrist"] = capture_wrist(prims, scene, cfg, session, speed=speed, max_views=max_views, prior=prior,
                                  log=log)
    (session / "views.json").write_text(json.dumps(data, indent=1, default=list) + "\n", encoding="utf-8")
    rep = solve(data, cfg, prims.table_z_mm, session)
    rep["session"] = str(session)
    rep["skipped_poses"] = len(data["wrist"]["skipped"])
    rep["start_tip"] = [round(v, 2) for v in data["wrist"]["start_tip"]]    # where to look from next time
    rep["saved"] = [k for k in ("wrist", "env") if rep[f"{k}_accepted"]] if apply else []
    if rep["saved"]:
        save(rep, env_path, stamp, scene)
    (session / "report.json").write_text(json.dumps(public(rep), indent=1) + "\n", encoding="utf-8")
    return rep


def solve_session(session: Path, cfg: CalibrationConfig) -> dict:
    """Solve a saved session again (offline, no robot)."""
    data = json.loads((session / "views.json").read_text(encoding="utf-8"))
    rep = solve(data, cfg, data.get("table_z_mm"), session)
    rep["session"] = str(session)
    return rep


def last_session(result_file: Path = CALIBRATION_FILE) -> Path | None:
    """The session directory of the last saved calibration (to aim the next run from), if it is still there."""
    try:
        session = json.loads(result_file.read_text(encoding="utf-8")).get("session")
    except (OSError, ValueError):
        return None
    path = Path(session) if session else None
    if path is not None and not path.is_absolute():
        path = SESSIONS_DIR.parent.parent / path
    return path if path is not None and (path / "views.json").exists() else None


def last_start_tip(result_file: Path = CALIBRATION_FILE) -> list[float] | None:
    """Fingertip pose the last saved calibration started from (the wrist camera saw the board there)."""
    try:
        return json.loads(result_file.read_text(encoding="utf-8")).get("start_tip")
    except (OSError, ValueError):
        return None
