"""Pure validation + safety checks. No I/O, so it is fully unit-testable."""
from __future__ import annotations

import math

from common.config import JointStepLimits, SpeedLimits, WorkspaceLimits
from common.errors import SafetyError, ValidationError
from common.types import Feedback

JPEG_MAGIC = b"\xff\xd8"


def check_speed(speed: int, limits: SpeedLimits) -> int:
    if not limits.min <= speed <= limits.max:
        raise ValidationError(f"Tốc độ phải trong {limits.min}–{limits.max}%.")
    return speed


def check_joint_step(joint: int, degrees: float, limits: JointStepLimits) -> None:
    if not 1 <= joint <= 6 or not limits.min_deg <= abs(degrees) <= limits.max_deg:
        raise ValidationError(f"Chỉ hỗ trợ J1–J6 với bước {limits.min_deg:g}–{limits.max_deg:g} độ.")


def check_joints(joints: list) -> list[float]:
    if not isinstance(joints, list) or len(joints) != 6 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in joints):
        raise ValidationError("Vị trí lưu phải gồm đúng 6 góc khớp.")
    return [float(v) for v in joints]


def workspace_violations(pose: list[float], ws: WorkspaceLimits) -> dict[str, tuple[float, str]]:
    """Every workspace limit this point breaks: {key: (by how much — mm, or degrees for the sector,
    message)}. Empty = inside."""
    if len(pose) < 3:
        raise ValidationError("Pose phải có ít nhất X, Y, Z.")
    out = {}
    for axis, value, (lo, hi) in zip("XYZ", pose[:3], (ws.x, ws.y, ws.z)):
        if not lo <= value <= hi:
            key = f"{axis.lower()}_{'lo' if value < lo else 'hi'}"
            out[key] = (lo - value if value < lo else value - hi,
                        f"{axis}={value:.1f} mm nằm ngoài workspace [{lo}, {hi}].")
    x, y, z = (float(c) for c in pose[:3])
    r = (x * x + y * y) ** 0.5
    if ws.max_radius_mm is not None and r > ws.max_radius_mm:
        out["radius"] = (r - ws.max_radius_mm,
                         f"{r:.0f} mm from the base axis: outside the reach limit {ws.max_radius_mm:g} mm.")
    if ws.sector_deg is not None and r >= ws.sector_min_radius_mm:
        lo, hi = ws.sector_deg
        ang = math.degrees(math.atan2(y, x))
        span, off = (hi - lo) % 360, (ang - lo) % 360
        if off > span:
            out["sector"] = (min(off - span, 360 - off),
                             f"direction {ang:.1f}° from the base: outside the working sector "
                             f"{lo:g}° → {hi:g}° (counter-clockwise).")
    for i, k in enumerate(ws.keep_out or []):
        (kx, ky), zlo, zhi = k["xy"], *k.get("z", (-1e9, 1e9))
        d = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
        if d < k["radius_mm"] and zlo <= z <= zhi:
            out[f"keep_out{i}"] = (k["radius_mm"] - d,
                                   f"inside the keep-out zone around {k.get('name', 'an obstacle')} "
                                   f"({d:.0f} mm from it, keep ≥ {k['radius_mm']:g} mm).")
    return out


# never tolerated on an escape path: the table floor (z_lo) keeps its own, stricter rule
ESCAPE_TOL = {"sector": 1.0}          # degrees; everything else 5 mm
NO_ESCAPE = {"z_lo"}


def check_pose_in_workspace(pose: list[float], ws: WorkspaceLimits, tolerate: dict | None = None) -> None:
    """Reject Cartesian points (mm, base frame) outside the workspace. tolerate = the violations of
    the START of a path (workspace_violations amounts): a point on that path may break the same limit
    by no more than the start did (+ ESCAPE_TOL), so an arm hand-guided out of the workspace can be
    driven back in — never further out. Targets are always checked with tolerate=None."""
    for key, (amount, message) in workspace_violations(pose, ws).items():
        allowed = (tolerate or {}).get(key)
        if allowed is None or key in NO_ESCAPE or amount > allowed + ESCAPE_TOL.get(key, 5.0):
            raise ValidationError(message)


def check_pose6(pose: list) -> list[float]:
    if not isinstance(pose, list) or len(pose) != 6 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in pose):
        raise ValidationError("Pose phải gồm 6 số: X, Y, Z (mm), Rx, Ry, Rz (độ).")
    return [float(v) for v in pose]


def check_joint_limits(joints: list[float], limits_deg: list[float]) -> None:
    for i, (value, limit) in enumerate(zip(joints, limits_deg), start=1):
        if abs(value) > limit:
            raise ValidationError(f"J{i}={value:.1f}° vượt giới hạn ±{limit}°.")


def check_relative(delta: list[float], max_mm: float, max_deg: float) -> None:
    translation = sum(d * d for d in delta[:3]) ** 0.5
    if translation > max_mm:
        raise ValidationError(f"Dịch chuyển {translation:.1f} mm > tối đa {max_mm} mm mỗi lệnh.")
    if any(abs(d) > max_deg for d in delta[3:]):
        raise ValidationError(f"Xoay tối đa {max_deg}° mỗi lệnh.")


def check_name(name: str, max_len: int = 40) -> str:
    name = str(name).strip()
    if not 1 <= len(name) <= max_len:
        raise ValidationError(f"Tên phải dài 1–{max_len} ký tự.")
    return name


def check_jpeg(binary: bytes) -> bytes:
    if not 512 <= len(binary) <= 4_000_000 or not binary.startswith(JPEG_MAGIC):
        raise ValidationError("Ảnh webcam không hợp lệ hoặc quá lớn.")
    return binary


def require_confirm(confirm: object, message: str = "Cần xác nhận vùng làm việc an toàn.") -> None:
    if confirm is not True:
        raise SafetyError(message)


def require_motion_ready(feedback: Feedback) -> Feedback:
    """Mode 5, enabled, no alarm, not already moving."""
    if not (feedback["mode"] == 5 and feedback["enabled"] == 1
            and feedback["error"] == 0 and feedback["running"] == 0):
        raise SafetyError("Robot phải ở Mode 5, Enabled, không alarm và đang đứng yên trước khi chuyển động.")
    return feedback
