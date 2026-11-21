"""What the agent sees: camera frames -> objects with robot-base coordinates.

    scene = Scene.from_config()
    scene.find("red")       -> [SceneObject(label="red", base_xyz_mm=[x, y, z], ...)]
    scene.look("env")       -> {"objects": [...], "image_jpeg_b64": "..."}

Frame sources (no camera is opened here, so it never fights the UI for a device):
  * env   -> D455 depth helper (colour + aligned depth + intrinsics)   scripts/run_ui.sh --depth
  * other -> latest JPEG from the running UI (colour only)             /api/cameras/<name>/frame.jpg

3D needs depth (env) and a calibration (scripts/calibrate.sh cameras);
without them objects are still reported in pixels.
"""
from __future__ import annotations

import math

import base64
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np

from common.config import PerceptionConfig, load_perception_config
from common.errors import RobotError
from drivers.camera import Frame, RealSenseRemoteCamera
from perception.detection import detect_colored_objects
from perception.transforms import HandEye, deproject_pixel


@dataclass
class SceneObject:
    label: str
    pixel: tuple[int, int]               # u, v in the colour image
    bbox: tuple[int, int, int, int]      # x, y, w, h
    area_px: float
    depth_m: float | None = None         # distance from the camera
    camera_xyz_m: list[float] | None = None
    base_xyz_mm: list[float] | None = None   # robot base frame — what motion skills need
    angle_deg: float | None = None       # edge angle of the blob in the image (0–90, minAreaRect);
                                         # absent when not measurable (small blob, or cut by the image edge)
    cut_by_edge: bool = False            # only part of the object is in the image

    def summary(self) -> dict:
        d = asdict(self)
        for key in ("camera_xyz_m", "base_xyz_mm"):
            if d[key] is not None:
                d[key] = [round(v, 4 if key == "camera_xyz_m" else 1) for v in d[key]]
        if d["depth_m"] is not None:
            d["depth_m"] = round(d["depth_m"], 4)
        if d["angle_deg"] is None:
            d.pop("angle_deg")
        if not d["cut_by_edge"]:
            d.pop("cut_by_edge")
        return d


class UIFrameSource:
    """Colour frames from the UI server's camera hub (no depth)."""

    def __init__(self, ui_url: str, name: str, timeout: float = 2.0):
        self.url, self.timeout = f"{ui_url.rstrip('/')}/api/cameras/{name}/frame.jpg", timeout

    def read(self) -> Frame:
        import cv2
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                data = np.frombuffer(r.read(), np.uint8)
                captured = r.headers.get("X-Capture-Monotonic-Ns")
                sequence = r.headers.get("X-Frame-Sequence")
                stream = r.headers.get("X-Frame-Stream")
        except (urllib.error.URLError, OSError) as exc:
            raise RobotError(f"camera frame unavailable ({self.url}) — is scripts/run_ui.sh running? {exc}") from exc
        return Frame(color=cv2.imdecode(data, cv2.IMREAD_COLOR),
                     captured_ns=int(captured) if captured else None,
                     sequence_id=int(sequence) if sequence else None, stream_id=stream)


class Scene:
    def __init__(self, cfg: PerceptionConfig, sources: dict, env_calibration: HandEye | None = None,
                 workspace=None):
        self.cfg = cfg
        # Robot workspace box (common.config.WorkspaceLimits): once calibrated, objects
        # outside it (shelves, people in the background) are dropped.
        self.workspace = workspace
        # colour -> last env sighting with robot coordinates; used when the arm hides the object
        self.last_seen: dict[str, dict] = {}
        # Objects higher than this (robot base Z, mm) are parts of the robot itself
        # (coloured tape on the arm/wrist camera), not things on the table. None = off.
        self.object_z_max: float | None = None
        self.last_frames: dict = {}                    # camera -> (frame, time) of the last look
        self.wrist_overlay = None                      # perception.wrist_overlay.WristOverlay (fingertips in the wrist picture)
        self.sources = sources                       # camera name -> object with read() -> Frame
        self.env_calibration = env_calibration       # None = not calibrated yet
        self.ranges = {k: [tuple(map(tuple, r)) for r in v] for k, v in cfg.colors.items()} or None

    @classmethod
    def from_config(cls, cfg: PerceptionConfig | None = None, cameras=("env", "wrist"), workspace=None) -> "Scene":
        cfg = cfg or load_perception_config()
        sources = {}
        for name in cameras:
            if name == "env":
                sources[name] = _LazyHelper(cfg.env_helper_url)
            else:
                sources[name] = UIFrameSource(cfg.ui_url, name)
        calib = HandEye.load(cfg.calibration_path) if cfg.calibration_path.exists() else None
        return cls(cfg, sources, calib, workspace)

    @property
    def colors(self) -> list[str]:
        return sorted(self.ranges or {})

    def observe(self, camera: str = "env") -> Frame:
        if camera not in self.sources:
            raise RobotError(f"unknown camera '{camera}' (have {sorted(self.sources)})")
        return self.sources[camera].read()

    # -- geometry ---------------------------------------------------------
    def _depth_at(self, depth_m: np.ndarray, u: int, v: int) -> float | None:
        n = self.cfg.depth_window_px
        patch = depth_m[max(v - n, 0):v + n + 1, max(u - n, 0):u + n + 1]
        valid = patch[patch > 0]
        return float(np.median(valid)) if valid.size else None

    def locate(self, frame: Frame, u: int, v: int) -> tuple[float | None, list | None, list | None]:
        """Pixel -> (depth m, camera xyz m, base xyz mm); None where not computable."""
        if frame.depth_m is None or frame.intrinsics is None:
            return None, None, None
        depth = self._depth_at(frame.depth_m, u, v)
        if depth is None:
            return None, None, None
        k = frame.intrinsics
        cam = deproject_pixel(u, v, depth, k.fx, k.fy, k.cx, k.cy)
        base = self.env_calibration.camera_to_base(cam).tolist() if self.env_calibration else None
        return depth, cam.tolist(), base

    # -- queries the agent uses -------------------------------------------
    def find(self, color: str, camera: str = "env", frame: Frame | None = None) -> list[SceneObject]:
        frame = frame or self.observe(camera)
        found = []
        for det in detect_colored_objects(frame.color, [color], self.cfg.min_area_px, self.ranges):
            u, v = det.center
            depth, cam, base = self.locate(frame, u, v) if camera == "env" else (None, None, None)
            if base is not None and not self.in_workspace(base):
                continue
            if base is not None and self.object_z_max is not None and base[2] > self.object_z_max:
                continue                                     # on the robot, not on the table
            found.append(SceneObject(det.label, (u, v), det.bbox, det.area, depth, cam, base,
                                     getattr(det, "angle_deg", None), getattr(det, "cut_by_edge", False)))
        if found and found[0].base_xyz_mm is not None:
            self.last_seen[color] = {**found[0].summary(), "seen_at": time.time()}
        return found

    def last_seen_info(self, color: str) -> dict | None:
        """Where `color` was last seen in robot coordinates, and how long ago."""
        seen = self.last_seen.get(color)
        if seen is None:
            return None
        return {"base_xyz_mm": seen["base_xyz_mm"], "seconds_ago": round(time.time() - seen["seen_at"], 1)}

    def in_workspace(self, xyz_mm: list[float], margin_mm: float = 50.0) -> bool:
        if self.workspace is None:
            return True
        bounds = (self.workspace.x, self.workspace.y, self.workspace.z)
        r_max = getattr(self.workspace, "max_radius_mm", None)
        r = (xyz_mm[0] ** 2 + xyz_mm[1] ** 2) ** 0.5
        if r_max is not None and r > r_max + margin_mm:
            return False
        sector = getattr(self.workspace, "sector_deg", None)
        if sector is not None:
            # outside the working sector (+ margin_mm along the arc), or next to the base axis (the robot's
            # own base: its E-stop button came back as a red object at (84, -18), 85 mm out, on 20 Nov)
            if r < getattr(self.workspace, "sector_min_radius_mm", 150.0):
                return False
            lo, hi = sector
            ang = math.degrees(math.atan2(xyz_mm[1], xyz_mm[0]))
            span, off = (hi - lo) % 360, (ang - lo) % 360
            if off > span and min(off - span, 360 - off) > math.degrees(margin_mm / r):
                return False
        return all(lo - margin_mm <= c <= hi + margin_mm for c, (lo, hi) in zip(xyz_mm, bounds))

    def look(self, camera: str = "env", zoom: list[int] | None = None, **_ignored) -> dict:
        """The camera picture and nothing else: no detection here (20 Nov, operator: Astra finds the
        objects in the image itself). zoom=[x, y, w, h] (full-image pixels) returns that region enlarged.
        The frame is kept so depth_at answers for exactly the pixels of this picture."""
        frame = self.observe(camera)
        if camera == "env" and frame.depth_m is not None and self.depth_frames > 1:
            frame = self._median_depth(camera, frame)
        self.last_frames[camera] = (frame, time.time())
        h, w = frame.color.shape[:2]
        result = {"camera": camera, "image_size": [w, h], "has_depth": frame.depth_m is not None,
                  "calibrated": camera == "env" and self.env_calibration is not None,
                  "frame": frame.sequence_id, "captured_ns": frame.captured_ns,
                  "sequence_id": frame.sequence_id, "stream_id": frame.stream_id,
                  "legacy": frame.captured_ns is None}
        picture = frame.color
        if camera == "wrist" and self.wrist_overlay is not None:
            try:
                tips = self.wrist_overlay()["fingertips_px"]
            except Exception as exc:  # noqa: BLE001 — no robot state: the picture without the marks
                result["fingertips_px"] = f"not available ({exc})"[:120]
            else:
                picture = frame.color.copy()
                self.wrist_overlay.draw(picture, tips)
                result["fingertips_px"] = tips
        if zoom and (int(zoom[2]) <= 0 or int(zoom[3]) <= 0 or
                     (int(zoom[0]) <= 0 and int(zoom[1]) <= 0 and int(zoom[2]) >= w and int(zoom[3]) >= h)):
            zoom = None                                  # [0,0,0,0] / whole image: no crop (was an 8x8 crop)
        if zoom:
            x, y, zw, zh = (int(v) for v in zoom)
            x, y = max(0, min(x, w - 8)), max(0, min(y, h - 8))
            zw, zh = max(8, min(zw, w - x)), max(8, min(zh, h - y))
            result["zoom"] = [x, y, zw, zh]
            result["image_jpeg_b64"] = self.encode_image(picture[y:y + zh, x:x + zw], [], upscale=True)
        else:
            result["image_jpeg_b64"] = self.encode_image(picture, [])
        return result

    depth_frames = 1          # env look: per-pixel median over N frames. 20 Nov: frame-to-frame noise is only
                              # ±0–1.3 mm on the cubes; the error is systematic, so 1 (no extra frames)

    def _median_depth(self, camera: str, frame):
        """The picture's frame with its depth replaced by the per-pixel median over depth_frames frames
        (zeros = no depth are ignored). 20 Nov: single-frame D455 depth moved the same cube by ±5 mm."""
        from dataclasses import replace
        stack = [frame.depth_m]
        for _ in range(self.depth_frames - 1):
            try:
                f = self.observe(camera)
            except Exception:  # noqa: BLE001 — fewer frames, still a median
                break
            if f.depth_m is not None and f.depth_m.shape == frame.depth_m.shape:
                stack.append(f.depth_m)
        d = np.stack(stack).astype(np.float32)
        d[~np.isfinite(d) | (d <= 0)] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(d, axis=0)
        return replace(frame, depth_m=np.nan_to_num(med, nan=0.0).astype(np.float32))

    def depth_at(self, pixels: list, window_px: int = 2, plane_z: float | None = None) -> dict:
        """Depth of the env camera at the given pixels of the LAST env look (same frame as that picture):
        per pixel the median of the valid depth in a (2*window_px+1)^2 window, the 3D point in the camera
        frame and, when calibrated, in the robot base frame. No detection, no filtering."""
        if "env" not in self.last_frames:
            raise ValueError("look at the env camera first: depth_at answers for the pixels of that picture")
        frame, at = self.last_frames["env"]
        if frame.depth_m is None or frame.intrinsics is None:
            raise ValueError("the env picture has no depth (D455 depth helper not running)")
        if not isinstance(pixels, list) or not 1 <= len(pixels) <= 50:
            raise ValueError("pixels: a list of 1–50 [u, v] pairs in full-image coordinates")
        h, w = frame.depth_m.shape[:2]
        k, n = frame.intrinsics, max(0, min(int(window_px), 10))
        out = []
        for p in pixels:
            u, v = int(round(float(p[0]))), int(round(float(p[1])))
            if not (0 <= u < w and 0 <= v < h):
                out.append({"pixel": [u, v], "error": f"outside the image {w}x{h}"})
                continue
            win = frame.depth_m[max(0, v - n):v + n + 1, max(0, u - n):u + n + 1]
            valid = win[np.isfinite(win) & (win > 0)]
            row = {"pixel": [u, v], "valid_fraction": round(valid.size / win.size, 2)}
            if plane_z is not None and self.env_calibration is not None:
                ray = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
                T = np.asarray(self.env_calibration.T, float)
                o, dirb = T[:3, 3] * (1000.0 if np.abs(T[:3, 3]).max() < 5 else 1.0), T[:3, :3] @ ray
                if abs(dirb[2]) > 1e-9 and (float(plane_z) - o[2]) / dirb[2] > 0:
                    row["at_plane_xyz_mm"] = [round(float(c), 1) for c in o + dirb * (float(plane_z) - o[2]) / dirb[2]]
            if valid.size:
                d = float(np.median(valid))
                row["depth_spread_mm"] = round(float(np.percentile(valid, 90) - np.percentile(valid, 10)) * 1000, 1)
                cam = deproject_pixel(u, v, d, k.fx, k.fy, k.cx, k.cy)
                row.update(depth_mm=round(d * 1000, 1), camera_xyz_mm=[round(float(c) * 1000, 1) for c in cam])
                if self.env_calibration is not None:
                    row["base_xyz_mm"] = [round(float(c), 1) for c in self.env_calibration.camera_to_base(cam).tolist()]
            out.append(row)
        # the same picture with the asked pixels numbered: in a bounded context the look image is gone by the
        # time the depths arrive (20 Nov 164149: look -> depth_at -> look again, same pixels, 3x)
        import cv2
        img = frame.color.copy()
        for i, row in enumerate(out):
            u, v = row["pixel"]
            if 0 <= u < w and 0 <= v < h:
                cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
                cv2.putText(img, str(i), (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        return {"camera": "env", "frame": frame.sequence_id, "picture_age_s": round(time.time() - at, 1),
                "points": out, "image_jpeg_b64": self.encode_image(img, [])}

    def find_report(self, color: str, camera: str = "env") -> dict:
        """Tool-friendly result: current detections + last sighting when hidden."""
        frame = self.observe(camera)
        objects = [o.summary() for o in self.find(color, camera, frame)]
        report = {"objects": objects, "captured_ns": frame.captured_ns, "sequence_id": frame.sequence_id,
                  "stream_id": frame.stream_id, "legacy": frame.captured_ns is None}
        if not objects:
            report["last_seen"] = self.last_seen_info(color)
            report["hint"] = ("not visible now — often hidden by the arm. Use last_seen, move the arm "
                              "away, or call search_object.")
        return report

    def encode_image(self, bgr: np.ndarray, objects: list[dict], upscale: bool = False) -> str:
        """Annotated, downsized JPEG for the model (boxes + labels drawn on)."""
        import cv2
        img = bgr.copy()
        if upscale and img.shape[1] < self.cfg.image_max_width:
            s = self.cfg.image_max_width / img.shape[1]
            img = cv2.resize(img, (self.cfg.image_max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            objects = [{**o, "bbox": [int(b * s) for b in o["bbox"]]} for o in objects]
        for o in objects:
            x, y, w, h = o["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, o["label"], (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if img.shape[1] > self.cfg.image_max_width:
            s = self.cfg.image_max_width / img.shape[1]
            img = cv2.resize(img, (self.cfg.image_max_width, int(img.shape[0] * s)))
        return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


class _LazyHelper:
    """Connect to the depth helper on first use, so a Scene can exist before it runs."""

    def __init__(self, url: str):
        self.url, self._cam = url, None

    def read(self) -> Frame:
        if self._cam is None:
            self._cam = RealSenseRemoteCamera(self.url)
        return self._cam.read()
