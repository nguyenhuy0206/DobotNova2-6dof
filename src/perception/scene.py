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
        except (urllib.error.URLError, OSError) as exc:
            raise RobotError(f"camera frame unavailable ({self.url}) — is scripts/run_ui.sh running? {exc}") from exc
        return Frame(color=cv2.imdecode(data, cv2.IMREAD_COLOR))


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
        if r_max is not None and (xyz_mm[0] ** 2 + xyz_mm[1] ** 2) ** 0.5 > r_max + margin_mm:
            return False
        return all(lo - margin_mm <= c <= hi + margin_mm for c, (lo, hi) in zip(xyz_mm, bounds))

    def look(self, camera: str = "env", colors: list[str] | None = None,
             zoom: list[int] | None = None, image: str = "auto") -> dict:
        """zoom=[x, y, w, h] (full-image pixels): return that region enlarged — small
        objects far from the camera are only a few pixels wide in the full view.
        Object pixels are always in full-image coordinates. ``image`` is read by the
        harness (auto | always | never: whether the picture goes to the model)."""
        frame = self.observe(camera)
        objects = []
        for color in colors or self.colors:
            objects += [o.summary() for o in self.find(color, camera, frame)]
        h, w = frame.color.shape[:2]
        missing = {c: self.last_seen_info(c) for c in (colors or self.colors)
                   if not any(o["label"] == c for o in objects) and c in self.last_seen}
        result = {"camera": camera, "image_size": [w, h], "has_depth": frame.depth_m is not None,
                  "calibrated": camera == "env" and self.env_calibration is not None,
                  "objects": objects, "not_visible_last_seen": missing}
        if zoom and (int(zoom[2]) <= 0 or int(zoom[3]) <= 0 or
                     (int(zoom[0]) <= 0 and int(zoom[1]) <= 0 and int(zoom[2]) >= w and int(zoom[3]) >= h)):
            zoom = None                                  # [0,0,0,0] / whole image: no crop (was an 8x8 crop)
        if zoom:
            x, y, zw, zh = (int(v) for v in zoom)
            x, y = max(0, min(x, w - 8)), max(0, min(y, h - 8))
            zw, zh = max(8, min(zw, w - x)), max(8, min(zh, h - y))
            crop = frame.color[y:y + zh, x:x + zw]
            shifted = [{**o, "bbox": [o["bbox"][0] - x, o["bbox"][1] - y, *o["bbox"][2:]]} for o in objects
                       if x <= o["pixel"][0] < x + zw and y <= o["pixel"][1] < y + zh]
            result["zoom"] = [x, y, zw, zh]
            result["image_jpeg_b64"] = self.encode_image(crop, shifted, upscale=True)
        else:
            result["image_jpeg_b64"] = self.encode_image(frame.color, objects)
        return result

    def find_report(self, color: str, camera: str = "env") -> dict:
        """Tool-friendly result: current detections + last sighting when hidden."""
        objects = [o.summary() for o in self.find(color, camera)]
        report = {"objects": objects}
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
