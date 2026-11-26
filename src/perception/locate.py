"""Height-connected top faces in aligned depth, measured in robot base millimetres."""
from __future__ import annotations

import math

import cv2
import numpy as np


def _points(depth, intrinsics, calibration, x0, y0):
    v, u = np.indices(depth.shape)
    k = intrinsics
    camera = np.stack([(u + x0 - k.cx) * depth / k.fx,
                       (v + y0 - k.cy) * depth / k.fy, depth], axis=-1)
    origin = calibration.camera_to_base(np.zeros(3))
    axes = np.stack([calibration.camera_to_base(axis) - origin for axis in np.eye(3)], axis=1)
    return camera @ axes.T + origin


def _extent(points):
    rect = cv2.minAreaRect(np.asarray(points[:, :2], np.float32))
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    long, short = float(lengths.max()), float(lengths.min())
    edge = edges[int(np.argmax(lengths))]
    yaw = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    period = 90 if long <= 1.15 * short else 180
    yaw = (yaw + period / 2) % period - period / 2
    if yaw <= -period / 2:
        yaw += period
    return [long, short], yaw


def _filled(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def _measure(points, valid, mask, seed, bounds, image_shape, table_z, ring_px,
             max_object_size_mm, workspace):
    x0, y0, x1, y1 = bounds
    h, w = image_shape
    top = points[mask]
    centre = np.r_[np.mean(top[:, :2], axis=0), np.median(top[:, 2])]
    footprint, yaw = _extent(top)
    filled = _filled(mask)
    ring = cv2.dilate(filled, np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) > 0
    support = points[ring & ~filled.astype(bool) & valid, 2]
    support = support[support <= centre[2] + 6]
    support_z = float(np.median(support)) if support.size >= 10 else table_z
    height = float(centre[2] - support_z) if support_z is not None else None
    spread = float(np.percentile(top[:, 2], 90) - np.percentile(top[:, 2], 10))
    quality = []
    if len(top) < 30:
        quality.append("few points")
    ys, xs = np.nonzero(mask)
    if np.any(xs + x0 == 0) or np.any(xs + x0 == w - 1) or np.any(ys + y0 == 0) or np.any(ys + y0 == h - 1):
        quality.append("cut by image edge")
    if ((x0 > 0 and mask[:, 0].any()) or (x1 < w and mask[:, -1].any()) or
            (y0 > 0 and mask[0].any()) or (y1 < h and mask[-1].any())):
        quality.append("mask touches ROI border")
    if spread > 4:
        quality.append("large spread")
    if footprint[1] > 0 and footprint[0] <= max_object_size_mm and footprint[0] > 1.4 * footprint[1]:
        # review 23 Nov: cubes packed side by side in a box have the same top height and merge into one face
        quality.append(f"footprint not square ({footprint[0]:.0f} x {footprint[1]:.0f} mm): "
                       "possibly several objects touching")
    if support.size < 10:
        quality.append("support not found")
    if workspace is not None:
        from actions.validation import workspace_violations
        if workspace_violations(centre.tolist(), workspace):
            quality.append("outside workspace")
    return {"ok": True, "pixel": seed, "top_center_mm": centre.tolist(),
            "support_z_mm": support_z, "height_mm": height, "footprint_mm": footprint,
            "yaw_deg": yaw, "n_points": len(top),
            "valid_fraction": float(np.count_nonzero(valid & (filled > 0)) / np.count_nonzero(filled)),
            "depth_spread_mm": spread,
            "kind": "object" if height is not None and height >= 10 and footprint[0] <= max_object_size_mm else "surface",
            "quality": quality}


def locate_object(frame, calibration, pixel, table_z=None, *, roi_radius_px=None,
                  seed_radius_px=4, top_tolerance_mm=6.0, ring_px=8,
                  max_object_size_mm=80.0, workspace=None) -> dict:
    """Locate a top face near full-image [u,v]. No image or hardware is modified.

    Depth is metres; all returned lengths and table_z are robot base millimetres.
    Yaw is the long XY edge angle, modulo 90 for aspect ratio <=1.15, else 180.
    Faces longer than max_object_size_mm are surfaces even when raised. Without
    a measured support, table_z is the fallback and quality records the omission.
    roi_radius_px=None measures the full image, including large place surfaces.
    """
    try:
        return _locate(frame, calibration, pixel, table_z, roi_radius_px, seed_radius_px,
                       top_tolerance_mm, ring_px, max_object_size_mm, workspace)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, OverflowError, cv2.error) as exc:
        return {"ok": False, "error": str(exc)}


def _locate(frame, calibration, pixel, table_z, radius, seed_radius, tolerance, ring_px, max_size, workspace):
    if calibration is None or calibration.eye_in_hand:
        raise ValueError("env camera is uncalibrated: a fixed camera-to-base calibration is required")
    T = np.asarray(calibration.T, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("invalid camera-to-base calibration")
    depth = np.asarray(frame.depth_m, float)
    if depth.ndim != 2 or not depth.size or frame.intrinsics is None:
        raise ValueError("env frame has no aligned depth or intrinsics")
    k = frame.intrinsics
    if not np.isfinite([k.fx, k.fy, k.cx, k.cy]).all() or min(k.fx, k.fy) <= 0:
        raise ValueError("invalid camera intrinsics")
    if len(pixel) != 2:
        raise ValueError("pixel must be a full-image [u, v] pair")
    u, v = (int(round(float(c))) for c in pixel)
    h, w = depth.shape
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"pixel outside the image {w}x{h}")
    seed_radius, ring_px = int(seed_radius), int(ring_px)
    if seed_radius < 0 or ring_px < 1 or not np.isfinite([tolerance, max_size]).all() or min(tolerance, max_size) <= 0:
        raise ValueError("invalid segmentation parameters")
    if table_z is not None:
        table_z = float(table_z)
        if not math.isfinite(table_z):
            raise ValueError("table_z must be finite")
    radius = max(w, h) if radius is None else int(radius)
    if radius < 1:
        raise ValueError("roi_radius_px must be positive")
    x0, y0, x1, y1 = max(0, u - radius), max(0, v - radius), min(w, u + radius + 1), min(h, v + radius + 1)
    crop = depth[y0:y1, x0:x1]
    valid = np.isfinite(crop) & (crop > 0)
    points = _points(np.where(valid, crop, 0), k, calibration, x0, y0)
    points[~valid] = np.nan
    ys, xs = np.indices(crop.shape)
    near = valid & (abs(xs + x0 - u) <= seed_radius) & (abs(ys + y0 - v) <= seed_radius)
    if not near.any():
        raise ValueError("no depth at pixel")
    seed_z = float(np.percentile(points[near, 2], 75))
    candidates = near & (abs(points[..., 2] - seed_z) <= tolerance)
    if not candidates.any():
        raise ValueError("no consistent top height near pixel")
    distance = np.where(candidates, (xs + x0 - u) ** 2 + (ys + y0 - v) ** 2, np.inf)
    sv, su = np.unravel_index(np.argmin(distance), crop.shape)
    mask = (valid & (abs(points[..., 2] - seed_z) <= tolerance)).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    mask = labels == labels[sv, su]
    return _measure(points, valid, mask, [int(su + x0), int(sv + y0)], (x0, y0, x1, y1),
                    depth.shape, table_z, ring_px, max_size, workspace)
