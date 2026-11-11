"""Baseline HSV colour detector (red / green / blue blocks).

Good enough for tabletop pick-and-place demos; replace ``detect_colored_objects``
with a learned detector later while keeping the ``Detection`` shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OpenCV HSV: H 0–179. Red wraps around 0, so it has two ranges.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue": [((95, 120, 60), (130, 255, 255))],
}


@dataclass
class Detection:
    label: str
    bbox: tuple[int, int, int, int]   # x, y, w, h in pixels
    center: tuple[int, int]           # u, v
    area: float
    score: float = 1.0


def detect_colored_objects(bgr: np.ndarray, colors: list[str] | None = None,
                           min_area: float = 400.0) -> list[Detection]:
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    found: list[Detection] = []
    for label in colors or list(COLOR_RANGES):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOR_RANGES[label]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            found.append(Detection(label, (x, y, w, h), (x + w // 2, y + h // 2), float(area)))
    return sorted(found, key=lambda d: -d.area)
