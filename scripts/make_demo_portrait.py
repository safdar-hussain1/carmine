"""Generates the single demo portrait the web app ships (web/public/demo/portrait.jpg).

The source is `skimage.data.astronaut()` -- a NASA photograph of astronaut
Eileen Collins, a work of the U.S. federal government and therefore in the
public domain. It is the same image every figure in reports/figures uses, so
the sample face in the browser is the face the measurements were taken on.

The crop is driven by the detected face hull rather than hardcoded pixels: it
centres the face, pads out to a 3:4 portrait frame, and resizes to a size the
mirror can display without visible softness.

Usage:
    python scripts/make_demo_portrait.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carmine.landmarks import FaceLandmarker  # noqa: E402

OUT_PATH = ROOT / "web" / "public" / "demo" / "portrait.jpg"
OUT_SIZE = (720, 960)  # width, height -- 3:4
PAD_FACTOR = 1.35  # how much of the frame beyond the face hull to keep
JPEG_QUALITY = 92


def _face_box(landmarks: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def build() -> Path:
    image = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]

    landmarker = FaceLandmarker()
    landmarks = landmarker.detect(image)
    if landmarks is None:
        raise RuntimeError("no face detected in the source portrait")

    x0, y0, x1, y1 = _face_box(landmarks)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    face_h = (y1 - y0) * PAD_FACTOR * 2

    # Fit a 3:4 window around the face, then clamp it inside the source. The
    # window is shrunk (not shifted first) when it cannot fit, so the face
    # stays centred rather than drifting to a corner.
    crop_h = min(face_h, height)
    crop_w = crop_h * OUT_SIZE[0] / OUT_SIZE[1]
    if crop_w > width:
        crop_w = width
        crop_h = crop_w * OUT_SIZE[1] / OUT_SIZE[0]

    left = int(round(min(max(cx - crop_w / 2, 0), width - crop_w)))
    top = int(round(min(max(cy - crop_h / 2, 0), height - crop_h)))
    crop = image[top : top + int(round(crop_h)), left : left + int(round(crop_w))]

    resized = cv2.resize(crop, OUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"wrote {path.relative_to(ROOT)}")
