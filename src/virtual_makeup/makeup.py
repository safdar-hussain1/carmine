"""High-level makeup application pipeline."""

from __future__ import annotations

import numpy as np

from . import blend, masks
from .config import MakeupLook, parse_hex_color
from .landmarks import FaceLandmarker, _validate_image

_default_landmarker: FaceLandmarker | None = None


def _get_landmarker() -> FaceLandmarker:
    global _default_landmarker
    if _default_landmarker is None:
        _default_landmarker = FaceLandmarker()
    return _default_landmarker


def apply_makeup(
    image_bgr: np.ndarray,
    look: MakeupLook,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a full makeup look to a BGR image and return a new image.

    Order matters: smoothing first (it must not blur applied pigment),
    then blush, eyeshadow, lipstick, and eyeliner on top.
    """
    if not isinstance(look, MakeupLook):
        raise ValueError(f"look must be a MakeupLook, got {type(look).__name__}")
    _validate_image(image_bgr)
    if landmarks is None:
        landmarks = _get_landmarker().detect(image_bgr)

    shape = image_bgr.shape[:2]
    out = image_bgr

    if look.smoothing > 0:
        out = blend.smooth(out, masks.skin_mask(landmarks, shape), look.smoothing)
    if look.blush_intensity > 0:
        out = blend.tint(
            out, masks.blush_mask(landmarks, shape),
            parse_hex_color(look.blush_color), look.blush_intensity,
            lightness_pull=0.15,
        )
    if look.eyeshadow_intensity > 0:
        out = blend.tint(
            out, masks.eyeshadow_mask(landmarks, shape),
            parse_hex_color(look.eyeshadow_color), look.eyeshadow_intensity,
            lightness_pull=0.30,
        )
    if look.lipstick_intensity > 0:
        out = blend.tint(
            out, masks.lip_mask(landmarks, shape),
            parse_hex_color(look.lipstick_color), look.lipstick_intensity,
            lightness_pull=0.35,
        )
    if look.eyeliner_intensity > 0:
        out = blend.paint(
            out, masks.eyeliner_mask(landmarks, shape),
            parse_hex_color(look.eyeliner_color), look.eyeliner_intensity,
        )
    return out if out is not image_bgr else image_bgr.copy()
