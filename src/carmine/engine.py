"""Assembles masks, pigment ops, and landmark smoothing into a full pipeline.

`apply_look` runs a single still image through every product in a `Look` in
a fixed order chosen so later products layer correctly over earlier ones
(e.g. blush and highlighter go down before eyeshadow so the eye region isn't
dimmed by a cheek effect bleeding across, and lipstick/eyeliner go last since
they're the highest-contrast, most texture-erasing products). `VideoEngine`
wraps that per-frame pipeline with a `OneEuroFilter` over the landmark
stream so a Look painted on live video doesn't shimmer with per-frame
detector jitter.
"""

from __future__ import annotations

import numpy as np

from . import masks, pigment
from .filters import OneEuroFilter
from .landmarks import FaceLandmarker, NoFaceError, validate_image
from .look import Look, Product

__all__ = ["apply_look", "VideoEngine"]

# Lazily constructed on first use by apply_look() when no landmarks are
# supplied -- FaceLandmarker construction loads a model file, so a
# module-level singleton avoids paying that cost on every call.
_default_landmarker: FaceLandmarker | None = None


def _get_default_landmarker() -> FaceLandmarker:
    global _default_landmarker
    if _default_landmarker is None:
        _default_landmarker = FaceLandmarker()
    return _default_landmarker


def _apply_product(
    image: np.ndarray,
    mask: np.ndarray,
    product: Product,
    lightness_pull: float,
) -> np.ndarray:
    """Tint `image` with `product` inside `mask`, skipping zero-intensity products."""
    if product.intensity <= 0:
        return image
    color = pigment.parse_hex_color(product.color)
    return pigment.tint(image, mask, color, product.intensity, lightness_pull)


def apply_look(
    image_bgr: np.ndarray,
    look: Look,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Paint every product in `look` onto `image_bgr` and return a new array.

    Args:
        image_bgr: BGR uint8 image of shape (H, W, 3).
        look: The `Look` to apply.
        landmarks: Precomputed (478, 2) landmarks. If omitted, they're
            detected from `image_bgr` with a cached module-level
            `FaceLandmarker`.

    Returns:
        A new BGR uint8 array -- never the same object as `image_bgr`, even
        when `look` is entirely zero-intensity (an all-zero look still
        yields a bit-identical copy).

    Raises:
        ValueError: If `look` is not a `Look`, or `image_bgr` fails
            `validate_image`.
        NoFaceError: If landmarks aren't supplied and none can be detected.
    """
    if not isinstance(look, Look):
        raise ValueError(f"look must be a carmine.look.Look, got {type(look).__name__}")
    validate_image(image_bgr)

    if landmarks is None:
        landmarks = _get_default_landmarker().detect(image_bgr)

    shape = image_bgr.shape[:2]
    out = image_bgr.copy()

    # Skin smoothing first so every later effect paints on top of it.
    if look.smoothing > 0:
        out = pigment.smooth(out, masks.skin_mask(landmarks, shape), look.smoothing)

    # Blush before highlighter/eyeshadow/brows so cheek color sits under
    # the higher-contrast features layered on top of it.
    out = _apply_product(out, masks.blush_mask(landmarks, shape), look.blush, 0.15)

    # Highlighter: tint toward the color, then a gloss finish scaled by
    # intensity gives it a sheen tint alone can't produce.
    if look.highlighter.intensity > 0:
        h_mask = masks.highlighter_mask(landmarks, shape)
        out = _apply_product(out, h_mask, look.highlighter, 0.10)
        out = pigment.finish_gloss(out, h_mask, strength=look.highlighter.intensity * 0.5)

    out = _apply_product(out, masks.eyeshadow_mask(landmarks, shape), look.eyeshadow, 0.30)
    out = _apply_product(out, masks.brow_mask(landmarks, shape), look.brows, 0.20)

    # Lipstick: lightness_pull and finish both depend on the requested
    # finish, so this can't reuse the generic _apply_product helper as-is.
    if look.lipstick.intensity > 0:
        lip_mask = masks.lip_mask(landmarks, shape)
        color = pigment.parse_hex_color(look.lipstick.color)
        lightness_pull = 0.30 if look.lipstick.finish == "matte" else 0.35
        out = pigment.tint(out, lip_mask, color, look.lipstick.intensity, lightness_pull)
        if look.lipstick.finish == "matte":
            out = pigment.finish_matte(out, lip_mask, strength=0.35)
        elif look.lipstick.finish == "gloss":
            out = pigment.finish_gloss(out, lip_mask, strength=look.lipstick.intensity)
        # satin: tint alone, no finish pass.

    # Eyeliner is a flat, hard-edged paint rather than a texture-preserving
    # tint, and goes last as the highest-contrast, most opaque product.
    if look.eyeliner.intensity > 0:
        eyeliner_mask = masks.eyeliner_mask(landmarks, shape)
        color = pigment.parse_hex_color(look.eyeliner.color)
        out = pigment.paint(out, eyeliner_mask, color, look.eyeliner.intensity)

    return out


class VideoEngine:
    """Applies a fixed `Look` to a stream of video frames.

    Landmarks are detected per frame with `FaceLandmarker.detect_video`
    (which requires monotonically increasing timestamps) and optionally
    smoothed with a `OneEuroFilter` before `apply_look` paints them, so a
    still face doesn't visibly shimmer from per-frame detector jitter.
    """

    def __init__(self, look: Look, smooth_landmarks: bool = True) -> None:
        if not isinstance(look, Look):
            raise ValueError(f"look must be a carmine.look.Look, got {type(look).__name__}")
        self.look = look
        self.smooth_landmarks = smooth_landmarks
        self._landmarker = FaceLandmarker()
        self._filter = OneEuroFilter() if smooth_landmarks else None

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Paint `self.look` onto one video frame.

        Args:
            frame_bgr: BGR uint8 frame of shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds, strictly
                increasing across successive calls.

        Returns:
            A new BGR uint8 array. If no face is detected, an unmodified
            copy of `frame_bgr` is returned and the landmark filter (if
            any) is reset so a later good frame doesn't get smoothed
            against stale, pre-gap state.
        """
        try:
            landmarks = self._landmarker.detect_video(frame_bgr, timestamp_ms)
        except NoFaceError:
            if self._filter is not None:
                self._filter.reset()
            return frame_bgr.copy()

        if self._filter is not None:
            landmarks = self._filter(landmarks, timestamp_ms / 1000.0)

        return apply_look(frame_bgr, self.look, landmarks=landmarks)
