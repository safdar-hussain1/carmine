"""Color blending that recolors skin without destroying its texture.

The original 2024 project used ``cv2.addWeighted(image, 1, mask, 0.6, 0)``
— an *additive* overlay that blows out brightness and shifts every channel
— or hard pixel assignment (``image[mask] = color``), which erases pores,
highlights and shading entirely.

Here makeup is applied in CIELAB: the a/b (chroma) channels are pulled
toward the target color while L (lightness) keeps most of the original
detail, so lips still look like lips after recoloring.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_lab(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def _color_to_lab(color_rgb: tuple[int, int, int]) -> np.ndarray:
    swatch = np.array([[color_rgb[::-1]]], dtype=np.float32) / 255.0  # RGB -> BGR
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)[0, 0]


def tint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    intensity: float,
    lightness_pull: float = 0.35,
) -> np.ndarray:
    """Blend the masked region toward ``color_rgb`` in Lab space.

    ``intensity`` scales the whole effect; ``lightness_pull`` controls how
    much of the target color's lightness is adopted (kept well below 1 so
    the original shading — texture — survives).
    """
    if intensity <= 0:
        return image_bgr.copy()
    lab = _to_lab(image_bgr)
    target = _color_to_lab(color_rgb)
    w = (mask * float(intensity))[..., None]
    l_w = w * lightness_pull
    lab[..., 1:] += (target[1:] - lab[..., 1:]) * w
    lab[..., :1] += (target[0] - lab[..., :1]) * l_w
    out = _from_lab(lab)
    # The Lab round-trip drifts pixels by +/-1-3 even where w == 0; keep
    # untouched regions bit-identical to the input.
    untouched = mask <= 0
    out[untouched] = image_bgr[untouched]
    return out


def paint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """Straight alpha blend of a flat pigment — used for eyeliner, where
    covering the underlying texture is the point."""
    if intensity <= 0:
        return image_bgr.copy()
    color = np.array(color_rgb[::-1], dtype=np.float32)
    w = (mask * float(intensity))[..., None]
    out = image_bgr.astype(np.float32) * (1 - w) + color * w
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth(image_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Edge-preserving skin smoothing blended in only within the mask."""
    if amount <= 0:
        return image_bgr.copy()
    softened = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=45, sigmaSpace=9)
    w = (mask * float(amount))[..., None]
    out = image_bgr.astype(np.float32) * (1 - w) + softened.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)
