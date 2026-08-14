"""Pigment application in CIELAB space.

Painting a flat RGB color straight onto a masked region (or blending it in
additively) either erases skin texture or blows out brightness. Every op
here does its work in Lab: chroma (a/b) moves toward the target color while
lightness (L) mostly keeps the original per-pixel detail, so tinted skin
still looks like skin instead of a sticker. ``paint`` is the deliberate
exception -- it collapses the region to a flat color, which is exactly what
a hard-edged product like eyeliner needs.

The finish functions (``finish_matte``, ``finish_gloss``) reshape the L
channel's distribution within a mask to fake a product's surface behavior:
matte damps local highlights toward their blurred neighborhood, gloss lifts
the brightest highlights further above the rest.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse a ``#RRGGBB`` (or bare ``RRGGBB``) string into an (R, G, B) tuple.

    Raises:
        ValueError: `value` is not a string, or doesn't match the pattern.
        The offending value is embedded in the message.
    """
    if not isinstance(value, str):
        raise ValueError(f"color must be a '#RRGGBB' string, got {value!r}")
    match = _HEX_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid hex color {value!r}, expected '#RRGGBB'")
    hex_digits = match.group(1)
    return (
        int(hex_digits[0:2], 16),
        int(hex_digits[2:4], 16),
        int(hex_digits[4:6], 16),
    )


def _to_lab(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _from_lab(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


def _color_to_lab(color_rgb: tuple[int, int, int]) -> np.ndarray:
    swatch = np.array([[color_rgb[::-1]]], dtype=np.float32) / 255.0  # RGB -> BGR
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)[0, 0]


def _restore_untouched(out: np.ndarray, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy original pixels back wherever the mask is zero.

    The Lab round-trip (float conversion, color-space matrix multiply,
    conversion back to uint8) drifts pixels by a few sRGB levels even at
    weight 0, purely from floating-point rounding. Callers rely on
    unmasked pixels staying bit-identical to the input, so we paper over
    that drift explicitly rather than chase it out of the color math.
    """
    untouched = mask <= 0
    out[untouched] = image_bgr[untouched]
    return out


def tint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    intensity: float,
    lightness_pull: float = 0.35,
) -> np.ndarray:
    """Pull the masked region's chroma toward `color_rgb` in Lab space.

    `intensity` scales the whole effect. `lightness_pull` caps how much of
    the target's own lightness bleeds in -- kept well under 1 so shading
    and highlights already in the image survive the tint.
    """
    if intensity <= 0:
        return image_bgr.copy()
    lab = _to_lab(image_bgr)
    target = _color_to_lab(color_rgb)
    weight = (mask * float(intensity))[..., None]
    lightness_weight = weight * lightness_pull
    lab[..., 1:] += (target[1:] - lab[..., 1:]) * weight
    lab[..., :1] += (target[0] - lab[..., :1]) * lightness_weight
    out = _from_lab(lab)
    return _restore_untouched(out, image_bgr, mask)


def paint(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    intensity: float,
) -> np.ndarray:
    """Alpha-blend a flat pigment over the masked region.

    Unlike `tint`, this discards the underlying texture entirely -- the
    right behavior for a hard-edged product like eyeliner, where covering
    what's underneath is the point rather than a side effect.
    """
    if intensity <= 0:
        return image_bgr.copy()
    color = np.array(color_rgb[::-1], dtype=np.float32)
    weight = (mask * float(intensity))[..., None]
    out = image_bgr.astype(np.float32) * (1 - weight) + color * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth(image_bgr: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Blend in an edge-preserving bilateral filter within the masked region."""
    if amount <= 0:
        return image_bgr.copy()
    softened = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=45, sigmaSpace=9)
    weight = (mask * float(amount))[..., None]
    out = image_bgr.astype(np.float32) * (1 - weight) + softened.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def finish_matte(image_bgr: np.ndarray, mask: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Flatten micro-highlights within the mask toward their local blur.

    Pulls each masked pixel's L value toward a sigma=5 Gaussian blur of L,
    which damps small specular highlights while leaving the broad shading
    that reads as skin/lip shape intact.
    """
    if strength <= 0:
        return image_bgr.copy()
    lab = _to_lab(image_bgr)
    l_channel = lab[..., 0]
    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=5)
    weight = mask * float(strength)
    lab[..., 0] = l_channel + (blurred - l_channel) * weight
    out = _from_lab(lab)
    return _restore_untouched(out, image_bgr, mask)


def finish_gloss(image_bgr: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Boost the brightest highlights within the mask to fake specular shine.

    Builds a highlight map from where each masked pixel's L falls between
    the 75th and 99th percentile of L within the mask, then adds to L in
    proportion to that map, `mask`, and `strength`. Regions too small or
    too flat to have a meaningful highlight spread are left untouched.
    """
    if strength <= 0:
        return image_bgr.copy()
    lab = _to_lab(image_bgr)
    l_channel = lab[..., 0]
    inside = mask > 0.5
    if int(np.count_nonzero(inside)) < 10:
        return image_bgr.copy()
    masked_l = l_channel[inside]
    p75, p99 = np.percentile(masked_l, [75, 99])
    spread = p99 - p75
    if spread < 1e-6:
        return image_bgr.copy()
    highlight = np.clip((l_channel - p75) / spread, 0.0, 1.0)
    lab[..., 0] = np.clip(
        l_channel + highlight * mask * 18.0 * float(strength), 0, 100
    )
    out = _from_lab(lab)
    return _restore_untouched(out, image_bgr, mask)
