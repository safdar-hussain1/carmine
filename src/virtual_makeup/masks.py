"""Soft, face-scaled region masks.

Every mask is a float32 array in [0, 1] with the image's height/width.
All sizes (feather radii, liner thickness, blush axes) are expressed as
fractions of the interocular distance, so the same look renders
identically on a 400 px selfie and a 4000 px portrait — the original
2024 project used fixed pixel offsets and broke on anything but the one
test image.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import regions
from .landmarks import interocular_distance


def _polygon(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1.0)
    return mask


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), sigma)


def lip_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Lip region: outer contour minus the mouth opening.

    The subtraction is what keeps lipstick off teeth when the mouth is
    open — the original project filled landmarks 48-68 as one polygon
    and painted the whole mouth interior.
    """
    iod = interocular_distance(landmarks)
    outer = _polygon(shape, landmarks[regions.LIPS_OUTER])
    inner = _polygon(shape, landmarks[regions.LIPS_INNER])
    mask = np.clip(outer - inner, 0.0, 1.0)
    return _feather(mask, sigma=iod * 0.02)


def eyeshadow_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Upper-eyelid region between the lash line and the eyebrow.

    Built per eye from the upper eyelid arc and points interpolated 60%
    of the way toward the brow, then feathered heavily so the color
    fades out instead of ending at a polygon edge.
    """
    iod = interocular_distance(landmarks)
    mask = np.zeros(shape, dtype=np.float32)
    pairs = [
        (regions.RIGHT_EYE_UPPER, regions.RIGHT_BROW_LOWER),
        (regions.LEFT_EYE_UPPER, regions.LEFT_BROW_LOWER),
    ]
    for lid_idx, brow_idx in pairs:
        lid = landmarks[lid_idx]
        brow = landmarks[brow_idx]
        # Resample the brow arc so each lid point has a matching brow point.
        t = np.linspace(0, 1, len(lid))
        src_t = np.linspace(0, 1, len(brow))
        brow_matched = np.stack(
            [np.interp(t, src_t, brow[:, 0]), np.interp(t, src_t, brow[:, 1])], axis=1
        )
        upper = lid + (brow_matched - lid) * 0.60
        poly = np.concatenate([lid, upper[::-1]], axis=0)
        mask = np.maximum(mask, _polygon(shape, poly))
        # Keep shadow out of the eye itself.
    for eye_idx in (regions.RIGHT_EYE, regions.LEFT_EYE):
        mask = np.clip(mask - _polygon(shape, landmarks[eye_idx]), 0.0, 1.0)
    return _feather(mask, sigma=iod * 0.045)


def eyeliner_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Thin line along each upper lash line with a small outer wing."""
    iod = interocular_distance(landmarks)
    thickness = max(1, int(round(iod * 0.012)))
    mask = np.zeros(shape, dtype=np.float32)
    for arc_idx in (regions.RIGHT_EYE_UPPER, regions.LEFT_EYE_UPPER):
        arc = landmarks[arc_idx].copy()
        # Extend the outer corner along the last segment, lifted upward,
        # to form the wing.
        direction = arc[-1] - arc[-2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            unit = direction / norm + np.array([0.0, -0.45], dtype=np.float32)
            unit /= np.linalg.norm(unit)
            wing = arc[-1] + unit * iod * 0.06
            arc = np.vstack([arc, wing])
        cv2.polylines(
            mask,
            [np.round(arc).astype(np.int32)],
            isClosed=False,
            color=1.0,
            thickness=thickness,
        )
    return _feather(mask, sigma=max(1.0, iod * 0.006))


def blush_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Soft ellipse on each mid-cheek, axes scaled to the face.

    Clipped to the face oval so the feathered falloff cannot bleed onto
    hair or background at the cheek silhouette.
    """
    iod = interocular_distance(landmarks)
    axes = (int(round(iod * 0.20)), int(round(iod * 0.14)))
    mask = np.zeros(shape, dtype=np.float32)
    for idx, angle in ((regions.RIGHT_CHEEK, -15), (regions.LEFT_CHEEK, 15)):
        center = tuple(np.round(landmarks[idx]).astype(int))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
    mask = _feather(mask, sigma=iod * 0.10)
    face = _feather(_polygon(shape, landmarks[regions.FACE_OVAL]), sigma=iod * 0.02)
    return mask * face


def skin_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Face oval minus eyes, brows and mouth — the smoothing region."""
    iod = interocular_distance(landmarks)
    mask = _polygon(shape, landmarks[regions.FACE_OVAL])
    exclusions = [
        regions.LIPS_OUTER,
        regions.RIGHT_EYE,
        regions.LEFT_EYE,
    ]
    for idx in exclusions:
        excl = _feather(_polygon(shape, landmarks[idx]), sigma=iod * 0.02)
        mask = np.clip(mask - excl, 0.0, 1.0)
    for brow_idx in (regions.RIGHT_BROW_LOWER, regions.LEFT_BROW_LOWER):
        pts = np.round(landmarks[brow_idx]).astype(np.int32)
        cv2.polylines(mask, [pts], False, 0.0, thickness=max(3, int(iod * 0.05)))
    return _feather(mask, sigma=iod * 0.03)
