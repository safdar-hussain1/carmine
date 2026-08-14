"""Soft, face-scaled product masks.

Each mask function takes the (478, 2) landmark array and the target image
shape and returns a float32 array in [0, 1] with that shape: 1.0 means
"apply the product at full strength here", 0.0 means "leave untouched",
and values in between blend the two. Every size that matters -- feather
radii, line thickness, ellipse axes -- is expressed as a fraction of the
interocular distance (`geometry.interocular_distance`) rather than a raw
pixel count, so a mask built for a 200px thumbnail and one built for a
4000px portrait of the same face place makeup in the same relative spot
with the same relative softness.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import regions
from .geometry import interocular_distance


def _polygon(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Rasterize a closed polygon as a hard 0/1 mask."""
    mask = np.zeros(shape, dtype=np.float32)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1.0)
    return mask


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a mask's edges with a Gaussian kernel sized from sigma."""
    if sigma <= 0:
        return mask
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), sigma)


def lip_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Lipstick region: outer mouth contour minus the inner opening.

    Filling only the outer contour would paint straight across an open
    mouth; subtracting the inner ring keeps color off teeth and the
    mouth's interior regardless of how open it is.
    """
    iod = interocular_distance(landmarks)
    outer = _polygon(shape, landmarks[regions.LIPS_OUTER])
    inner = _polygon(shape, landmarks[regions.LIPS_INNER])
    mask = np.clip(outer - inner, 0.0, 1.0)
    return _feather(mask, sigma=iod * 0.02)


def eyeshadow_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Eyeshadow region: upper lid up to 60% of the way to the brow.

    Built per eye from the upper lash-line arc and points interpolated
    toward the matching brow arc, so the shape stays proportional to the
    eye/brow gap on any face. On top of the polygon we apply a crease
    gradient: full strength at the lash line, fading to 0.35 at the top
    of the polygon, since real eyeshadow application is heaviest at the
    lash line and lightest near the brow bone. The gradient is computed
    per row within each eye's own bounding box so it's independent of
    image size, then the eye opening itself is cut out before the final
    feather.
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
        eye_mask = _polygon(shape, poly)

        # Crease gradient: 1.0 at the lash-line row, 0.35 at the upper
        # polygon-edge row, clipped outside that span. Rows are found from
        # the mean y of the lid arc and of the upper (brow-ward) edge.
        lash_row = float(np.mean(lid[:, 1]))
        crease_row = float(np.mean(upper[:, 1]))
        row_min = int(np.floor(min(lash_row, crease_row)))
        row_max = int(np.ceil(max(lash_row, crease_row)))
        row_min = max(row_min, 0)
        row_max = min(row_max, shape[0] - 1)
        if row_max >= row_min:
            rows = np.arange(row_min, row_max + 1)
            # np.interp needs increasing xp; crease_row is above (smaller y)
            # than lash_row in image coordinates, so order accordingly.
            factors = np.interp(rows, [crease_row, lash_row], [0.35, 1.0])
            gradient = np.ones(shape[0], dtype=np.float32)
            gradient[row_min : row_max + 1] = factors
            eye_mask = eye_mask * gradient[:, None]

        mask = np.maximum(mask, eye_mask)

    for eye_idx in (regions.RIGHT_EYE, regions.LEFT_EYE):
        mask = np.clip(mask - _polygon(shape, landmarks[eye_idx]), 0.0, 1.0)
    return _feather(mask, sigma=iod * 0.045)


def eyeliner_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A narrow stroke traced along each upper lash line, winged at the corner."""
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
    """A feathered ellipse centered on each cheek, sized relative to the face.

    Multiplying by a feathered face-oval mask keeps the ellipse's soft
    edge from spilling onto hair or the background near the jawline,
    which a plain ellipse fill would not prevent.
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


def brow_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Brow-filling region: each eyebrow's lower and upper edges as a polygon.

    Per brow, the polygon is `*_BROW_LOWER` followed by `*_BROW_UPPER`
    reversed, which traces the brow's perimeter (the two arcs share a
    junction landmark at the outer end, which cv2.fillPoly tolerates
    fine as a repeated vertex).
    """
    iod = interocular_distance(landmarks)
    mask = np.zeros(shape, dtype=np.float32)
    pairs = [
        (regions.RIGHT_BROW_LOWER, regions.RIGHT_BROW_UPPER),
        (regions.LEFT_BROW_LOWER, regions.LEFT_BROW_UPPER),
    ]
    for lower_idx, upper_idx in pairs:
        poly = np.concatenate(
            [landmarks[lower_idx], landmarks[upper_idx][::-1]], axis=0
        )
        mask = np.maximum(mask, _polygon(shape, poly))
    return _feather(mask, sigma=iod * 0.015)


def highlighter_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Highlighter region: cheekbone crests plus the bridge of the nose.

    Each cheekbone arc and the nose bridge are drawn as thick feathered
    lines rather than filled polygons, since highlighter is applied as a
    strip along these features rather than over an enclosed area.
    """
    iod = interocular_distance(landmarks)
    mask = np.zeros(shape, dtype=np.float32)

    cheek_thickness = max(1, int(round(iod * 0.10)))
    for arc_idx in (regions.RIGHT_CHEEKBONE, regions.LEFT_CHEEKBONE):
        pts = np.round(landmarks[arc_idx]).astype(np.int32)
        cv2.polylines(
            mask, [pts], isClosed=False, color=1.0, thickness=cheek_thickness
        )
    mask = _feather(mask, sigma=iod * 0.06)

    nose_thickness = max(1, int(round(iod * 0.05)))
    nose_mask = np.zeros(shape, dtype=np.float32)
    pts = np.round(landmarks[regions.NOSE_BRIDGE]).astype(np.int32)
    cv2.polylines(
        nose_mask, [pts], isClosed=False, color=1.0, thickness=nose_thickness
    )
    nose_mask = _feather(nose_mask, sigma=iod * 0.06)

    return np.clip(np.maximum(mask, nose_mask), 0.0, 1.0)


def skin_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Face oval minus eyes, lips and brows -- the smoothing region."""
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
