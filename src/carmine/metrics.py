"""Benchmark metrics for scoring a makeup method's output against its input.

Each photo metric takes a `before` image, an `after` image produced by some
method, and a mask describing where that method was allowed to paint, and
returns a single float. They're deliberately narrow and honest about what
they measure -- no metric here compares against a "ground truth" with-makeup
photo, since no such photo can tell a well-placed edit from a merely
similar-looking one (see the project notebook for why reference-SSIM scoring
was rejected as a protocol).

* `pigment_on_target` -- of all the pixel change between `before` and
  `after`, what fraction of that change (weighted by how much each pixel
  changed, so a few rounding-noise pixels don't count the same as an opaque
  polygon) landed inside the legitimate target region. High is good;
  painting outside the mask (wrong region, background, chin) lowers it.
* `background_untouched` -- fraction of pixels outside the face that stayed
  bit-identical. 1.0 means the method never touched anything off-face.
* `lip_texture_kept` -- Pearson correlation of lip-region lightness (Lab L
  channel) before vs after. A texture-preserving tint keeps this high; an
  opaque flat fill erases the correlation.
* `identity_ssim` -- grayscale structural similarity of the whole image,
  before vs after. Makeup should restyle a photo, not replace it.

`mask_jitter` is a video-stability metric: given a sequence of per-frame
masks (e.g. the eyeshadow mask re-built from each frame's landmarks), it
reports how much the mask moves frame-to-frame -- mean adjacent-frame IoU
(1.0 = masks identical every frame) and mean centroid displacement in
pixels, normalized by interocular distance so it's comparable across face
sizes.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity

__all__ = [
    "pigment_on_target",
    "background_untouched",
    "lip_texture_kept",
    "identity_ssim",
    "mask_jitter",
]

# Threshold used to binarize a soft [0, 1] mask into "counts as this region"
# vs "doesn't", shared across the region-based metrics below. Matches the
# threshold the original protocol used when unioning product masks into a
# single valid-region mask.
_REGION_THRESHOLD = 0.05


def _resize_to_match(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if after.shape != before.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return after


def pigment_on_target(before: np.ndarray, after: np.ndarray, target_mask: np.ndarray) -> float:
    """Fraction of edit energy that landed inside `target_mask`.

    "Edit energy" at a pixel is the summed absolute per-channel difference
    between `before` and `after`; the metric is that energy's sum inside the
    mask divided by its sum everywhere, so a one-count rounding ripple in a
    feathered tail doesn't count the same as an opaque polygon painted in
    the wrong place.

    Args:
        before: BGR uint8 input image.
        after: BGR uint8 output image (resized to match `before` if needed).
        target_mask: Float or bool array of shape `before.shape[:2]`;
            values above 0.05 count as "inside" the legitimate target region.

    Returns:
        A float in [0, 1]. 1.0 if every pixel of change fell inside the
        mask; 1.0 also when nothing changed at all (trivially on-target).
    """
    after = _resize_to_match(before, after)
    energy = np.abs(after.astype(np.int16) - before.astype(np.int16)).sum(axis=2)
    total = float(energy.sum())
    if total <= 0:
        return 1.0
    inside = np.asarray(target_mask) > _REGION_THRESHOLD
    return float(energy[inside].sum() / total)


def background_untouched(before: np.ndarray, after: np.ndarray, face_mask: np.ndarray) -> float:
    """Fraction of pixels outside `face_mask` that are bit-identical.

    Args:
        before: BGR uint8 input image.
        after: BGR uint8 output image (resized to match `before` if needed).
        face_mask: Float or bool array of shape `before.shape[:2]`; values
            above 0.5 mark the face region. Everything else is "background".

    Returns:
        A float in [0, 1]. 1.0 means every background pixel is unchanged.
        1.0 also when the face mask covers the whole image (no background
        pixels to check).
    """
    after = _resize_to_match(before, after)
    face = np.asarray(face_mask) > 0.5
    outside = ~face
    if not np.any(outside):
        return 1.0
    identical = np.all(after[outside] == before[outside], axis=-1)
    return float(identical.mean())


def lip_texture_kept(before: np.ndarray, after: np.ndarray, lip_mask: np.ndarray) -> float:
    """Pearson correlation of lip-region lightness (Lab L) before vs after.

    A texture-preserving tint shifts chroma but leaves the per-pixel L
    detail (highlights, creases, the shape of the lip line) intact, which
    keeps this correlation high. A flat, opaque fill collapses that detail
    to a near-constant value and drives the correlation toward zero.

    Args:
        before: BGR uint8 input image.
        after: BGR uint8 output image (resized to match `before` if needed).
        lip_mask: Float or bool array of shape `before.shape[:2]`; values
            above 0.5 mark the lip region scored.

    Returns:
        A float, typically in [-1, 1]. 0.0 if either side has no lightness
        variance inside the mask (a degenerate, texture-free region on at
        least one side).

    Raises:
        ValueError: If `lip_mask` selects no pixels.
    """
    after = _resize_to_match(before, after)
    inside = np.asarray(lip_mask) > 0.5
    if not np.any(inside):
        raise ValueError("lip_mask selects no pixels")
    l_before = cv2.cvtColor(before, cv2.COLOR_BGR2Lab)[..., 0][inside].astype(float)
    l_after = cv2.cvtColor(after, cv2.COLOR_BGR2Lab)[..., 0][inside].astype(float)
    if l_before.std() < 1e-6 or l_after.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(l_before, l_after)[0, 1])


def identity_ssim(before: np.ndarray, after: np.ndarray) -> float:
    """Grayscale structural similarity of `after` vs `before`.

    Args:
        before: BGR uint8 input image.
        after: BGR uint8 output image (resized to match `before` if needed).

    Returns:
        A float in [-1, 1] (in practice close to [0, 1] for photographic
        images); 1.0 for identical images.
    """
    after = _resize_to_match(before, after)
    return float(
        structural_similarity(
            cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
        )
    )


def _centroid(binary_mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(binary_mask)
    if ys.size == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def mask_jitter(mask_seq: list[np.ndarray], iod: float) -> dict:
    """Frame-to-frame stability of a sequence of masks.

    Each mask is binarized at 0.5, then adjacent frames are compared:
    intersection-over-union of the binary masks, and Euclidean displacement
    of their centroids, normalized by `iod` so the number is comparable
    across face/frame sizes.

    Args:
        mask_seq: A list of at least 2 float or bool masks, all the same
            shape.
        iod: Interocular distance in pixels, used to normalize centroid
            displacement into a face-relative unit.

    Returns:
        A dict with:
            "mean_iou": mean of adjacent-frame IoU across the sequence.
                1.0 means every consecutive pair of masks was pixel-identical
                (after thresholding); pairs where both masks are empty also
                score 1.0 (agreement that nothing is there).
            "centroid_drift": mean of adjacent-frame centroid displacement
                in pixels / `iod`. Frames with an empty mask are skipped for
                this part (there is no centroid to compare); 0.0 if no pair
                has two non-empty masks.

    Raises:
        ValueError: If `mask_seq` has fewer than 2 elements, or `iod` is not
            positive.
    """
    if len(mask_seq) < 2:
        raise ValueError(f"mask_seq needs at least 2 frames, got {len(mask_seq)}")
    if iod <= 0:
        raise ValueError(f"iod must be positive, got {iod}")

    binaries = [np.asarray(m) > 0.5 for m in mask_seq]
    centroids = [_centroid(b) for b in binaries]

    ious = []
    drifts = []
    for a, b, ca, cb in zip(binaries, binaries[1:], centroids, centroids[1:]):
        union = int(np.count_nonzero(a | b))
        if union == 0:
            ious.append(1.0)
        else:
            inter = int(np.count_nonzero(a & b))
            ious.append(inter / union)
        if ca is not None and cb is not None:
            drifts.append(float(np.linalg.norm(ca - cb)) / iod)

    return {
        "mean_iou": float(np.mean(ious)),
        "centroid_drift": float(np.mean(drifts)) if drifts else 0.0,
    }
