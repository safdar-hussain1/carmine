"""Behavioral tests for face-scaled product masks."""

import numpy as np
import pytest

from carmine import masks, regions
from carmine.geometry import interocular_distance
from carmine.landmarks import FaceLandmarker


@pytest.fixture(scope="module")
def astronaut_landmarks():
    from skimage import data

    rgb = data.astronaut()
    bgr = rgb[:, :, ::-1].copy()
    landmarker = FaceLandmarker()
    return landmarker.detect(bgr)


@pytest.fixture(scope="module")
def shape(astronaut_landmarks):
    return (512, 512)


MASK_FUNCS = [
    "lip_mask",
    "eyeshadow_mask",
    "eyeliner_mask",
    "blush_mask",
    "brow_mask",
    "highlighter_mask",
    "skin_mask",
]


class TestMaskBasics:
    @pytest.mark.parametrize("name", MASK_FUNCS)
    def test_mask_is_float32_in_unit_range_and_correct_shape(
        self, name, astronaut_landmarks, shape
    ):
        mask_fn = getattr(masks, name)
        mask = mask_fn(astronaut_landmarks, shape)
        assert mask.dtype == np.float32
        assert mask.shape == shape
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.max() > 0.0


def _open_mouth_landmarks(base_landmarks):
    """Build a synthetic landmark set with LIPS_INNER pulled open into a ring.

    Starts from real detected landmarks (so every other index used by the
    masks is realistic) and displaces the inner-lip ring vertically to open
    a real gap between the upper and lower inner lip points.
    """
    lm = base_landmarks.copy()
    inner = regions.LIPS_INNER
    n = len(inner)
    half = n // 2
    # First half of the ring runs across the top of the opening, second half
    # across the bottom (matches the ordered contour in regions.py).
    for i, idx in enumerate(inner):
        if i < half:
            lm[idx, 1] -= 6.0
        else:
            lm[idx, 1] += 6.0
    return lm


class TestLipMask:
    def test_near_zero_at_mouth_opening_centroid(self, astronaut_landmarks, shape):
        lm = _open_mouth_landmarks(astronaut_landmarks)
        mask = masks.lip_mask(lm, shape)
        centroid = lm[regions.LIPS_INNER].mean(axis=0)
        x, y = int(round(centroid[0])), int(round(centroid[1]))
        assert mask[y, x] < 0.05


# Eyeliner is a thin line rather than a filled region: its area is
# length x thickness, and integer-rounded thickness at small feather
# sigmas doesn't track the quadratic area scaling that filled polygons
# and ellipses show. studio-v1 excluded it from the scaling check for
# the same reason; the remaining masks are all filled/thick-stroke
# regions and do scale quadratically.
SCALING_MASK_FUNCS = [name for name in MASK_FUNCS if name != "eyeliner_mask"]


class TestScaling:
    @pytest.mark.parametrize("name", SCALING_MASK_FUNCS)
    def test_area_scales_with_doubled_landmarks_and_shape(
        self, name, astronaut_landmarks, shape
    ):
        mask_fn = getattr(masks, name)
        base_mask = mask_fn(astronaut_landmarks, shape)
        base_area = float(base_mask.sum())

        doubled_landmarks = astronaut_landmarks * 2.0
        doubled_shape = (shape[0] * 2, shape[1] * 2)
        doubled_mask = mask_fn(doubled_landmarks, doubled_shape)
        doubled_area = float(doubled_mask.sum())

        assert base_area > 0
        ratio = doubled_area / base_area
        assert 3.4 <= ratio <= 4.6, f"{name}: area ratio {ratio} out of range"


class TestBlushMask:
    def test_zero_outside_face_oval(self, astronaut_landmarks, shape):
        mask = masks.blush_mask(astronaut_landmarks, shape)
        # Corner of the image is guaranteed to be outside the face oval.
        assert mask[0, 0] == 0.0
        assert mask[0, shape[1] - 1] == 0.0


class TestBrowMask:
    def test_covers_brow_arc_midpoints(self, astronaut_landmarks, shape):
        mask = masks.brow_mask(astronaut_landmarks, shape)
        for arc in (regions.RIGHT_BROW_LOWER, regions.LEFT_BROW_LOWER):
            pts = astronaut_landmarks[arc]
            mid = pts[len(pts) // 2]
            x, y = int(round(mid[0])), int(round(mid[1]))
            assert mask[y, x] > 0.3, f"brow mask too weak at {arc} midpoint"

    def test_near_zero_at_forehead_center(self, astronaut_landmarks, shape):
        iod = interocular_distance(astronaut_landmarks)
        right_inner = astronaut_landmarks[regions.RIGHT_BROW_LOWER[0]]
        left_inner = astronaut_landmarks[regions.LEFT_BROW_LOWER[0]]
        midpoint = (right_inner + left_inner) / 2.0
        forehead = midpoint.copy()
        forehead[1] -= 0.5 * iod
        x, y = int(round(forehead[0])), int(round(forehead[1]))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask = masks.brow_mask(astronaut_landmarks, shape)
            assert mask[y, x] < 0.05


class TestHighlighterMask:
    def test_near_zero_at_chin(self, astronaut_landmarks, shape):
        mask = masks.highlighter_mask(astronaut_landmarks, shape)
        chin = astronaut_landmarks[152]
        x, y = int(round(chin[0])), int(round(chin[1]))
        assert mask[y, x] < 0.05


class TestEyeshadowCreaseGradient:
    def test_lash_line_rows_brighter_than_brow_edge_rows(
        self, astronaut_landmarks, shape
    ):
        mask = masks.eyeshadow_mask(astronaut_landmarks, shape)
        for lid_idx, brow_idx in (
            (regions.RIGHT_EYE_UPPER, regions.RIGHT_BROW_LOWER),
            (regions.LEFT_EYE_UPPER, regions.LEFT_BROW_LOWER),
        ):
            lid = astronaut_landmarks[lid_idx]
            brow = astronaut_landmarks[brow_idx]
            lash_row = int(round(np.mean(lid[:, 1])))
            crease_row = int(round(np.mean(brow[:, 1])))
            x_min = int(np.min(lid[:, 0])) - 5
            x_max = int(np.max(lid[:, 0])) + 5
            x_min = max(x_min, 0)
            x_max = min(x_max, shape[1])

            near_lash = mask[max(lash_row - 2, 0) : lash_row + 3, x_min:x_max]
            near_crease = mask[
                max(crease_row - 2, 0) : crease_row + 3, x_min:x_max
            ]
            assert near_lash.size > 0 and near_crease.size > 0
            assert near_lash.mean() > near_crease.mean()
