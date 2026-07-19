import cv2
import numpy as np
import pytest

from virtual_makeup import masks, regions


@pytest.fixture(scope="module")
def shape(astronaut_bgr):
    return astronaut_bgr.shape[:2]


ALL_MASKS = [
    masks.lip_mask,
    masks.eyeshadow_mask,
    masks.eyeliner_mask,
    masks.blush_mask,
    masks.skin_mask,
]


@pytest.mark.parametrize("fn", ALL_MASKS)
def test_masks_are_normalized_float(fn, astronaut_landmarks, shape):
    m = fn(astronaut_landmarks, shape)
    assert m.shape == shape
    assert m.dtype == np.float32
    assert 0.0 <= m.min() and m.max() <= 1.0
    assert m.sum() > 0, "mask is empty"


def test_lip_mask_covers_lips_not_mouth_interior(astronaut_landmarks, shape):
    m = masks.lip_mask(astronaut_landmarks, shape)
    # Mid upper lip: halfway between the outer (0) and inner (13) top points.
    lip_pt = (astronaut_landmarks[0] + astronaut_landmarks[13]) / 2
    x, y = np.round(lip_pt).astype(int)
    assert m[y, x] > 0.5
    # Mouth interior: centroid of the inner ring must stay clean —
    # painting the mouth interior is the classic naive-fill mistake.
    interior = astronaut_landmarks[regions.LIPS_INNER].mean(axis=0)
    x, y = np.round(interior).astype(int)
    assert m[y, x] < 0.25


def test_eyeshadow_stays_out_of_the_eye(astronaut_landmarks, shape):
    m = masks.eyeshadow_mask(astronaut_landmarks, shape)
    for eye in (regions.RIGHT_EYE, regions.LEFT_EYE):
        center = astronaut_landmarks[eye].mean(axis=0)
        x, y = np.round(center).astype(int)
        assert m[y, x] < 0.3
    # But it is present between the lid and the brow.
    for upper, brow in (
        (regions.RIGHT_EYE_UPPER, regions.RIGHT_BROW_LOWER),
        (regions.LEFT_EYE_UPPER, regions.LEFT_BROW_LOWER),
    ):
        lid_mid = astronaut_landmarks[upper][len(upper) // 2]
        brow_mid = astronaut_landmarks[brow][len(brow) // 2]
        probe = lid_mid + (brow_mid - lid_mid) * 0.3
        x, y = np.round(probe).astype(int)
        assert m[y, x] > 0.3


def test_masks_scale_with_image_size(astronaut_bgr, landmarker):
    """Same face at 2x resolution -> mask area ~4x (scale invariance).

    Naive filters use fixed pixel offsets (eyeshadow) and fixed radii
    (blush circles of r=20), which do not scale.
    """
    big = cv2.resize(astronaut_bgr, None, fx=2.0, fy=2.0)
    lm_small = landmarker.detect(astronaut_bgr)
    lm_big = landmarker.detect(big)
    for fn in (masks.lip_mask, masks.blush_mask, masks.eyeshadow_mask):
        area_small = fn(lm_small, astronaut_bgr.shape[:2]).sum()
        area_big = fn(lm_big, big.shape[:2]).sum()
        ratio = area_big / area_small
        assert 3.0 < ratio < 5.5, f"{fn.__name__}: area ratio {ratio:.2f}, expected ~4"


def test_skin_mask_excludes_features(astronaut_landmarks, shape):
    m = masks.skin_mask(astronaut_landmarks, shape)
    lips_center = astronaut_landmarks[regions.LIPS_OUTER].mean(axis=0)
    x, y = np.round(lips_center).astype(int)
    assert m[y, x] < 0.4
    for eye in (regions.RIGHT_EYE, regions.LEFT_EYE):
        cx, cy = np.round(astronaut_landmarks[eye].mean(axis=0)).astype(int)
        assert m[cy, cx] < 0.4
