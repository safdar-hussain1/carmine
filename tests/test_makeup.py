import cv2
import numpy as np
import pytest

from virtual_makeup import PRESETS, MakeupLook, apply_makeup, masks, regions
from virtual_makeup.landmarks import NoFaceDetectedError


@pytest.fixture(scope="module")
def classic_result(astronaut_bgr, astronaut_landmarks):
    return apply_makeup(astronaut_bgr, PRESETS["classic"], landmarks=astronaut_landmarks)


class TestInputValidation:
    def test_rejects_non_array(self):
        with pytest.raises(ValueError, match="numpy array"):
            apply_makeup("selfie.jpg", MakeupLook())

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="H x W x 3"):
            apply_makeup(np.zeros((64, 64), dtype=np.uint8), MakeupLook())

    def test_rejects_wrong_dtype(self):
        with pytest.raises(ValueError, match="uint8"):
            apply_makeup(np.zeros((64, 64, 3), dtype=np.float32), MakeupLook())

    def test_rejects_non_look(self, astronaut_bgr):
        with pytest.raises(ValueError, match="MakeupLook"):
            apply_makeup(astronaut_bgr, {"lipstick": "#FF0000"})

    def test_no_face_raises(self):
        blank = np.full((256, 256, 3), 128, dtype=np.uint8)
        with pytest.raises(NoFaceDetectedError):
            apply_makeup(blank, MakeupLook())


class TestApplication:
    def test_returns_new_image_same_shape(self, astronaut_bgr, classic_result):
        assert classic_result.shape == astronaut_bgr.shape
        assert classic_result.dtype == np.uint8
        assert classic_result is not astronaut_bgr

    def test_zero_intensity_look_is_identity(self, astronaut_bgr, astronaut_landmarks):
        look = MakeupLook(
            lipstick_intensity=0, eyeshadow_intensity=0, eyeliner_intensity=0,
            blush_intensity=0, smoothing=0,
        )
        out = apply_makeup(astronaut_bgr, look, landmarks=astronaut_landmarks)
        assert np.array_equal(out, astronaut_bgr)
        assert out is not astronaut_bgr

    def test_input_image_is_not_mutated(self, astronaut_bgr, astronaut_landmarks):
        before = astronaut_bgr.copy()
        apply_makeup(astronaut_bgr, PRESETS["bold"], landmarks=astronaut_landmarks)
        assert np.array_equal(astronaut_bgr, before)

    def test_pixels_far_from_face_are_untouched(
        self, astronaut_bgr, astronaut_landmarks, classic_result
    ):
        """Makeup must be contained: the background stays bit-identical.

        Naive additive addWeighted compositing brightens blurred mask
        halos well outside the face; a channel swap recolors the entire
        image.
        """
        h, w = astronaut_bgr.shape[:2]
        face = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(np.round(astronaut_landmarks).astype(np.int32))
        cv2.fillConvexPoly(face, hull, 255)
        margin = int(cv2.arcLength(hull, True) * 0.05)
        face = cv2.dilate(face, np.ones((margin, margin), np.uint8))
        outside = face == 0
        assert outside.sum() > 0.3 * h * w
        assert np.array_equal(classic_result[outside], astronaut_bgr[outside])

    def test_lip_texture_survives_recoloring(
        self, astronaut_bgr, astronaut_landmarks, classic_result
    ):
        """Lightness detail inside the lips must correlate strongly with
        the input — naive flat fills drop this to ~0."""
        m = masks.lip_mask(astronaut_landmarks, astronaut_bgr.shape[:2]) > 0.5
        before = cv2.cvtColor(astronaut_bgr, cv2.COLOR_BGR2Lab)[..., 0][m].astype(float)
        after = cv2.cvtColor(classic_result, cv2.COLOR_BGR2Lab)[..., 0][m].astype(float)
        corr = np.corrcoef(before, after)[0, 1]
        assert corr > 0.85

    def test_lips_actually_change_color(
        self, astronaut_bgr, astronaut_landmarks, classic_result
    ):
        m = masks.lip_mask(astronaut_landmarks, astronaut_bgr.shape[:2]) > 0.5
        diff = np.abs(
            classic_result[m].astype(int) - astronaut_bgr[m].astype(int)
        ).mean()
        assert diff > 5, "lipstick had no visible effect"

    def test_all_presets_run(self, astronaut_bgr, astronaut_landmarks):
        for name, look in PRESETS.items():
            out = apply_makeup(astronaut_bgr, look, landmarks=astronaut_landmarks)
            assert not np.array_equal(out, astronaut_bgr), name
