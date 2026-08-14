"""Behavioral tests for apply_look and VideoEngine."""

import numpy as np
import pytest

from carmine import masks
from carmine.engine import VideoEngine, apply_look
from carmine.landmarks import FaceLandmarker
from carmine.look import Look, Product, PRESETS


@pytest.fixture(scope="module")
def astronaut_bgr():
    from skimage import data

    rgb = data.astronaut()
    return rgb[:, :, ::-1].copy()


@pytest.fixture(scope="module")
def astronaut_landmarks(astronaut_bgr):
    return FaceLandmarker().detect(astronaut_bgr)


class TestApplyLookZeroLook:
    def test_all_zero_look_is_bit_identical_copy(self, astronaut_bgr, astronaut_landmarks):
        look = Look()  # every product defaults to intensity 0.0
        out = apply_look(astronaut_bgr, look, landmarks=astronaut_landmarks)
        assert out is not astronaut_bgr
        np.testing.assert_array_equal(out, astronaut_bgr)


class TestApplyLookLipstickOnly:
    def test_changes_pixels_only_within_lip_mask_support(self, astronaut_bgr, astronaut_landmarks):
        look = Look(lipstick=Product("#B03A5B", intensity=0.8, finish="satin"))
        out = apply_look(astronaut_bgr, look, landmarks=astronaut_landmarks)

        lip_mask = masks.lip_mask(astronaut_landmarks, astronaut_bgr.shape[:2])
        outside = lip_mask <= 0

        np.testing.assert_array_equal(out[outside], astronaut_bgr[outside])
        # Sanity: something inside the mask's support actually changed.
        inside = lip_mask > 0
        assert not np.array_equal(out[inside], astronaut_bgr[inside])


class TestApplyLookValidation:
    def test_wrong_look_type_raises_value_error(self, astronaut_bgr, astronaut_landmarks):
        with pytest.raises(ValueError):
            apply_look(astronaut_bgr, "not a look", landmarks=astronaut_landmarks)

    def test_invalid_image_raises_value_error(self, astronaut_landmarks):
        with pytest.raises(ValueError):
            apply_look(np.zeros((10, 10), dtype=np.uint8), Look(), landmarks=astronaut_landmarks)


class TestApplyLookPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_preset_runs_without_error(self, name, astronaut_bgr, astronaut_landmarks):
        out = apply_look(astronaut_bgr, PRESETS[name], landmarks=astronaut_landmarks)
        assert out.shape == astronaut_bgr.shape
        assert out.dtype == astronaut_bgr.dtype


class TestApplyLookLazyLandmarks:
    def test_detects_landmarks_when_not_supplied(self, astronaut_bgr):
        look = Look(lipstick=Product("#B03A5B", intensity=0.5))
        out = apply_look(astronaut_bgr, look)
        assert out.shape == astronaut_bgr.shape
        assert not np.array_equal(out, astronaut_bgr)


class TestVideoEngine:
    def test_processes_a_stream_of_frames(self, astronaut_bgr):
        engine = VideoEngine(PRESETS["everyday"])
        timestamps = [0, 33, 66]
        for ts in timestamps:
            out = engine.process(astronaut_bgr, ts)
            assert isinstance(out, np.ndarray)
            assert out.shape == astronaut_bgr.shape

    def test_no_face_frame_returns_unmodified_copy_and_recovers(self, astronaut_bgr):
        engine = VideoEngine(PRESETS["everyday"])
        engine.process(astronaut_bgr, 0)

        black_frame = np.zeros_like(astronaut_bgr)
        out = engine.process(black_frame, 33)
        np.testing.assert_array_equal(out, black_frame)
        assert out is not black_frame

        # Filter was reset on the no-face gap; the next good frame at a
        # later timestamp should still process without raising.
        out2 = engine.process(astronaut_bgr, 66)
        assert isinstance(out2, np.ndarray)
        assert out2.shape == astronaut_bgr.shape
