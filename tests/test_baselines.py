"""Tests that pin down the defect each standard failure-mode baseline exhibits."""

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from carmine import baselines, masks, pigment, regions
from carmine.landmarks import FaceLandmarker
from carmine.look import Look, Product


@pytest.fixture(scope="module")
def astronaut_bgr():
    from skimage import data

    rgb = data.astronaut()
    return rgb[:, :, ::-1].copy()


@pytest.fixture(scope="module")
def astronaut_landmarks(astronaut_bgr):
    return FaceLandmarker().detect(astronaut_bgr)


def _l_detail(bgr, inside, sigma=5):
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    l_channel = lab[..., 0]
    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma)
    return (l_channel - blurred)[inside]


class TestOpaqueFill:
    def test_erases_lip_texture(self, astronaut_bgr, astronaut_landmarks):
        look = Look(lipstick=Product("#8E1B3A", intensity=0.9, finish="matte"))
        lip_mask = masks.lip_mask(astronaut_landmarks, astronaut_bgr.shape[:2])
        inside = lip_mask > 0.5

        out = baselines.opaque_fill(astronaut_bgr, astronaut_landmarks, look)
        before = _l_detail(astronaut_bgr, inside)
        after = _l_detail(out, inside)
        correlation = np.corrcoef(before, after)[0, 1]
        assert correlation < 0.95

        # The engine's own texture-preserving tint, applied to the same
        # region, keeps far more of the original detail correlated.
        color = pigment.parse_hex_color(look.lipstick.color)
        honest = pigment.tint(astronaut_bgr, lip_mask, color, look.lipstick.intensity)
        honest_after = _l_detail(honest, inside)
        honest_correlation = np.corrcoef(before, honest_after)[0, 1]
        assert honest_correlation >= 0.95

    def test_changes_nothing_when_look_is_all_zero(self, astronaut_bgr, astronaut_landmarks):
        out = baselines.opaque_fill(astronaut_bgr, astronaut_landmarks, Look())
        np.testing.assert_array_equal(out, astronaut_bgr)


class TestChannelSwap:
    def test_changes_most_background_pixels(self, astronaut_bgr, astronaut_landmarks):
        # "Background" here is the strip above the face -- unambiguously
        # not the subject, unlike everything below the chin (spacesuit,
        # arms), which is desaturated enough that plenty of pixels have
        # R == B and are naturally no-ops under a channel swap regardless
        # of what the baseline does.
        face_top = astronaut_landmarks[regions.FACE_OVAL][:, 1].min()
        h, w = astronaut_bgr.shape[:2]
        rows, _ = np.mgrid[0:h, 0:w]
        background = rows < face_top

        out = baselines.channel_swap(astronaut_bgr, astronaut_landmarks, Look())
        changed = np.any(out != astronaut_bgr, axis=2)

        fraction = (changed & background).sum() / background.sum()
        assert fraction > 0.90

    def test_correct_application_matches_engine_before_swap(
        self, astronaut_bgr, astronaut_landmarks
    ):
        from carmine.engine import apply_look

        look = Look(lipstick=Product("#B03A5B", intensity=0.6))
        honest = apply_look(astronaut_bgr, look, landmarks=astronaut_landmarks)
        swapped = baselines.channel_swap(astronaut_bgr, astronaut_landmarks, look)
        np.testing.assert_array_equal(swapped, cv2.cvtColor(honest, cv2.COLOR_RGB2BGR))


class TestUntrainedGan:
    def test_output_uncorrelated_with_input_and_identity_destroyed(self, astronaut_bgr):
        out = baselines.untrained_gan(astronaut_bgr, Look())
        assert out.shape == (256, 256, 3)

        src = cv2.resize(astronaut_bgr, (256, 256))
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
        out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()

        correlation = np.corrcoef(src_gray, out_gray)[0, 1]
        assert abs(correlation) < 0.5

        score = structural_similarity(
            cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        )
        assert score < 0.5

    def test_deterministic_for_fixed_seed(self, astronaut_bgr):
        first = baselines.untrained_gan(astronaut_bgr, Look(), seed=0)
        second = baselines.untrained_gan(astronaut_bgr, Look(), seed=0)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_differ(self, astronaut_bgr):
        a = baselines.untrained_gan(astronaut_bgr, Look(), seed=0)
        b = baselines.untrained_gan(astronaut_bgr, Look(), seed=1)
        assert not np.array_equal(a, b)


class TestMismatchedIndices:
    def test_places_most_edit_energy_outside_the_lip_region(
        self, astronaut_bgr, astronaut_landmarks
    ):
        look = Look(lipstick=Product("#8E1B3A", intensity=0.9))
        out = baselines.mismatched_indices(astronaut_bgr, astronaut_landmarks, look)

        changed = np.any(out != astronaut_bgr, axis=2)
        assert changed.sum() > 0

        lip_mask = masks.lip_mask(astronaut_landmarks, astronaut_bgr.shape[:2])
        true_lip = lip_mask > 0.05

        containment = (changed & true_lip).sum() / changed.sum()
        assert containment < 0.60

    def test_changes_nothing_when_look_is_all_zero(self, astronaut_bgr, astronaut_landmarks):
        out = baselines.mismatched_indices(astronaut_bgr, astronaut_landmarks, Look())
        np.testing.assert_array_equal(out, astronaut_bgr)
