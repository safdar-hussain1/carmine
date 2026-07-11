"""Tests that pin down the 2024 bugs the legacy module reproduces."""

import numpy as np
import pytest

from virtual_makeup import legacy, masks


def test_legacy_mediapipe_paints_mostly_outside_valid_regions(
    astronaut_bgr, astronaut_landmarks
):
    """dlib indices on the 468-point mesh: most painted pixels land
    outside every legitimate makeup region."""
    out = legacy.legacy_mediapipe(astronaut_bgr, astronaut_landmarks)
    changed = np.any(out != astronaut_bgr, axis=2)
    shape = astronaut_bgr.shape[:2]
    valid = np.zeros(shape, dtype=bool)
    for fn in (masks.lip_mask, masks.eyeshadow_mask, masks.blush_mask,
               masks.eyeliner_mask):
        valid |= fn(astronaut_landmarks, shape) > 0.05
    containment = (changed & valid).sum() / max(1, changed.sum())
    assert changed.sum() > 0
    assert containment < 0.5, (
        f"legacy mediapipe containment {containment:.2f}; "
        "expected most pigment outside valid regions"
    )


def test_legacy_gan_output_is_noise(astronaut_bgr):
    """An untrained generator cannot reconstruct its input; its output
    should not correlate with the source image."""
    out = legacy.legacy_gan(astronaut_bgr)
    assert out.shape == (256, 256, 3)
    import cv2
    from skimage.metrics import structural_similarity

    src = cv2.resize(astronaut_bgr, (256, 256))
    score = structural_similarity(
        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY),
    )
    # A faithful makeup application scores ~0.9+ SSIM against its own
    # input; the untrained net destroys most structure. (The notebook's
    # 0.043 was measured against *different* reference photos.)
    assert score < 0.5


def test_legacy_metrics_precision_is_always_one():
    """With y_true all ones, 'precision' is 1.0 no matter the scores —
    the original notebook's headline metric was unfalsifiable."""
    for scores in ([0.9, 0.8], [0.01, 0.02], [0.5, 0.1, 0.99]):
        assert legacy.legacy_metrics(scores)["Precision"] == 1.0


def test_legacy_metrics_accuracy_is_just_threshold_counting():
    m = legacy.legacy_metrics([0.5, 0.5, 0.1, 0.9], threshold=0.45)
    assert m["Accuracy"] == pytest.approx(0.75)
    assert m["Recall"] == m["Accuracy"]


def test_legacy_dlib_channel_swap_recolors_whole_image(astronaut_bgr):
    """The RGB2BGR-on-BGR save bug changes pixels everywhere, including
    the background far from any makeup."""
    lm68 = _dlibish_landmarks(astronaut_bgr)
    honest = legacy.legacy_dlib(astronaut_bgr, lm68, channel_swap_bug=False)
    swapped = legacy.legacy_dlib(astronaut_bgr, lm68, channel_swap_bug=True)
    corner = (slice(0, 40), slice(0, 40))
    assert np.array_equal(honest[corner], astronaut_bgr[corner])
    assert not np.array_equal(swapped[corner], astronaut_bgr[corner])


def _dlibish_landmarks(image):
    """A rough but geometrically ordered 68-point layout (left-to-right,
    positive face width) — the channel-swap test only cares about pixels
    far from the face."""
    h, w = image.shape[:2]
    x = np.linspace(w * 0.35, h * 0.65, 68)
    y = np.full(68, h * 0.5) + 20 * np.sin(np.linspace(0, np.pi, 68))
    return np.stack([x, y], axis=1).astype(np.float32)
