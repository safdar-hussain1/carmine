"""Behavioral tests for CIELAB tinting, flat paint, smoothing, and finishes."""

import cv2
import numpy as np
import pytest

from carmine import pigment


def _textured_patch(shape=(64, 64), base=140.0, amplitude=40.0, seed=0):
    """A grayscale-ish BGR image with a gradient plus per-pixel noise.

    Gives GaussianBlur something real to smooth away, so texture-preservation
    checks (correlating detail before/after an op) aren't trivially perfect.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(-amplitude, amplitude, shape[1], dtype=np.float32)
    gray = base + np.tile(ramp, (shape[0], 1))
    gray += rng.normal(0, 10.0, size=shape).astype(np.float32)
    gray = np.clip(gray, 5, 250).astype(np.uint8)
    return cv2.merge([gray, gray, gray])


def _colorful_patch(shape=(64, 64), seed=1):
    rng = np.random.default_rng(seed)
    image = rng.integers(60, 200, size=(*shape, 3), dtype=np.uint8)
    return image


def _rect_mask(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


SHAPE = (64, 64)
INNER = (16, 48, 16, 48)  # y0, y1, x0, x1 -- the "product" region


def _hard_mask():
    return _rect_mask(SHAPE, *INNER)


def _outside_slice(mask):
    return mask <= 0


OPS = [
    ("tint", dict(color_rgb=(176, 58, 91), intensity=0), "intensity"),
    ("paint", dict(color_rgb=(20, 20, 20), intensity=0), "intensity"),
    ("smooth", dict(amount=0), "amount"),
    ("finish_matte", dict(strength=0), "strength"),
    ("finish_gloss", dict(strength=0), "strength"),
]


class TestZeroIsNoOp:
    @pytest.mark.parametrize("name, kwargs, param", OPS)
    def test_zero_returns_bit_identical_copy(self, name, kwargs, param):
        image = _colorful_patch()
        mask = _hard_mask()
        fn = getattr(pigment, name)
        out = fn(image, mask, **kwargs)
        assert out is not image
        np.testing.assert_array_equal(out, image)


FULL_OPS = [
    ("tint", dict(color_rgb=(176, 58, 91), intensity=1.0)),
    ("paint", dict(color_rgb=(20, 20, 20), intensity=1.0)),
    ("smooth", dict(amount=1.0)),
    ("finish_matte", dict(strength=1.0)),
    ("finish_gloss", dict(strength=1.0)),
]


class TestUntouchedRegionIsPreserved:
    @pytest.mark.parametrize("name, kwargs", FULL_OPS)
    def test_full_strength_leaves_unmasked_pixels_bit_identical(self, name, kwargs):
        image = _textured_patch()
        mask = _hard_mask()
        fn = getattr(pigment, name)
        out = fn(image, mask, **kwargs)
        outside = _outside_slice(mask)
        np.testing.assert_array_equal(out[outside], image[outside])


class TestTint:
    def test_masked_ab_mean_moves_toward_target_monotonically(self):
        image = _colorful_patch()
        mask = _hard_mask()
        target_rgb = (176, 58, 91)
        target_lab = pigment._color_to_lab(target_rgb)
        inside = mask > 0

        def ab_distance(intensity):
            out = pigment.tint(image, mask, target_rgb, intensity)
            lab = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            mean_ab = lab[inside][:, 1:].mean(axis=0)
            return float(np.linalg.norm(mean_ab - target_lab[1:]))

        d1, d2, d3 = ab_distance(0.3), ab_distance(0.6), ab_distance(0.9)
        assert d1 > d2 > d3

    def test_preserves_texture_detail(self):
        # A larger patch with a mask well inland from its own border: the
        # sigma=5 blur used for the detail signal has a ~15px reach, so a
        # mask edge that close to the patch boundary would mix in
        # untouched (unblended) pixels and break the correlation on
        # boundary effects alone rather than on the tint itself.
        shape = (200, 200)
        image = _textured_patch(shape=shape)
        mask = _rect_mask(shape, 60, 140, 60, 140)
        inside = mask > 0

        def l_detail(bgr):
            lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            l_channel = lab[..., 0]
            blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=5)
            return (l_channel - blurred)[inside]

        before = l_detail(image)
        out = pigment.tint(image, mask, (176, 58, 91), 0.8)
        after = l_detail(out)

        correlation = np.corrcoef(before, after)[0, 1]
        assert correlation >= 0.95


class TestPaint:
    def test_full_intensity_gives_exact_flat_color_inside_hard_mask(self):
        image = _colorful_patch()
        mask = _hard_mask()
        color_rgb = (20, 20, 20)
        out = pigment.paint(image, mask, color_rgb, 1.0)
        inside = mask > 0
        expected = np.array(color_rgb[::-1], dtype=np.uint8)
        assert np.all(out[inside] == expected)


class TestFinishMatte:
    def test_reduces_masked_lightness_variance(self):
        image = _textured_patch()
        mask = _hard_mask()
        inside = mask > 0

        def l_variance(bgr):
            lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            return float(np.var(lab[..., 0][inside]))

        before = l_variance(image)
        out = pigment.finish_matte(image, mask, strength=1.0)
        after = l_variance(out)
        assert after < before


class TestFinishGloss:
    def test_raises_p99_without_shifting_median_much(self):
        rng = np.random.default_rng(2)
        gray = rng.integers(80, 140, size=SHAPE, dtype=np.uint8).astype(np.float32)
        # A handful of bright specular-highlight pixels give p99 something
        # real to sit above p75, without dragging the median around.
        gray[20:24, 20:24] = 250
        image = cv2.merge([gray, gray, gray]).astype(np.uint8)
        mask = _hard_mask()
        inside = mask > 0

        def l_channel(bgr):
            lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            return lab[..., 0][inside]

        before = l_channel(image)
        out = pigment.finish_gloss(image, mask, strength=1.0)
        after = l_channel(out)

        assert np.percentile(after, 99) > np.percentile(before, 99)
        assert abs(np.median(after) - np.median(before)) < 2.0

    def test_degenerate_flat_patch_returns_exact_copy(self):
        image = np.full((*SHAPE, 3), 120, dtype=np.uint8)
        mask = _hard_mask()
        out = pigment.finish_gloss(image, mask, strength=1.0)
        assert out is not image
        np.testing.assert_array_equal(out, image)


class TestParseHexColor:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("#AABBCC", (0xAA, 0xBB, 0xCC)),
            ("aabbcc", (0xAA, 0xBB, 0xCC)),
            ("#000000", (0, 0, 0)),
            ("#FFFFFF", (255, 255, 255)),
        ],
    )
    def test_valid_forms(self, value, expected):
        assert pigment.parse_hex_color(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["", "#GGGGGG", "#AABBC", "not-a-color", "#AABBCCDD", 12345],
    )
    def test_invalid_forms_raise_with_offending_value_in_message(self, value):
        with pytest.raises(ValueError) as exc_info:
            pigment.parse_hex_color(value)
        assert str(value) in str(exc_info.value)
