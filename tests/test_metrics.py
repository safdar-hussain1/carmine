"""Constructed-case tests for `carmine.metrics`, plus a schema test that pins
the committed `reports/benchmark.json` artifact."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from carmine import metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def _flat_image(shape, color):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:] = color
    return img


def _textured_patch(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(80, 180, size=(*shape, 3), dtype=np.uint8)


class TestPigmentOnTarget:
    def test_perfect_containment_scores_one(self):
        before = _flat_image((20, 20), (120, 120, 120))
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        after = before.copy()
        after[5:15, 5:15] = (10, 200, 40)
        assert metrics.pigment_on_target(before, after, mask) == pytest.approx(1.0)

    def test_all_edits_outside_mask_scores_near_zero(self):
        before = _flat_image((20, 20), (120, 120, 120))
        mask = np.zeros((20, 20), dtype=np.float32)
        mask[5:15, 5:15] = 1.0
        after = before.copy()
        after[0:3, 0:3] = (10, 200, 40)
        assert metrics.pigment_on_target(before, after, mask) == pytest.approx(0.0, abs=1e-6)

    def test_no_change_is_trivially_on_target(self):
        before = _flat_image((10, 10), (50, 60, 70))
        mask = np.zeros((10, 10), dtype=np.float32)
        assert metrics.pigment_on_target(before, before.copy(), mask) == pytest.approx(1.0)


class TestBackgroundUntouched:
    def test_identical_images_score_one(self):
        before = _textured_patch((20, 20))
        face_mask = np.zeros((20, 20), dtype=np.float32)
        face_mask[5:15, 5:15] = 1.0
        assert metrics.background_untouched(before, before.copy(), face_mask) == pytest.approx(1.0)

    def test_channel_swapped_background_scores_near_zero(self):
        before = _textured_patch((20, 20), seed=1)
        face_mask = np.zeros((20, 20), dtype=np.float32)
        face_mask[5:15, 5:15] = 1.0  # small face region, most of image is "background"
        after = before[:, :, ::-1].copy()  # channel swap everywhere, including background
        score = metrics.background_untouched(before, after, face_mask)
        assert score == pytest.approx(0.0, abs=0.05)


class TestLipTextureKept:
    def test_mild_tint_keeps_high_correlation(self):
        before = _textured_patch((20, 20), seed=2)
        mask = np.ones((20, 20), dtype=np.float32)
        lab = cv2.cvtColor(before, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab[..., 1] = np.clip(lab[..., 1] + 15, 0, 255)  # shift chroma only
        after = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
        score = metrics.lip_texture_kept(before, after, mask)
        assert score > 0.9

    def test_flat_fill_destroys_correlation(self):
        before = _textured_patch((20, 20), seed=3)
        mask = np.ones((20, 20), dtype=np.float32)
        after = _flat_image((20, 20), (90, 90, 90))
        score = metrics.lip_texture_kept(before, after, mask)
        assert score < 0.3

    def test_empty_mask_raises(self):
        before = _textured_patch((10, 10))
        with pytest.raises(ValueError):
            metrics.lip_texture_kept(before, before.copy(), np.zeros((10, 10), dtype=np.float32))


class TestLipDetailRetention:
    def test_pure_chroma_shift_keeps_detail_near_one(self):
        # Lab L is untouched by a chroma-only shift, so the L high-pass
        # (and its std) is unchanged -- ratio should land almost exactly 1.0.
        before = _textured_patch((20, 20), seed=6)
        mask = np.ones((20, 20), dtype=np.float32)
        lab = cv2.cvtColor(before, cv2.COLOR_BGR2Lab).astype(np.float32)
        lab[..., 1] = np.clip(lab[..., 1] + 15, 0, 255)
        after = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
        score = metrics.lip_detail_retention(before, after, mask)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_flat_fill_drives_detail_to_zero(self):
        before = _textured_patch((20, 20), seed=7)
        mask = np.ones((20, 20), dtype=np.float32)
        after = _flat_image((20, 20), (90, 90, 90))
        score = metrics.lip_detail_retention(before, after, mask)
        assert score < 0.05

    def test_empty_mask_raises(self):
        before = _textured_patch((10, 10))
        with pytest.raises(ValueError):
            metrics.lip_detail_retention(before, before.copy(), np.zeros((10, 10), dtype=np.float32))


class TestIdentitySsim:
    def test_identical_images_score_one(self):
        before = _textured_patch((32, 32), seed=4)
        assert metrics.identity_ssim(before, before.copy()) == pytest.approx(1.0)

    def test_very_different_images_score_low(self):
        before = _flat_image((32, 32), (10, 10, 10))
        after = _flat_image((32, 32), (250, 250, 250))
        rng = np.random.default_rng(5)
        noisy_after = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        score = metrics.identity_ssim(before, noisy_after)
        assert score < 0.5


class TestMaskJitter:
    def test_static_masks_score_one_iou_zero_drift(self):
        mask = np.zeros((30, 30), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        seq = [mask.copy() for _ in range(5)]
        result = metrics.mask_jitter(seq, iod=50.0)
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["centroid_drift"] == pytest.approx(0.0)

    def test_alternating_disjoint_masks_score_zero_iou(self):
        a = np.zeros((30, 30), dtype=np.float32)
        a[0:5, 0:5] = 1.0
        b = np.zeros((30, 30), dtype=np.float32)
        b[25:30, 25:30] = 1.0
        seq = [a, b, a, b, a, b]
        result = metrics.mask_jitter(seq, iod=50.0)
        assert result["mean_iou"] == pytest.approx(0.0)
        assert result["centroid_drift"] > 0.0

    def test_too_few_frames_raises(self):
        with pytest.raises(ValueError):
            metrics.mask_jitter([np.zeros((5, 5))], iod=10.0)

    def test_nonpositive_iod_raises(self):
        with pytest.raises(ValueError):
            metrics.mask_jitter([np.zeros((5, 5)), np.zeros((5, 5))], iod=0.0)


class TestBenchmarkJsonSchema:
    """Pins the shape and plausibility of the committed reports/benchmark.json."""

    @pytest.fixture
    def bench(self):
        path = REPO_ROOT / "reports" / "benchmark.json"
        if not path.is_file():
            pytest.skip("reports/benchmark.json not generated in this checkout")
        return json.loads(path.read_text())

    METRIC_KEYS = (
        "pigment_on_target",
        "background_untouched",
        "lip_texture_kept",
        "lip_detail_retention",
        "identity_ssim",
    )
    METHODS = {"mismatched_indices", "opaque_fill", "channel_swap", "untrained_gan", "carmine"}

    def test_photo_section_schema(self, bench):
        photo = bench["photo"]
        rows = photo["rows"]
        methods = {row["method"] for row in rows}
        assert methods == self.METHODS
        for row in rows:
            for key in (*self.METRIC_KEYS, "ms_per_image"):
                assert key in row, f"{row['method']} missing {key}"
                value = row[key]
                assert np.isfinite(value), f"{row['method']}.{key} is not finite"
            assert 0.0 <= row["pigment_on_target"] <= 1.0
            assert 0.0 <= row["background_untouched"] <= 1.0
            assert row["identity_ssim"] <= 1.0 + 1e-6
            assert row["ms_per_image"] > 0.0

            # per-method spread [min, max] across images, alongside the mean
            spread = row["spread"]
            for key in (*self.METRIC_KEYS, "ms_per_image"):
                lo, hi = spread[key]
                assert np.isfinite(lo) and np.isfinite(hi)
                assert lo <= row[key] <= hi, f"{row['method']}.{key} mean outside its own spread"

    def test_photo_timing_is_per_method_not_summed(self, bench):
        # Regression guard for the timing bug this fix addresses: all five
        # methods used to be evaluated inside every timed block, so every
        # method's reported ms_per_image was roughly the sum of all five
        # (~600ms each). carmine and channel_swap both call apply_look and
        # so should cost far more than the three cheap/no-landmark methods,
        # but none of them should be anywhere near the old ~600ms-for-everyone
        # regime.
        rows = {row["method"]: row for row in bench["photo"]["rows"]}
        for method in ("mismatched_indices", "opaque_fill", "untrained_gan"):
            assert rows["carmine"]["ms_per_image"] > rows[method]["ms_per_image"], (
                f"carmine should cost more per image than {method}, given carmine "
                "and channel_swap both run the full apply_look pipeline and the "
                "others don't"
            )
        # None of the cheap methods should be anywhere close to carmine's cost.
        assert rows["mismatched_indices"]["ms_per_image"] < rows["carmine"]["ms_per_image"] / 5
        assert rows["opaque_fill"]["ms_per_image"] < rows["carmine"]["ms_per_image"] / 5

    def test_lip_detail_retention_ordering_matches_measured_reality(self, bench):
        # Pins the actual measured ranking (see meta.lip_detail_retention_note
        # for why it isn't monotonic with texture-preservation quality) so a
        # regenerated benchmark.json that flips this ordering fails loudly
        # instead of silently.
        rows = {row["method"]: row["lip_detail_retention"] for row in bench["photo"]["rows"]}
        assert (
            rows["mismatched_indices"]
            > rows["opaque_fill"]
            > rows["channel_swap"]
            > rows["carmine"]
            > rows["untrained_gan"]
        )

    def test_meta_schema(self, bench):
        meta = bench["meta"]
        assert meta["n_images"] >= 20
        assert meta["look_preset"] == "velvet"
        assert isinstance(meta["skipped"], list)
        assert meta["smoothing_override"] == pytest.approx(0.0)
        assert isinstance(meta["smoothing_override_rationale"], str) and meta["smoothing_override_rationale"]
        assert isinstance(meta["lip_detail_retention_note"], str) and meta["lip_detail_retention_note"]

        protocol = meta["protocol"]
        for key in (
            "pigment_on_target",
            "pigment_on_target_caveat",
            "background_untouched",
            "lip_texture_kept",
            "lip_detail_retention",
            "identity_ssim",
        ):
            assert key in protocol
            assert isinstance(protocol[key], str) and protocol[key]

    def test_stability_section_schema(self, bench):
        stability = bench["stability"]
        assert stability["protocol"] == "ground_truth_affine"

        for variant in ("raw", "one_euro"):
            entry = stability[variant]
            assert np.isfinite(entry["deviation_px_iod"])
            assert entry["deviation_px_iod"] >= 0.0
            assert np.isfinite(entry["jitter_px_iod"])
            assert entry["jitter_px_iod"] >= 0.0

        grid = stability["grid_search"]
        expected_n = len(grid["min_cutoff_values"]) * len(grid["beta_values"])
        assert len(grid["results"]) == expected_n
        for entry in grid["results"]:
            assert np.isfinite(entry["deviation_px_iod"])
            assert np.isfinite(entry["jitter_px_iod"])
            assert isinstance(entry["meets_deviation_constraint"], bool)

        selected = stability["selected_params"]
        assert selected is None or {"min_cutoff", "beta"} <= selected.keys()
        assert isinstance(stability["meets_improvement_threshold"], bool)
        # A non-null selection must actually meet the improvement bar; a null
        # one must not claim to.
        if selected is not None:
            assert stability["meets_improvement_threshold"] is True
        else:
            assert stability["meets_improvement_threshold"] is False

        assert np.isfinite(stability["jitter_reduction_pct"])
        assert isinstance(stability["note"], str) and stability["note"]
        assert np.isfinite(stability["video_ms_per_frame"])
        assert stability["video_ms_per_frame"] > 0.0
        assert stability["n_frames_with_face"] >= 2

        clip_params = stability["clip_params"]
        assert clip_params["noise_sigma"] > 0
        assert clip_params["fps"] > 0

    def test_carmine_scores_match_expected_shape(self, bench):
        rows = {row["method"]: row for row in bench["photo"]["rows"]}
        carmine = rows["carmine"]
        assert carmine["pigment_on_target"] >= 0.75
        assert carmine["background_untouched"] == pytest.approx(1.0, abs=1e-6)
        assert carmine["lip_texture_kept"] >= 0.95
        assert carmine["identity_ssim"] >= 0.95
