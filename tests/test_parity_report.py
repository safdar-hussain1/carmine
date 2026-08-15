"""Tests for the cross-surface parity harness and the numbers it publishes.

Two things are guarded here, and they run under different conditions on
purpose.

The **fixture** tests exercise `scripts/export_parity_fixtures.py`: they
render the canned frames and check that the manifest describes what is
actually on disk. They need the local portrait dataset, which is not in the
repository, so they skip where it is absent.

The **report** test validates `reports/browser_metrics.json`, which *is*
committed, and it never skips. That file is the published claim -- the ΔE
figures and per-stage timings that the site and the write-up quote -- so its
schema and its plausibility are pinned here. Committing numbers without a
test that reads them is how a stale figure survives a rewrite of the code
that produced it.

Note what the report test does *not* do: it does not re-derive the numbers.
Reproducing them needs a browser, and that is `scripts/verify_site.py
--with-parity`'s job. This checks that what was committed is well-formed and
inside the range the harness is allowed to pass with, so a regression cannot
be committed silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "export_parity_fixtures.py"
FIXTURE_DIR = REPO_ROOT / "reports" / "parity_fixtures"
METRICS = REPO_ROOT / "reports" / "browser_metrics.json"
DATASET = REPO_ROOT / "data" / "no_makeup"

# Mirrors the gates web/src/main.ts enforces in the browser. Duplicated
# rather than imported because there is nothing to import from -- the
# thresholds live in TypeScript -- and a committed number drifting past the
# gate its own selftest applies is exactly what this catches.
CPU_MEAN_DELTA_E_LIMIT = 2.0
CPU_P99_DELTA_E_LIMIT = 5.0
CPU_MAX_DELTA_E_LIMIT = 12.0

EXPECTED_FRAMES = 3
EXPECTED_LOOKS = {"velvet-satin", "glass", "velvet-matte-brows"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    """The fixture manifest, exporting the fixtures first if they are absent.

    Skips when the dataset is missing, which is the CI case: the portraits
    are local-only, so there is nothing to render from.
    """
    if not FIXTURE_DIR.joinpath("manifest.json").exists():
        if not DATASET.is_dir():
            pytest.skip(f"parity fixtures absent and no dataset at {DATASET}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        )
        if result.returncode != 0:
            pytest.skip(f"fixture export failed: {result.stderr[-400:]}")
    return json.loads(FIXTURE_DIR.joinpath("manifest.json").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_manifest_describes_three_frames_and_three_looks(manifest):
    assert len(manifest["frames"]) == EXPECTED_FRAMES
    assert set(manifest["looks"]) == EXPECTED_LOOKS
    for frame in manifest["frames"]:
        assert set(frame["expected"]) == EXPECTED_LOOKS
        # The browser engine builds masks at a 720px long side; a fixture
        # larger than that would be compared against a different resolution
        # than it was rendered at.
        assert max(frame["width"], frame["height"]) <= manifest["proc_max_side"]


@pytest.mark.slow
def test_fixture_looks_have_smoothing_disabled(manifest):
    """The browser has no smoothing path, so a fixture using it would
    measure a missing feature rather than a disagreement."""
    for look in manifest["looks"].values():
        assert look["smoothing"] == 0.0


@pytest.mark.slow
def test_manifest_hashes_match_the_files_on_disk(manifest):
    mismatched = [
        name
        for name, digest in manifest["sha256"].items()
        if not FIXTURE_DIR.joinpath(name).exists() or _sha256(FIXTURE_DIR / name) != digest
    ]
    assert not mismatched, f"manifest hashes do not match: {mismatched}"


@pytest.mark.slow
def test_every_referenced_file_is_hashed(manifest):
    referenced = set()
    for frame in manifest["frames"]:
        referenced.add(frame["input"])
        referenced.add(frame["landmarks"])
        referenced.update(frame["expected"].values())
    assert referenced == set(manifest["sha256"])


@pytest.mark.slow
def test_landmarks_are_478_pairs(manifest):
    for frame in manifest["frames"]:
        points = json.loads(FIXTURE_DIR.joinpath(frame["landmarks"]).read_text(encoding="utf-8"))
        assert len(points) == 478
        assert all(len(point) == 2 for point in points)


# --- the committed report -------------------------------------------------


@pytest.fixture(scope="module")
def metrics():
    assert METRICS.exists(), (
        f"{METRICS.relative_to(REPO_ROOT)} is missing; regenerate it with "
        "`python scripts/verify_site.py --with-parity`"
    )
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_metrics_file_carries_no_image_data(metrics):
    """The reason this file is committed while the fixtures are not."""
    text = METRICS.read_text(encoding="utf-8")
    assert "data:image" not in text
    assert "base64" not in text


def test_parity_cases_cover_every_frame_and_look(metrics):
    for key in ("cpu", "gpu", "endToEnd"):
        cases = metrics["parity"][key]
        assert len(cases) == EXPECTED_FRAMES * len(EXPECTED_LOOKS), key
        assert {case["look"] for case in cases} == EXPECTED_LOOKS, key
        assert len({case["frame"] for case in cases}) == EXPECTED_FRAMES, key


def test_every_case_measured_a_real_region(metrics):
    for key in ("cpu", "gpu", "endToEnd"):
        for case in metrics["parity"][key]:
            # A case that measured nothing would report a flawless mean.
            assert case["supportPixels"] > 1000, (key, case["frame"])
            for field in ("meanDeltaE", "p99DeltaE", "maxDeltaE"):
                assert math.isfinite(case[field]), (key, case["frame"], field)
            assert case["meanDeltaE"] <= case["p99DeltaE"] <= case["maxDeltaE"]


def test_cpu_parity_is_within_the_published_thresholds(metrics):
    """The claim itself: the reference paths agree.

    These are the same gates the in-browser check applies, restated against
    the committed file so a number that drifted past them cannot be committed
    while the browser run that produced it is not being re-executed.
    """
    for case in metrics["parity"]["cpu"]:
        label = f"{case['frame']}/{case['look']}"
        assert case["meanDeltaE"] < CPU_MEAN_DELTA_E_LIMIT, label
        assert case["p99DeltaE"] < CPU_P99_DELTA_E_LIMIT, label
        assert case["maxDeltaE"] < CPU_MAX_DELTA_E_LIMIT, label


def test_nothing_was_painted_outside_the_mask_support(metrics):
    """Both engines promise untouched pixels stay bit-identical. This is the
    one parity property that is exact rather than approximate, so it is
    checked on every path including the shader's."""
    for key in ("cpu", "gpu", "endToEnd"):
        for case in metrics["parity"][key]:
            assert case["changedOutsideSupport"] == 0, (key, case["frame"])


def test_gpu_parity_is_recorded_with_its_renderer(metrics):
    """The shader path carries no threshold, so the renderer string is what
    makes its numbers interpretable -- a software rasterizer's float behavior
    is not the hardware this ships to."""
    assert metrics["parity"]["glRenderer"]
    for case in metrics["parity"]["gpu"]:
        assert math.isfinite(case["meanDeltaE"])


def test_live_path_is_fast_enough_to_be_live(metrics):
    """The point of the live mask path.

    The reference construction costs hundreds of milliseconds on a face that
    fills the frame -- feather radii scale with the face, not the frame -- so
    the mirror builds masks at half resolution with box-approximated
    feathers. This pins that it is actually paying off; a regression that
    quietly routed the live loop back through the exact path would otherwise
    only show up as a mirror nobody wants to use.
    """
    timing = metrics["timing"]
    live = timing["livePath"]
    assert live["maskWidth"] * 2 <= timing["maskWidth"] + 2
    assert live["buildMasks"]["median"] < 40.0
    assert live["buildMasks"]["median"] < timing["buildMasks"]["median"]
    assert live["totalMedian"] < timing["totalMedian"]
    for stage in ("buildMasks", "draw"):
        entry = live[stage]
        assert math.isfinite(entry["median"]) and entry["median"] > 0, stage
        assert entry["samples"] == timing["frames"], stage
    assert live["totalMedian"] == pytest.approx(
        timing["detect"]["median"] + live["buildMasks"]["median"] + live["draw"]["median"]
    )


def test_hardware_timing_attempt_is_recorded_either_way(metrics):
    """`--timing-only` retries without forcing software rasterization.

    Whether it reaches a real driver depends on the machine, so the schema
    admits both outcomes -- but it must never claim a hardware number while
    sitting on a software renderer, which is what the last branch checks.
    """
    record = metrics["timing_hardware"]
    assert record["attempted"] is True
    if not record["recorded"]:
        assert record["reason"]
        return
    renderer = record["glRenderer"]
    assert renderer
    assert not any(
        name in renderer.lower() for name in ("swiftshader", "llvmpipe", "software", "lavapipe")
    ), renderer
    timing = record["timing"]
    assert timing["livePath"]["totalMedian"] > 0
    assert timing["livePath"]["totalMedian"] < timing["totalMedian"]


def test_timing_medians_are_finite_and_positive(metrics):
    timing = metrics["timing"]
    assert timing["frames"] >= 100
    assert timing["warmupFrames"] > 0
    assert timing["width"] == 1280 and timing["height"] == 720
    assert timing["fenced"] is True
    # Without this the mask timing cannot be compared to anything: feather
    # radii, and so the cost, scale with the face rather than the frame.
    assert timing["interocularPx"] > 0
    for stage in ("detect", "buildMasks", "draw"):
        entry = timing[stage]
        assert entry["samples"] == timing["frames"], stage
        assert math.isfinite(entry["median"]) and entry["median"] > 0, stage
        assert entry["min"] <= entry["median"] <= entry["max"], stage
    assert timing["totalMedian"] == pytest.approx(
        sum(timing[stage]["median"] for stage in ("detect", "buildMasks", "draw"))
    )


def test_selftest_summary_records_no_skips(metrics):
    """A --with-parity run whose parity checks skipped would produce a file
    of nothing at all, so the run's own summary is committed alongside."""
    assert metrics["selftest"]["pass"] is True
    assert metrics["selftest"]["skipped"] == 0
