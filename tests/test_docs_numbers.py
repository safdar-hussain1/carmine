"""Pins the load-bearing numbers quoted in README.md and DESIGN_CARD.md.

Those two files hand-transcribe figures out of `reports/benchmark.json` and
`reports/browser_metrics.json` -- there was previously no test connecting the
prose to the data, so a regenerated report could silently drift away from
what the docs claim. This derives each published figure from the committed
JSON (with the same rounding the docs use) and asserts the resulting string
appears verbatim in the doc text. It intentionally does not re-derive the
JSON itself -- that is `test_parity_report.py`'s job -- only that the docs
still say what the JSON currently supports.

Kept deliberately loose about layout: every assertion is a substring check,
so reformatting a table or rewording a sentence around a number does not
break this, only changing the number (or letting it drift from the data)
does.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "reports" / "benchmark.json"
BROWSER_METRICS = REPO_ROOT / "reports" / "browser_metrics.json"
README = REPO_ROOT / "README.md"
DESIGN_CARD = REPO_ROOT / "web" / "public" / "DESIGN_CARD.md"


def _read(path: Path) -> str:
    assert path.exists(), f"{path.relative_to(REPO_ROOT)} is missing"
    return path.read_text(encoding="utf-8")


def _load(path: Path) -> dict:
    assert path.exists(), f"{path.relative_to(REPO_ROOT)} is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def _carmine_row(benchmark: dict) -> dict:
    rows = benchmark["photo"]["rows"]
    matches = [r for r in rows if r["method"] == "carmine"]
    assert len(matches) == 1, "expected exactly one 'carmine' row in benchmark.json"
    return matches[0]


def _opaque_fill_row(benchmark: dict) -> dict:
    rows = benchmark["photo"]["rows"]
    matches = [r for r in rows if r["method"] == "opaque_fill"]
    assert len(matches) == 1, "expected exactly one 'opaque_fill' row in benchmark.json"
    return matches[0]


def test_readme_and_design_card_exist():
    assert README.exists()
    assert DESIGN_CARD.exists()


def test_carmine_pigment_on_target_and_luminance_shift_vs_opaque():
    benchmark = _load(BENCHMARK)
    carmine = _carmine_row(benchmark)
    opaque = _opaque_fill_row(benchmark)

    pigment_on_target = f"{carmine['pigment_on_target']:.3f}"
    lip_luminance_shift = f"{carmine['lip_luminance_shift']:.1f}"
    opaque_luminance_shift = f"{opaque['lip_luminance_shift']:.1f}"

    assert pigment_on_target == "0.838"
    assert lip_luminance_shift == "13.2"
    assert opaque_luminance_shift == "45.7"

    readme = _read(README)
    design_card = _read(DESIGN_CARD)

    assert pigment_on_target in readme
    assert lip_luminance_shift in readme
    assert opaque_luminance_shift in readme

    assert lip_luminance_shift in design_card
    assert opaque_luminance_shift in design_card


def test_cpu_parity_worst_case_mean_p99_max():
    """README/DESIGN_CARD quote the worst case across all cpu comparisons
    (max of each metric across every frame/look pair), not an average."""
    metrics = _load(BROWSER_METRICS)
    cpu = metrics["parity"]["cpu"]

    worst_mean = f"{max(c['meanDeltaE'] for c in cpu):.3f}"
    worst_p99 = f"{max(c['p99DeltaE'] for c in cpu):.3f}"
    worst_max = f"{max(c['maxDeltaE'] for c in cpu):.3f}"

    assert worst_mean == "0.747"
    assert worst_p99 == "2.763"
    assert worst_max == "11.434"

    readme = _read(README)
    design_card = _read(DESIGN_CARD)

    for value in (worst_mean, worst_p99, worst_max):
        assert value in readme
    for value in (worst_mean, worst_p99):
        assert value in design_card


def test_end_to_end_parity_worst_case_mean():
    metrics = _load(BROWSER_METRICS)
    end_to_end = metrics["parity"]["endToEnd"]

    worst_mean = f"{max(c['meanDeltaE'] for c in end_to_end):.3f}"
    assert worst_mean == "2.717"

    readme = _read(README)
    design_card = _read(DESIGN_CARD)
    assert worst_mean in readme
    assert worst_mean in design_card


def test_live_path_total_ms_and_fps():
    metrics = _load(BROWSER_METRICS)
    live_total = metrics["timing_hardware"]["timing"]["livePath"]["totalMedian"]

    total_ms = f"{live_total:.1f}"
    fps = f"{1000.0 / live_total:.1f}"

    assert total_ms == "26.6"
    assert fps == "37.6"

    readme = _read(README)
    design_card = _read(DESIGN_CARD)

    assert total_ms in readme
    assert fps in readme
    assert total_ms in design_card
    assert fps in design_card
