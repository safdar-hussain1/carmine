"""Guards the generated files under web/src/gen against drifting from Python.

Both files are regenerated to a temp path with the same generator that
produced the committed copy and compared against it, so a Python change that
touches an exported value (or a hand-edit of a committed JSON) fails loudly
instead of quietly diverging from the browser mirror.

`constants.json` carries the values both engines read at runtime. They are
configuration, not computation, so the comparison is byte-for-byte.

`test_vectors.json` carries expected outputs the TypeScript test suite
asserts against -- and those matter even more, because a stale fixture does
not make a test fail, it makes the test keep passing while measuring the
wrong thing. Its floats are *computed* (OpenCV's float32 Lab conversion, the
One-Euro filter's arithmetic), and the last digits of those differ between
CPUs: regenerated on linux/x86_64 against a file written on macOS/arm64, ten
floats move by up to ~7e-5 while every integer pixel is identical. So this
file is compared as data: the same keys, the same list lengths, integers
exact, and floats within `FLOAT_REL_TOL` / `FLOAT_ABS_TOL` -- tight enough
that swapping two pixel values or breaking the filter's derivative still
fails, loose enough for another CPU's rounding.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_constants  # noqa: E402
import export_test_vectors  # noqa: E402

GEN_DIR = REPO_ROOT / "web" / "src" / "gen"
COMMITTED_PATH = GEN_DIR / "constants.json"
VECTORS_PATH = GEN_DIR / "test_vectors.json"

FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-3


def _assert_matches(committed_path, regenerated_path, regenerate_command):
    assert committed_path.exists(), f"missing committed file: {committed_path}"
    committed_bytes = committed_path.read_bytes()
    regenerated_bytes = regenerated_path.read_bytes()

    assert regenerated_bytes == committed_bytes, (
        f"{committed_path.relative_to(REPO_ROOT)} is out of sync with its generator; "
        f"regenerate it with: {regenerate_command}"
    )


def _data_mismatches(committed, regenerated, path="$"):
    """Every place two JSON values differ, with floats compared to tolerance."""
    if type(committed) is not type(regenerated):
        return [f"{path}: {type(committed).__name__} != {type(regenerated).__name__}"]
    if isinstance(committed, float):
        if math.isclose(committed, regenerated, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return []
        return [f"{path}: {committed!r} != {regenerated!r}"]
    if isinstance(committed, dict):
        if committed.keys() != regenerated.keys():
            return [f"{path}: keys {sorted(committed)} != {sorted(regenerated)}"]
        return [
            problem
            for key in committed
            for problem in _data_mismatches(committed[key], regenerated[key], f"{path}.{key}")
        ]
    if isinstance(committed, list):
        if len(committed) != len(regenerated):
            return [f"{path}: length {len(committed)} != {len(regenerated)}"]
        return [
            problem
            for index, (a, b) in enumerate(zip(committed, regenerated))
            for problem in _data_mismatches(a, b, f"{path}[{index}]")
        ]
    return [] if committed == regenerated else [f"{path}: {committed!r} != {regenerated!r}"]


def test_data_comparison_tolerates_rounding_but_not_changes():
    """The comparison itself: CPU-level float noise passes, real edits fail."""
    base = {"lab": [0.008035869, 53.2409], "rgb": [150, 79, 220], "n": 4}
    rounding = {"lab": [0.008022091, 53.24090001], "rgb": [150, 79, 220], "n": 4}
    assert _data_mismatches(base, rounding) == []
    for changed in (
        {"lab": [0.008035869, 53.2509], "rgb": [150, 79, 220], "n": 4},  # float off by 0.01
        {"lab": [0.008035869, 53.2409], "rgb": [220, 79, 150], "n": 4},  # pixels swapped
        {"lab": [0.008035869, 53.2409], "rgb": [150, 79], "n": 4},  # value dropped
        {"lab": [0.008035869, 53.2409], "rgb": [150, 79, 220]},  # key dropped
        {"lab": [0.008035869, 53.2409], "rgb": [150, 79, 220], "n": 4.0},  # int became float
    ):
        assert _data_mismatches(base, changed), changed


def test_constants_json_matches_generator(tmp_path):
    regenerated_path = tmp_path / "constants.json"
    export_constants.write_constants(regenerated_path)
    _assert_matches(
        COMMITTED_PATH,
        regenerated_path,
        "PYTHONPATH=src python scripts/export_constants.py",
    )


def test_test_vectors_json_matches_generator(tmp_path):
    regenerated_path = tmp_path / "test_vectors.json"
    export_test_vectors.write_vectors(regenerated_path)
    assert VECTORS_PATH.exists(), f"missing committed file: {VECTORS_PATH}"
    committed = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    regenerated = json.loads(regenerated_path.read_text(encoding="utf-8"))
    mismatches = _data_mismatches(committed, regenerated)
    assert not mismatches, (
        f"{VECTORS_PATH.relative_to(REPO_ROOT)} is out of sync with its generator "
        f"({len(mismatches)} values differ, first: {mismatches[:5]}); regenerate it with: "
        "PYTHONPATH=src python scripts/export_test_vectors.py"
    )
