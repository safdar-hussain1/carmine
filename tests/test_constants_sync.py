"""Guards the generated files under web/src/gen against drifting from Python.

Both files are regenerated to a temp path with the same generator that
produced the committed copy and compared byte-for-byte, so a Python change
that touches an exported value (or a hand-edit of a committed JSON) fails
loudly instead of quietly diverging from the browser mirror.

`constants.json` carries the values both engines read at runtime.
`test_vectors.json` carries expected outputs the TypeScript test suite
asserts against -- and those matter even more, because a stale fixture does
not make a test fail, it makes the test keep passing while measuring the
wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_constants  # noqa: E402
import export_test_vectors  # noqa: E402

GEN_DIR = REPO_ROOT / "web" / "src" / "gen"
COMMITTED_PATH = GEN_DIR / "constants.json"
VECTORS_PATH = GEN_DIR / "test_vectors.json"


def _assert_matches(committed_path, regenerated_path, regenerate_command):
    assert committed_path.exists(), f"missing committed file: {committed_path}"
    committed_bytes = committed_path.read_bytes()
    regenerated_bytes = regenerated_path.read_bytes()

    assert regenerated_bytes == committed_bytes, (
        f"{committed_path.relative_to(REPO_ROOT)} is out of sync with its generator; "
        f"regenerate it with: {regenerate_command}"
    )


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
    _assert_matches(
        VECTORS_PATH,
        regenerated_path,
        "PYTHONPATH=src python scripts/export_test_vectors.py",
    )
