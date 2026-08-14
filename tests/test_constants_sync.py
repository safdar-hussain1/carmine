"""Guards against web/src/gen/constants.json drifting from its generator.

Regenerates the constants JSON to a temp file with the same generator that
produced the committed copy and asserts byte-for-byte equality, so a Python
change that touches an exported value (or a hand-edit of the committed JSON)
fails loudly instead of quietly diverging from the browser mirror.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_constants  # noqa: E402

COMMITTED_PATH = REPO_ROOT / "web" / "src" / "gen" / "constants.json"


def test_constants_json_matches_generator(tmp_path):
    regenerated_path = tmp_path / "constants.json"
    export_constants.write_constants(regenerated_path)

    assert COMMITTED_PATH.exists(), f"missing committed constants file: {COMMITTED_PATH}"
    committed_bytes = COMMITTED_PATH.read_bytes()
    regenerated_bytes = regenerated_path.read_bytes()

    assert regenerated_bytes == committed_bytes, (
        "web/src/gen/constants.json is out of sync with scripts/export_constants.py; "
        "regenerate it with: PYTHONPATH=src python scripts/export_constants.py"
    )
