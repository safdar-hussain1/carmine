"""Guards the sample portrait's shipped landmarks against drifting.

The page draws every picture of its sample photo from
`web/src/gen/sample_landmarks.json` instead of running the face model, so
that file must stay what the model finds in that photo. It is regenerated
here with the same script and compared point by point. The model's
arithmetic differs in the last digits between CPUs, so a coordinate may move
a little; `TOLERANCE_PX` allows two pixels in a 1600-pixel photo, far less
than another photo, another model or a hand edit moves the points.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2

from carmine.regions import NUM_LANDMARKS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_sample_landmarks  # noqa: E402

COMMITTED_PATH = export_sample_landmarks.OUTPUT_PATH
TOLERANCE_PX = 2.0


def _committed() -> dict:
    return json.loads(COMMITTED_PATH.read_text(encoding="utf-8"))


def test_committed_landmarks_describe_the_sample():
    data = _committed()
    image = cv2.imread(str(export_sample_landmarks.SAMPLE_PATH), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    assert data["source"] == "demo/model.jpg"
    assert (data["width"], data["height"]) == (width, height)
    assert len(data["points"]) == NUM_LANDMARKS
    # The model places a face's points even where they run off the edge (the
    # sample's chin ends a few pixels below the frame), so allow a margin.
    for x, y, z in data["points"]:
        assert -0.05 * width <= x <= 1.05 * width
        assert -0.05 * height <= y <= 1.05 * height
        assert math.isfinite(z)


def test_committed_landmarks_match_the_model(tmp_path):
    regenerated_path = tmp_path / "sample_landmarks.json"
    sample = export_sample_landmarks.detect_with_depth(export_sample_landmarks.SAMPLE_PATH)
    export_sample_landmarks.write(sample, regenerated_path)
    committed = _committed()
    regenerated = json.loads(regenerated_path.read_text(encoding="utf-8"))
    assert (committed["width"], committed["height"]) == (regenerated["width"], regenerated["height"])
    worst = max(
        abs(a - b)
        for p, q in zip(committed["points"], regenerated["points"], strict=True)
        for a, b in zip(p, q, strict=True)
    )
    assert worst <= TOLERANCE_PX, (
        f"{COMMITTED_PATH.relative_to(REPO_ROOT)} is out of sync with the model "
        f"(a coordinate moved {worst:.2f} px); regenerate it with: "
        "PYTHONPATH=src python scripts/export_sample_landmarks.py"
    )
