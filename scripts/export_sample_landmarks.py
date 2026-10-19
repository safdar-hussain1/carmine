"""Finds the face in the sample portrait once and writes its landmarks to the site.

The page opens on `web/public/demo/model.jpg`. Every picture it draws of that
photo -- the opening scan, the looks, the how-it-works animations -- starts
from its 478 face landmarks. Detecting them in the browser means downloading
the face model and its wasm runtime (about 16 MB together) before anything
can move, so the landmarks are found here, with the same model file, and
shipped as a small JSON file. The browser builds the masks and paints the
makeup from them with its own engine; the model is only fetched when a
visitor turns on a camera or opens a photo.

Each point is `[x, y, z]` in the photo's pixels. `z` is MediaPipe's depth,
which uses roughly the same scale as `x` (so it is multiplied by the width
here); smaller means closer to the camera. The page uses it to turn the mesh
in 3D. Values are rounded to two decimals, which is far below a pixel and
keeps the file small.

Run as: PYTHONPATH=src python scripts/export_sample_landmarks.py
(set CARMINE_MODEL to the bundled model to skip the download).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from carmine.landmarks import NUM_LANDMARKS, model_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "web" / "public" / "demo" / "model.jpg"
OUTPUT_PATH = REPO_ROOT / "web" / "src" / "gen" / "sample_landmarks.json"


def detect_with_depth(image_path: Path) -> dict:
    """The sample's landmarks as pixel coordinates with depth."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read {image_path}")
    height, width = image_bgr.shape[:2]
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path())),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise RuntimeError(f"no face found in {image_path}")
    face = result.face_landmarks[0]
    if len(face) != NUM_LANDMARKS:
        raise RuntimeError(f"expected {NUM_LANDMARKS} landmarks, got {len(face)}")

    points = [
        [round(p.x * width, 2), round(p.y * height, 2), round(p.z * width, 2)] for p in face
    ]
    return {"source": "demo/model.jpg", "width": width, "height": height, "points": points}


def write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One point per line: compact, and a diff shows which points moved.
    rows = ",\n    ".join(json.dumps(point, separators=(",", ":")) for point in data["points"])
    text = (
        "{\n"
        f'  "source": {json.dumps(data["source"])},\n'
        f'  "width": {data["width"]},\n'
        f'  "height": {data["height"]},\n'
        f'  "points": [\n    {rows}\n  ]\n'
        "}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="where to write the JSON")
    args = parser.parse_args()
    data = detect_with_depth(SAMPLE_PATH)
    write(data, args.out)
    print(f"wrote {args.out} ({len(data['points'])} points, {data['width']}x{data['height']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
