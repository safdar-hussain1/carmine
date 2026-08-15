"""Exports the canned frames the cross-surface parity harness compares against.

The two engines -- `carmine` (NumPy + OpenCV) and the browser engine
(TypeScript + WebGL2) -- are independent implementations of the same math,
and the only honest way to claim they agree is to measure it. This script
produces the Python side of that measurement: a small set of frames, the
landmarks detected for each, and the reference render of each look.

**Why the landmarks are exported too.** Each side ships its own face
landmarker (MediaPipe's Python task API here, its wasm build there), and the
two do not return bit-identical points. If the browser detected its own
landmarks before rendering, a parity number would fold two unrelated
differences -- landmarker drift and rendering-math drift -- into one figure,
and a regression in either would be indistinguishable. So the fixtures pin
the landmarks: both sides render from the *same* points, and the resulting
number is purely about rendering. The landmarker gap is measured separately
(the browser re-detects on the same input PNG and renders again), which is
why the input PNG is exported rather than only the landmarks.

**Why the frames are downscaled to a 720px long side.** The browser engine
builds masks at a processing resolution capped at 720 on the long side
(`masks.ts`, `PROC_MAX_SIDE`). Feeding it a larger frame would make it
render at a different resolution than Python did, and comparing those means
comparing two resamplings rather than two engines. At 720 the two paths line
up 1:1.

**Why smoothing is off in every fixture look.** Skin smoothing is a
bilateral filter; the browser has no counterpart to it on either path (see
renderer.ts), so a look with smoothing on would measure a missing feature
rather than a disagreement.

The fixtures contain dataset faces, so `reports/parity_fixtures/` is
git-ignored and served only over localhost during verification -- never
shipped with the site.

Usage:
    python scripts/export_parity_fixtures.py [--dataset data/no_makeup]
                                             [--out reports/parity_fixtures]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carmine.engine import apply_look  # noqa: E402
from carmine.landmarks import FaceLandmarker, NoFaceError  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

# Matches masks.ts PROC_MAX_SIDE. Kept as a literal with this note rather
# than imported from the generated constants because it is a property of the
# browser's processing pipeline, not of the Python engine.
PROC_MAX_SIDE = 720

DEMO_PORTRAIT = ROOT / "web" / "public" / "demo" / "portrait.jpg"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

#: How many dataset portraits join the demo portrait in the fixture set.
DATASET_FRAMES = 2


def fixture_looks() -> dict[str, object]:
    """The looks each frame is rendered through.

    Two looks, chosen to cover disjoint halves of the engine:

    * ``velvet-satin`` is the benchmark look (`scripts/benchmark.py` uses the
      same overrides), exercising blush, eyeshadow, a lipstick tint and the
      flat eyeliner paint.
    * ``glass`` exercises the two gloss passes -- highlighter and a gloss
      lipstick -- which are the only places a whole-region percentile
      reduction feeds the per-pixel math.

    Both drop smoothing to zero. ``velvet``'s lipstick is forced to satin so
    the pair does not double up on gloss while leaving the plain tint path
    unmeasured; the matte finish is consequently *not* covered by these
    fixtures, and neither are brows (zero intensity in both presets).
    """
    velvet = PRESETS["velvet"]
    velvet_satin = dataclasses.replace(
        velvet,
        lipstick=dataclasses.replace(velvet.lipstick, finish="satin"),
        smoothing=0.0,
    )
    glass = dataclasses.replace(PRESETS["glass"], smoothing=0.0)
    return {"velvet-satin": velvet_satin, "glass": glass}


def downscale(image: np.ndarray, max_side: int = PROC_MAX_SIDE) -> np.ndarray:
    """Shrink `image` so its long side is at most `max_side`, or pass it through.

    INTER_AREA is the right filter for a downscale (it averages the source
    pixels a destination pixel covers rather than point-sampling them), and
    it is what the rest of the project uses.
    """
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image.copy()
    scale = max_side / long_side
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def select_frames(dataset: Path) -> list[tuple[str, Path]]:
    """The demo portrait plus the first `DATASET_FRAMES` dataset images.

    "First" is by sorted filename, so the fixture set is the same on every
    machine and every run -- a fixture set that drifted with directory order
    would make two parity numbers incomparable.

    Raises:
        FileNotFoundError: The demo portrait or the dataset directory is
            missing, or the dataset holds too few images.
    """
    if not DEMO_PORTRAIT.exists():
        raise FileNotFoundError(f"demo portrait not found at {DEMO_PORTRAIT}")
    frames: list[tuple[str, Path]] = [("portrait", DEMO_PORTRAIT)]

    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset}")
    candidates = sorted(
        (p for p in dataset.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if len(candidates) < DATASET_FRAMES:
        raise FileNotFoundError(
            f"need at least {DATASET_FRAMES} images in {dataset}, found {len(candidates)}"
        )
    for path in candidates[:DATASET_FRAMES]:
        frames.append((f"dataset-{path.stem}", path))
    return frames


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def export(dataset: Path, out_dir: Path) -> dict:
    """Render every fixture and write the manifest. Returns the manifest."""
    frames = select_frames(dataset)
    looks = fixture_looks()
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarker = FaceLandmarker()
    manifest_frames = []

    for name, source_path in frames:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read {source_path}")
        image = downscale(image)

        try:
            landmarks = landmarker.detect(image)
        except NoFaceError as error:
            raise ValueError(f"no face detected in {source_path}: {error}") from error

        input_name = f"{name}_input.png"
        # PNG, not JPEG: the browser has to read back the *same* bytes the
        # Python side rendered from, and a second JPEG round trip would put
        # a few levels of its own between the two engines.
        cv2.imwrite(str(out_dir / input_name), image)

        landmarks_name = f"{name}_landmarks.json"
        (out_dir / landmarks_name).write_text(
            json.dumps([[float(x), float(y)] for x, y in landmarks]),
            encoding="utf-8",
        )

        height, width = image.shape[:2]
        entry = {
            "name": name,
            "input": input_name,
            "landmarks": landmarks_name,
            "width": int(width),
            "height": int(height),
            "expected": {},
        }
        for look_name, look in looks.items():
            rendered = apply_look(image, look, landmarks=landmarks)
            expected_name = f"{name}_{look_name}_expected.png"
            cv2.imwrite(str(out_dir / expected_name), rendered)
            entry["expected"][look_name] = expected_name
        manifest_frames.append(entry)

    files: dict[str, str] = {}
    for entry in manifest_frames:
        for rel in [entry["input"], entry["landmarks"], *entry["expected"].values()]:
            files[rel] = sha256_of(out_dir / rel)

    manifest = {
        "version": 1,
        "proc_max_side": PROC_MAX_SIDE,
        "looks": {name: look.to_dict() for name, look in looks.items()},
        "frames": manifest_frames,
        "sha256": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "no_makeup",
        help="Directory of source portraits (default: data/no_makeup)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "parity_fixtures",
        help="Output directory (default: reports/parity_fixtures)",
    )
    args = parser.parse_args()

    manifest = export(args.dataset, args.out)
    print(f"wrote {len(manifest['sha256'])} files to {args.out}")
    for entry in manifest["frames"]:
        print(f"  {entry['name']}: {entry['width']}x{entry['height']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
