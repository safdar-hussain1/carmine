"""Benchmark: the engine vs four standard failure-mode baselines, on photos.

Runs all five methods (`carmine.baselines.mismatched_indices`,
`opaque_fill`, `channel_swap`, `untrained_gan`, and the real engine,
`carmine.engine.apply_look`) over every portrait in a local no-makeup
photo directory, scoring each output on four honest, per-image metrics
from `carmine.metrics`:

* pigment_on_target   -- edit energy inside the legitimate product regions
                          (lips/eyes/blush), energy-weighted.
* background_untouched -- fraction of off-face pixels left bit-identical.
* lip_texture_kept    -- lip-region lightness correlation, before vs after.
* identity_ssim       -- grayscale SSIM of the whole image, before vs after.

Aggregate (mean per method) results are written to `reports/benchmark.json`
under the "photo" key.

Every method is scored under the "velvet" look preset (strong pigment
across every measurable product) so all five methods paint the same
requested colors and intensities. `carmine`'s own skin smoothing is
disabled for this run only: smoothing legitimately edits the whole face
and has no counterpart in any baseline, so leaving it on would make
pigment_on_target an apples-to-oranges comparison between a method that
edits the whole face and four that don't.

Usage:
    python scripts/benchmark.py --dataset data/no_makeup --out reports
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carmine import baselines, masks, metrics  # noqa: E402
from carmine.engine import apply_look  # noqa: E402
from carmine.landmarks import FaceLandmarker, NoFaceError  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

METHODS = ["mismatched_indices", "opaque_fill", "channel_swap", "untrained_gan", "carmine"]

PROTOCOL = {
    "pigment_on_target": (
        "fraction of total pixel-change energy that falls inside the union "
        "of the lip, eyeshadow, blush, and eyeliner masks (threshold 0.05)"
    ),
    "background_untouched": (
        "fraction of pixels outside a dilated face-hull mask that are "
        "bit-identical between input and output"
    ),
    "lip_texture_kept": (
        "Pearson correlation of Lab-L lightness inside the lip mask "
        "(threshold 0.5), before vs after"
    ),
    "identity_ssim": "grayscale structural similarity of output vs input, whole image",
}


def _valid_region_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Union of the legitimate product-placement masks, thresholded."""
    valid = np.zeros(shape, dtype=bool)
    for fn in (masks.lip_mask, masks.eyeshadow_mask, masks.blush_mask, masks.eyeliner_mask):
        valid |= fn(landmarks, shape) > 0.05
    return valid


def _face_hull_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convex hull of every landmark, dilated by 5% of its perimeter."""
    hull = cv2.convexHull(np.round(landmarks).astype(np.int32))
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    margin = max(3, int(cv2.arcLength(hull, True) * 0.05))
    return cv2.dilate(m, np.ones((margin, margin), np.uint8)) > 0


def _score(before: np.ndarray, after: np.ndarray, landmarks: np.ndarray) -> dict:
    shape = before.shape[:2]
    valid = _valid_region_mask(landmarks, shape)
    face = _face_hull_mask(landmarks, shape)
    lip = masks.lip_mask(landmarks, shape)
    return {
        "pigment_on_target": metrics.pigment_on_target(before, after, valid),
        "background_untouched": metrics.background_untouched(before, after, face),
        "lip_texture_kept": metrics.lip_texture_kept(before, after, lip),
        "identity_ssim": metrics.identity_ssim(before, after),
    }


def _run_methods(image: np.ndarray, landmarks: np.ndarray, look) -> dict[str, np.ndarray]:
    """Run every method on one image, returning {method: output}.

    Per-method input semantics are faithful to the studio-v1 driver this was
    ported from, adapted to this codebase's baseline signatures:

    * `mismatched_indices` there detected real 68-point dlib landmarks and
      applied 68-point region slices to them (correctly, since that IS the
      68-point scheme). Here, `carmine.baselines.mismatched_indices` takes
      the *478-point mesh* landmarks and misapplies the 68-point slices to
      them directly -- the bug it reproduces (68-point index ranges run
      against a 478-point array) is already baked into the function
      signature, so no separate dlib detection is needed or meaningful for
      this baseline; the mesh landmarks already detected for every other
      method are passed straight through.
    * `opaque_fill` and `channel_swap` take the correct mesh landmarks, same
      as the real engine.
    * `untrained_gan` takes only the image (and `look`, unused, for
      interface parity); no landmarks needed.
    """
    return {
        "mismatched_indices": baselines.mismatched_indices(image, landmarks, look),
        "opaque_fill": baselines.opaque_fill(image, landmarks, look),
        "channel_swap": baselines.channel_swap(image, landmarks, look),
        "untrained_gan": baselines.untrained_gan(image, look),
        "carmine": apply_look(image, look, landmarks=landmarks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "no_makeup")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--max-side", type=int, default=1200, help="downscale very large photos to this max side"
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"dataset directory not found: {args.dataset}")

    landmarker = FaceLandmarker()
    benchmark_look = dataclasses.replace(PRESETS["velvet"], smoothing=0.0)

    per_image: dict[str, list[dict]] = {m: [] for m in METHODS}
    runtimes: dict[str, list[float]] = {m: [] for m in METHODS}
    skipped: list[dict] = []

    files = sorted(args.dataset.glob("*.jpg"), key=lambda p: int(p.stem))
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            skipped.append({"name": path.name, "reason": "unreadable"})
            continue
        h, w = image.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        try:
            landmarks = landmarker.detect(image)
        except NoFaceError:
            skipped.append({"name": path.name, "reason": "no face detected"})
            continue

        for method in METHODS:
            t0 = time.perf_counter()
            out = _run_methods(image, landmarks, benchmark_look)[method]
            elapsed_ms = (time.perf_counter() - t0) * 1000
            runtimes[method].append(elapsed_ms)
            per_image[method].append(_score(image, out, landmarks))
        print(f"processed {path.name}", flush=True)

    rows = []
    for method in METHODS:
        scores = per_image[method]
        row = {"method": method}
        for key in ("pigment_on_target", "background_untouched", "lip_texture_kept", "identity_ssim"):
            row[key] = float(np.mean([s[key] for s in scores]))
        row["ms_per_image"] = float(np.mean(runtimes[method]))
        rows.append(row)

    n_images = len(per_image[METHODS[0]])
    result = {
        "photo": {"rows": rows},
        "meta": {
            "n_images": n_images,
            "skipped": skipped,
            "look_preset": "velvet",
            "protocol": PROTOCOL,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "benchmark.json"
    if out_json.is_file():
        existing = json.loads(out_json.read_text())
        existing.pop("photo", None)
        existing_meta = existing.pop("meta", {})
        result["meta"] = {**existing_meta, **result["meta"]}
        result = {**existing, **result}
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    print(f"\n{'method':<20}{'on_target':>10}{'backgnd':>9}{'lip_tex':>9}{'ssim':>7}{'ms':>8}")
    for row in rows:
        print(
            f"{row['method']:<20}"
            f"{row['pigment_on_target']:>10.3f}"
            f"{row['background_untouched']:>9.3f}"
            f"{row['lip_texture_kept']:>9.3f}"
            f"{row['identity_ssim']:>7.3f}"
            f"{row['ms_per_image']:>8.1f}"
        )
    if skipped:
        print(f"\nskipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
