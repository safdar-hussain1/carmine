"""Benchmark: 2024 legacy pipelines vs the rewrite, on real photos.

Runs every method on the no-makeup half of the paired makeup dataset and
scores them on four honest, per-image metrics:

* containment   — fraction of the total edit energy (summed absolute
                  pixel change) that falls inside legitimate makeup
                  regions (lips/lids/lashes/cheeks); pigment painted on
                  chins or backgrounds lowers it. Energy-weighted so a
                  one-count rounding ripple in a feathered tail doesn't
                  count the same as an opaque purple polygon.
* background    — fraction of pixels outside the face that are
                  bit-identical to the input (1.0 = untouched).
* lip texture   — Pearson correlation of lip-region lightness before vs
                  after; flat fills erase texture and drive this to ~0.
* identity SSIM — grayscale SSIM of output vs input; makeup should
                  restyle, not replace, the photo.

It also re-runs the original notebook's own evaluation protocol (SSIM
against the with-makeup reference photos + threshold "accuracy") to
verify the 2024 report's numbers and demonstrate why they were
meaningless.

Usage:
    python scripts/benchmark.py --dataset DIR --shape-predictor FILE.dat \
        [--out reports]

The dataset directory must contain no_makeup/ and with_makeup/ folders
(the paired makeup dataset the 2024 course project used). Neither the
dataset nor the dlib model is committed to this repo.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skimage.metrics import structural_similarity  # noqa: E402

from virtual_makeup import PRESETS, apply_makeup, legacy, masks  # noqa: E402
from virtual_makeup.landmarks import FaceLandmarker, NoFaceDetectedError  # noqa: E402


def valid_region_mask(landmarks: np.ndarray, shape) -> np.ndarray:
    valid = np.zeros(shape, dtype=bool)
    for fn in (masks.lip_mask, masks.eyeshadow_mask, masks.blush_mask,
               masks.eyeliner_mask):
        valid |= fn(landmarks, shape) > 0.05
    return valid


def face_hull_mask(landmarks: np.ndarray, shape) -> np.ndarray:
    hull = cv2.convexHull(np.round(landmarks).astype(np.int32))
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    margin = max(3, int(cv2.arcLength(hull, True) * 0.05))
    return cv2.dilate(m, np.ones((margin, margin), np.uint8)) > 0


def score(image: np.ndarray, output: np.ndarray, mesh_lm: np.ndarray) -> dict:
    shape = image.shape[:2]
    if output.shape != image.shape:
        output = cv2.resize(output, (image.shape[1], image.shape[0]))
    energy = np.abs(output.astype(np.int16) - image.astype(np.int16)).sum(axis=2)
    valid = valid_region_mask(mesh_lm, shape)
    face = face_hull_mask(mesh_lm, shape)

    total = float(energy.sum())
    containment = float(energy[valid].sum() / total) if total > 0 else 1.0
    outside = ~face
    background = float(np.all(output[outside] == image[outside], axis=1).mean())

    lip = masks.lip_mask(mesh_lm, shape) > 0.5
    l_before = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[..., 0][lip].astype(float)
    l_after = cv2.cvtColor(output, cv2.COLOR_BGR2Lab)[..., 0][lip].astype(float)
    if l_before.std() < 1e-6 or l_after.std() < 1e-6:
        texture = 0.0
    else:
        texture = float(np.corrcoef(l_before, l_after)[0, 1])

    identity = float(structural_similarity(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),
    ))
    return {
        "containment": containment,
        "background_integrity": background,
        "lip_texture_corr": texture,
        "identity_ssim": identity,
    }


def detect_dlib68(detector, predictor, image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return None
    shape = predictor(gray, faces[0])
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)],
                    dtype=np.float32)


def original_protocol(outputs: dict[str, dict[str, np.ndarray]],
                      reference_dir: Path) -> dict:
    """The 2024 notebook's evaluation, reproduced verbatim: grayscale
    SSIM of each output against the *with-makeup reference photo*, then
    'accuracy/precision/recall' from an all-ones y_true."""
    results = {}
    for method, images in outputs.items():
        scores = []
        for name, out in images.items():
            ref = cv2.imread(str(reference_dir / name))
            if ref is None:
                continue
            if out.shape != ref.shape:
                ref = cv2.resize(ref, (out.shape[1], out.shape[0]))
            scores.append(structural_similarity(
                cv2.cvtColor(out, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY),
            ))
        results[method] = {
            "mean_ssim_vs_reference": float(np.mean(scores)),
            "legacy_metrics@0.45": legacy.legacy_metrics(scores),
            "n": len(scores),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shape-predictor", type=Path, required=True,
                        help="dlib shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--max-side", type=int, default=1200,
                        help="downscale very large photos to this max side")
    args = parser.parse_args()

    no_makeup = args.dataset / "no_makeup"
    with_makeup = args.dataset / "with_makeup"
    if not no_makeup.is_dir() or not with_makeup.is_dir():
        parser.error(f"{args.dataset} must contain no_makeup/ and with_makeup/")
    if not args.shape_predictor.is_file():
        parser.error(f"shape predictor not found: {args.shape_predictor}")

    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.shape_predictor))
    landmarker = FaceLandmarker()

    methods = ["legacy_mediapipe", "legacy_dlib", "legacy_dlib_swap",
               "legacy_gan", "new_classic"]
    per_image: dict[str, list[dict]] = {m: [] for m in methods}
    runtimes: dict[str, list[float]] = {m: [] for m in methods}
    outputs: dict[str, dict[str, np.ndarray]] = {m: {} for m in methods}
    skipped = []

    files = sorted(no_makeup.glob("*.jpg"), key=lambda p: int(p.stem))
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            skipped.append((path.name, "unreadable"))
            continue
        h, w = image.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        try:
            mesh_lm = landmarker.detect(image)
        except NoFaceDetectedError:
            skipped.append((path.name, "no face (mediapipe)"))
            continue
        lm68 = detect_dlib68(detector, predictor, image)
        if lm68 is None:
            skipped.append((path.name, "no face (dlib)"))
            continue

        # The new pipeline runs with smoothing disabled: skin smoothing
        # legitimately edits the whole face and has no legacy
        # counterpart, so including it would make the containment metric
        # an apples-to-oranges comparison.
        benchmark_look = dataclasses.replace(PRESETS["classic"], smoothing=0.0)
        runs = {
            "legacy_mediapipe": lambda: legacy.legacy_mediapipe(image, mesh_lm),
            "legacy_dlib": lambda: legacy.legacy_dlib(image, lm68),
            "legacy_dlib_swap": lambda: legacy.legacy_dlib(image, lm68,
                                                           channel_swap_bug=True),
            "legacy_gan": lambda: legacy.legacy_gan(image),
            "new_classic": lambda: apply_makeup(image, benchmark_look,
                                                landmarks=mesh_lm),
        }
        for method, run in runs.items():
            t0 = time.perf_counter()
            out = run()
            runtimes[method].append(time.perf_counter() - t0)
            outputs[method][path.name] = out
            per_image[method].append(score(image, out, mesh_lm))
        print(f"processed {path.name}", flush=True)

    summary = {}
    for method in methods:
        rows = per_image[method]
        summary[method] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in rows])),
                "min": float(np.min([r[metric] for r in rows])),
                "max": float(np.max([r[metric] for r in rows])),
            }
            for metric in rows[0]
        }
        summary[method]["runtime_ms"] = float(np.mean(runtimes[method]) * 1000)
        summary[method]["n_images"] = len(rows)

    protocol = original_protocol(outputs, with_makeup)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": summary,
        "original_protocol_reproduction": protocol,
        "per_image": {m: rows for m, rows in per_image.items()},
        "skipped": skipped,
    }
    out_json = args.out / "benchmark.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    print(f"\n{'method':<20}{'contain':>9}{'backgnd':>9}{'texture':>9}"
          f"{'ssim':>7}{'ms':>8}")
    for method in methods:
        s = summary[method]
        print(f"{method:<20}"
              f"{s['containment']['mean']:>9.3f}"
              f"{s['background_integrity']['mean']:>9.3f}"
              f"{s['lip_texture_corr']['mean']:>9.3f}"
              f"{s['identity_ssim']['mean']:>7.3f}"
              f"{s['runtime_ms']:>8.1f}")
    print("\noriginal protocol (SSIM vs reference + all-ones metrics):")
    for method, r in protocol.items():
        m = r["legacy_metrics@0.45"]
        print(f"{method:<20} ssim={r['mean_ssim_vs_reference']:.3f} "
              f"acc={m['Accuracy']:.3f} prec={m['Precision']:.3f}")
    if skipped:
        print(f"\nskipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
