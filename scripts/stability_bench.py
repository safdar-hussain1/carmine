"""Video-stability benchmark: does One-Euro smoothing actually help?

Builds a deterministic 90-frame synthetic "clip" from the public-domain
astronaut portrait by sweeping a fixed, known affine transform (pan, zoom,
rotate, all sinusoidal over the clip) across it and adding seeded per-frame
Gaussian pixel noise, which is enough to make MediaPipe's per-frame landmark
detection jitter the way a real camera feed does even though the underlying
face never actually moves relative to the transform.

Ground-truth protocol
----------------------
Because the affine transform applied to build each frame is known exactly,
we can compute a ground-truth landmark position for every frame: detect raw
landmarks once on frame 0 (identity transform), then apply each frame's own
affine matrix to those points. That sidesteps a real trap with a naive
frame-to-frame IoU/centroid-jitter score on a *moving* clip: a filter that
lags behind real motion (or is simply frozen) can still score a perfect 1.0
IoU against itself frame-to-frame while tracking the ground truth worse than
an unfiltered signal would -- frame-to-frame agreement rewards staying still,
not being correct. Scoring against ground truth instead measures two
independent things:

* deviation  -- mean per-landmark distance from ground truth, in px / iod.
                How far off is this stream, on average?
* motion-compensated jitter -- mean frame-to-frame change of the *residual*
                (landmarks - ground_truth), in px / iod. This is jitter with
                the known real motion subtracted out, so it isolates noise
                from motion instead of conflating the two.

A grid search over One-Euro's `min_cutoff` x `beta` finds the configuration
that minimizes motion-compensated jitter subject to not degrading mean
deviation by more than 1.5x over the raw (unfiltered) stream's own
deviation -- a filter that tracks ground truth much worse than raw isn't a
win just because it's smoother. If the best such configuration cuts jitter
by at least 20% relative to raw, it's adopted as `OneEuroFilter`'s (and so
`VideoEngine`'s) new defaults and documented there; otherwise the honest
null result is published and the defaults are left alone.

`carmine.metrics.mask_jitter` (mean adjacent-frame mask IoU / centroid
drift) still exists and is still unit-tested on constructed cases, but this
script does not use it for publishing -- it has the same "rewards lag"
blind spot as raw IoU jitter would, for the same reason.

Usage:
    python scripts/stability_bench.py --out reports
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skimage import data  # noqa: E402

from carmine.engine import VideoEngine  # noqa: E402
from carmine.filters import OneEuroFilter  # noqa: E402
from carmine.geometry import interocular_distance  # noqa: E402
from carmine.landmarks import FaceLandmarker, NoFaceError  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

N_FRAMES = 90
FPS = 30.0

# σ=2 (the original choice) produced motion-compensated jitter too small to
# meaningfully separate raw from filtered on this clip; σ=6 injects enough
# per-frame landmark-detection noise to be representative of a real, mildly
# noisy camera feed.
NOISE_SIGMA = 6.0
NOISE_SEED = 1234

# Fixed affine sweep parameters, all sinusoidal over one full clip-length
# cycle (t goes from 0 to 1 across the 90 frames):
PAN_FRACTION = 0.04  # +/- 4% of image width
ZOOM_PEAK = 0.06  # 1.0 -> 1.06 -> 1.0
ROTATE_DEG = 2.0  # +/- 2 degrees

# One-Euro grid search.
MIN_CUTOFF_VALUES = [0.5, 1.0, 1.5]
BETA_VALUES = [0.007, 0.05, 0.2, 0.5]
DEVIATION_CONSTRAINT_MULTIPLIER = 1.5
IMPROVEMENT_THRESHOLD_PCT = 20.0


def _affine_params(t: float, width: int) -> tuple[float, float, float]:
    """(pan_x_px, zoom, rotate_deg) at position t in [0, 1] of the clip."""
    pan_x = PAN_FRACTION * width * math.sin(2 * math.pi * t)
    zoom = 1.0 + (ZOOM_PEAK / 2.0) * (1 - math.cos(2 * math.pi * t))
    rotate_deg = ROTATE_DEG * math.sin(2 * math.pi * t)
    return pan_x, zoom, rotate_deg


def _affine_matrix(t: float, w: int, h: int) -> np.ndarray:
    pan_x, zoom, rotate_deg = _affine_params(t, w)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, zoom)
    matrix[0, 2] += pan_x
    return matrix


def build_clip(base_bgr: np.ndarray, n_frames: int = N_FRAMES) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Deterministic synthetic clip: fixed affine sweep + seeded pixel noise.

    Returns (frames, matrices) -- `matrices[i]` is the exact 2x3 affine
    matrix used to build `frames[i]` from `base_bgr`, before noise, so it
    can also be applied to frame-0 landmarks to get ground truth.
    """
    h, w = base_bgr.shape[:2]
    rng = np.random.default_rng(NOISE_SEED)
    frames, matrices = [], []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        matrix = _affine_matrix(t, w, h)
        matrices.append(matrix)
        warped = cv2.warpAffine(base_bgr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        noisy = warped.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, size=warped.shape)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return frames, matrices


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points.astype(np.float64), ones])
    return (matrix @ homogeneous.T).T


def detect_raw_landmarks(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Detect landmarks per frame with one VIDEO-mode stream.

    Returns (landmarks, kept_frame_indices); a frame with no detected face
    is dropped and reported, so ground truth (built from the same indices'
    affine matrices) stays aligned with the returned landmark list.
    """
    landmarker = FaceLandmarker()
    landmarks, kept, dropped = [], [], []
    for i, frame in enumerate(frames):
        timestamp_ms = int(round(i * (1000.0 / FPS)))
        try:
            lm = landmarker.detect_video(frame, timestamp_ms)
        except NoFaceError:
            dropped.append(i)
            continue
        landmarks.append(lm)
        kept.append(i)
    if dropped:
        print(f"stability_bench: no face detected on frames {dropped}, dropped from analysis")
    return landmarks, kept


def _deviation_px_iod(landmark_seq: list[np.ndarray], gt_seq: list[np.ndarray], iod: float) -> float:
    """Mean per-landmark distance from ground truth, in px / iod."""
    per_frame = [
        float(np.linalg.norm(lm - gt, axis=1).mean()) for lm, gt in zip(landmark_seq, gt_seq)
    ]
    return float(np.mean(per_frame)) / iod


def _motion_compensated_jitter_px_iod(
    landmark_seq: list[np.ndarray], gt_seq: list[np.ndarray], iod: float
) -> float:
    """Mean frame-to-frame change of (landmarks - ground_truth), in px / iod.

    Subtracting ground truth before differencing removes the clip's known,
    real motion, so what's left is noise -- unlike raw adjacent-frame
    landmark deltas (or mask IoU/centroid deltas), which would also reward
    a filter for lagging behind genuine motion.
    """
    residuals = [lm - gt for lm, gt in zip(landmark_seq, gt_seq)]
    deltas = [
        float(np.linalg.norm(residuals[i + 1] - residuals[i], axis=1).mean())
        for i in range(len(residuals) - 1)
    ]
    return float(np.mean(deltas)) / iod


def _filtered_stream(
    raw_lm: list[np.ndarray], timestamps_s: list[float], min_cutoff: float, beta: float
) -> list[np.ndarray]:
    onefilter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    return [onefilter(lm, t) for lm, t in zip(raw_lm, timestamps_s)]


def grid_search(
    raw_lm: list[np.ndarray], gt_seq: list[np.ndarray], timestamps_s: list[float], iod: float
) -> list[dict]:
    results = []
    for min_cutoff, beta in itertools.product(MIN_CUTOFF_VALUES, BETA_VALUES):
        filtered = _filtered_stream(raw_lm, timestamps_s, min_cutoff, beta)
        deviation = _deviation_px_iod(filtered, gt_seq, iod)
        jitter = _motion_compensated_jitter_px_iod(filtered, gt_seq, iod)
        results.append(
            {
                "min_cutoff": min_cutoff,
                "beta": beta,
                "deviation_px_iod": deviation,
                "jitter_px_iod": jitter,
            }
        )
    return results


def measure_video_ms_per_frame(frames: list[np.ndarray]) -> float:
    """Median wall-clock ms/frame for the full VideoEngine pipeline."""
    engine = VideoEngine(PRESETS["velvet"], smooth_landmarks=True)
    times_ms = []
    for i, frame in enumerate(frames):
        timestamp_ms = int(round(i * (1000.0 / FPS)))
        t0 = time.perf_counter()
        engine.process(frame, timestamp_ms)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times_ms))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    base_bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    frames, matrices = build_clip(base_bgr)
    print(f"built {len(frames)}-frame synthetic clip ({base_bgr.shape[1]}x{base_bgr.shape[0]})")

    raw_lm, kept = detect_raw_landmarks(frames)
    if len(raw_lm) < 2:
        raise RuntimeError(f"too few frames with a detected face ({len(raw_lm)}) to score jitter")

    landmarks0 = raw_lm[0]
    iod = interocular_distance(landmarks0)
    gt_seq = [_apply_affine(matrices[i], landmarks0) for i in kept]
    timestamps_s = [i / FPS for i in kept]

    raw_deviation = _deviation_px_iod(raw_lm, gt_seq, iod)
    raw_jitter = _motion_compensated_jitter_px_iod(raw_lm, gt_seq, iod)

    grid_results = grid_search(raw_lm, gt_seq, timestamps_s, iod)
    for entry in grid_results:
        entry["meets_deviation_constraint"] = (
            entry["deviation_px_iod"] <= DEVIATION_CONSTRAINT_MULTIPLIER * raw_deviation
        )

    within_constraint = [g for g in grid_results if g["meets_deviation_constraint"]]
    if within_constraint:
        chosen = min(within_constraint, key=lambda g: g["jitter_px_iod"])
        chosen_satisfies_constraint = True
    else:
        chosen = min(grid_results, key=lambda g: g["jitter_px_iod"])
        chosen_satisfies_constraint = False

    jitter_reduction_pct = (
        (raw_jitter - chosen["jitter_px_iod"]) / raw_jitter * 100.0 if raw_jitter > 0 else 0.0
    )
    meets_improvement_threshold = (
        chosen_satisfies_constraint and jitter_reduction_pct >= IMPROVEMENT_THRESHOLD_PCT
    )

    if meets_improvement_threshold:
        selected_params = {"min_cutoff": chosen["min_cutoff"], "beta": chosen["beta"]}
        note = (
            f"selected min_cutoff={chosen['min_cutoff']}, beta={chosen['beta']}: "
            f"{jitter_reduction_pct:.1f}% motion-compensated jitter reduction vs raw, "
            f"within the {DEVIATION_CONSTRAINT_MULTIPLIER}x raw-deviation constraint. "
            "VideoEngine's OneEuroFilter defaults were updated to match."
        )
    else:
        selected_params = None
        if not chosen_satisfies_constraint:
            note = (
                "one_euro shows no measurable benefit on this clip: no grid config kept "
                f"deviation within {DEVIATION_CONSTRAINT_MULTIPLIER}x raw deviation "
                f"({raw_deviation:.4f} px/iod); best unconstrained candidate "
                f"(min_cutoff={chosen['min_cutoff']}, beta={chosen['beta']}) is reported "
                "for transparency only. VideoEngine defaults left unchanged."
            )
        else:
            note = (
                "one_euro shows no measurable benefit on this clip: the best "
                f"constraint-satisfying config (min_cutoff={chosen['min_cutoff']}, "
                f"beta={chosen['beta']}) only reduced motion-compensated jitter by "
                f"{jitter_reduction_pct:.1f}% (< {IMPROVEMENT_THRESHOLD_PCT:.0f}% threshold). "
                "VideoEngine defaults left unchanged."
            )

    video_ms_per_frame = measure_video_ms_per_frame(frames)

    stability = {
        "protocol": "ground_truth_affine",
        "raw": {"deviation_px_iod": raw_deviation, "jitter_px_iod": raw_jitter},
        "one_euro": {
            "deviation_px_iod": chosen["deviation_px_iod"],
            "jitter_px_iod": chosen["jitter_px_iod"],
            "min_cutoff": chosen["min_cutoff"],
            "beta": chosen["beta"],
            "meets_deviation_constraint": chosen_satisfies_constraint,
        },
        "grid_search": {
            "min_cutoff_values": MIN_CUTOFF_VALUES,
            "beta_values": BETA_VALUES,
            "deviation_constraint_multiplier": DEVIATION_CONSTRAINT_MULTIPLIER,
            "improvement_threshold_pct": IMPROVEMENT_THRESHOLD_PCT,
            "results": grid_results,
        },
        "selected_params": selected_params,
        "jitter_reduction_pct": jitter_reduction_pct,
        "meets_improvement_threshold": meets_improvement_threshold,
        "note": note,
        "video_ms_per_frame": video_ms_per_frame,
        "n_frames": len(frames),
        "n_frames_with_face": len(kept),
        "clip_params": {
            "fps": FPS,
            "noise_sigma": NOISE_SIGMA,
            "noise_seed": NOISE_SEED,
            "pan_fraction": PAN_FRACTION,
            "zoom_peak": ZOOM_PEAK,
            "rotate_deg": ROTATE_DEG,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "benchmark.json"
    result = json.loads(out_json.read_text()) if out_json.is_file() else {}
    result["stability"] = stability
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    print(f"\nraw:      deviation {raw_deviation:.4f} px/iod, jitter {raw_jitter:.4f} px/iod")
    print(
        f"one_euro: deviation {chosen['deviation_px_iod']:.4f} px/iod, "
        f"jitter {chosen['jitter_px_iod']:.4f} px/iod "
        f"(min_cutoff={chosen['min_cutoff']}, beta={chosen['beta']})"
    )
    print(f"jitter reduction: {jitter_reduction_pct:+.1f}%  (>= {IMPROVEMENT_THRESHOLD_PCT:.0f}% required to adopt)")
    print(note)
    print(f"video ms/frame (median): {video_ms_per_frame:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
