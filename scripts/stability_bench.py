"""Video-stability benchmark: does One-Euro smoothing actually help?

Builds a deterministic 90-frame synthetic "clip" from the public-domain
astronaut portrait by sweeping a fixed affine transform (pan, zoom, rotate,
all sinusoidal over the clip) across it and adding seeded per-frame Gaussian
pixel noise, which is enough to make MediaPipe's per-frame landmark
detection jitter the way a real camera feed does even though the underlying
face never actually moves relative to the transform.

For each frame we build the lip+eyeshadow region (the union of
`carmine.masks.lip_mask` and `carmine.masks.eyeshadow_mask`) from two
landmark streams -- raw per-frame detections, and the same stream run
through `carmine.filters.OneEuroFilter` (the smoothing `VideoEngine` uses by
default) -- and score each stream's frame-to-frame stability with
`carmine.metrics.mask_jitter`. The results (and the relative improvement
one_euro gives over raw) are written into `reports/benchmark.json` under the
"stability" key, alongside a median per-frame wall-clock cost for the full
`VideoEngine` pipeline (detection + smoothing + painting).

Usage:
    python scripts/stability_bench.py --out reports
"""

from __future__ import annotations

import argparse
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

from carmine import masks, metrics  # noqa: E402
from carmine.engine import VideoEngine  # noqa: E402
from carmine.filters import OneEuroFilter  # noqa: E402
from carmine.geometry import interocular_distance  # noqa: E402
from carmine.landmarks import FaceLandmarker, NoFaceError  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

N_FRAMES = 90
FPS = 30.0
NOISE_SIGMA = 2.0
NOISE_SEED = 1234

# Fixed affine sweep parameters, all sinusoidal over one full clip-length
# cycle (t goes from 0 to 1 across the 90 frames):
PAN_FRACTION = 0.04  # +/- 4% of image width
ZOOM_PEAK = 0.06  # 1.0 -> 1.06 -> 1.0
ROTATE_DEG = 2.0  # +/- 2 degrees


def _affine_params(t: float, width: int) -> tuple[float, float, float]:
    """(pan_x_px, zoom, rotate_deg) at position t in [0, 1] of the clip."""
    pan_x = PAN_FRACTION * width * math.sin(2 * math.pi * t)
    zoom = 1.0 + (ZOOM_PEAK / 2.0) * (1 - math.cos(2 * math.pi * t))
    rotate_deg = ROTATE_DEG * math.sin(2 * math.pi * t)
    return pan_x, zoom, rotate_deg


def build_clip(base_bgr: np.ndarray, n_frames: int = N_FRAMES) -> list[np.ndarray]:
    """Deterministic synthetic clip: fixed affine sweep + seeded pixel noise."""
    h, w = base_bgr.shape[:2]
    rng = np.random.default_rng(NOISE_SEED)
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        pan_x, zoom, rotate_deg = _affine_params(t, w)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, zoom)
        matrix[0, 2] += pan_x
        warped = cv2.warpAffine(base_bgr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        noisy = warped.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, size=warped.shape)
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return frames


def _union_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.maximum(masks.lip_mask(landmarks, shape), masks.eyeshadow_mask(landmarks, shape))


def detect_landmark_streams(frames: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Detect raw landmarks per frame, then a One-Euro-filtered copy.

    Returns (raw_landmarks, filtered_landmarks, kept_frame_indices). Frames
    where detection fails are dropped from both streams (and reported by
    their index) so a stray miss can't desync raw vs filtered.
    """
    landmarker = FaceLandmarker()
    onefilter = OneEuroFilter()
    raw, filtered, kept = [], [], []
    dropped = []
    for i, frame in enumerate(frames):
        timestamp_ms = int(round(i * (1000.0 / FPS)))
        try:
            lm = landmarker.detect_video(frame, timestamp_ms)
        except NoFaceError:
            dropped.append(i)
            continue
        raw.append(lm)
        filtered.append(onefilter(lm, timestamp_ms / 1000.0))
        kept.append(i)
    if dropped:
        print(f"stability_bench: no face detected on frames {dropped}, dropped from jitter analysis")
    return raw, filtered, kept


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


def _improvement_pct(raw: dict, one_euro: dict) -> dict:
    """Percent improvement of one_euro over raw; positive is always better."""

    def pct(before: float, after: float, higher_is_better: bool) -> float:
        if before == 0:
            return 0.0
        change = (after - before) / abs(before) * 100.0
        return change if higher_is_better else -change

    return {
        "mean_iou": pct(raw["mean_iou"], one_euro["mean_iou"], higher_is_better=True),
        "centroid_drift": pct(raw["centroid_drift"], one_euro["centroid_drift"], higher_is_better=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    base_bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    frames = build_clip(base_bgr)
    print(f"built {len(frames)}-frame synthetic clip ({base_bgr.shape[1]}x{base_bgr.shape[0]})")

    raw_lm, filtered_lm, kept = detect_landmark_streams(frames)
    if len(raw_lm) < 2:
        raise RuntimeError(f"too few frames with a detected face ({len(raw_lm)}) to score jitter")

    shape = base_bgr.shape[:2]
    raw_masks = [_union_mask(lm, shape) for lm in raw_lm]
    one_euro_masks = [_union_mask(lm, shape) for lm in filtered_lm]

    iod = interocular_distance(raw_lm[0])
    raw_stats = metrics.mask_jitter(raw_masks, iod)
    one_euro_stats = metrics.mask_jitter(one_euro_masks, iod)
    improvement = _improvement_pct(raw_stats, one_euro_stats)

    video_ms_per_frame = measure_video_ms_per_frame(frames)

    stability = {
        "raw": raw_stats,
        "one_euro": one_euro_stats,
        "improvement_pct": improvement,
        "video_ms_per_frame": video_ms_per_frame,
        "n_frames": len(frames),
        "n_frames_with_face": len(kept),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "benchmark.json"
    result = json.loads(out_json.read_text()) if out_json.is_file() else {}
    result["stability"] = stability
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")

    print(f"\n{'':10}{'mean_iou':>10}{'drift':>10}")
    print(f"{'raw':10}{raw_stats['mean_iou']:>10.4f}{raw_stats['centroid_drift']:>10.4f}")
    print(f"{'one_euro':10}{one_euro_stats['mean_iou']:>10.4f}{one_euro_stats['centroid_drift']:>10.4f}")
    print(
        f"improvement: mean_iou {improvement['mean_iou']:+.1f}%, "
        f"centroid_drift {improvement['centroid_drift']:+.1f}%"
    )
    print(f"video ms/frame (median): {video_ms_per_frame:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
