"""Bakes Python-computed expected values into web/src/gen/test_vectors.json.

The browser engine reimplements the color math, the pigment ops and the
One-Euro filter in TypeScript. Reimplementations drift, and a TypeScript
test can only ever check TypeScript against itself -- so the numbers the
vitest suite compares against are produced *here*, by the real OpenCV /
NumPy code paths in `carmine`, and committed alongside the generated
constants. `tests/test_constants_sync.py` regenerates this file and asserts
byte-for-byte equality, so a Python-side change that moves any of these
numbers fails loudly in pytest instead of silently invalidating the
TypeScript tests.

Every array of image pixels in this file is RGB (not the BGR that
`carmine.pigment` works in internally) because that is the channel order
the browser's ImageData uses; the conversion happens here so the
TypeScript side never has to think about it.

Run as: PYTHONPATH=src python scripts/export_test_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from carmine import pigment
from carmine.filters import OneEuroFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "web" / "src" / "gen" / "test_vectors.json"

# Probe colors for the sRGB->Lab conversion: the three primaries, the ends
# and middle of the grey ramp, a near-black (where Lab's linear branch takes
# over from the cube-root branch), and a spread of skin/lip/makeup tones --
# the region of color space the engine actually spends its time in.
LAB_PROBES = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 255),
    (128, 128, 128),
    (0, 0, 0),
    (20, 20, 20),
    (230, 190, 170),
    (160, 110, 90),
    (90, 60, 45),
    (176, 58, 91),
    (245, 217, 200),
]


def _round_trip_probe_rgb() -> list[tuple[int, int, int]]:
    """Colors used for the Lab -> sRGB direction (same set, kept explicit)."""
    return list(LAB_PROBES)


def _rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb[..., ::-1].copy()


def _synthetic_patch(size: int, seed: int) -> np.ndarray:
    """A deterministic RGB uint8 patch with enough tonal spread to exercise
    the gloss percentiles (a flat patch would trip the `spread < 1e-6`
    early-out and test nothing)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(30, 226, size=(size, size, 3), dtype=np.int64)
    return base.astype(np.uint8)


def _synthetic_mask(size: int) -> np.ndarray:
    """Mask with a genuinely-zero first row (so the untouched-restore path
    is exercised) and >=10 pixels above 0.5 (so gloss does not early-out)."""
    mask = np.full((size, size), 0.8, dtype=np.float32)
    mask[0, :] = 0.0
    if size >= 4:
        # A couple of partial-weight pixels so the blend is not all-or-nothing.
        mask[1, 0] = 0.25
        mask[1, 1] = 0.5
    return mask


def _pixels(image_rgb: np.ndarray) -> list[int]:
    return [int(v) for v in image_rgb.reshape(-1)]


def _floats(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1)]


def _lab_probes() -> dict:
    entries = []
    for rgb in LAB_PROBES:
        swatch = np.array([[rgb[::-1]]], dtype=np.float32) / 255.0  # RGB -> BGR
        lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)[0, 0]
        entries.append({"rgb": list(rgb), "lab": _floats(lab)})
    round_trip = []
    for rgb in _round_trip_probe_rgb():
        swatch = np.array([[rgb[::-1]]], dtype=np.float32) / 255.0
        lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)
        back = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0][::-1]
        round_trip.append({"rgb": list(rgb), "rgb_back": _floats(back * 255.0)})
    return {"forward": entries, "round_trip": round_trip}


def _pigment_cases() -> dict:
    """Expected outputs for tint / paint / finish_matte / finish_gloss.

    The 4x4 patch is the headline case (small enough to eyeball in the JSON
    when a test fails); the 16x16 patch exists because finish_matte's
    sigma=5 blur and finish_gloss's percentiles are both close to degenerate
    on a 4x4 image, and a parity test that only ever ran on the degenerate
    size would not catch a real blur bug.
    """
    cases: dict[str, list] = {"tint": [], "paint": [], "finish_matte": [], "finish_gloss": []}

    patches = {4: _synthetic_patch(4, seed=12), 16: _synthetic_patch(16, seed=13)}
    masks = {size: _synthetic_mask(size) for size in patches}

    tint_combos = [
        {"color": [176, 58, 91], "intensity": 0.55, "lightness_pull": 0.35},
        {"color": [138, 90, 68], "intensity": 0.35, "lightness_pull": 0.30},
        {"color": [245, 217, 200], "intensity": 1.0, "lightness_pull": 0.10},
    ]

    for size in (4, 16):
        rgb = patches[size]
        bgr = _rgb_to_bgr(rgb)
        mask = masks[size]
        for combo in tint_combos:
            out = pigment.tint(
                bgr,
                mask,
                tuple(combo["color"]),
                combo["intensity"],
                combo["lightness_pull"],
            )
            cases["tint"].append(
                {
                    "size": size,
                    "color": combo["color"],
                    "intensity": combo["intensity"],
                    "lightness_pull": combo["lightness_pull"],
                    "expected_rgb": _pixels(_rgb_to_bgr(out)),
                }
            )

        paint_out = pigment.paint(bgr, mask, (27, 27, 27), 0.8)
        cases["paint"].append(
            {
                "size": size,
                "color": [27, 27, 27],
                "intensity": 0.8,
                "expected_rgb": _pixels(_rgb_to_bgr(paint_out)),
            }
        )

        matte_out = pigment.finish_matte(bgr, mask, strength=0.35)
        cases["finish_matte"].append(
            {"size": size, "strength": 0.35, "expected_rgb": _pixels(_rgb_to_bgr(matte_out))}
        )

        gloss_out = pigment.finish_gloss(bgr, mask, strength=0.6)
        lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        inside = mask > 0.5
        p75, p99 = np.percentile(lab[..., 0][inside], [75, 99])
        cases["finish_gloss"].append(
            {
                "size": size,
                "strength": 0.6,
                "p75": float(p75),
                "p99": float(p99),
                "expected_rgb": _pixels(_rgb_to_bgr(gloss_out)),
            }
        )

    return {
        "patches": {
            str(size): {"size": size, "rgb": _pixels(patches[size])} for size in patches
        },
        "masks": {str(size): _floats(masks[size]) for size in masks},
        "cases": cases,
    }


def _one_euro_trace() -> dict:
    """One filtered pass over a seeded noisy sine.

    The signal is a (4, 2) array per step -- not a scalar -- because the
    filter's adaptive cutoff is elementwise, and a scalar trace would pass
    even against an implementation that collapsed the whole array to a
    single speed estimate.
    """
    rng = np.random.default_rng(7)
    steps = 30
    dt = 1.0 / 30.0
    inputs = []
    for i in range(steps):
        t = i * dt
        phase = np.array([0.0, 0.7, 1.4, 2.1], dtype=np.float64)[:, None]
        base = 100.0 + 40.0 * np.sin(2.0 * np.pi * 0.8 * t + phase)
        base = np.concatenate([base, base * 0.5 + 20.0], axis=1)
        noise = rng.normal(0.0, 1.5, size=base.shape)
        inputs.append(base + noise)

    filt = OneEuroFilter()
    trace = []
    for i, x in enumerate(inputs):
        t = i * dt
        out = filt(x, t)
        trace.append({"t": t, "x": _floats(x), "y": _floats(out)})

    # A repeated timestamp must reuse the previous output rather than
    # divide by dt == 0.
    stale_out = filt(inputs[-1] * 2.0, (steps - 1) * dt)
    return {
        "shape": [4, 2],
        "params": {
            "freq": filt.freq,
            "min_cutoff": filt.min_cutoff,
            "beta": filt.beta,
            "d_cutoff": filt.d_cutoff,
        },
        "trace": trace,
        "non_increasing_timestamp": {
            "t": (steps - 1) * dt,
            "x": _floats(inputs[-1] * 2.0),
            "y": _floats(stale_out),
        },
    }


def build_vectors() -> dict:
    return {
        "lab": _lab_probes(),
        "one_euro": _one_euro_trace(),
        "pigment": _pigment_cases(),
    }


def write_vectors(output_path: Path) -> None:
    data = build_vectors()
    text = json.dumps(data, sort_keys=True, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    write_vectors(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
