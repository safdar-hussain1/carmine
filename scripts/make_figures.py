"""Generate the report figures and the dashboard's baked data.

Inputs:  reports/benchmark.json (produced by scripts/benchmark.py)
Outputs: reports/figures/*.png, docs/data.js

Demo imagery uses the public-domain NASA portrait bundled with
scikit-image (skimage.data.astronaut), so no dataset photos of private
individuals are committed to the repo.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skimage import data  # noqa: E402

from virtual_makeup import PRESETS, apply_makeup, legacy, masks  # noqa: E402
from virtual_makeup.landmarks import FaceLandmarker  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"

# dataviz reference palette (light mode)
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = {  # fixed categorical slot per method, never re-ranked
    "legacy_mediapipe": "#2a78d6",
    "legacy_dlib": "#1baf7a",
    "legacy_dlib_swap": "#eda100",
    "legacy_gan": "#e34948",
    "new_classic": "#4a3aa7",
}
LABELS = {
    "legacy_mediapipe": "Mismatched indices",
    "legacy_dlib": "Opaque fill",
    "legacy_dlib_swap": "Channel swap",
    "legacy_gan": "Untrained GAN",
    "new_classic": "This engine",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.dpi": 150,
})


def _bars(ax, summary, metric, title, fmt="{:.2f}", limit=(0, 1.05)):
    methods = list(SERIES)
    values = [summary[m][metric]["mean"] for m in methods]
    y = np.arange(len(methods))
    ax.barh(y, values, height=0.55,
            color=[SERIES[m] for m in methods], zorder=3)
    ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(*limit)
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(max(v, 0) + 0.02, yi, fmt.format(v), va="center",
                fontsize=8.5, color=SECONDARY)


def fig_benchmark(summary) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4))
    _bars(axes[0, 0], summary, "containment",
          "Edit energy inside valid makeup regions")
    _bars(axes[0, 1], summary, "background_integrity",
          "Background pixels left bit-identical")
    _bars(axes[1, 0], summary, "lip_texture_corr",
          "Lip texture preserved (lightness corr.)", limit=(-0.3, 1.1))
    _bars(axes[1, 1], summary, "identity_ssim",
          "Identity SSIM vs input")
    fig.suptitle("Engine vs naive baselines — 25 photos, paired makeup dataset",
                 fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGDIR / "benchmark_metrics.png")
    plt.close(fig)


def fig_original_protocol(protocol) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    methods = list(SERIES)
    values = [protocol[m]["mean_ssim_vs_reference"] for m in methods]
    y = np.arange(len(methods))
    ax.barh(y, values, height=0.55, color=[SERIES[m] for m in methods], zorder=3)
    ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.7)
    ax.axvline(0.45, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.text(0.452, 4.4, "0.45 “accuracy” threshold", fontsize=8,
            color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5,
                color=SECONDARY)
    ax.set_title(
        "Why we don’t score with reference SSIM — it can’t tell them apart",
        fontsize=10, loc="left", color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "original_protocol.png")
    plt.close(fig)


def _label_strip(images, titles, path, height=340):
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        tiles.append(cv2.resize(img, (int(w * height / h), height)))
    fig, axes = plt.subplots(1, len(tiles), figsize=(2.6 * len(tiles), 3.2))
    for ax, tile, title in zip(axes, tiles, titles):
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def demo_figures(landmarker) -> np.ndarray:
    astro = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    lm = landmarker.detect(astro)

    _label_strip(
        [astro] + [apply_makeup(astro, PRESETS[p], landmarks=lm)
                   for p in ("natural", "classic", "bold")],
        ["original", "natural", "classic", "bold"],
        FIGDIR / "presets_demo.png",
    )

    lm68 = _dlib68(astro)
    strip = [
        astro,
        legacy.legacy_mediapipe(astro, lm),
        legacy.legacy_dlib(astro, lm68, channel_swap_bug=True),
        cv2.resize(legacy.legacy_gan(astro), astro.shape[1::-1]),
        apply_makeup(astro, PRESETS["classic"], landmarks=lm),
    ]
    _label_strip(
        strip,
        ["original", "mismatched indices", "channel swap", "untrained GAN",
         "this engine (classic)"],
        FIGDIR / "legacy_vs_new.png",
    )

    # Region masks overlay
    shape = astro.shape[:2]
    overlay = astro.copy().astype(np.float32)
    tint_colors = {
        "lips": ((60, 60, 220), masks.lip_mask),
        "shadow": ((200, 120, 60), masks.eyeshadow_mask),
        "liner": ((60, 200, 60), masks.eyeliner_mask),
        "blush": ((180, 60, 180), masks.blush_mask),
    }
    for color, fn in tint_colors.values():
        m = fn(lm, shape)[..., None]
        overlay = overlay * (1 - 0.85 * m) + np.array(color) * 0.85 * m
    _label_strip(
        [astro, overlay.astype(np.uint8)],
        ["input", "soft face-scaled masks"],
        FIGDIR / "masks_demo.png",
        height=420,
    )
    return astro


def _dlib68(image):
    import dlib

    predictor_path = Path(
        "/Users/safdarhussain/Desktop/SEM 5/Digital Image Processing/"
        "Virtual Makeup (CS2 Group 15)/Requirements/"
        "shape_predictor_68_face_landmarks.dat"
    )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = detector(gray)[0]
    shape = predictor(gray, face)
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)],
                    dtype=np.float32)


def export_dashboard_data(summary, protocol) -> None:
    payload = {
        "labels": LABELS,
        "summary": {
            m: {k: (v if isinstance(v, (int, float)) else v["mean"])
                for k, v in s.items()}
            for m, s in summary.items()
        },
        "original_protocol": {
            m: {
                "ssim": p["mean_ssim_vs_reference"],
                "accuracy": p["legacy_metrics@0.45"]["Accuracy"],
                "precision": p["legacy_metrics@0.45"]["Precision"],
            }
            for m, p in protocol.items()
        },
    }
    out = ROOT / "docs" / "data.js"
    out.write_text("const BENCH = " + json.dumps(payload, indent=1) + ";\n")
    print(f"wrote {out}")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads((ROOT / "reports" / "benchmark.json").read_text())
    summary = bench["summary"]
    protocol = bench["original_protocol_reproduction"]
    fig_benchmark(summary)
    fig_original_protocol(protocol)
    landmarker = FaceLandmarker()
    demo_figures(landmarker)
    export_dashboard_data(summary, protocol)
    print(f"figures in {FIGDIR}")


if __name__ == "__main__":
    main()
