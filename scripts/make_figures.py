"""Generate the report figures from reports/benchmark.json.

Inputs:  reports/benchmark.json (produced by scripts/benchmark.py and
         scripts/stability_bench.py)
Outputs: reports/figures/benchmark_metrics.png, reports/figures/presets_demo.png

Demo imagery uses the public-domain NASA portrait bundled with scikit-image
(skimage.data.astronaut), so no dataset photos of private individuals are
committed to the repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skimage import data  # noqa: E402

from carmine.engine import apply_look  # noqa: E402
from carmine.landmarks import FaceLandmarker  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"

# dataviz reference palette (light mode)
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = {  # fixed categorical slot per method, never re-ranked
    "mismatched_indices": "#2a78d6",
    "opaque_fill": "#1baf7a",
    "channel_swap": "#eda100",
    "untrained_gan": "#e34948",
    "carmine": "#4a3aa7",
}
LABELS = {
    "mismatched_indices": "Mismatched indices",
    "opaque_fill": "Opaque fill",
    "channel_swap": "Channel swap",
    "untrained_gan": "Untrained GAN",
    "carmine": "This engine",
}
METRICS = ["pigment_on_target", "background_untouched", "lip_texture_kept", "identity_ssim"]
METRIC_TITLES = {
    "pigment_on_target": "Edit energy inside legitimate product regions",
    "background_untouched": "Background pixels left bit-identical",
    "lip_texture_kept": "Lip texture preserved (lightness corr.)",
    "identity_ssim": "Identity SSIM vs input",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.dpi": 150,
    }
)


def _bars(ax, rows_by_method, metric, title, limit=(0, 1.05)):
    methods = list(SERIES)
    values = [rows_by_method[m][metric] for m in methods]
    y = np.arange(len(methods))
    ax.barh(y, values, height=0.55, color=[SERIES[m] for m in methods], zorder=3)
    ax.set_yticks(y, [LABELS[m] for m in methods], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(*limit)
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(max(v, limit[0]) + 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color=SECONDARY)


def fig_benchmark(rows_by_method: dict, n_images: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4))
    for ax, metric in zip(axes.flat, METRICS):
        limit = (-0.4, 1.05) if metric == "lip_texture_kept" else (0, 1.05)
        _bars(ax, rows_by_method, metric, METRIC_TITLES[metric], limit=limit)
    fig.suptitle(
        f"Engine vs standard failure-mode baselines -- {n_images} photos, paired portrait dataset",
        fontsize=11,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGDIR / "benchmark_metrics.png")
    plt.close(fig)


def fig_presets_demo(landmarker: FaceLandmarker) -> None:
    astro = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    lm = landmarker.detect(astro)
    preset_names = ["bare", "everyday", "velvet", "glass"]
    images = [astro] + [apply_look(astro, PRESETS[p], landmarks=lm) for p in preset_names]
    titles = ["original"] + preset_names

    height = 340
    tiles = []
    for img in images:
        h, w = img.shape[:2]
        tiles.append(cv2.resize(img, (int(w * height / h), height)))

    fig, axes = plt.subplots(1, len(tiles), figsize=(2.6 * len(tiles), 3.2))
    for ax, tile, title in zip(axes, tiles, titles):
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle("carmine.look presets on the demo portrait", fontsize=10, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "presets_demo.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads((ROOT / "reports" / "benchmark.json").read_text())
    rows_by_method = {row["method"]: row for row in bench["photo"]["rows"]}
    n_images = bench["meta"]["n_images"]

    fig_benchmark(rows_by_method, n_images)

    landmarker = FaceLandmarker()
    fig_presets_demo(landmarker)

    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
