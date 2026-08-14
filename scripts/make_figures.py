"""Generate the report figures from reports/benchmark.json.

Inputs:  reports/benchmark.json (produced by scripts/benchmark.py and
         scripts/stability_bench.py)
Outputs: reports/figures/benchmark_metrics.png, reports/figures/presets_demo.png,
         reports/figures/opacity_compare.png

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skimage import data  # noqa: E402

from carmine import baselines, regions  # noqa: E402
from carmine.engine import apply_look  # noqa: E402
from carmine.landmarks import FaceLandmarker  # noqa: E402
from carmine.look import PRESETS  # noqa: E402

from benchmark import benchmark_look  # noqa: E402

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
# Order: the two containment metrics first, then lip_luminance_shift in the
# primary (top-right) slot -- it's the metric that actually distinguishes a
# hard fill from a soft tint by magnitude, not just direction -- followed by
# the two blind-spot-prone/diagnostic metrics and identity last.
METRICS = [
    "pigment_on_target",
    "background_untouched",
    "lip_luminance_shift",
    "lip_texture_kept",
    "lip_detail_retention",
    "identity_ssim",
]
METRIC_TITLES = {
    "pigment_on_target": "Edit energy inside legitimate product regions",
    "background_untouched": "Background pixels left bit-identical",
    "lip_texture_kept": "Lip texture preserved (lightness corr.)",
    "lip_detail_retention": (
        "Lip detail ratio (diagnostic -- additive fills\npreserve highpass until saturation)"
    ),
    "lip_luminance_shift": "Lip brightness shift, |ΔL| (lower is better)",
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


def _bars(ax, rows_by_method, metric, title, limit=(0, 1.05), fmt="{:.2f}"):
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
    span = limit[1] - limit[0]
    for yi, v in zip(y, values):
        ax.text(max(v, limit[0]) + span * 0.015, yi, fmt.format(v), va="center", fontsize=8.5, color=SECONDARY)


def fig_benchmark(rows_by_method: dict, n_images: int) -> None:
    luminance_max = max(rows_by_method[m]["lip_luminance_shift"] for m in SERIES)
    luminance_limit = (0, luminance_max * 1.15)

    limits = {
        "lip_texture_kept": (-0.4, 1.05),
        "lip_detail_retention": (0, max(1.5, max(rows_by_method[m]["lip_detail_retention"] for m in SERIES) * 1.1)),
        "lip_luminance_shift": luminance_limit,
    }
    fmts = {"lip_luminance_shift": "{:.1f}"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 5.8))
    for ax, metric in zip(axes.flat, METRICS):
        limit = limits.get(metric, (0, 1.05))
        fmt = fmts.get(metric, "{:.2f}")
        _bars(ax, rows_by_method, metric, METRIC_TITLES[metric], limit=limit, fmt=fmt)
    fig.suptitle(
        f"Engine vs standard failure-mode baselines -- {n_images} photos, "
        "no-makeup half of a local paired-portrait photo set",
        fontsize=11,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3.5)
    fig.subplots_adjust(wspace=0.45)
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


def fig_opacity_compare(landmarker: FaceLandmarker) -> None:
    """Lip-region crops: original vs carmine vs opaque_fill, the exact
    benchmarked configuration (velvet + lipstick finish forced to satin +
    smoothing forced to 0 -- see `benchmark_look()`), so this figure shows
    the same setup the `lip_detail_retention` / `lip_luminance_shift`
    numbers were measured under, not a look that merely looks similar.

    `pigment_on_target` and `lip_texture_kept` can't tell a hard fill from a
    soft tint when both share the same, correctly-indexed lip region (see
    `carmine.metrics.lip_detail_retention`'s docstring) -- this figure makes
    that difference visible directly, at the pixel level, instead of only
    through a metric.
    """
    astro = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    lm = landmarker.detect(astro)
    look = benchmark_look()

    carmine_out = apply_look(astro, look, landmarks=lm)
    opaque_out = baselines.opaque_fill(astro, lm, look)

    lip_pts = lm[regions.LIPS_OUTER]
    x0, y0 = lip_pts.min(axis=0)
    x1, y1 = lip_pts.max(axis=0)
    pad_x, pad_y = (x1 - x0) * 0.6, (y1 - y0) * 0.8
    x0i = int(max(0, x0 - pad_x))
    x1i = int(min(astro.shape[1], x1 + pad_x))
    y0i = int(max(0, y0 - pad_y))
    y1i = int(min(astro.shape[0], y1 + pad_y))

    def crop(img: np.ndarray) -> np.ndarray:
        return img[y0i:y1i, x0i:x1i]

    height = 320
    tiles = []
    for img in (astro, carmine_out, opaque_out):
        tile = crop(img)
        h, w = tile.shape[:2]
        tiles.append(cv2.resize(tile, (int(w * height / h), height), interpolation=cv2.INTER_NEAREST))
    titles = ["original", "carmine (velvet, satin lipstick)", "opaque_fill (velvet, satin lipstick)"]

    fig, axes = plt.subplots(1, 3, figsize=(3.4 * 3, 2.5))
    for ax, tile, title in zip(axes, tiles, titles):
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10, color=INK)
        ax.axis("off")
    fig.suptitle(
        "Same lip region, same benchmarked look -- hard fill vs texture-preserving tint",
        fontsize=10.5,
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "opacity_compare.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads((ROOT / "reports" / "benchmark.json").read_text())
    rows_by_method = {row["method"]: row for row in bench["photo"]["rows"]}
    n_images = bench["meta"]["n_images"]

    fig_benchmark(rows_by_method, n_images)

    landmarker = FaceLandmarker()
    fig_presets_demo(landmarker)
    fig_opacity_compare(landmarker)

    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
