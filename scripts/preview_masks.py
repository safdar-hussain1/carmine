"""Render every product mask over the demo portrait for a visual sanity check.

Loads the scikit-image astronaut portrait, detects landmarks, builds each
mask in `carmine.masks`, and writes a labeled grid of overlays to
`reports/figures/masks_demo.png` so a human can eyeball whether each mask
sits where it should (brow on the eyebrows, highlighter on the cheekbone
crests and nose bridge, blush on the mid-cheeks, eyeshadow fading toward
the brow, eyeliner a thin winged lash line, lipstick excluding the mouth
opening).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage import data

from carmine import masks
from carmine.landmarks import FaceLandmarker

# BGR colors, chosen for contrast against skin tones rather than product
# realism -- this is a diagnostic render, not a preview of the final look.
# highlighter_mask in particular used a pale (220, 220, 240) that nearly
# disappears over light skin; a saturated magenta reads clearly instead.
_TILE_COLORS = {
    "lip_mask": (60, 60, 220),
    "eyeshadow_mask": (200, 100, 40),
    "eyeliner_mask": (20, 20, 20),
    "blush_mask": (120, 90, 220),
    "brow_mask": (60, 40, 30),
    "highlighter_mask": (220, 0, 220),
    "skin_mask": (150, 200, 240),
}

# Floor on the blend alpha so faint mask values (e.g. the crease-gradient
# tail or feathered edges) still show up as visible color in the render
# instead of fading to near-invisible.
_MIN_VISIBLE_ALPHA = 0.55


def _overlay(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    color_arr = np.array(color, dtype=np.float32)
    visible = mask > 0.02
    alpha = np.where(visible, np.maximum(mask, _MIN_VISIBLE_ALPHA), mask)[:, :, None]
    blended = image_bgr.astype(np.float32) * (1 - alpha) + color_arr * alpha
    return blended.astype(np.uint8)


def main() -> None:
    rgb = data.astronaut()
    bgr = rgb[:, :, ::-1].copy()

    landmarker = FaceLandmarker()
    landmarks = landmarker.detect(bgr)
    shape = bgr.shape[:2]

    tiles = []
    for name, color in _TILE_COLORS.items():
        mask_fn = getattr(masks, name)
        mask = mask_fn(landmarks, shape)
        tile = _overlay(bgr, mask, color)
        cv2.putText(
            tile,
            name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    h, w = shape
    grid = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile

    out_path = Path(__file__).resolve().parent.parent / "reports" / "figures" / "masks_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
