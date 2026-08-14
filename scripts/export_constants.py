"""Dumps region indices and mask/pigment/filter constants to web/src/gen/constants.json.

The browser mirror needs the same region index lists, mask geometry
fractions, pigment lightness pulls, finish parameters, One-Euro filter
defaults, and preset Looks as the Python engine, so its output matches
frame-for-frame. Values that are exposed as real Python objects (region
index lists, `OneEuroFilter` constructor defaults, `Look` presets) are
imported and read directly here so drift between the two sides is
impossible. Values that only exist as literals buried inside function
bodies in `carmine/masks.py`, `carmine/pigment.py`, and `carmine/engine.py`
are mirrored by hand below, each with a comment citing the source line;
`tests/test_constants_sync.py` catches this script drifting from the
committed JSON, and Task 12's parity tests catch the mirrored literals
drifting from the Python source.

Run as: PYTHONPATH=src python scripts/export_constants.py
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from carmine import regions
from carmine.filters import OneEuroFilter
from carmine.look import PRESETS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "web" / "src" / "gen" / "constants.json"


def _regions() -> dict:
    """Region index lists actually consumed by mask functions (masks.py)."""
    return {
        "LIPS_OUTER": regions.LIPS_OUTER,
        "LIPS_INNER": regions.LIPS_INNER,
        "RIGHT_EYE": regions.RIGHT_EYE,
        "LEFT_EYE": regions.LEFT_EYE,
        "RIGHT_EYE_UPPER": regions.RIGHT_EYE_UPPER,
        "LEFT_EYE_UPPER": regions.LEFT_EYE_UPPER,
        "RIGHT_BROW_LOWER": regions.RIGHT_BROW_LOWER,
        "LEFT_BROW_LOWER": regions.LEFT_BROW_LOWER,
        "RIGHT_BROW_UPPER": regions.RIGHT_BROW_UPPER,
        "LEFT_BROW_UPPER": regions.LEFT_BROW_UPPER,
        "FACE_OVAL": regions.FACE_OVAL,
        "RIGHT_CHEEK": regions.RIGHT_CHEEK,
        "LEFT_CHEEK": regions.LEFT_CHEEK,
        "RIGHT_EYE_OUTER": regions.RIGHT_EYE_OUTER,
        "LEFT_EYE_OUTER": regions.LEFT_EYE_OUTER,
        "NOSE_BRIDGE": regions.NOSE_BRIDGE,
        "RIGHT_CHEEKBONE": regions.RIGHT_CHEEKBONE,
        "LEFT_CHEEKBONE": regions.LEFT_CHEEKBONE,
        "NUM_LANDMARKS": regions.NUM_LANDMARKS,
    }


def _masks() -> dict:
    """Mask geometry fractions mirrored from carmine/masks.py literals.

    Every fraction below is relative to `geometry.interocular_distance`,
    matching the Python side's convention.
    """
    return {
        "lip": {
            # masks.py lip_mask: _feather(mask, sigma=iod * 0.02)
            "feather": 0.02,
        },
        "eyeshadow": {
            # masks.py eyeshadow_mask: upper = lid + (brow_matched - lid) * 0.60
            "lid_to_brow": 0.60,
            # masks.py eyeshadow_mask: crease gradient factors [0.35, 1.0]
            "crease_lash_factor": 1.0,
            "crease_top_factor": 0.35,
            # masks.py eyeshadow_mask: _feather(mask, sigma=iod * 0.045)
            "feather": 0.045,
        },
        "eyeliner": {
            # masks.py eyeliner_mask: thickness = max(1, round(iod * 0.012))
            "thickness_factor": 0.012,
            "thickness_min": 1,
            # masks.py eyeliner_mask: unit + (0.0, -0.45)
            "wing_dy": -0.45,
            # masks.py eyeliner_mask: wing = arc[-1] + unit * iod * 0.06
            "wing_length_factor": 0.06,
            # masks.py eyeliner_mask: _feather(mask, sigma=max(1.0, iod * 0.006))
            "feather_factor": 0.006,
            "feather_min": 1.0,
        },
        "blush": {
            # masks.py blush_mask: axes = (round(iod * 0.20), round(iod * 0.14))
            "axis_x": 0.20,
            "axis_y": 0.14,
            # masks.py blush_mask: angle -15 (right cheek), 15 (left cheek)
            "angle": 15,
            # masks.py blush_mask: _feather(mask, sigma=iod * 0.10)
            "feather": 0.10,
            # masks.py blush_mask: face oval feather sigma=iod * 0.02
            "oval_feather": 0.02,
        },
        "brow": {
            # masks.py brow_mask: _feather(mask, sigma=iod * 0.015)
            "feather": 0.015,
        },
        "highlighter": {
            # masks.py highlighter_mask: cheek_thickness = max(1, round(iod * 0.10))
            "cheek_thickness_factor": 0.10,
            # masks.py highlighter_mask: _feather(mask, sigma=iod * 0.06)
            "cheek_feather": 0.06,
            # masks.py highlighter_mask: nose_thickness = max(1, round(iod * 0.05))
            "nose_thickness_factor": 0.05,
            # masks.py highlighter_mask: _feather(nose_mask, sigma=iod * 0.06)
            "nose_feather": 0.06,
        },
        "skin": {
            # masks.py skin_mask: exclusion feather sigma=iod * 0.02 (lips/eyes)
            "exclusion_feather": 0.02,
            # masks.py skin_mask: brow cutout thickness=max(3, int(iod * 0.05))
            "brow_cutout_thickness_factor": 0.05,
            "brow_cutout_thickness_min": 3,
            # masks.py skin_mask: _feather(mask, sigma=iod * 0.03)
            "feather": 0.03,
        },
    }


def _pigment() -> dict:
    """Pigment lightness pulls and finish parameters mirrored from
    carmine/engine.py and carmine/pigment.py literals.
    """
    return {
        "lightness_pull": {
            # engine.py apply_look: _apply_product(..., look.blush, 0.15)
            "blush": 0.15,
            # engine.py apply_look: _apply_product(..., look.highlighter, 0.10)
            "highlighter": 0.10,
            # engine.py apply_look: _apply_product(..., look.eyeshadow, 0.30)
            "eyeshadow": 0.30,
            # engine.py apply_look: _apply_product(..., look.brows, 0.20)
            "brows": 0.20,
            # engine.py apply_look: lightness_pull = 0.30 if matte else 0.35
            "lipstick_matte": 0.30,
            "lipstick_satin": 0.35,
            "lipstick_gloss": 0.35,
        },
        "finish_matte": {
            # engine.py apply_look: finish_matte(..., strength=0.35)
            "default_strength": 0.35,
            # pigment.py finish_matte: cv2.GaussianBlur(l_channel, (0, 0), sigmaX=5)
            "blur_sigma": 5,
        },
        "finish_gloss": {
            # pigment.py finish_gloss: p75, p99 = np.percentile(masked_l, [75, 99])
            "percentile_low": 75,
            "percentile_high": 99,
            # pigment.py finish_gloss: highlight * mask * 18.0 * strength
            "strength_factor": 18.0,
            # engine.py apply_look: finish_gloss(..., strength=intensity * 0.5)
            "highlighter_strength_factor": 0.5,
        },
    }


def _one_euro() -> dict:
    """One-Euro filter defaults, read from the live constructor signature
    so this can never drift from carmine/filters.py.
    """
    params = inspect.signature(OneEuroFilter.__init__).parameters
    return {
        "freq": params["freq"].default,
        "min_cutoff": params["min_cutoff"].default,
        "beta": params["beta"].default,
        "d_cutoff": params["d_cutoff"].default,
    }


def _presets() -> dict:
    """Preset Looks, serialized via Look.to_dict() so this can never drift
    from carmine/look.py.
    """
    return {name: look.to_dict() for name, look in PRESETS.items()}


def build_constants() -> dict:
    return {
        "masks": _masks(),
        "one_euro": _one_euro(),
        "pigment": _pigment(),
        "presets": _presets(),
        "regions": _regions(),
    }


def write_constants(output_path: Path) -> None:
    data = build_constants()
    text = json.dumps(data, sort_keys=True, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    write_constants(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
