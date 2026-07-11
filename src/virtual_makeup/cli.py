"""Command-line interface: ``vmakeup apply``, ``vmakeup landmarks``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .config import PRESETS, MakeupLook


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vmakeup",
        description="Landmark-driven virtual makeup for portrait photos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    apply_p = sub.add_parser("apply", help="apply a makeup look to a photo")
    apply_p.add_argument("input", type=Path, help="input image (JPEG/PNG)")
    apply_p.add_argument("output", type=Path, help="output image path")
    apply_p.add_argument(
        "--preset", choices=sorted(PRESETS), default="classic",
        help="base look to start from (default: classic)",
    )
    apply_p.add_argument("--lipstick", metavar="#RRGGBB", help="lipstick color")
    apply_p.add_argument("--lipstick-intensity", type=float, metavar="0..1")
    apply_p.add_argument("--eyeshadow", metavar="#RRGGBB", help="eyeshadow color")
    apply_p.add_argument("--eyeshadow-intensity", type=float, metavar="0..1")
    apply_p.add_argument("--eyeliner", metavar="#RRGGBB", help="eyeliner color")
    apply_p.add_argument("--eyeliner-intensity", type=float, metavar="0..1")
    apply_p.add_argument("--blush", metavar="#RRGGBB", help="blush color")
    apply_p.add_argument("--blush-intensity", type=float, metavar="0..1")
    apply_p.add_argument("--smoothing", type=float, metavar="0..1")

    lm_p = sub.add_parser("landmarks", help="render detected landmarks for debugging")
    lm_p.add_argument("input", type=Path)
    lm_p.add_argument("output", type=Path)
    return parser


def _look_from_args(args: argparse.Namespace) -> MakeupLook:
    look = PRESETS[args.preset]
    overrides = {}
    mapping = {
        "lipstick": "lipstick_color",
        "lipstick_intensity": "lipstick_intensity",
        "eyeshadow": "eyeshadow_color",
        "eyeshadow_intensity": "eyeshadow_intensity",
        "eyeliner": "eyeliner_color",
        "eyeliner_intensity": "eyeliner_intensity",
        "blush": "blush_color",
        "blush_intensity": "blush_intensity",
        "smoothing": "smoothing",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    return dataclasses.replace(look, **overrides) if overrides else look


def _read_image(path: Path):
    import cv2

    if not path.exists():
        raise FileNotFoundError(f"input image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "apply":
            _cmd_apply(args)
        elif args.command == "landmarks":
            _cmd_landmarks(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_apply(args: argparse.Namespace) -> None:
    import cv2

    from .makeup import apply_makeup

    look = _look_from_args(args)
    image = _read_image(args.input)
    result = apply_makeup(image, look)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise ValueError(f"could not write output: {args.output}")
    print(f"wrote {args.output}")


def _cmd_landmarks(args: argparse.Namespace) -> None:
    import cv2

    from .landmarks import FaceLandmarker

    image = _read_image(args.input)
    pts = FaceLandmarker().detect(image)
    out = image.copy()
    radius = max(1, image.shape[1] // 800)
    for x, y in pts.astype(int):
        cv2.circle(out, (x, y), radius, (80, 220, 80), -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), out):
        raise ValueError(f"could not write output: {args.output}")
    print(f"wrote {args.output} ({len(pts)} landmarks)")


if __name__ == "__main__":
    raise SystemExit(main())
