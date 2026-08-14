"""Command-line interface for carmine.

Subcommands:
    apply      -- paint a Look onto a still image.
    video      -- paint a Look onto every frame of a video.
    landmarks  -- render the 478 detected landmarks as dots (debug aid).
    looks      -- list the built-in presets.

Every subcommand's flags live on its own subparser -- there are no top-level
flags that could collide with a subcommand's flag of the same name.

Error contract: user-facing problems (missing files, bad hex colors, invalid
JSON, no face detected, an unreadable/unwritable video) are reported as a
single ``error: ...`` line on stderr and exit code 2 -- never a traceback.
When a `Look` fails validation with several problems at once, all of them are
listed (one per line) since `Look.__post_init__` already collects them into a
single `ValueError` message.
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

from carmine.engine import VideoEngine, apply_look
from carmine.landmarks import FaceLandmarker, NoFaceError
from carmine.look import Look, PRESETS, Product

__all__ = ["main"]

PRODUCT_NAMES = ["lipstick", "eyeshadow", "eyeliner", "brows", "blush", "highlighter"]
FINISH_CHOICES = ("matte", "satin", "gloss")

# Overlay rule for `apply --lipstick HEX` (etc.) without a matching
# `--<product>-intensity`: the product becomes visible at this intensity
# unless the base look already had a non-zero intensity for it, in which
# case that intensity is kept. This keeps "just recolor this preset's
# lipstick" a one-flag operation instead of requiring two flags.
DEFAULT_FLAG_INTENSITY = 0.7


class CliError(Exception):
    """A user-facing error: caught in main(), reported without a traceback."""


def _read_image(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.is_file():
        raise CliError(f"input file not found: {path_str}")
    image = cv2.imread(str(path))
    if image is None:
        raise CliError(f"could not read image: {path_str}")
    return image


def _resolve_base_look(args: argparse.Namespace) -> Look:
    """Resolve the starting `Look` from `--preset` or `--look-json`, else default."""
    if getattr(args, "preset", None) is not None:
        try:
            return PRESETS[args.preset]
        except KeyError:
            raise CliError(
                f"unknown preset {args.preset!r}, expected one of {sorted(PRESETS)}"
            ) from None

    look_json = getattr(args, "look_json", None)
    if look_json is not None:
        path = Path(look_json)
        if not path.is_file():
            raise CliError(f"look-json file not found: {look_json}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise CliError(f"invalid JSON in {look_json}: {exc}") from exc
        try:
            return Look.from_dict(data)
        except (ValueError, TypeError, AttributeError) as exc:
            raise CliError(str(exc)) from exc

    return Look()


def _overlay_product_flags(base: Look, args: argparse.Namespace) -> Look:
    """Overlay any `--<product>[-intensity][-finish]` flags on top of `base`.

    See module docstring / --help for the exact overlay rule.
    """
    updates: dict = {}

    for name in PRODUCT_NAMES:
        color = getattr(args, name, None)
        intensity = getattr(args, f"{name}_intensity", None)
        finish = getattr(args, f"{name}_finish", None)

        if color is None and intensity is None and finish is None:
            continue

        base_product: Product = getattr(base, name)
        new_color = color if color is not None else base_product.color
        if intensity is not None:
            new_intensity = intensity
        elif base_product.intensity > 0:
            new_intensity = base_product.intensity
        else:
            new_intensity = DEFAULT_FLAG_INTENSITY
        new_finish = finish if finish is not None else base_product.finish

        updates[name] = Product(color=new_color, intensity=new_intensity, finish=new_finish)

    smoothing = getattr(args, "smoothing", None)
    if smoothing is not None:
        updates["smoothing"] = smoothing

    if not updates:
        return base

    try:
        return dataclasses.replace(base, **updates)
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def _add_look_source_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Start from a built-in preset look.",
    )
    group.add_argument(
        "--look-json",
        metavar="FILE",
        help="Start from a Look serialized as JSON (see `carmine looks --json`).",
    )


def _add_product_flags(parser: argparse.ArgumentParser) -> None:
    for name in PRODUCT_NAMES:
        parser.add_argument(f"--{name}", metavar="HEX", help=f"{name} color as #RRGGBB.")
        parser.add_argument(
            f"--{name}-intensity",
            type=float,
            metavar="F",
            help=f"{name} intensity in [0, 1].",
        )
    parser.add_argument(
        "--lipstick-finish",
        choices=FINISH_CHOICES,
        help="Lipstick finish.",
    )
    parser.add_argument("--smoothing", type=float, metavar="F", help="Skin smoothing in [0, 1].")


def cmd_apply(args: argparse.Namespace) -> int:
    image = _read_image(args.input)
    base = _resolve_base_look(args)
    look = _overlay_product_flags(base, args)

    try:
        out = apply_look(image, look)
    except NoFaceError as exc:
        raise CliError(str(exc)) from exc
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    if not cv2.imwrite(args.output, out):
        raise CliError(f"failed to write output image: {args.output}")

    return 0


def cmd_video(args: argparse.Namespace) -> int:
    if not Path(args.input).is_file():
        raise CliError(f"input file not found: {args.input}")

    look = _resolve_base_look(args)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise CliError(f"could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise CliError(f"could not open output video for writing: {args.output}")

    engine = VideoEngine(look, smooth_landmarks=not args.no_smooth_landmarks)

    frame_count = 0
    faces_found = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_count * 1000.0 / fps))
            out = engine.process(frame, timestamp_ms)
            # engine.process() returns an unmodified copy on a no-face frame;
            # a changed frame is our proxy for "a face was found and painted".
            if not np.array_equal(out, frame):
                faces_found += 1
            writer.write(out)
            frame_count += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    print(
        f"processed {frame_count} frames, {faces_found} with a detected face, "
        f"in {elapsed:.2f}s"
    )
    return 0


def cmd_landmarks(args: argparse.Namespace) -> int:
    image = _read_image(args.input)

    try:
        landmarks = FaceLandmarker().detect(image)
    except NoFaceError as exc:
        raise CliError(str(exc)) from exc

    out = image.copy()
    for x, y in landmarks:
        cv2.circle(out, (int(round(x)), int(round(y))), radius=1, color=(0, 255, 0), thickness=-1)

    if not cv2.imwrite(args.output, out):
        raise CliError(f"failed to write output image: {args.output}")

    return 0


def _format_looks_table() -> str:
    lines = []
    for name in sorted(PRESETS):
        look = PRESETS[name]
        lines.append(f"{name}:")
        for product_name in PRODUCT_NAMES:
            product: Product = getattr(look, product_name)
            if product.intensity <= 0:
                continue
            lines.append(
                f"  {product_name:<12} {product.color}  "
                f"intensity={product.intensity:.2f}  finish={product.finish}"
            )
        lines.append(f"  smoothing={look.smoothing:.2f}")
    return "\n".join(lines)


def cmd_looks(args: argparse.Namespace) -> int:
    if args.json:
        payload = {name: PRESETS[name].to_dict() for name in sorted(PRESETS)}
        print(json.dumps(payload, indent=2))
    else:
        print(_format_looks_table())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carmine",
        description="Texture-preserving virtual makeup engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply a look to a still image.",
        description=(
            "Apply a look to a still image. The look starts from --preset "
            "(or --look-json, or the all-zero default look if neither is "
            "given), then any per-product flags are overlaid on top: giving "
            "--<product>-intensity and/or --lipstick-finish overrides just "
            "that attribute. Giving a --<product> color without its "
            "matching --<product>-intensity flag sets that product's "
            f"intensity to {DEFAULT_FLAG_INTENSITY} -- unless the base look "
            "already had a non-zero intensity for it, in which case that "
            "intensity is kept."
        ),
    )
    apply_parser.add_argument("input", help="Input image path.")
    apply_parser.add_argument("output", help="Output image path.")
    _add_look_source_flags(apply_parser)
    _add_product_flags(apply_parser)
    apply_parser.set_defaults(func=cmd_apply)

    video_parser = subparsers.add_parser(
        "video",
        help="Apply a look to every frame of a video.",
        description="Apply a look to every frame of a video, writing an mp4 at the source fps/size.",
    )
    video_parser.add_argument("input", help="Input video path.")
    video_parser.add_argument("output", help="Output video path (.mp4).")
    _add_look_source_flags(video_parser)
    video_parser.add_argument(
        "--no-smooth-landmarks",
        action="store_true",
        help="Disable the One-Euro landmark smoothing filter across frames.",
    )
    video_parser.set_defaults(func=cmd_video)

    landmarks_parser = subparsers.add_parser(
        "landmarks",
        help="Render detected landmarks as dots (debug aid).",
        description="Detect the 478 face landmarks and render them as small green dots.",
    )
    landmarks_parser.add_argument("input", help="Input image path.")
    landmarks_parser.add_argument("output", help="Output image path.")
    landmarks_parser.set_defaults(func=cmd_landmarks)

    looks_parser = subparsers.add_parser(
        "looks",
        help="List the built-in preset looks.",
        description="List the built-in preset looks and their non-zero products.",
    )
    looks_parser.add_argument(
        "--json", action="store_true", help="Emit {name: look.to_dict()} as JSON."
    )
    looks_parser.set_defaults(func=cmd_looks)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
