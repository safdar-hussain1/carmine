"""Makeup look configuration with fail-fast validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an ``(R, G, B)`` tuple. Raises ValueError."""
    if not isinstance(value, str):
        raise ValueError(f"color must be a '#RRGGBB' string, got {value!r}")
    m = _HEX_RE.match(value.strip())
    if not m:
        raise ValueError(f"invalid hex color {value!r}, expected '#RRGGBB'")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass(frozen=True)
class MakeupLook:
    """One complete makeup configuration.

    Colors are ``#RRGGBB`` strings, intensities are in [0, 1] where 0
    disables the effect entirely.
    """

    lipstick_color: str = "#B03A5B"
    lipstick_intensity: float = 0.75

    eyeshadow_color: str = "#8A5A44"
    eyeshadow_intensity: float = 0.45

    eyeliner_color: str = "#1B1B1B"
    eyeliner_intensity: float = 0.0

    blush_color: str = "#D96C6C"
    blush_intensity: float = 0.35

    smoothing: float = 0.0

    def __post_init__(self) -> None:
        errors: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith("_color"):
                try:
                    parse_hex_color(value)
                except ValueError as exc:
                    errors.append(str(exc))
            else:  # intensities and smoothing
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{f.name} must be a number, got {value!r}")
                elif not 0.0 <= float(value) <= 1.0:
                    errors.append(f"{f.name} must be in [0, 1], got {value}")
        if errors:
            raise ValueError("invalid MakeupLook: " + "; ".join(errors))


# Named presets used by the CLI and the demo notebook.
PRESETS: dict[str, MakeupLook] = {
    "natural": MakeupLook(
        lipstick_color="#C4707F", lipstick_intensity=0.45,
        eyeshadow_color="#A87860", eyeshadow_intensity=0.30,
        blush_color="#E08A7A", blush_intensity=0.25,
        smoothing=0.20,
    ),
    "classic": MakeupLook(
        lipstick_color="#B03A5B", lipstick_intensity=0.75,
        eyeshadow_color="#8A5A44", eyeshadow_intensity=0.45,
        eyeliner_color="#1B1B1B", eyeliner_intensity=0.8,
        blush_color="#D96C6C", blush_intensity=0.35,
        smoothing=0.25,
    ),
    "bold": MakeupLook(
        lipstick_color="#8E1B3A", lipstick_intensity=0.9,
        eyeshadow_color="#5C3A6E", eyeshadow_intensity=0.6,
        eyeliner_color="#101010", eyeliner_intensity=1.0,
        blush_color="#C25B5B", blush_intensity=0.45,
        smoothing=0.30,
    ),
}
