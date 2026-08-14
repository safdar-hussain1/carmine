"""Look configuration module.

A Look defines the complete makeup application: product type (color, intensity,
finish) for each facial feature, plus skin smoothing intensity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any

from carmine.pigment import parse_hex_color


@dataclass(frozen=True)
class Product:
    """A single makeup product application.

    Attributes:
        color: Hex color string (#RRGGBB).
        intensity: Opacity/strength in [0, 1]. Default 0.0 (fully transparent).
        finish: Surface finish: "matte", "satin", or "gloss". Default "satin".
    """

    color: str
    intensity: float = 0.0
    finish: str = "satin"


@dataclass(frozen=True)
class Look:
    """A complete makeup Look.

    All products (lipstick, eyeshadow, etc.) are Product instances. Smoothing
    is a global skin-smoothing intensity in [0, 1].

    Attributes:
        lipstick: Lip product.
        eyeshadow: Eye shadow product.
        eyeliner: Eyeliner product.
        brows: Eyebrow product.
        blush: Blush/rouge product.
        highlighter: Highlight/luminizer product.
        smoothing: Skin smoothing intensity in [0, 1].
    """

    lipstick: Product = field(default_factory=lambda: Product("#B03A5B", 0.0))
    eyeshadow: Product = field(default_factory=lambda: Product("#8A5A44", 0.0))
    eyeliner: Product = field(default_factory=lambda: Product("#1B1B1B", 0.0))
    brows: Product = field(default_factory=lambda: Product("#4A3728", 0.0))
    blush: Product = field(default_factory=lambda: Product("#D96C6C", 0.0))
    highlighter: Product = field(default_factory=lambda: Product("#F5D9C8", 0.0))
    smoothing: float = 0.0

    def __post_init__(self) -> None:
        """Validate all fields and collect errors.

        Raises:
            ValueError: If any field is invalid. The message contains all errors found.
        """
        errors = []

        # Check smoothing
        if isinstance(self.smoothing, bool):
            errors.append("smoothing must be a number, not a bool")
        elif isinstance(self.smoothing, str):
            errors.append(f"smoothing must be a number, not a string")
        else:
            try:
                smoothing_val = float(self.smoothing)
                if not (0 <= smoothing_val <= 1):
                    errors.append(f"smoothing must be in [0, 1], got {smoothing_val}")
            except (TypeError, ValueError):
                errors.append(f"smoothing must be a number, got {type(self.smoothing).__name__}")

        # Validate all products
        for field_obj in fields(self):
            if field_obj.name == "smoothing":
                continue

            product = getattr(self, field_obj.name)
            errors.extend(self._validate_product(field_obj.name, product))

        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def _validate_product(field_name: str, product: Product) -> list[str]:
        """Validate a single Product and return list of error messages."""
        errors = []

        # Check hex color
        try:
            parse_hex_color(product.color)
        except ValueError as e:
            errors.append(str(e))

        # Check intensity type (reject bools and strings)
        if isinstance(product.intensity, bool):
            errors.append(f"{field_name}: intensity must be a number, not a bool")
        elif isinstance(product.intensity, str):
            errors.append(f"{field_name}: intensity must be a number, not a string")
        else:
            try:
                intensity_val = float(product.intensity)
                if not (0 <= intensity_val <= 1):
                    errors.append(
                        f"{field_name}: intensity must be in [0, 1], got {intensity_val}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"{field_name}: intensity must be a number, got {type(product.intensity).__name__}"
                )

        # Check finish
        if product.finish not in {"matte", "satin", "gloss"}:
            errors.append(
                f"{field_name}: finish must be one of {{'matte', 'satin', 'gloss'}}, got {product.finish!r}"
            )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert Look to a plain nested dict.

        Returns:
            A dict with product dicts (each with 'color', 'intensity', 'finish')
            and 'smoothing'.
        """
        return {
            "lipstick": asdict(self.lipstick),
            "eyeshadow": asdict(self.eyeshadow),
            "eyeliner": asdict(self.eyeliner),
            "brows": asdict(self.brows),
            "blush": asdict(self.blush),
            "highlighter": asdict(self.highlighter),
            "smoothing": self.smoothing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Look:
        """Reconstruct a Look from a dict.

        Args:
            data: A dict with keys for each product field and 'smoothing'.
                  Each product value should be a dict with 'color', 'intensity', 'finish'.

        Returns:
            A Look instance.

        Raises:
            ValueError: If data contains unknown keys or is malformed.
        """
        expected_keys = {"lipstick", "eyeshadow", "eyeliner", "brows", "blush", "highlighter", "smoothing"}
        provided_keys = set(data.keys())

        unknown_keys = provided_keys - expected_keys
        if unknown_keys:
            raise ValueError(f"Unknown keys in Look dict: {unknown_keys}")

        products = {}
        for field_name in ["lipstick", "eyeshadow", "eyeliner", "brows", "blush", "highlighter"]:
            product_data = data.get(field_name, {})
            products[field_name] = Product(
                color=product_data.get("color", "#000000"),
                intensity=product_data.get("intensity", 0.0),
                finish=product_data.get("finish", "satin"),
            )

        return cls(
            lipstick=products["lipstick"],
            eyeshadow=products["eyeshadow"],
            eyeliner=products["eyeliner"],
            brows=products["brows"],
            blush=products["blush"],
            highlighter=products["highlighter"],
            smoothing=data.get("smoothing", 0.0),
        )


# Preset Looks
PRESETS: dict[str, Look] = {
    "bare": Look(
        lipstick=Product("#C4707F", intensity=0.3, finish="satin"),
        eyeshadow=Product("#8A5A44", intensity=0.0),
        eyeliner=Product("#1B1B1B", intensity=0.0),
        brows=Product("#4A3728", intensity=0.25),
        blush=Product("#D96C6C", intensity=0.0),
        highlighter=Product("#F5D9C8", intensity=0.0),
        smoothing=0.15,
    ),
    "everyday": Look(
        lipstick=Product("#B03A5B", intensity=0.55, finish="satin"),
        eyeshadow=Product("#8A5A44", intensity=0.35),
        eyeliner=Product("#1B1B1B", intensity=0.0),
        brows=Product("#4A3728", intensity=0.35),
        blush=Product("#D96C6C", intensity=0.3),
        highlighter=Product("#F5D9C8", intensity=0.0),
        smoothing=0.2,
    ),
    "velvet": Look(
        lipstick=Product("#8E1B3A", intensity=0.85, finish="matte"),
        eyeshadow=Product("#5C3A6E", intensity=0.5),
        eyeliner=Product("#1B1B1B", intensity=0.8),
        brows=Product("#4A3728", intensity=0.0),
        blush=Product("#D96C6C", intensity=0.35),
        highlighter=Product("#F5D9C8", intensity=0.0),
        smoothing=0.25,
    ),
    "glass": Look(
        lipstick=Product("#B03A5B", intensity=0.6, finish="gloss"),
        eyeshadow=Product("#8A5A44", intensity=0.0),
        eyeliner=Product("#1B1B1B", intensity=0.0),
        brows=Product("#4A3728", intensity=0.0),
        blush=Product("#D96C6C", intensity=0.3),
        highlighter=Product("#F5D9C8", intensity=0.6),
        smoothing=0.3,
    ),
}
