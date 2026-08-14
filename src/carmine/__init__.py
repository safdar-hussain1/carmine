"""Carmine: texture-preserving virtual makeup engine."""

from carmine.engine import VideoEngine, apply_look
from carmine.filters import OneEuroFilter
from carmine.landmarks import FaceLandmarker, NoFaceError
from carmine.look import Product, Look, PRESETS

__version__ = "1.0.0"

__all__ = [
    "Product",
    "Look",
    "PRESETS",
    "apply_look",
    "VideoEngine",
    "OneEuroFilter",
    "FaceLandmarker",
    "NoFaceError",
]
