"""virtual_makeup: landmark-driven virtual makeup that preserves skin texture.

Public API::

    from virtual_makeup import MakeupLook, PRESETS, apply_makeup, FaceLandmarker
"""

from .config import PRESETS, MakeupLook, parse_hex_color
from .landmarks import FaceLandmarker, NoFaceDetectedError
from .makeup import apply_makeup

__version__ = "1.0.0"

__all__ = [
    "MakeupLook",
    "PRESETS",
    "apply_makeup",
    "FaceLandmarker",
    "NoFaceDetectedError",
    "parse_hex_color",
    "__version__",
]
