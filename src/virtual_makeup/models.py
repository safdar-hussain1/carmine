"""Model asset management for the MediaPipe FaceLandmarker task."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_ENV_VAR = "VMAKEUP_MODEL"
# ~3.7 MB; anything much smaller is a failed/partial download.
_MIN_MODEL_BYTES = 1_000_000


def default_model_path() -> Path:
    return Path.home() / ".cache" / "virtual_makeup" / "face_landmarker.task"


def get_model_path(auto_download: bool = True) -> Path:
    """Resolve the landmarker model, downloading it on first use.

    Set the VMAKEUP_MODEL environment variable to use a pre-downloaded
    copy (e.g. in offline environments).
    """
    env = os.environ.get(MODEL_ENV_VAR)
    if env:
        path = Path(env)
        if not path.is_file():
            raise FileNotFoundError(f"{MODEL_ENV_VAR} points to a missing file: {path}")
        return path

    path = default_model_path()
    if path.is_file() and path.stat().st_size >= _MIN_MODEL_BYTES:
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"face landmarker model not found at {path}; "
            f"download it from {MODEL_URL} or set {MODEL_ENV_VAR}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.partial")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if tmp.stat().st_size < _MIN_MODEL_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded model looks truncated ({tmp} < 1 MB)")
    tmp.replace(path)
    return path
