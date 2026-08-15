"""FaceLandmarker wrapper with pinned model management.

Wraps the MediaPipe Tasks FaceLandmarker (the legacy ``mp.solutions`` API is
not available in this environment) behind a small BGR-in / pixel-coords-out
interface consistent with the rest of the engine.
"""

import hashlib
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .regions import NUM_LANDMARKS

# sha256 of the known-good face_landmarker.task, computed from the file
# cached at ~/.cache/virtual_makeup/face_landmarker.task. The download URL
# below serves "latest", which can drift out from under this pin -- that
# drift is exactly what the checksum check catches.
_CHECKSUM = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

_CACHE_PATH = Path.home() / ".cache" / "carmine" / "face_landmarker.task"

_ENV_OVERRIDE = "CARMINE_MODEL"


class NoFaceError(RuntimeError):
    """Raised when no face is detected in an image."""


def validate_image(image) -> None:
    """Fail fast on a malformed image, listing every problem found.

    Args:
        image: Candidate image to validate.

    Raises:
        ValueError: With a message listing every problem found (not an
            ndarray, wrong ndim, wrong channel count, non-uint8 dtype, or
            empty array).
    """
    problems = []

    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Invalid image: expected a numpy ndarray, got {type(image).__name__}"
        )

    if image.ndim != 3:
        problems.append(f"ndim must be 3 (H, W, C), got {image.ndim}")
    elif image.shape[2] != 3:
        problems.append(f"channel count must be 3, got {image.shape[2]}")

    if image.dtype != np.uint8:
        problems.append(f"dtype must be uint8, got {image.dtype}")

    if image.size == 0:
        problems.append("image is empty")

    if problems:
        raise ValueError("Invalid image: " + "; ".join(problems))


def _verify_checksum(path: Path, expected_hex: str) -> None:
    """Verify a downloaded file's sha256 digest, deleting it on mismatch.

    Args:
        path: Path to the file to verify.
        expected_hex: Expected sha256 hex digest.

    Raises:
        RuntimeError: If the digest does not match; the file is deleted.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hex:
        path.unlink()
        raise RuntimeError(
            f"Model checksum mismatch for {path}: expected {expected_hex}, "
            f"got {digest}. Downloaded file deleted."
        )


def _download_model(tmp_path: Path) -> None:
    """Download the model to ``tmp_path``, wrapping failures in RuntimeError.

    Verify-then-promote: the caller is responsible for checksumming
    ``tmp_path`` *before* it is renamed into the cache path, so a
    partial/corrupt/wrong download never becomes the cached model. Any
    temp file left behind by a failed download is removed here so a retry
    starts clean.

    Raises:
        RuntimeError: If the download fails for any reason (network error,
            TLS/certificate error, or any other OS-level failure), with a
            message identifying the model URL, the cache path, the
            ``CARMINE_MODEL`` override, and (for SSL failures specifically)
            the macOS "Install Certificates.command" fix.
    """
    try:
        urllib.request.urlretrieve(_MODEL_URL, tmp_path)
    except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        ssl_hint = ""
        if isinstance(exc, ssl.SSLError) or isinstance(
            getattr(exc, "reason", None), ssl.SSLError
        ):
            ssl_hint = (
                " This looks like an SSL/certificate error; on macOS, running "
                "'/Applications/Python 3.x/Install Certificates.command' "
                "(matching your Python version) often fixes it."
            )
        raise RuntimeError(
            f"Failed to download the FaceLandmarker model from {_MODEL_URL} "
            f"to {_CACHE_PATH}: {exc}. Set the {_ENV_OVERRIDE} environment "
            f"variable to point at a local .task file to bypass the "
            f"download entirely.{ssl_hint}"
        ) from exc


def model_path() -> Path:
    """Resolve the path to the FaceLandmarker model file.

    If the ``CARMINE_MODEL`` environment variable is set, its value is used
    verbatim as the model path -- the pinned checksum is NOT enforced on an
    explicit override, since the pin exists to protect the download path,
    not to second-guess a caller-supplied model.

    Otherwise, the model is downloaded (if not already cached) to
    ``~/.cache/carmine/face_landmarker.task`` and its sha256 checksum is
    verified *before* the downloaded file is promoted into the cache path
    (verify-then-promote), so a checksum failure never leaves a bad file at
    the cache path -- only a rejected temp file, which is deleted.

    Returns:
        Path to a usable FaceLandmarker ``.task`` model file.

    Raises:
        RuntimeError: If the download fails (see `_download_model`), or if
            the downloaded file's checksum does not match the pin. The
            temp file is deleted in either case, and the cache path is
            never left holding an unverified file.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)

    if not _CACHE_PATH.exists():
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _CACHE_PATH.with_suffix(".task.tmp")
        _download_model(tmp_path)
        # Verify before promoting: a checksum failure here deletes tmp_path
        # (see _verify_checksum) and never touches _CACHE_PATH.
        _verify_checksum(tmp_path, _CHECKSUM)
        tmp_path.rename(_CACHE_PATH)

    _verify_checksum(_CACHE_PATH, _CHECKSUM)
    return _CACHE_PATH


def _landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    """Convert normalized MediaPipe landmarks to float32 pixel coordinates."""
    coords = np.empty((NUM_LANDMARKS, 2), dtype=np.float32)
    for i, landmark in enumerate(face_landmarks):
        coords[i, 0] = landmark.x * width
        coords[i, 1] = landmark.y * height
    return coords


class FaceLandmarker:
    """Wraps MediaPipe Tasks FaceLandmarker for single-face detection.

    Images are supplied as BGR uint8 arrays (consistent with the rest of
    the engine) and converted to the RGB format the MediaPipe Tasks API
    expects. The underlying landmarker(s) are created lazily on first use
    since initialization is slow (roughly one second), and reused across
    calls.
    """

    def __init__(self):
        self._image_landmarker = None
        self._video_landmarker = None

    def _make_options(self, running_mode: "vision.RunningMode") -> "vision.FaceLandmarkerOptions":
        return vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path())),
            running_mode=running_mode,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

    def _get_image_landmarker(self):
        if self._image_landmarker is None:
            options = self._make_options(vision.RunningMode.IMAGE)
            self._image_landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._image_landmarker

    def _get_video_landmarker(self):
        if self._video_landmarker is None:
            options = self._make_options(vision.RunningMode.VIDEO)
            self._video_landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._video_landmarker

    @staticmethod
    def _to_mp_image(image_bgr: np.ndarray) -> "mp.Image":
        rgb = image_bgr[:, :, ::-1]
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Detect facial landmarks in a single BGR image.

        Args:
            image_bgr: BGR uint8 image of shape (H, W, 3).

        Returns:
            Float32 array of shape (478, 2) with pixel coordinates.

        Raises:
            ValueError: If the image fails validation.
            NoFaceError: If no face is detected.
        """
        validate_image(image_bgr)
        height, width = image_bgr.shape[:2]

        mp_image = self._to_mp_image(image_bgr)
        result = self._get_image_landmarker().detect(mp_image)

        if not result.face_landmarks:
            raise NoFaceError("No face detected in image")

        return _landmarks_to_pixels(result.face_landmarks[0], width, height)

    def detect_video(self, image_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Detect facial landmarks in a video frame using VIDEO running mode.

        Timestamps must be monotonically increasing across successive calls
        on the same FaceLandmarker instance -- this is a requirement of the
        underlying MediaPipe Tasks VIDEO running mode, which tracks state
        between frames.

        Args:
            image_bgr: BGR uint8 image of shape (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds; must be strictly
                greater than the timestamp of the previous call.

        Returns:
            Float32 array of shape (478, 2) with pixel coordinates.

        Raises:
            ValueError: If the image fails validation.
            NoFaceError: If no face is detected.
        """
        validate_image(image_bgr)
        height, width = image_bgr.shape[:2]

        mp_image = self._to_mp_image(image_bgr)
        result = self._get_video_landmarker().detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            raise NoFaceError("No face detected in image")

        return _landmarks_to_pixels(result.face_landmarks[0], width, height)
