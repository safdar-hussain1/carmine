"""Face landmark detection built on MediaPipe FaceMesh."""

from __future__ import annotations

import numpy as np

from . import regions


class NoFaceDetectedError(RuntimeError):
    """Raised when no face is found in the input image."""


class FaceLandmarker:
    """Detects the 468-point face mesh and returns pixel coordinates.

    Wraps the MediaPipe Tasks FaceLandmarker (the legacy ``solutions``
    API was removed in mediapipe 0.10.x). The task graph is created
    lazily on first use so importing this module stays cheap; the model
    file is auto-downloaded on first run (see ``models.get_model_path``).
    """

    def __init__(self, model_path=None) -> None:
        self._model_path = model_path
        self._task = None

    def _ensure_task(self):
        if self._task is None:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            from .models import get_model_path

            path = self._model_path or get_model_path()
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self._task = vision.FaceLandmarker.create_from_options(options)
        return self._task

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a (468, 2) float32 array of pixel-space landmarks.

        Raises NoFaceDetectedError if MediaPipe finds no face, and
        ValueError for inputs that are not H x W x 3 uint8 images.
        """
        _validate_image(image_bgr)
        import cv2
        import mediapipe as mp

        task = self._ensure_task()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            raise NoFaceDetectedError("no face detected in image")
        h, w = image_bgr.shape[:2]
        pts = np.array(
            [(p.x * w, p.y * h) for p in result.face_landmarks[0][: regions.NUM_LANDMARKS]],
            dtype=np.float32,
        )
        return pts

    def close(self) -> None:
        if self._task is not None:
            self._task.close()
            self._task = None


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be H x W x 3 (BGR), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError(f"image too small for face detection: {image.shape[:2]}")


def interocular_distance(landmarks: np.ndarray) -> float:
    """Distance between the outer eye corners — the scale unit for effects."""
    a = landmarks[regions.RIGHT_EYE_OUTER]
    b = landmarks[regions.LEFT_EYE_OUTER]
    return float(np.hypot(*(b - a)))
