"""Geometric calculations for facial landmarks."""

import math

import numpy as np

from .regions import RIGHT_EYE_OUTER, LEFT_EYE_OUTER


def interocular_distance(landmarks: np.ndarray) -> float:
    """Euclidean distance between outer eye corners.

    Measures the distance between the right and left outer eye corners,
    used to scale all makeup effects to face size in the image.

    Args:
        landmarks: Float32 array of shape (N, 2) containing landmark positions
            in pixel coordinates. Must have at least 478 landmarks.

    Returns:
        Distance in pixels between right and left outer eye corners.

    Raises:
        ValueError: If the distance is less than 1e-6 (degenerate/collapsed face).
    """
    right_corner = landmarks[RIGHT_EYE_OUTER]
    left_corner = landmarks[LEFT_EYE_OUTER]

    delta = left_corner - right_corner
    distance = float(np.linalg.norm(delta))

    if distance < 1e-6:
        raise ValueError(
            f"Degenerate landmarks: interocular distance {distance} is too small"
        )

    return distance


def face_roll_degrees(landmarks: np.ndarray) -> float:
    """Rotation angle of the face in degrees.

    Measures the roll (rotation around the camera's optical axis) by computing
    the angle of the vector from the right eye corner to the left eye corner.
    Positive angles indicate counterclockwise rotation; negative angles indicate
    clockwise rotation.

    Args:
        landmarks: Float32 array of shape (N, 2) containing landmark positions
            in pixel coordinates. Must have at least 478 landmarks.

    Returns:
        Roll angle in degrees, in the range [-180, 180].

    Raises:
        ValueError: If eye corners are identical (degenerate landmarks).
    """
    right_corner = landmarks[RIGHT_EYE_OUTER]
    left_corner = landmarks[LEFT_EYE_OUTER]

    delta = left_corner - right_corner
    dx, dy = delta[0], delta[1]

    # Check for degenerate case
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        raise ValueError("Degenerate landmarks: eye corners are identical")

    # atan2 returns angle in radians; convert to degrees
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
