"""Tests for geometric calculations on facial landmarks."""

import math

import numpy as np
import pytest

from carmine.geometry import interocular_distance, face_roll_degrees
from carmine.regions import RIGHT_EYE_OUTER, LEFT_EYE_OUTER


class TestInterocularDistance:
    """Tests for interocular_distance function."""

    def test_known_distance_value(self):
        """interocular_distance should return correct euclidean distance."""
        # Create synthetic landmarks: all zeros except the eye corners
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [3.0, 4.0]
        # Distance: sqrt(3^2 + 4^2) = sqrt(25) = 5.0
        distance = interocular_distance(landmarks)
        assert distance == pytest.approx(5.0)

    def test_distance_scales_linearly(self):
        """Distance should scale linearly when coordinates scale."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [3.0, 4.0]
        d1 = interocular_distance(landmarks)

        # Double all coordinates
        landmarks_doubled = landmarks * 2.0
        d2 = interocular_distance(landmarks_doubled)

        assert d2 == pytest.approx(2.0 * d1)

    def test_horizontal_distance(self):
        """Test with pure horizontal separation."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [10.0, 0.0]
        distance = interocular_distance(landmarks)
        assert distance == pytest.approx(10.0)

    def test_vertical_distance(self):
        """Test with pure vertical separation."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [0.0, 10.0]
        distance = interocular_distance(landmarks)
        assert distance == pytest.approx(10.0)

    def test_degenerate_landmarks_raise_error(self):
        """Should raise ValueError when landmarks are too close (< 1e-6)."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        # Eye corners are identical or very close
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [1e-7, 0.0]
        with pytest.raises(ValueError):
            interocular_distance(landmarks)

    def test_exactly_at_threshold(self):
        """Distance just above 1e-6 threshold should be valid."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        # Use 2e-6 to avoid float32 precision issues with 1e-6
        landmarks[LEFT_EYE_OUTER] = [2e-6, 0.0]
        distance = interocular_distance(landmarks)
        assert distance == pytest.approx(2e-6)

    def test_invalid_shape_raises_error(self):
        """Should raise error for invalid landmark array shape."""
        landmarks = np.zeros((477, 2), dtype=np.float32)  # Wrong number of landmarks
        with pytest.raises((ValueError, IndexError)):
            interocular_distance(landmarks)


class TestFaceRollDegrees:
    """Tests for face_roll_degrees function."""

    def test_level_face_zero_roll(self):
        """A level face should have 0 degrees roll."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 50.0]
        landmarks[LEFT_EYE_OUTER] = [100.0, 50.0]
        roll = face_roll_degrees(landmarks)
        assert roll == pytest.approx(0.0, abs=1e-5)

    def test_45_degree_counterclockwise_roll(self):
        """A 45° counterclockwise rotation should show as 45°."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        # Vector from right to left: (100, 100) which is 45° from horizontal
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [100.0, 100.0]
        roll = face_roll_degrees(landmarks)
        assert roll == pytest.approx(45.0, abs=1e-5)

    def test_negative_45_degree_roll(self):
        """A 45° clockwise rotation (negative roll)."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [0.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [100.0, -100.0]
        roll = face_roll_degrees(landmarks)
        assert roll == pytest.approx(-45.0, abs=1e-5)

    def test_upside_down_face(self):
        """A face rotated 180° (eyes swap positions, vector points left and down)."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [100.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [0.0, 0.0]
        roll = face_roll_degrees(landmarks)
        # Vector from (100, 0) to (0, 0) is (-100, 0), which is 180°
        assert roll == pytest.approx(180.0, abs=1e-5)

    def test_vertical_eyes(self):
        """Eyes arranged vertically (90° rotation)."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [50.0, 0.0]
        landmarks[LEFT_EYE_OUTER] = [50.0, 100.0]
        roll = face_roll_degrees(landmarks)
        assert roll == pytest.approx(90.0, abs=1e-5)

    def test_negative_90_degree_roll(self):
        """Eyes arranged vertically downward (-90°)."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [50.0, 100.0]
        landmarks[LEFT_EYE_OUTER] = [50.0, 0.0]
        roll = face_roll_degrees(landmarks)
        assert roll == pytest.approx(-90.0, abs=1e-5)

    def test_degenerate_landmarks_raise_error(self):
        """Should raise ValueError when eye corners are identical."""
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[RIGHT_EYE_OUTER] = [50.0, 50.0]
        landmarks[LEFT_EYE_OUTER] = [50.0, 50.0]
        with pytest.raises(ValueError):
            face_roll_degrees(landmarks)
