"""Tests for face topology and landmark index sets."""

import pytest
from carmine.regions import (
    LIPS_OUTER,
    LIPS_INNER,
    RIGHT_EYE,
    LEFT_EYE,
    RIGHT_EYE_UPPER,
    LEFT_EYE_UPPER,
    RIGHT_BROW_LOWER,
    LEFT_BROW_LOWER,
    RIGHT_BROW_UPPER,
    LEFT_BROW_UPPER,
    FACE_OVAL,
    RIGHT_CHEEK,
    LEFT_CHEEK,
    RIGHT_EYE_OUTER,
    LEFT_EYE_OUTER,
    NOSE_BRIDGE,
    RIGHT_CHEEKBONE,
    LEFT_CHEEKBONE,
    NUM_LANDMARKS,
)


class TestConstants:
    """Verify landmark constant values."""

    def test_num_landmarks(self):
        """NUM_LANDMARKS should be 478 (468 mesh + 10 iris)."""
        assert NUM_LANDMARKS == 478

    def test_scalar_indices(self):
        """Verify scalar (single-point) indices are within bounds."""
        assert 0 <= RIGHT_CHEEK < NUM_LANDMARKS
        assert 0 <= LEFT_CHEEK < NUM_LANDMARKS
        assert 0 <= RIGHT_EYE_OUTER < NUM_LANDMARKS
        assert 0 <= LEFT_EYE_OUTER < NUM_LANDMARKS


class TestIndexBounds:
    """All exported index lists must have valid bounds."""

    INDEX_LISTS = [
        ("LIPS_OUTER", LIPS_OUTER),
        ("LIPS_INNER", LIPS_INNER),
        ("RIGHT_EYE", RIGHT_EYE),
        ("LEFT_EYE", LEFT_EYE),
        ("RIGHT_EYE_UPPER", RIGHT_EYE_UPPER),
        ("LEFT_EYE_UPPER", LEFT_EYE_UPPER),
        ("RIGHT_BROW_LOWER", RIGHT_BROW_LOWER),
        ("LEFT_BROW_LOWER", LEFT_BROW_LOWER),
        ("RIGHT_BROW_UPPER", RIGHT_BROW_UPPER),
        ("LEFT_BROW_UPPER", LEFT_BROW_UPPER),
        ("FACE_OVAL", FACE_OVAL),
        ("NOSE_BRIDGE", NOSE_BRIDGE),
        ("RIGHT_CHEEKBONE", RIGHT_CHEEKBONE),
        ("LEFT_CHEEKBONE", LEFT_CHEEKBONE),
    ]

    @pytest.mark.parametrize("name,indices", INDEX_LISTS)
    def test_all_indices_in_bounds(self, name, indices):
        """All indices must satisfy 0 <= i < 478."""
        for idx in indices:
            assert (
                0 <= idx < NUM_LANDMARKS
            ), f"{name} contains out-of-bounds index {idx}"

    @pytest.mark.parametrize("name,indices", INDEX_LISTS)
    def test_no_duplicates_within_ring(self, name, indices):
        """Each ring should have no duplicate indices."""
        unique = set(indices)
        assert len(unique) == len(
            indices
        ), f"{name} has duplicate indices: {len(indices)} total, {len(unique)} unique"


class TestBrowPolygons:
    """Verify that brow arcs form valid closed polygons."""

    def test_right_brow_forms_polygon(self):
        """RIGHT_BROW_LOWER + reversed RIGHT_BROW_UPPER should form a closed polygon."""
        # The polygon is formed by concatenating lower + reversed(upper), which
        # may share connection indices at the boundaries.
        combined = RIGHT_BROW_LOWER + RIGHT_BROW_UPPER[::-1]
        # Allow exactly one or two shared indices for polygon closure
        unique_count = len(set(combined))
        total_count = len(combined)
        # Either all unique, or sharing one junction point
        assert unique_count >= total_count - 2, (
            f"Right brow polygon has too many duplicates: "
            f"{total_count} total, {unique_count} unique"
        )

    def test_left_brow_forms_polygon(self):
        """LEFT_BROW_LOWER + reversed LEFT_BROW_UPPER should form a closed polygon."""
        combined = LEFT_BROW_LOWER + LEFT_BROW_UPPER[::-1]
        unique_count = len(set(combined))
        total_count = len(combined)
        assert unique_count >= total_count - 2, (
            f"Left brow polygon has too many duplicates: "
            f"{total_count} total, {unique_count} unique"
        )

    def test_brow_lower_and_upper_no_overlap(self):
        """Brow lower and upper arcs may share junction points."""
        right_lower_set = set(RIGHT_BROW_LOWER)
        right_upper_set = set(RIGHT_BROW_UPPER)
        # They should share at most one junction point (at the ends)
        overlap = len(right_lower_set & right_upper_set)
        assert overlap <= 1, (
            f"Right brow lower and upper share {overlap} indices; expected at most 1"
        )

        left_lower_set = set(LEFT_BROW_LOWER)
        left_upper_set = set(LEFT_BROW_UPPER)
        overlap = len(left_lower_set & left_upper_set)
        assert overlap <= 1, (
            f"Left brow lower and upper share {overlap} indices; expected at most 1"
        )
