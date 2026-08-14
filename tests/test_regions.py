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
    """Verify that brow arcs form valid closed polygons with correct traversal order."""

    def test_right_brow_upper_exact_values(self):
        """RIGHT_BROW_UPPER must have exact ordering for proper polygon closure."""
        # Stored as inner→outer (107→70); when reversed, becomes outer→inner (70→107)
        # allowing LOWER + reversed(UPPER) to form a closed path
        assert RIGHT_BROW_UPPER == [107, 66, 105, 63, 70]

    def test_left_brow_upper_exact_values(self):
        """LEFT_BROW_UPPER must have exact ordering for proper polygon closure."""
        # Stored as inner→outer (336→300); when reversed, becomes outer→inner (300→336)
        # allowing LOWER + reversed(UPPER) to form a closed path
        assert LEFT_BROW_UPPER == [336, 296, 334, 293, 300]

    def test_right_brow_traversal_order(self):
        """RIGHT_BROW arcs connect inner endpoint to shared junction point."""
        # RIGHT_BROW_LOWER: 55 (inner) → 70 (junction)
        assert RIGHT_BROW_LOWER[0] == 55, "Right brow lower starts at inner point"
        assert RIGHT_BROW_LOWER[-1] == 70, "Right brow lower ends at junction point"

        # RIGHT_BROW_UPPER: 107 (inner) → 70 (junction), same junction as lower
        assert RIGHT_BROW_UPPER[0] == 107, "Right brow upper starts at inner point"
        assert RIGHT_BROW_UPPER[-1] == 70, "Right brow upper ends at junction point"

    def test_left_brow_traversal_order(self):
        """LEFT_BROW arcs connect inner endpoint to shared junction point."""
        # LEFT_BROW_LOWER: 285 (inner) → 300 (junction)
        assert LEFT_BROW_LOWER[0] == 285, "Left brow lower starts at inner point"
        assert LEFT_BROW_LOWER[-1] == 300, "Left brow lower ends at junction point"

        # LEFT_BROW_UPPER: 336 (inner) → 300 (junction), same junction as lower
        assert LEFT_BROW_UPPER[0] == 336, "Left brow upper starts at inner point"
        assert LEFT_BROW_UPPER[-1] == 300, "Left brow upper ends at junction point"

    def test_right_brow_polygon_closure(self):
        """Verify RIGHT_BROW_LOWER + reversed(UPPER) forms a complete perimeter."""
        # Polygon path: 55→...→70→70→...→107 (with shared junction at index 70)
        combined = RIGHT_BROW_LOWER + RIGHT_BROW_UPPER[::-1]
        # The junction point (70) appears exactly twice (once at end of LOWER, once at
        # start of reversed UPPER), which is valid for polygon construction
        assert combined.count(70) == 2, (
            f"Junction point 70 should appear twice in polygon; got {combined.count(70)}"
        )

    def test_left_brow_polygon_closure(self):
        """Verify LEFT_BROW_LOWER + reversed(UPPER) forms a complete perimeter."""
        # Polygon path: 285→...→300→300→...→336 (with shared junction at index 300)
        combined = LEFT_BROW_LOWER + LEFT_BROW_UPPER[::-1]
        assert combined.count(300) == 2, (
            f"Junction point 300 should appear twice in polygon; got {combined.count(300)}"
        )
