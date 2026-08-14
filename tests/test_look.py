"""Tests for Look configuration and presets."""

import pytest

from carmine.look import Look, Product, PRESETS


class TestProduct:
    def test_product_creation_with_defaults(self):
        """Product initializes with sensible defaults."""
        p = Product("#AABBCC")
        assert p.color == "#AABBCC"
        assert p.intensity == 0.0
        assert p.finish == "satin"

    def test_product_creation_with_custom_values(self):
        """Product accepts custom intensity and finish."""
        p = Product("#FF0000", intensity=0.5, finish="matte")
        assert p.color == "#FF0000"
        assert p.intensity == 0.5
        assert p.finish == "matte"

    def test_product_is_frozen(self):
        """Product is immutable."""
        p = Product("#AABBCC")
        with pytest.raises(AttributeError):
            p.color = "#BBCCAA"


class TestLookDefaults:
    def test_look_creation_with_all_defaults(self):
        """Look initializes with default products."""
        look = Look()
        assert look.lipstick.color == "#B03A5B"
        assert look.lipstick.intensity == 0.0
        assert look.eyeshadow.color == "#8A5A44"
        assert look.eyeliner.color == "#1B1B1B"
        assert look.brows.color == "#4A3728"
        assert look.blush.color == "#D96C6C"
        assert look.highlighter.color == "#F5D9C8"
        assert look.smoothing == 0.0

    def test_look_is_frozen(self):
        """Look is immutable."""
        look = Look()
        with pytest.raises(AttributeError):
            look.smoothing = 0.5


class TestLookValidation:
    def test_valid_look_passes_validation(self):
        """A well-formed Look should not raise during __post_init__."""
        look = Look(
            lipstick=Product("#FF0000", intensity=0.5),
            smoothing=0.3
        )
        # If we get here without exception, validation passed
        assert look.smoothing == 0.3

    def test_bad_hex_color_in_product_raises_valueerror(self):
        """Invalid hex color in any product raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("not-a-color"))
        assert "invalid hex color" in str(exc_info.value).lower()

    def test_intensity_out_of_range_raises_valueerror(self):
        """Intensity outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("#FF0000", intensity=1.5))
        error_msg = str(exc_info.value)
        assert "lipstick" in error_msg.lower()
        assert "intensity" in error_msg.lower()
        assert "1.5" in error_msg

    def test_negative_intensity_raises_valueerror(self):
        """Negative intensity raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("#FF0000", intensity=-0.1))
        error_msg = str(exc_info.value)
        assert "lipstick" in error_msg.lower()
        assert "intensity" in error_msg.lower()
        assert "-0.1" in error_msg

    def test_smoothing_out_of_range_raises_valueerror(self):
        """Smoothing outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(smoothing=1.5)
        error_msg = str(exc_info.value)
        assert "smoothing" in error_msg.lower()
        assert "1.5" in error_msg

    def test_invalid_finish_raises_valueerror(self):
        """Finish not in {matte, satin, gloss} raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("#FF0000", finish="shimmer"))
        error_msg = str(exc_info.value)
        assert "lipstick" in error_msg.lower()
        assert "finish" in error_msg.lower()
        assert "shimmer" in error_msg.lower()

    def test_non_numeric_intensity_raises_valueerror(self):
        """Non-numeric intensity raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("#FF0000", intensity="0.5"))
        error_msg = str(exc_info.value)
        assert "lipstick" in error_msg.lower()
        assert "intensity" in error_msg.lower()
        assert "string" in error_msg.lower()

    def test_bool_intensity_rejected(self):
        """Boolean values for intensity are rejected."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick=Product("#FF0000", intensity=True))
        error_msg = str(exc_info.value)
        assert "lipstick" in error_msg.lower()
        assert "intensity" in error_msg.lower()
        assert "bool" in error_msg.lower()

    def test_multiple_errors_collected_in_single_exception(self):
        """All validation errors are collected and reported in one ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(
                lipstick=Product("bad-hex", intensity=1.5, finish="shimmer"),
                eyeshadow=Product("#FF0000", intensity=-0.1),
                smoothing=2.0
            )
        error_msg = str(exc_info.value)
        # Verify all error sources are mentioned
        assert "lipstick" in error_msg.lower()
        assert "eyeshadow" in error_msg.lower()
        assert "smoothing" in error_msg.lower()

    def test_non_product_field_and_smoothing_error_collected(self):
        """Non-Product field and smoothing error are collected in one ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Look(lipstick="red", smoothing=5)
        error_msg = str(exc_info.value)
        # Should mention both the lipstick type error and smoothing range error
        assert "lipstick" in error_msg.lower()
        assert "Product" in error_msg
        assert "smoothing" in error_msg.lower()
        assert "5" in error_msg


class TestPresets:
    def test_bare_preset_exists_and_validates(self):
        """The bare preset should exist and be valid."""
        bare = PRESETS["bare"]
        assert bare.lipstick.color == "#C4707F"
        assert bare.lipstick.intensity == 0.3
        assert bare.lipstick.finish == "satin"
        assert bare.brows.intensity == 0.25
        assert bare.smoothing == 0.15

    def test_everyday_preset_exists_and_validates(self):
        """The everyday preset should exist and be valid."""
        everyday = PRESETS["everyday"]
        assert everyday.lipstick.intensity == 0.55
        assert everyday.lipstick.finish == "satin"
        assert everyday.eyeshadow.intensity == 0.35
        assert everyday.blush.intensity == 0.3
        assert everyday.brows.intensity == 0.35
        assert everyday.smoothing == 0.2

    def test_velvet_preset_exists_and_validates(self):
        """The velvet preset should exist and be valid."""
        velvet = PRESETS["velvet"]
        assert velvet.lipstick.color == "#8E1B3A"
        assert velvet.lipstick.intensity == 0.85
        assert velvet.lipstick.finish == "matte"
        assert velvet.eyeshadow.color == "#5C3A6E"
        assert velvet.eyeshadow.intensity == 0.5
        assert velvet.eyeliner.intensity == 0.8
        assert velvet.blush.intensity == 0.35
        assert velvet.smoothing == 0.25

    def test_glass_preset_exists_and_validates(self):
        """The glass preset should exist and be valid."""
        glass = PRESETS["glass"]
        assert glass.lipstick.color == "#B03A5B"
        assert glass.lipstick.intensity == 0.6
        assert glass.lipstick.finish == "gloss"
        assert glass.highlighter.intensity == 0.6
        assert glass.blush.intensity == 0.3
        assert glass.smoothing == 0.3

    def test_all_presets_validate(self):
        """All presets in PRESETS dict should be valid Looks."""
        for name, look in PRESETS.items():
            assert isinstance(look, Look)
            # If we get here, the Look passed validation in __post_init__


class TestLookDictConversion:
    def test_to_dict_produces_nested_dicts(self):
        """to_dict converts a Look to a plain nested dict."""
        look = Look(
            lipstick=Product("#FF0000", intensity=0.5),
            smoothing=0.2
        )
        d = look.to_dict()

        # Check structure
        assert isinstance(d, dict)
        assert "lipstick" in d
        assert "eyeshadow" in d
        assert "eyeliner" in d
        assert "brows" in d
        assert "blush" in d
        assert "highlighter" in d
        assert "smoothing" in d

        # Check that products are dicts, not Product objects
        assert isinstance(d["lipstick"], dict)
        assert d["lipstick"]["color"] == "#FF0000"
        assert d["lipstick"]["intensity"] == 0.5
        assert d["lipstick"]["finish"] == "satin"

    def test_from_dict_recreates_look_from_dict(self):
        """from_dict reconstructs a Look from a plain dict."""
        d = {
            "lipstick": {"color": "#FF0000", "intensity": 0.5, "finish": "satin"},
            "eyeshadow": {"color": "#8A5A44", "intensity": 0.0, "finish": "satin"},
            "eyeliner": {"color": "#1B1B1B", "intensity": 0.0, "finish": "satin"},
            "brows": {"color": "#4A3728", "intensity": 0.0, "finish": "satin"},
            "blush": {"color": "#D96C6C", "intensity": 0.0, "finish": "satin"},
            "highlighter": {"color": "#F5D9C8", "intensity": 0.0, "finish": "satin"},
            "smoothing": 0.2
        }
        look = Look.from_dict(d)

        assert look.lipstick.color == "#FF0000"
        assert look.lipstick.intensity == 0.5
        assert look.smoothing == 0.2

    def test_round_trip_to_dict_from_dict(self):
        """to_dict followed by from_dict should produce an equal Look."""
        original = Look(
            lipstick=Product("#FF0000", intensity=0.75, finish="gloss"),
            smoothing=0.4
        )

        d = original.to_dict()
        restored = Look.from_dict(d)

        assert original == restored

    def test_round_trip_all_presets(self):
        """All presets should round-trip through to_dict/from_dict."""
        for name, original in PRESETS.items():
            d = original.to_dict()
            restored = Look.from_dict(d)
            assert original == restored, f"Preset '{name}' failed round-trip"

    def test_from_dict_rejects_unknown_keys(self):
        """from_dict should reject dicts with unknown keys."""
        d = {
            "lipstick": {"color": "#FF0000", "intensity": 0.5, "finish": "satin"},
            "eyeshadow": {"color": "#8A5A44", "intensity": 0.0, "finish": "satin"},
            "eyeliner": {"color": "#1B1B1B", "intensity": 0.0, "finish": "satin"},
            "brows": {"color": "#4A3728", "intensity": 0.0, "finish": "satin"},
            "blush": {"color": "#D96C6C", "intensity": 0.0, "finish": "satin"},
            "highlighter": {"color": "#F5D9C8", "intensity": 0.0, "finish": "satin"},
            "smoothing": 0.2,
            "unknown_field": "bad"
        }
        with pytest.raises(ValueError) as exc_info:
            Look.from_dict(d)
        error_msg = str(exc_info.value)
        assert "unknown" in error_msg.lower()
        assert "unknown_field" in error_msg
