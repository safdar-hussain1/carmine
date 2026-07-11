import pytest

from virtual_makeup.config import PRESETS, MakeupLook, parse_hex_color


class TestParseHexColor:
    def test_parses_with_and_without_hash(self):
        assert parse_hex_color("#B03A5B") == (176, 58, 91)
        assert parse_hex_color("b03a5b") == (176, 58, 91)

    @pytest.mark.parametrize("bad", ["", "#FFF", "#GGGGGG", "#12345", "red", 123, None])
    def test_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_hex_color(bad)


class TestMakeupLookValidation:
    def test_default_look_is_valid(self):
        MakeupLook()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lipstick_color": "not-a-color"},
            {"eyeshadow_color": "#12345G"},
            {"lipstick_intensity": -0.1},
            {"lipstick_intensity": 1.5},
            {"blush_intensity": "high"},
            {"smoothing": 2.0},
            {"eyeliner_intensity": True},
        ],
    )
    def test_rejects_invalid_field(self, kwargs):
        with pytest.raises(ValueError, match="invalid MakeupLook"):
            MakeupLook(**kwargs)

    def test_collects_all_errors_in_one_message(self):
        with pytest.raises(ValueError) as exc:
            MakeupLook(lipstick_color="xx", blush_intensity=7, smoothing=-1)
        message = str(exc.value)
        assert "xx" in message
        assert "blush_intensity" in message
        assert "smoothing" in message

    def test_all_presets_construct(self):
        assert set(PRESETS) == {"natural", "classic", "bold"}
