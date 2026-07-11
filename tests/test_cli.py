import cv2
import numpy as np
import pytest

from virtual_makeup.cli import main


@pytest.fixture()
def astronaut_file(tmp_path, astronaut_bgr):
    path = tmp_path / "face.png"
    cv2.imwrite(str(path), astronaut_bgr)
    return path


def test_apply_writes_output(astronaut_file, tmp_path, capsys):
    out = tmp_path / "out" / "result.png"
    rc = main(["apply", str(astronaut_file), str(out), "--preset", "natural"])
    assert rc == 0
    assert out.exists()
    result = cv2.imread(str(out))
    assert result is not None
    assert not np.array_equal(result, cv2.imread(str(astronaut_file)))


def test_apply_with_overrides(astronaut_file, tmp_path):
    out = tmp_path / "red.png"
    rc = main([
        "apply", str(astronaut_file), str(out),
        "--lipstick", "#CC0044", "--lipstick-intensity", "0.9",
        "--eyeliner-intensity", "0",
    ])
    assert rc == 0
    assert out.exists()


def test_invalid_color_fails_cleanly(astronaut_file, tmp_path, capsys):
    rc = main(["apply", str(astronaut_file), str(tmp_path / "x.png"), "--lipstick", "red"])
    assert rc == 1
    assert "invalid hex color" in capsys.readouterr().err


def test_missing_input_fails_cleanly(tmp_path, capsys):
    rc = main(["apply", str(tmp_path / "nope.jpg"), str(tmp_path / "x.png")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_out_of_range_intensity_fails_cleanly(astronaut_file, tmp_path, capsys):
    rc = main(["apply", str(astronaut_file), str(tmp_path / "x.png"),
               "--lipstick-intensity", "1.5"])
    assert rc == 1
    assert "invalid MakeupLook" in capsys.readouterr().err


def test_landmarks_command(astronaut_file, tmp_path):
    out = tmp_path / "lm.png"
    rc = main(["landmarks", str(astronaut_file), str(out)])
    assert rc == 0
    assert out.exists()
