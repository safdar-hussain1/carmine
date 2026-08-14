"""Subprocess tests for the `carmine` CLI (src/carmine/cli.py)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from carmine.look import Look, PRESETS

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(REPO_ROOT / "src")


def run_cli(*args):
    """Run `python -m carmine.cli <args>` as a subprocess.

    Uses `sys.executable` (the interpreter already running this test suite,
    which is the project's venv python) rather than hard-coding a path to
    the venv, and sets PYTHONPATH=src explicitly the way the rest of this
    suite is invoked -- the CLI isn't installed in a way that's importable
    without it (see other tests' invocation via `PYTHONPATH=src pytest`).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR
    return subprocess.run(
        [sys.executable, "-m", "carmine.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture(scope="module")
def astronaut_png(tmp_path_factory):
    from skimage import data

    rgb = data.astronaut()
    bgr = rgb[:, :, ::-1].copy()
    path = tmp_path_factory.mktemp("cli-fixtures") / "astronaut.png"
    cv2.imwrite(str(path), bgr)
    return path


@pytest.fixture(scope="module")
def astronaut_bgr():
    from skimage import data

    rgb = data.astronaut()
    return rgb[:, :, ::-1].copy()


class TestApply:
    def test_preset_writes_output_that_differs_from_input(self, astronaut_png, tmp_path):
        out_path = tmp_path / "out.png"
        result = run_cli("apply", str(astronaut_png), str(out_path), "--preset", "velvet")

        assert result.returncode == 0, result.stderr
        assert out_path.exists()

        original = cv2.imread(str(astronaut_png))
        produced = cv2.imread(str(out_path))
        assert not np.array_equal(original, produced)

    def test_bad_hex_exits_2_with_error_and_no_traceback(self, astronaut_png, tmp_path):
        out_path = tmp_path / "out.png"
        result = run_cli(
            "apply", str(astronaut_png), str(out_path), "--lipstick", "#ZZZZZZ"
        )

        assert result.returncode == 2
        assert "error:" in result.stderr
        assert "#ZZZZZZ" in result.stderr
        assert "Traceback" not in result.stderr
        assert not out_path.exists()

    def test_missing_input_exits_2(self, tmp_path):
        missing = tmp_path / "does-not-exist.png"
        out_path = tmp_path / "out.png"
        result = run_cli("apply", str(missing), str(out_path), "--preset", "bare")

        assert result.returncode == 2
        assert "error:" in result.stderr
        assert "Traceback" not in result.stderr

    def test_invalid_look_json_exits_2(self, astronaut_png, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json")
        out_path = tmp_path / "out.png"

        result = run_cli(
            "apply", str(astronaut_png), str(out_path), "--look-json", str(bad_json)
        )

        assert result.returncode == 2
        assert "error:" in result.stderr
        assert "Traceback" not in result.stderr

    def test_color_flag_without_intensity_defaults_to_visible(self, astronaut_png, tmp_path):
        # Default Look() has lipstick intensity 0, so applying --lipstick alone
        # (no --lipstick-intensity) must still visibly paint it per the
        # documented overlay rule.
        out_path = tmp_path / "out.png"
        result = run_cli(
            "apply", str(astronaut_png), str(out_path), "--lipstick", "#B03A5B"
        )

        assert result.returncode == 0, result.stderr
        original = cv2.imread(str(astronaut_png))
        produced = cv2.imread(str(out_path))
        assert not np.array_equal(original, produced)

    def test_multiple_errors_are_all_listed(self, astronaut_png, tmp_path):
        out_path = tmp_path / "out.png"
        result = run_cli(
            "apply",
            str(astronaut_png),
            str(out_path),
            "--lipstick",
            "#ZZZZZZ",
            "--eyeshadow",
            "not-a-color",
        )

        assert result.returncode == 2
        assert "#ZZZZZZ" in result.stderr
        assert "not-a-color" in result.stderr


class TestLandmarks:
    def test_writes_output_image(self, astronaut_png, tmp_path):
        out_path = tmp_path / "out.png"
        result = run_cli("landmarks", str(astronaut_png), str(out_path))

        assert result.returncode == 0, result.stderr
        assert out_path.exists()


class TestLooks:
    def test_table_lists_all_preset_names(self):
        result = run_cli("looks")

        assert result.returncode == 0, result.stderr
        for name in PRESETS:
            assert name in result.stdout

    def test_json_round_trips_through_look_from_dict(self):
        result = run_cli("looks", "--json")

        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert len(data) == 4
        assert set(data.keys()) == set(PRESETS.keys())

        for name, look_dict in data.items():
            look = Look.from_dict(look_dict)
            assert look.to_dict() == look_dict


@pytest.mark.slow
class TestVideo:
    def test_video_subcommand_processes_all_frames(self, astronaut_bgr, tmp_path):
        height, width = astronaut_bgr.shape[:2]
        video_path = tmp_path / "in.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
        for _ in range(10):
            writer.write(astronaut_bgr)
        writer.release()

        out_path = tmp_path / "out.mp4"
        result = run_cli("video", str(video_path), str(out_path), "--preset", "bare")

        assert result.returncode == 0, result.stderr
        assert out_path.exists()

        cap = cv2.VideoCapture(str(out_path))
        assert cap.isOpened()
        frame_count = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            frame_count += 1
        cap.release()

        assert frame_count >= 1
