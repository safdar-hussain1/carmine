"""Tests for FaceLandmarker wrapper and pinned model management."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import urllib.error

from carmine.landmarks import (
    FaceLandmarker,
    NoFaceError,
    model_path,
    validate_image,
    _CHECKSUM,
    _verify_checksum,
)
from carmine import landmarks as landmarks_module
from carmine.regions import NUM_LANDMARKS


class TestValidateImage:
    """Tests for validate_image fail-fast checks."""

    def test_accepts_valid_image(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image(image) is None

    def test_rejects_non_ndarray(self):
        with pytest.raises(ValueError, match="ndarray"):
            validate_image([[1, 2, 3]])

    def test_rejects_wrong_ndim(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="ndim"):
            validate_image(image)

    def test_rejects_wrong_channel_count(self):
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel"):
            validate_image(image)

    def test_rejects_non_uint8_dtype(self):
        image = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            validate_image(image)

    def test_rejects_empty_image(self):
        image = np.zeros((0, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            validate_image(image)

    def test_lists_every_problem_in_one_message(self):
        # Wrong ndim AND wrong dtype at once.
        image = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError) as exc_info:
            validate_image(image)
        message = str(exc_info.value)
        assert "ndim" in message
        assert "dtype" in message


class TestChecksum:
    """Tests for model checksum verification (no network access)."""

    def test_verify_checksum_accepts_matching_file(self, tmp_path):
        target = tmp_path / "model.task"
        target.write_bytes(b"hello world")
        digest = hashlib.sha256(b"hello world").hexdigest()
        # Should not raise.
        _verify_checksum(target, digest)

    def test_verify_checksum_rejects_mismatch_and_deletes_file(self, tmp_path):
        target = tmp_path / "model.task"
        target.write_bytes(b"corrupted contents")
        wrong_digest = hashlib.sha256(b"something else").hexdigest()
        with pytest.raises(RuntimeError, match="checksum"):
            _verify_checksum(target, wrong_digest)
        assert not target.exists()

    def test_pinned_checksum_is_64_hex_chars(self):
        assert len(_CHECKSUM) == 64
        int(_CHECKSUM, 16)  # raises ValueError if not valid hex


class TestModelPath:
    """Tests for model_path env override behavior."""

    def test_env_override_used_verbatim(self, tmp_path, monkeypatch):
        override = tmp_path / "custom_model.task"
        override.write_bytes(b"not the real model")
        monkeypatch.setenv("CARMINE_MODEL", str(override))
        result = model_path()
        assert result == override

    def test_wrong_bytes_from_download_raise_and_leave_no_cache_file(
        self, tmp_path, monkeypatch
    ):
        """Verify-then-promote: a download that writes the wrong bytes must
        fail the checksum *before* anything is renamed into the cache path,
        so the cache path is never left holding a bad/unverified file."""
        fake_cache = tmp_path / "cache" / "face_landmarker.task"
        monkeypatch.delenv("CARMINE_MODEL", raising=False)
        monkeypatch.setattr(landmarks_module, "_CACHE_PATH", fake_cache)

        def fake_urlretrieve(url, filename):
            Path(filename).write_bytes(b"totally not the model")

        monkeypatch.setattr(
            landmarks_module.urllib.request, "urlretrieve", fake_urlretrieve
        )

        with pytest.raises(RuntimeError, match="checksum"):
            model_path()

        assert not fake_cache.exists()
        assert not fake_cache.with_suffix(".task.tmp").exists()

    def test_download_failure_is_wrapped_in_a_clear_runtime_error(
        self, tmp_path, monkeypatch
    ):
        """A network/OS-level failure during download must not propagate as
        a bare urllib exception -- it should become a RuntimeError naming
        the model URL, the cache path, and the CARMINE_MODEL override, so
        the CLI's `error: ...` contract can report it in one line."""
        fake_cache = tmp_path / "cache" / "face_landmarker.task"
        monkeypatch.delenv("CARMINE_MODEL", raising=False)
        monkeypatch.setattr(landmarks_module, "_CACHE_PATH", fake_cache)

        def fake_urlretrieve(url, filename):
            raise urllib.error.URLError("simulated network failure")

        monkeypatch.setattr(
            landmarks_module.urllib.request, "urlretrieve", fake_urlretrieve
        )

        with pytest.raises(RuntimeError) as exc_info:
            model_path()

        message = str(exc_info.value)
        assert landmarks_module._MODEL_URL in message
        assert str(fake_cache) in message
        assert "CARMINE_MODEL" in message
        assert not fake_cache.exists()
        assert not fake_cache.with_suffix(".task.tmp").exists()

    def test_ssl_download_failure_mentions_install_certificates(
        self, tmp_path, monkeypatch
    ):
        import ssl

        fake_cache = tmp_path / "cache" / "face_landmarker.task"
        monkeypatch.delenv("CARMINE_MODEL", raising=False)
        monkeypatch.setattr(landmarks_module, "_CACHE_PATH", fake_cache)

        def fake_urlretrieve(url, filename):
            raise ssl.SSLError("simulated certificate verify failed")

        monkeypatch.setattr(
            landmarks_module.urllib.request, "urlretrieve", fake_urlretrieve
        )

        with pytest.raises(RuntimeError, match="Install Certificates"):
            model_path()


class TestFaceLandmarkerDetect:
    """Tests for FaceLandmarker.detect on real images."""

    def test_detect_returns_478_by_2_landmarks(self):
        from skimage import data

        rgb = data.astronaut()
        bgr = rgb[:, :, ::-1].copy()

        landmarker = FaceLandmarker()
        result = landmarker.detect(bgr)

        assert result.shape == (NUM_LANDMARKS, 2)
        assert result.dtype == np.float32

    def test_detect_no_face_raises_no_face_error(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        landmarker = FaceLandmarker()
        with pytest.raises(NoFaceError):
            landmarker.detect(bgr)

    def test_detect_rejects_invalid_image(self):
        landmarker = FaceLandmarker()
        with pytest.raises(ValueError):
            landmarker.detect(np.zeros((10, 10), dtype=np.uint8))


class TestFaceLandmarkerDetectVideo:
    """Tests for FaceLandmarker.detect_video with monotonically increasing timestamps."""

    def test_detect_video_returns_478_by_2_landmarks(self):
        from skimage import data

        rgb = data.astronaut()
        bgr = rgb[:, :, ::-1].copy()

        landmarker = FaceLandmarker()
        result = landmarker.detect_video(bgr, timestamp_ms=0)

        assert result.shape == (NUM_LANDMARKS, 2)
        assert result.dtype == np.float32

    def test_detect_video_no_face_raises_no_face_error(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        landmarker = FaceLandmarker()
        with pytest.raises(NoFaceError):
            landmarker.detect_video(bgr, timestamp_ms=0)

    def test_detect_video_accepts_increasing_timestamps(self):
        from skimage import data

        rgb = data.astronaut()
        bgr = rgb[:, :, ::-1].copy()

        landmarker = FaceLandmarker()
        landmarker.detect_video(bgr, timestamp_ms=0)
        result = landmarker.detect_video(bgr, timestamp_ms=33)

        assert result.shape == (NUM_LANDMARKS, 2)
