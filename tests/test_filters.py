"""Behavioral tests for the One-Euro landmark filter."""

import numpy as np
import pytest

from carmine.filters import OneEuroFilter


def _noisy_sine(n=120, dt=1.0 / 30.0, sigma=0.4, seed=7):
    """A noisy sine sampled at 30Hz over 4s, shaped like (n, 478, 2) landmarks.

    Every element carries the same signal so RMS comparisons stay simple,
    but the array shape matches what the filter will actually see in the
    engine. sigma is large enough that noise reduction dominates the
    default filter's own tracking lag against sin(t).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    clean = np.sin(t)
    noise = rng.normal(0, sigma, size=(n, 478, 2))
    noisy = clean[:, None, None] + noise
    return t, clean, noisy


class TestJitterReduction:
    def test_filtered_rms_error_at_least_40pct_below_raw(self):
        t, clean, noisy = _noisy_sine()
        f = OneEuroFilter(freq=30.0)
        filtered = np.stack([f(noisy[i], t[i]) for i in range(len(t))])

        clean_full = np.broadcast_to(clean[:, None, None], filtered.shape)
        raw_rms = np.sqrt(np.mean((noisy - clean_full) ** 2))
        filtered_rms = np.sqrt(np.mean((filtered - clean_full) ** 2))

        assert filtered_rms <= raw_rms * 0.60


class TestConstantInput:
    def test_converges_to_constant(self):
        f = OneEuroFilter(freq=30.0)
        constant = np.full((478, 2), 42.0)
        dt = 1.0 / 30.0
        out = None
        for i in range(60):
            out = f(constant, i * dt)
        np.testing.assert_allclose(out, constant, atol=1e-6)

    def test_first_call_returns_input_unchanged(self):
        f = OneEuroFilter()
        x = np.full((478, 2), 3.0)
        out = f(x, 0.0)
        np.testing.assert_array_equal(out, x)


class TestNonIncreasingTimestamp:
    def test_dt_le_zero_returns_previous_output(self):
        f = OneEuroFilter(freq=30.0)
        x0 = np.full((478, 2), 1.0)
        out0 = f(x0, 0.0)
        x1 = np.full((478, 2), 5.0)
        out1 = f(x1, 1.0 / 30.0)

        same_t = f(np.full((478, 2), 99.0), 1.0 / 30.0)
        np.testing.assert_array_equal(same_t, out1)

        earlier_t = f(np.full((478, 2), 100.0), 0.0)
        np.testing.assert_array_equal(earlier_t, out1)


class TestReset:
    def test_reset_reinitializes_state(self):
        f = OneEuroFilter(freq=30.0)
        f(np.full((478, 2), 1.0), 0.0)
        f(np.full((478, 2), 5.0), 1.0 / 30.0)

        f.reset()
        x = np.full((478, 2), 7.0)
        out = f(x, 10.0)
        np.testing.assert_array_equal(out, x)


class TestShapeAgnostic:
    def test_handles_478x2_array(self):
        f = OneEuroFilter()
        x = np.random.default_rng(0).normal(size=(478, 2)).astype(np.float32)
        out = f(x, 0.0)
        assert out.shape == (478, 2)
