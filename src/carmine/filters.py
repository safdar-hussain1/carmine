"""One-Euro filter for smoothing noisy, jittery landmark streams.

Raw per-frame landmark detections jitter even when the face is still --
small pixel-level noise in the detector's output. A naive low-pass filter
either lags behind fast motion or leaves the jitter mostly intact; the
One-Euro filter (Casiez, Pavia & Roussel, 2012) adapts its cutoff frequency
to the signal's own speed, so it damps jitter hard when the landmarks are
nearly still and relaxes to follow quick motion without much lag.
"""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(freq: float, cutoff: np.ndarray | float) -> np.ndarray:
    """Exponential-smoothing alpha for a given sampling freq and cutoff(s)."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + freq * tau)


def _lowpass(alpha: np.ndarray, value: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return alpha * value + (1.0 - alpha) * prev


class OneEuroFilter:
    """Adaptive low-pass filter for a stream of (478, 2) landmark arrays.

    Args:
        freq: Expected sampling frequency in Hz, used only to seed the
            first estimate -- every call after the first derives the
            instantaneous frequency from the actual timestamps.
        min_cutoff: Baseline cutoff frequency; lower values damp jitter on
            a still signal more aggressively.
        beta: Speed coefficient; higher values let the cutoff rise faster
            (less lag) as the signal moves quickly.
        d_cutoff: Cutoff frequency for the derivative's own low-pass.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        """Discard all filter state so the next call re-initializes it."""
        self._t_prev: float | None = None
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._out_prev: np.ndarray | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter one sample.

        Args:
            x: Array of shape (478, 2) (or any shape matching prior calls).
            t: Sample timestamp in seconds.

        Returns:
            The filtered array. The first call (or the first call after a
            `reset()`) returns `x` unchanged while seeding internal state.
        """
        x = np.asarray(x, dtype=np.float64)

        if self._t_prev is None:
            self._t_prev = float(t)
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._out_prev = x.copy()
            return x.copy()

        dt = float(t) - self._t_prev
        if dt <= 0:
            # Non-increasing timestamp: nothing sane to compute, reuse the
            # last output rather than divide by a zero/negative dt.
            return self._out_prev.copy()

        freq = 1.0 / dt

        dx = (x - self._x_prev) / dt
        d_alpha = _smoothing_factor(freq, self.d_cutoff)
        edx = _lowpass(d_alpha, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = _smoothing_factor(freq, cutoff)
        # The x lowpass chases the previous *filtered* output, not the raw
        # previous sample -- only the derivative estimate above uses raw
        # x_prev. Using raw x here would turn this into an unrelated,
        # much weaker filter.
        out = _lowpass(alpha, x, self._out_prev)

        self._t_prev = float(t)
        self._x_prev = x.copy()
        self._dx_prev = edx
        self._out_prev = out.copy()

        return out
