import { describe, expect, it } from "vitest";

import { ONE_EURO_DEFAULTS, OneEuroFilter } from "./oneEuro";
import vectors from "../gen/test_vectors.json";

/**
 * The trace comparison is the real test here.
 *
 * A One-Euro implementation with the derivative taken against the previous
 * *output* instead of the previous raw sample still smooths, still looks
 * plausible on video, and still passes every structural property you might
 * think to assert (output between input and previous output, converges on a
 * constant signal, and so on). Only a step-for-step comparison against the
 * Python filter's own numbers catches it -- hence 1e-6, which is float64
 * agreement rather than "roughly the same filter".
 */
const TRACE_TOLERANCE = 1e-6;

describe("OneEuroFilter", () => {
  it("uses the Python constructor's defaults", () => {
    const params = vectors.one_euro.params;
    expect(ONE_EURO_DEFAULTS.freq).toBe(params.freq);
    expect(ONE_EURO_DEFAULTS.minCutoff).toBe(params.min_cutoff);
    expect(ONE_EURO_DEFAULTS.beta).toBe(params.beta);
    expect(ONE_EURO_DEFAULTS.dCutoff).toBe(params.d_cutoff);
  });

  it("reproduces the Python filter's trace step for step", () => {
    const filter = new OneEuroFilter();
    expect(vectors.one_euro.trace.length).toBe(30);
    for (const [step, sample] of vectors.one_euro.trace.entries()) {
      const out = filter.filter(sample.x, sample.t);
      expect(out.length).toBe(sample.y.length);
      for (let i = 0; i < out.length; i++) {
        expect(
          Math.abs(out[i] - sample.y[i]),
          `step ${step} element ${i}: ${out[i]} vs ${sample.y[i]}`,
        ).toBeLessThanOrEqual(TRACE_TOLERANCE);
      }
    }
  });

  it("passes the first sample through untouched", () => {
    const filter = new OneEuroFilter();
    const first = vectors.one_euro.trace[0];
    expect(Array.from(filter.filter(first.x, first.t))).toEqual(first.x);
  });

  it("reuses the last output for a non-increasing timestamp", () => {
    const filter = new OneEuroFilter();
    for (const sample of vectors.one_euro.trace) {
      filter.filter(sample.x, sample.t);
    }
    const stale = vectors.one_euro.non_increasing_timestamp;
    const out = filter.filter(stale.x, stale.t);
    for (let i = 0; i < out.length; i++) {
      expect(Math.abs(out[i] - stale.y[i])).toBeLessThanOrEqual(TRACE_TOLERANCE);
    }
  });

  it("does not divide by a negative dt either", () => {
    const filter = new OneEuroFilter();
    const seed = [10, 20, 30];
    filter.filter(seed, 1.0);
    const out = filter.filter([1000, 2000, 3000], 0.5);
    expect(Array.from(out)).toEqual(seed);
    for (const value of out) {
      expect(Number.isFinite(value)).toBe(true);
    }
  });

  it("re-seeds from scratch after reset", () => {
    const filter = new OneEuroFilter();
    filter.filter([0, 0], 0);
    filter.filter([100, 100], 1 / 30);
    filter.reset();
    const after = filter.filter([50, 50], 2 / 30);
    expect(Array.from(after)).toEqual([50, 50]);
  });

  it("damps a jittery still signal far more than a moving one", () => {
    const still = new OneEuroFilter();
    const moving = new OneEuroFilter();
    let stillError = 0;
    let movingLag = 0;
    for (let i = 0; i < 60; i++) {
      const t = i / 30;
      const jitter = i % 2 === 0 ? 1 : -1;
      const stillOut = still.filter([100 + jitter], t);
      const target = 100 + i * 20;
      const movingOut = moving.filter([target], t);
      if (i > 10) {
        stillError += Math.abs(stillOut[0] - 100);
        movingLag += Math.abs(movingOut[0] - target);
      }
    }
    // Jitter on a still signal is almost entirely removed...
    expect(stillError / 49).toBeLessThan(0.35);
    // ...while a fast ramp is tracked with lag, not flattened.
    expect(movingLag / 49).toBeGreaterThan(1);
  });
});
