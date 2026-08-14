/**
 * One-Euro filter over a flat landmark array -- a direct port of
 * `carmine/filters.py`.
 *
 * Raw per-frame landmarks jitter even on a perfectly still face. A fixed
 * low-pass either lags visibly during head motion or leaves that jitter
 * mostly intact; the One-Euro filter (Casiez, Pavia & Roussel, 2012) raises
 * its own cutoff frequency in proportion to the signal's speed, so it damps
 * hard when the face is still and gets out of the way when it moves.
 *
 * Two details are load-bearing and easy to get subtly wrong, so they are
 * called out where they happen below: the derivative is measured against
 * the previous *raw* sample, while the signal low-pass chases the previous
 * *filtered* output. Swapping either one produces a filter that still looks
 * plausible frame to frame but is materially weaker.
 *
 * State is per-element: each coordinate carries its own derivative estimate
 * and therefore its own adaptive cutoff, exactly as the NumPy version's
 * elementwise arithmetic does.
 */

import constants from "../gen/constants.json";

export interface OneEuroParams {
  /** Sampling frequency in Hz. Only seeds the first estimate; every later
   * call derives its frequency from the actual timestamp delta. */
  freq: number;
  /** Baseline cutoff. Lower damps a still signal harder. */
  minCutoff: number;
  /** Speed coefficient. Higher lets the cutoff rise faster (less lag). */
  beta: number;
  /** Cutoff for the derivative estimate's own low-pass. */
  dCutoff: number;
}

/** Defaults read from the generated constants, which in turn read them from
 * `OneEuroFilter.__init__`'s live signature -- so they cannot drift. */
export const ONE_EURO_DEFAULTS: OneEuroParams = {
  freq: constants.one_euro.freq,
  minCutoff: constants.one_euro.min_cutoff,
  beta: constants.one_euro.beta,
  dCutoff: constants.one_euro.d_cutoff,
};

function smoothingFactor(freq: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + freq * tau);
}

export class OneEuroFilter {
  readonly freq: number;
  readonly minCutoff: number;
  readonly beta: number;
  readonly dCutoff: number;

  private tPrev: number | null = null;
  private xPrev: Float64Array | null = null;
  private dxPrev: Float64Array | null = null;
  private outPrev: Float64Array | null = null;

  constructor(params: Partial<OneEuroParams> = {}) {
    const merged = { ...ONE_EURO_DEFAULTS, ...params };
    this.freq = merged.freq;
    this.minCutoff = merged.minCutoff;
    this.beta = merged.beta;
    this.dCutoff = merged.dCutoff;
  }

  /** Discard all state so the next call re-seeds from scratch. Called when
   * face tracking drops out, so a later good frame is not smoothed against
   * landmarks from before the gap. */
  reset(): void {
    this.tPrev = null;
    this.xPrev = null;
    this.dxPrev = null;
    this.outPrev = null;
  }

  /**
   * Filter one sample.
   *
   * @param x Flat coordinate array; must keep the same length across calls.
   * @param t Sample timestamp in **seconds**.
   * @returns A newly allocated filtered array. The first call after
   *   construction or `reset()` passes `x` through untouched while seeding
   *   internal state.
   */
  filter(x: ArrayLike<number>, t: number): Float64Array {
    const current = Float64Array.from(x);

    if (this.tPrev === null) {
      this.tPrev = t;
      this.xPrev = current.slice();
      this.dxPrev = new Float64Array(current.length);
      this.outPrev = current.slice();
      return current.slice();
    }

    const dt = t - this.tPrev;
    if (dt <= 0) {
      // Non-increasing timestamp: there is nothing sane to compute from a
      // zero or negative dt, so reuse the last output rather than divide
      // by it. State is deliberately left untouched.
      return this.outPrev!.slice();
    }

    const freq = 1 / dt;
    const dAlpha = smoothingFactor(freq, this.dCutoff);
    const xPrev = this.xPrev!;
    const dxPrev = this.dxPrev!;
    const outPrev = this.outPrev!;

    const edx = new Float64Array(current.length);
    const out = new Float64Array(current.length);
    for (let i = 0; i < current.length; i++) {
      // Derivative against the previous RAW sample, not the previous output.
      const dx = (current[i] - xPrev[i]) / dt;
      const e = dAlpha * dx + (1 - dAlpha) * dxPrev[i];
      edx[i] = e;

      const cutoff = this.minCutoff + this.beta * Math.abs(e);
      const alpha = smoothingFactor(freq, cutoff);
      // Signal low-pass against the previous FILTERED output.
      out[i] = alpha * current[i] + (1 - alpha) * outPrev[i];
    }

    this.tPrev = t;
    this.xPrev = current;
    this.dxPrev = edx;
    this.outPrev = out;

    return out.slice();
  }
}
