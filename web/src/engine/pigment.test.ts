import { describe, expect, it } from "vitest";

import { finishGloss, finishMatte, glossPercentiles, paint, tint } from "./pigment";
import type { Rgb } from "./color";
import vectors from "../gen/test_vectors.json";

/**
 * Per-channel tolerance against the Python engine's byte output.
 *
 * 1.5/255 is not slack for a sloppy port -- it is the width of the known
 * disagreement, and it is deliberately narrower than 2 so that an actual
 * bug (a swapped channel, a missed lightness pull, a mis-signed weight)
 * cannot hide inside it. Two independent sources feed it:
 *
 * - OpenCV computes Lab through spline-interpolated lookup tables in
 *   float32 while this runs the closed-form math in float64, so the two
 *   Lab values a pixel takes differ by up to ~0.43 units before any product
 *   is applied (see color.test.ts).
 * - Both sides truncate to bytes at the end, so a sub-level difference in
 *   the float result can land on either side of an integer boundary and
 *   show up as a full level.
 */
const CHANNEL_TOLERANCE = 1.5;

type PatchSize = "4" | "16";

/** Widen a flat RGB byte list into the RGBA Float32Array the ops take. */
function toRgba(flat: readonly number[]): Float32Array {
  const count = flat.length / 3;
  const pixels = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    pixels[i * 4] = flat[i * 3];
    pixels[i * 4 + 1] = flat[i * 3 + 1];
    pixels[i * 4 + 2] = flat[i * 3 + 2];
    pixels[i * 4 + 3] = 255;
  }
  return pixels;
}

function patch(size: PatchSize): Float32Array {
  return toRgba(vectors.pigment.patches[size].rgb);
}

function mask(size: PatchSize): Float32Array {
  return Float32Array.from(vectors.pigment.masks[size]);
}

function expectMatchesRgb(actual: Float32Array, expected: readonly number[], label: string): void {
  const count = expected.length / 3;
  expect(actual.length).toBe(count * 4);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < 3; c++) {
      const got = actual[i * 4 + c];
      const want = expected[i * 3 + c];
      expect(
        Math.abs(got - want),
        `${label}: pixel ${i} channel ${c}: ${got} vs ${want}`,
      ).toBeLessThanOrEqual(CHANNEL_TOLERANCE);
    }
  }
}

describe("tint", () => {
  for (const [index, testCase] of vectors.pigment.cases.tint.entries()) {
    const size = String(testCase.size) as PatchSize;
    it(`matches Python for case ${index} (${size}x${size}, intensity ${testCase.intensity})`, () => {
      const out = tint(
        patch(size),
        mask(size),
        testCase.color as Rgb,
        testCase.intensity,
        testCase.lightness_pull,
      );
      expectMatchesRgb(out, testCase.expected_rgb, `tint case ${index}`);
    });
  }

  it("returns a bit-identical copy at zero intensity", () => {
    const pixels = patch("4");
    const out = tint(pixels, mask("4"), [176, 58, 91], 0, 0.35);
    expect(out).not.toBe(pixels);
    expect(Array.from(out)).toEqual(Array.from(pixels));
  });

  it("leaves zero-mask pixels bit-identical", () => {
    // The Lab round trip drifts pixels by a level or two even at weight 0,
    // so this is a real guarantee rather than a tautology.
    const pixels = patch("16");
    const m = mask("16");
    const out = tint(pixels, m, [245, 217, 200], 1.0, 0.35);
    let zeroPixels = 0;
    for (let i = 0; i < m.length; i++) {
      if (m[i] > 0) continue;
      zeroPixels++;
      for (let c = 0; c < 4; c++) {
        expect(out[i * 4 + c]).toBe(pixels[i * 4 + c]);
      }
    }
    expect(zeroPixels).toBeGreaterThan(0);
  });
});

describe("paint", () => {
  for (const [index, testCase] of vectors.pigment.cases.paint.entries()) {
    const size = String(testCase.size) as PatchSize;
    it(`matches Python for a ${size}x${size} patch`, () => {
      const out = paint(patch(size), mask(size), testCase.color as Rgb, testCase.intensity);
      expectMatchesRgb(out, testCase.expected_rgb, `paint case ${index}`);
    });
  }

  it("returns a bit-identical copy at zero intensity", () => {
    const pixels = patch("4");
    const out = paint(pixels, mask("4"), [27, 27, 27], 0);
    expect(Array.from(out)).toEqual(Array.from(pixels));
  });
});

describe("finishMatte", () => {
  for (const testCase of vectors.pigment.cases.finish_matte) {
    const size = String(testCase.size) as PatchSize;
    it(`matches Python for a ${size}x${size} patch`, () => {
      const out = finishMatte(
        patch(size),
        testCase.size,
        testCase.size,
        mask(size),
        testCase.strength,
      );
      expectMatchesRgb(out, testCase.expected_rgb, `finishMatte ${size}`);
    });
  }
});

describe("finishGloss", () => {
  for (const testCase of vectors.pigment.cases.finish_gloss) {
    const size = String(testCase.size) as PatchSize;

    it(`matches Python's percentiles for a ${size}x${size} patch`, () => {
      const stats = glossPercentiles(patch(size), mask(size));
      expect(stats).not.toBeNull();
      expect(Math.abs(stats!.p75 - testCase.p75)).toBeLessThanOrEqual(0.5);
      expect(Math.abs(stats!.p99 - testCase.p99)).toBeLessThanOrEqual(0.5);
    });

    it(`matches Python's output for a ${size}x${size} patch`, () => {
      const out = finishGloss(patch(size), mask(size), testCase.strength);
      expectMatchesRgb(out, testCase.expected_rgb, `finishGloss ${size}`);
    });
  }

  it("leaves the image alone when too few pixels sit inside the mask", () => {
    const pixels = patch("4");
    const sparse = new Float32Array(16);
    sparse[0] = 1;
    sparse[5] = 1;
    expect(glossPercentiles(pixels, sparse)).toBeNull();
    expect(Array.from(finishGloss(pixels, sparse, 0.6))).toEqual(Array.from(pixels));
  });

  it("leaves the image alone when the masked region has no highlight spread", () => {
    const flat = new Float32Array(4 * 4 * 4);
    for (let i = 0; i < 16; i++) {
      flat[i * 4] = 120;
      flat[i * 4 + 1] = 120;
      flat[i * 4 + 2] = 120;
      flat[i * 4 + 3] = 255;
    }
    const full = new Float32Array(16).fill(1);
    expect(glossPercentiles(flat, full)).toBeNull();
    expect(Array.from(finishGloss(flat, full, 0.6))).toEqual(Array.from(flat));
  });
});
