import { describe, expect, it } from "vitest";

import {
  approximateGaussianBlur,
  boxSizesForGaussian,
  gaussianBlur,
  reflect101,
} from "./blur";

/** A mask-shaped test image: a hard-edged blob, which is what a feather
 * actually gets handed. A smooth gradient would flatter any blur. */
function blob(width: number, height: number): Float32Array {
  const mask = new Float32Array(width * height);
  for (let y = height * 0.3; y < height * 0.7; y++) {
    for (let x = width * 0.25; x < width * 0.6; x++) {
      mask[Math.floor(y) * width + Math.floor(x)] = 1;
    }
  }
  return mask;
}

describe("boxSizesForGaussian", () => {
  it("returns one odd width per pass", () => {
    for (const sigma of [0.5, 1, 3, 7.5, 20, 61]) {
      const sizes = boxSizesForGaussian(sigma);
      expect(sizes).toHaveLength(3);
      for (const size of sizes) {
        expect(size % 2).toBe(1);
        expect(size).toBeGreaterThanOrEqual(1);
      }
    }
  });

  it("approaches the target variance, and does so better as sigma grows", () => {
    // Three independent box passes add variances; a box of width w has
    // variance (w^2 - 1) / 12. Integer widths cannot hit an arbitrary target
    // exactly, so the error is real and worth pinning rather than hiding:
    // it is a few percent at the radii that dominate the mask cost, and
    // coarser for the small feathers where a percentage of very little is
    // still very little.
    const error = (sigma: number) => {
      const total = boxSizesForGaussian(sigma).reduce(
        (sum, size) => sum + (size * size - 1) / 12,
        0,
      );
      return Math.abs(total - sigma * sigma) / (sigma * sigma);
    };
    expect(error(1)).toBeLessThan(0.4);
    expect(error(2)).toBeLessThan(0.2);
    for (const sigma of [5, 12, 30, 60]) {
      expect(error(sigma)).toBeLessThan(0.08);
    }
    expect(error(30)).toBeLessThan(error(2));
  });

  it("degrades to the identity rather than to a negative width", () => {
    expect(boxSizesForGaussian(0.01).every((size) => size >= 1)).toBe(true);
  });
});

describe("approximateGaussianBlur", () => {
  it("stays close to the exact Gaussian it stands in for", () => {
    const width = 64;
    const height = 48;
    const source = blob(width, height);
    for (const sigma of [2, 5, 12]) {
      const exact = gaussianBlur(source, width, height, sigma);
      const approx = approximateGaussianBlur(source, width, height, sigma);
      let worst = 0;
      for (let i = 0; i < exact.length; i++) {
        worst = Math.max(worst, Math.abs(exact[i] - approx[i]));
      }
      // Three boxes track a Gaussian's profile to a few percent. Mask
      // weights are quantized to 8 bits (1/255 ~ 0.004) before the GPU sees
      // them, so this is a couple of quantization steps' worth of shape
      // error, on the live path only.
      expect(worst).toBeLessThan(0.04);
    }
  });

  it("moves the same total weight the exact path does", () => {
    // Not "conserves": both paths reflect at the borders, so a blur that
    // reaches an edge folds weight back in and the total grows. What matters
    // is that the approximation grows it by the same amount the reference
    // does, so swapping one for the other cannot brighten or dim a mask.
    const width = 40;
    const height = 40;
    const source = blob(width, height);
    const sum = (values: Float32Array) => values.reduce((total, v) => total + v, 0);
    const exact = gaussianBlur(source, width, height, 6);
    const approx = approximateGaussianBlur(source, width, height, 6);
    expect(sum(approx) / sum(exact)).toBeCloseTo(1, 2);
  });

  it("leaves its input alone", () => {
    const width = 16;
    const height = 16;
    const source = blob(width, height);
    const copy = source.slice();
    approximateGaussianBlur(source, width, height, 4);
    expect(Array.from(source)).toEqual(Array.from(copy));
  });

  it("keeps every weight inside [0, 1]", () => {
    const width = 32;
    const height = 32;
    const approx = approximateGaussianBlur(blob(width, height), width, height, 9);
    for (const value of approx) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it("copies rather than blurs when sigma is non-positive", () => {
    const source = blob(8, 8);
    const out = approximateGaussianBlur(source, 8, 8, 0);
    expect(out).not.toBe(source);
    expect(Array.from(out)).toEqual(Array.from(source));
  });

  it("handles a blur radius wider than the image, as the face oval's is", () => {
    const width = 6;
    const height = 6;
    const source = blob(width, height);
    const approx = approximateGaussianBlur(source, width, height, 30);
    expect(approx.every((value) => Number.isFinite(value))).toBe(true);
    // Reflected borders mean a blur this wide flattens toward the mean.
    const mean = source.reduce((s, v) => s + v, 0) / source.length;
    for (const value of approx) {
      expect(Math.abs(value - mean)).toBeLessThan(0.2);
    }
  });

  it("uses the same border rule as the exact path", () => {
    // Both paths fold indices with reflect101; a mismatch here would show up
    // as a halo along the frame edge in the live view only.
    expect(reflect101(-1, 5)).toBe(1);
    expect(reflect101(5, 5)).toBe(3);
  });
});
