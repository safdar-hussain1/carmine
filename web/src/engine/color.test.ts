import { describe, expect, it } from "vitest";

import { hexToRgb, labToRgb, rgbToLab, type Lab, type Rgb } from "./color";
import vectors from "../gen/test_vectors.json";

/**
 * Tolerances.
 *
 * These numbers are not "close enough, probably" -- each one is the measured
 * ceiling of a known, bounded disagreement between this implementation and
 * OpenCV's:
 *
 * - `LAB_TOLERANCE` (0.5 Lab units). OpenCV's float32 color conversion does
 *   not evaluate the sRGB gamma decode or the Lab cube root directly; it
 *   interpolates both from 1024-entry spline tables. Sweeping every 5th
 *   level of the 8-bit RGB cube (~140k colors) puts the worst per-channel
 *   disagreement at 0.43 units, in the b channel of saturated colors. Half a
 *   Lab unit is roughly a fifth of a just-noticeable difference, and it
 *   disappears entirely at the 1/255 quantization both sides land on.
 * - `RGB_TOLERANCE` (1.5 / 255). The round trip through the same tables,
 *   plus the quantization itself.
 */
const LAB_TOLERANCE = 0.5;
const RGB_TOLERANCE = 1.5;

describe("rgbToLab", () => {
  it("matches OpenCV's float32 Lab for every probe color", () => {
    expect(vectors.lab.forward.length).toBeGreaterThanOrEqual(12);
    for (const probe of vectors.lab.forward) {
      const actual = rgbToLab(probe.rgb as Rgb);
      for (let c = 0; c < 3; c++) {
        expect(
          Math.abs(actual[c] - probe.lab[c]),
          `channel ${c} of rgb ${probe.rgb.join(",")}: ${actual[c]} vs ${probe.lab[c]}`,
        ).toBeLessThanOrEqual(LAB_TOLERANCE);
      }
    }
  });

  it("puts white at L=100 with no chroma and black at the origin", () => {
    const white = rgbToLab([255, 255, 255]);
    expect(white[0]).toBeCloseTo(100, 3);
    expect(white[1]).toBeCloseTo(0, 3);
    expect(white[2]).toBeCloseTo(0, 3);
    expect(rgbToLab([0, 0, 0])).toEqual([0, 0, 0]);
  });
});

describe("labToRgb", () => {
  it("round-trips every probe color back to its original sRGB", () => {
    for (const probe of vectors.lab.round_trip) {
      const back = labToRgb(rgbToLab(probe.rgb as Rgb));
      for (let c = 0; c < 3; c++) {
        expect(
          Math.abs(back[c] - probe.rgb_back[c]),
          `channel ${c} of rgb ${probe.rgb.join(",")}: ${back[c]} vs ${probe.rgb_back[c]}`,
        ).toBeLessThanOrEqual(RGB_TOLERANCE);
      }
    }
  });

  it("inverts rgbToLab across the grey ramp", () => {
    for (let v = 0; v <= 255; v += 15) {
      const back = labToRgb(rgbToLab([v, v, v]));
      for (const channel of back) {
        expect(Math.abs(channel - v)).toBeLessThanOrEqual(RGB_TOLERANCE);
      }
    }
  });

  it("is the exact inverse of rgbToLab for an arbitrary Lab triple", () => {
    const lab: Lab = [64, -22, 41];
    const round = rgbToLab(labToRgb(lab));
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(round[c] - lab[c])).toBeLessThanOrEqual(0.05);
    }
  });
});

describe("hexToRgb", () => {
  it("parses with and without the leading hash, and ignores case", () => {
    expect(hexToRgb("#B03A5B")).toEqual([176, 58, 91]);
    expect(hexToRgb("b03a5b")).toEqual([176, 58, 91]);
    expect(hexToRgb("  #FFFFFF  ")).toEqual([255, 255, 255]);
  });

  it("rejects anything that is not six hex digits", () => {
    // Shorthand is rejected on the Python side too; accepting it here would
    // let the web app mint looks the CLI cannot load.
    expect(() => hexToRgb("#FFF")).toThrow(/invalid hex color/);
    expect(() => hexToRgb("#GGGGGG")).toThrow(/invalid hex color/);
    expect(() => hexToRgb("")).toThrow(/invalid hex color/);
  });
});
