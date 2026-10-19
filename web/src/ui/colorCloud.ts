/**
 * A lip's pixels in colour space, three ways.
 *
 * Every pixel inside the lip mask is a point in CIELAB: lightness up the
 * middle, hue and colourfulness around it. Together they form a cloud whose
 * height is the lip's light and shadow. This module works out that cloud for
 * the untouched lip, for Carmine's tint, and for a flat fill of the same
 * lipstick, so the page can show what each does to it.
 *
 * Carmine's tint is the engine's own CPU reference (`pigment.tint`), with the
 * lightness pull the engine uses. The flat fill is the photo benchmark's
 * opaque-fill baseline for the lips: the colour added on top of each pixel,
 * channel by channel, clipped at white -- what `cv2.addWeighted` does in
 * `carmine.baselines`. Its signature failure is the one the benchmark
 * measures: it lifts the whole lip toward white.
 */

import { rgbToLab, type Rgb } from "../engine/color";
import { tint } from "../engine/pigment";

export interface CloudStates {
  original: Float32Array;
  carmine: Float32Array;
  flat: Float32Array;
}

export interface Cloud {
  count: number;
  /** Each state's sRGB per point, 0-255, three values per point. */
  rgb: CloudStates;
  /** Each state's Lab per point: L 0-100, then a and b. */
  lab: CloudStates;
}

/**
 * Samples the lip and computes all three states.
 *
 * @param pixels RGBA bytes as floats, the layout `pigment.ts` works on.
 * @param mask The lip mask at the same resolution.
 * @param limit Most points to keep; above it, an even spread is taken so the
 *   cloud keeps its shape.
 * @param threshold Mask weight a pixel needs to count as lip, so the soft
 *   edge (part skin) does not blur the cloud.
 */
export function buildCloud(
  pixels: Float32Array,
  mask: Float32Array,
  color: Rgb,
  intensity: number,
  lightnessPull: number,
  limit = 1600,
  threshold = 0.6,
): Cloud {
  const inside: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] >= threshold) {
      inside.push(i);
    }
  }
  const step = Math.max(1, inside.length / limit);
  const chosen: number[] = [];
  for (let k = 0; k < inside.length && chosen.length < limit; k += step) {
    chosen.push(inside[Math.floor(k)]);
  }

  const tinted = tint(pixels, mask, color, intensity, lightnessPull);
  const alpha = Math.min(1, intensity);
  const count = chosen.length;
  const rgb: CloudStates = {
    original: new Float32Array(count * 3),
    carmine: new Float32Array(count * 3),
    flat: new Float32Array(count * 3),
  };
  const lab: CloudStates = {
    original: new Float32Array(count * 3),
    carmine: new Float32Array(count * 3),
    flat: new Float32Array(count * 3),
  };

  chosen.forEach((index, n) => {
    const p = index * 4;
    for (let c = 0; c < 3; c++) {
      rgb.original[n * 3 + c] = pixels[p + c];
      rgb.carmine[n * 3 + c] = tinted[p + c];
      rgb.flat[n * 3 + c] = Math.min(255, pixels[p + c] + color[c] * alpha);
    }
    for (const state of ["original", "carmine", "flat"] as const) {
      const value = rgbToLab(rgb[state].subarray(n * 3, n * 3 + 3));
      lab[state][n * 3] = value[0];
      lab[state][n * 3 + 1] = value[1];
      lab[state][n * 3 + 2] = value[2];
    }
  });

  return { count, rgb, lab };
}

/** The mean and range of a cloud's lightness. */
export function lightnessStats(lab: Float32Array): { mean: number; min: number; max: number } {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  const count = lab.length / 3;
  for (let i = 0; i < lab.length; i += 3) {
    sum += lab[i];
    min = Math.min(min, lab[i]);
    max = Math.max(max, lab[i]);
  }
  return { mean: count > 0 ? sum / count : 0, min, max };
}
