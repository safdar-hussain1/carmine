/**
 * CPU pigment ops in CIELAB -- a port of `carmine/pigment.py`.
 *
 * Painting a flat RGB color into a masked region either erases skin texture
 * or blows out brightness, so every operation here works in Lab: chroma
 * (a/b) moves toward the target color while lightness (L) keeps most of its
 * per-pixel detail, and tinted skin still reads as skin. `paint` is the
 * deliberate exception -- it collapses the region to a flat color, which is
 * what a hard-edged product like eyeliner needs.
 *
 * This module is the CPU reference path. The live camera path runs the same
 * math in a fragment shader (`renderer.ts`); this implementation is what the
 * parity tests measure against the Python engine, and what the gloss
 * percentile helper feeds the shader from.
 *
 * **Pixel format.** Images are flat RGBA `Float32Array`s at processing
 * resolution holding *byte-valued* floats (0-255 integers), which is what
 * `ImageData` gives us once widened. Outputs are clamped to [0, 255] and
 * **truncated**, not rounded, because NumPy's `.astype(np.uint8)` truncates
 * -- rounding instead would put roughly half the pixels one level away from
 * the Python result.
 */

import { labToRgb, rgbToLab, type Rgb } from "./color";
import { gaussianBlur } from "./blur";
import constants from "../gen/constants.json";

const MATTE_BLUR_SIGMA = constants.pigment.finish_matte.blur_sigma;
const GLOSS_P_LOW = constants.pigment.finish_gloss.percentile_low;
const GLOSS_P_HIGH = constants.pigment.finish_gloss.percentile_high;
const GLOSS_STRENGTH_FACTOR = constants.pigment.finish_gloss.strength_factor;

/** Minimum masked-pixel count below which gloss has nothing meaningful to
 * measure a highlight spread from (pigment.py finish_gloss). */
const GLOSS_MIN_PIXELS = 10;

/** Percentile pair describing where a mask's highlights sit on the L axis. */
export interface GlossPercentiles {
  p75: number;
  p99: number;
}

function toByte(value: number): number {
  // clamp then truncate: NumPy's np.clip(x, 0, 255).astype(np.uint8).
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.trunc(value);
}

/**
 * Copy original pixels back wherever the mask is zero.
 *
 * The Lab round trip drifts pixels by a level or two even at weight 0,
 * purely from floating-point rounding. Callers rely on untouched regions
 * staying bit-identical to the input -- a look with one product enabled
 * must not perturb the rest of the frame -- so the drift is papered over
 * explicitly here rather than chased out of the color math.
 */
function restoreUntouched(out: Float32Array, source: Float32Array, mask: Float32Array): void {
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] <= 0) {
      const p = i * 4;
      out[p] = source[p];
      out[p + 1] = source[p + 1];
      out[p + 2] = source[p + 2];
      out[p + 3] = source[p + 3];
    }
  }
}

/**
 * Pull the masked region's chroma toward `colorRgb` in Lab space.
 *
 * `intensity` scales the whole effect; `lightnessPull` caps how much of the
 * target's own lightness bleeds in, kept well under 1 so the shading and
 * highlights already in the frame survive the tint.
 */
export function tint(
  pixels: Float32Array,
  mask: Float32Array,
  colorRgb: Rgb,
  intensity: number,
  lightnessPull: number,
): Float32Array {
  const out = pixels.slice();
  if (intensity <= 0) {
    return out;
  }
  const target = rgbToLab(colorRgb);

  for (let i = 0; i < mask.length; i++) {
    const weight = mask[i] * intensity;
    if (weight === 0) {
      continue;
    }
    const p = i * 4;
    const lab = rgbToLab([pixels[p], pixels[p + 1], pixels[p + 2]]);
    lab[1] += (target[1] - lab[1]) * weight;
    lab[2] += (target[2] - lab[2]) * weight;
    lab[0] += (target[0] - lab[0]) * weight * lightnessPull;
    const rgb = labToRgb(lab);
    out[p] = toByte(rgb[0]);
    out[p + 1] = toByte(rgb[1]);
    out[p + 2] = toByte(rgb[2]);
  }

  restoreUntouched(out, pixels, mask);
  return out;
}

/**
 * Alpha-blend a flat pigment over the masked region.
 *
 * Unlike `tint` this discards the underlying texture entirely, which is the
 * right behavior for eyeliner: covering what is beneath is the point rather
 * than a side effect. No Lab round trip is involved, so no untouched-restore
 * pass is needed -- a zero weight already leaves the pixel exact.
 */
export function paint(
  pixels: Float32Array,
  mask: Float32Array,
  colorRgb: Rgb,
  intensity: number,
): Float32Array {
  const out = pixels.slice();
  if (intensity <= 0) {
    return out;
  }
  for (let i = 0; i < mask.length; i++) {
    const weight = mask[i] * intensity;
    if (weight === 0) {
      continue;
    }
    const p = i * 4;
    for (let c = 0; c < 3; c++) {
      out[p + c] = toByte(pixels[p + c] * (1 - weight) + colorRgb[c] * weight);
    }
  }
  return out;
}

/** Extract the L channel of an RGBA image as a single-channel Float32Array. */
function lightnessChannel(pixels: Float32Array, count: number): Float32Array {
  const l = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const p = i * 4;
    l[i] = rgbToLab([pixels[p], pixels[p + 1], pixels[p + 2]])[0];
  }
  return l;
}

/** Rewrite an RGBA image's pixels from a modified L channel, keeping each
 * pixel's original a/b. */
function applyLightness(
  pixels: Float32Array,
  lightness: Float32Array,
  out: Float32Array,
  changed: (index: number) => boolean,
): void {
  for (let i = 0; i < lightness.length; i++) {
    if (!changed(i)) {
      continue;
    }
    const p = i * 4;
    const lab = rgbToLab([pixels[p], pixels[p + 1], pixels[p + 2]]);
    lab[0] = lightness[i];
    const rgb = labToRgb(lab);
    out[p] = toByte(rgb[0]);
    out[p + 1] = toByte(rgb[1]);
    out[p + 2] = toByte(rgb[2]);
  }
}

/**
 * Flatten micro-highlights within the mask toward their local blur.
 *
 * Pulls each masked pixel's L toward a sigma=5 Gaussian blur of the whole L
 * channel, damping small specular highlights while leaving the broad shading
 * that reads as lip shape intact. The blur is taken over the *entire*
 * channel, not just the mask, so the pull near a mask edge references real
 * neighboring skin rather than zeros.
 */
export function finishMatte(
  pixels: Float32Array,
  width: number,
  height: number,
  mask: Float32Array,
  strength: number,
): Float32Array {
  const out = pixels.slice();
  if (strength <= 0) {
    return out;
  }
  const lightness = lightnessChannel(pixels, width * height);
  const blurred = gaussianBlur(lightness, width, height, MATTE_BLUR_SIGMA);
  const adjusted = new Float32Array(lightness.length);
  for (let i = 0; i < lightness.length; i++) {
    adjusted[i] = lightness[i] + (blurred[i] - lightness[i]) * mask[i] * strength;
  }
  applyLightness(pixels, adjusted, out, () => true);
  restoreUntouched(out, pixels, mask);
  return out;
}

/** NumPy's default (`linear`) percentile over an already-sorted array. */
function percentileSorted(sorted: Float32Array, q: number): number {
  const position = (q / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/**
 * The 75th and 99th L percentiles inside `mask`, or null when the region is
 * too small or too flat for a highlight to mean anything.
 *
 * This is split out from `finishGloss` because the WebGL path needs the same
 * two numbers as uniforms: percentiles are a whole-region reduction, which a
 * fragment shader cannot do, so they are computed once per frame on the CPU
 * from the downscaled masked image and handed to the shader.
 */
export function glossPercentiles(
  pixels: Float32Array,
  mask: Float32Array,
): GlossPercentiles | null {
  const values: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0.5) {
      const p = i * 4;
      values.push(rgbToLab([pixels[p], pixels[p + 1], pixels[p + 2]])[0]);
    }
  }
  if (values.length < GLOSS_MIN_PIXELS) {
    return null;
  }
  const sorted = Float32Array.from(values).sort();
  const p75 = percentileSorted(sorted, GLOSS_P_LOW);
  const p99 = percentileSorted(sorted, GLOSS_P_HIGH);
  if (p99 - p75 < 1e-6) {
    return null;
  }
  return { p75, p99 };
}

/**
 * Boost the brightest highlights within the mask to fake specular shine.
 *
 * Builds a highlight map from where each masked pixel's L falls between the
 * 75th and 99th percentile of L inside the mask, then adds to L in
 * proportion to that map, the mask, and `strength`.
 */
export function finishGloss(
  pixels: Float32Array,
  mask: Float32Array,
  strength: number,
  percentiles?: GlossPercentiles | null,
): Float32Array {
  const out = pixels.slice();
  if (strength <= 0) {
    return out;
  }
  const stats = percentiles === undefined ? glossPercentiles(pixels, mask) : percentiles;
  if (stats === null) {
    return out;
  }
  const spread = stats.p99 - stats.p75;
  const count = mask.length;
  const lightness = lightnessChannel(pixels, count);
  const adjusted = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const highlight = Math.min(Math.max((lightness[i] - stats.p75) / spread, 0), 1);
    const boosted = lightness[i] + highlight * mask[i] * GLOSS_STRENGTH_FACTOR * strength;
    adjusted[i] = Math.min(Math.max(boosted, 0), 100);
  }
  applyLightness(pixels, adjusted, out, () => true);
  restoreUntouched(out, pixels, mask);
  return out;
}
