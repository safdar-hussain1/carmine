/**
 * The pieces the mirror and the selftest both need: one landmarker for the
 * whole page, the gloss percentiles the shader cannot compute for itself,
 * and a CPU fallback that renders a still frame without WebGL2 at all.
 */

import { createLandmarker, type Landmarker } from "../lib/landmarks";
import { buildMasks, processingSize, type MaskSet } from "../engine/masks";
import { glossPercentiles, tint, paint, finishMatte, finishGloss } from "../engine/pigment";
import { hexToRgb } from "../engine/color";
import { activeProducts, lightnessPullFor, PRODUCT_ORDER, type LookConfig } from "../engine/look";
import type { GlossInputs } from "../engine/renderer";
import constants from "../gen/constants.json";

const MATTE_STRENGTH = constants.pigment.finish_matte.default_strength;
const HIGHLIGHTER_GLOSS_FACTOR = constants.pigment.finish_gloss.highlighter_strength_factor;

export const MODEL_URL = "./models/face_landmarker.task";
export const DEMO_PORTRAIT_URL = "./demo/portrait.jpg";

let landmarkerPromise: Promise<Landmarker> | null = null;

/**
 * The page's single landmarker. Building one costs a model download and a
 * wasm instantiation, so the mirror, the photo path and the selftest all
 * share the same instance rather than paying that three times.
 */
export function sharedLandmarker(): Promise<Landmarker> {
  if (landmarkerPromise === null) {
    const promise = createLandmarker(MODEL_URL).catch((error: unknown) => {
      // Don't cache a rejected promise forever: a transient failure (a slow
      // network on first load, a WASM instantiation hiccup) would otherwise
      // wedge every future caller behind the same dead promise. Clear it so
      // the next call retries from scratch.
      if (landmarkerPromise === promise) {
        landmarkerPromise = null;
      }
      throw error;
    });
    landmarkerPromise = promise;
  }
  return landmarkerPromise;
}

/** Loads an image from a same-origin URL. */
export function loadImageElement(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${url}`));
    img.src = url;
  });
}

/** Draws any source into a 2D canvas at processing resolution. */
export function toProcessingCanvas(
  source: CanvasImageSource,
  width: number,
  height: number,
  scratch?: HTMLCanvasElement,
): HTMLCanvasElement {
  const size = processingSize(width, height);
  const canvas = scratch ?? document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  ctx.drawImage(source, 0, 0, size.width, size.height);
  return canvas;
}

/** Widens an ImageData's bytes into the byte-valued floats pigment.ts wants. */
export function toFloatPixels(data: ImageData): Float32Array {
  return Float32Array.from(data.data);
}

/**
 * The 75th/99th L percentiles the gloss finishes need, measured on the
 * processing-resolution frame. Only computed for products that actually use
 * gloss this frame -- it is a sort over the masked region, which is the most
 * expensive thing on the CPU side of a live frame.
 */
export function computeGloss(pixels: Float32Array, masks: MaskSet, look: LookConfig): GlossInputs {
  const gloss: GlossInputs = {};
  const highlighterMask = masks.masks.highlighter;
  if (look.highlighter.intensity > 0 && highlighterMask) {
    gloss.highlighter = glossPercentiles(pixels, highlighterMask);
  }
  const lipMask = masks.masks.lipstick;
  if (look.lipstick.intensity > 0 && look.lipstick.finish === "gloss" && lipMask) {
    gloss.lipstick = glossPercentiles(pixels, lipMask);
  }
  return gloss;
}

/** Builds the masks a look needs from a detected face. */
export function masksFor(
  landmarks: ArrayLike<number>,
  width: number,
  height: number,
  look: LookConfig,
): MaskSet {
  return buildMasks(landmarks, width, height, activeProducts(look));
}

/**
 * The whole look, on the CPU, at processing resolution.
 *
 * This exists for one situation: a browser without WebGL2, where the live
 * mirror cannot run but a still photo can still be made to work. It follows
 * `renderer.ts`'s product order and finish handling using the exact CPU
 * reference ops, so the result is the reference render rather than an
 * approximation of it.
 */
export function applyLookCpu(
  pixels: Float32Array,
  masks: MaskSet,
  look: LookConfig,
): Float32Array {
  let out = pixels;
  for (const name of PRODUCT_ORDER) {
    const product = look[name];
    const mask = masks.masks[name];
    if (!mask || product.intensity <= 0) {
      continue;
    }
    const rgb = hexToRgb(product.color);
    if (name === "eyeliner") {
      out = paint(out, mask, rgb, product.intensity);
      continue;
    }
    out = tint(out, mask, rgb, product.intensity, lightnessPullFor(name, product.finish));
    if (name === "highlighter") {
      out = finishGloss(out, mask, product.intensity * HIGHLIGHTER_GLOSS_FACTOR);
    } else if (name === "lipstick" && product.finish === "matte") {
      out = finishMatte(out, masks.width, masks.height, mask, MATTE_STRENGTH);
    } else if (name === "lipstick" && product.finish === "gloss") {
      out = finishGloss(out, mask, product.intensity);
    }
  }
  return out;
}

/** Narrows byte-valued floats back into an ImageData for putImageData. */
export function toImageData(pixels: Float32Array, width: number, height: number): ImageData {
  const bytes = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    bytes[i] = pixels[i];
  }
  return new ImageData(bytes, width, height);
}
