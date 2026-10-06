/**
 * A still picture, analysed once.
 *
 * A photo does not move, so everything the renderer needs apart from the
 * look can be worked out a single time: the landmarks, a mask for every
 * product, and the gloss percentiles. After that, changing a shade costs one
 * draw instead of a detection, six mask rasters and a percentile sort, which
 * is what lets the mirror blend smoothly from one look to the next on a
 * photo, and what lets the how-it-works story draw each step without
 * detecting the face again.
 *
 * Masks are the live construction, the same ones the mirror paints photos
 * with whenever WebGL2 is available. Gloss percentiles are measured on the
 * untinted picture, which is how the live shader path measures them too.
 */

import { PRODUCT_ORDER, type LookConfig } from "../engine/look";
import { buildMasks, PROC_MAX_SIDE_LIVE, type MaskSet } from "../engine/masks";
import { glossPercentiles, type GlossPercentiles } from "../engine/pigment";
import type { GlossInputs } from "../engine/renderer";
import { sharedLandmarker, toFloatPixels, toProcessingCanvas } from "./pipeline";

/** Something both the landmarker and the renderer can read pixels from. */
export type StillSource = HTMLImageElement | HTMLCanvasElement;

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  /** Centre of the box, which is where the mirror's crop is aimed. */
  cx: number;
  cy: number;
}

export interface PhotoScene {
  source: StillSource;
  width: number;
  height: number;
  /** 478 x/y pairs in the source's own pixels. */
  landmarks: Float32Array;
  /** Every product's mask, built once. */
  masks: MaskSet;
  gloss: { highlighter: GlossPercentiles | null; lipstick: GlossPercentiles | null };
  face: FaceBox;
}

const EVERY_PRODUCT = new Set<string>(PRODUCT_ORDER);

/** The bounding box of every landmark. */
export function faceBox(landmarks: ArrayLike<number>): FaceBox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i + 1 < landmarks.length; i += 2) {
    const x = landmarks[i];
    const y = landmarks[i + 1];
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
  };
}

/**
 * Detects the face in a still and builds everything a render needs.
 * Resolves to null when there is no face to paint.
 */
export async function analyzePhoto(
  source: StillSource,
  width: number,
  height: number,
): Promise<PhotoScene | null> {
  const landmarker = await sharedLandmarker();
  let landmarks = landmarker.detect(source, performance.now());
  // In video mode the landmarker follows the face from its previous frame.
  // On a different picture that track is lost, and the first call comes
  // back empty while it falls back to a fresh detection; asking again is
  // exactly what the next frame of a video would do.
  for (let attempt = 0; landmarks === null && attempt < 2; attempt++) {
    landmarks = landmarker.detect(source, performance.now());
  }
  if (landmarks === null) {
    return null;
  }
  const masks = buildMasks(landmarks, width, height, EVERY_PRODUCT, "live");

  // Sized to the mask set: the percentile reduction walks the mask and the
  // pixels with one index, so the two must share a resolution.
  const proc = toProcessingCanvas(source, width, height, undefined, PROC_MAX_SIDE_LIVE);
  const ctx = proc.getContext("2d", { willReadFrequently: true });
  let gloss: PhotoScene["gloss"] = { highlighter: null, lipstick: null };
  if (ctx) {
    const pixels = toFloatPixels(ctx.getImageData(0, 0, proc.width, proc.height));
    gloss = {
      highlighter: masks.masks.highlighter ? glossPercentiles(pixels, masks.masks.highlighter) : null,
      lipstick: masks.masks.lipstick ? glossPercentiles(pixels, masks.masks.lipstick) : null,
    };
  }

  return { source, width, height, landmarks, masks, gloss, face: faceBox(landmarks) };
}

/** The gloss inputs a look actually uses on this scene. */
export function glossFor(scene: PhotoScene, look: LookConfig): GlossInputs {
  return {
    highlighter: look.highlighter.intensity > 0 ? scene.gloss.highlighter : null,
    lipstick:
      look.lipstick.intensity > 0 && look.lipstick.finish === "gloss" ? scene.gloss.lipstick : null,
  };
}
