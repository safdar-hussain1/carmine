/**
 * Thin wrapper around @mediapipe/tasks-vision's FaceLandmarker.
 *
 * The wasm runtime is resolved from a locally-hosted copy under
 * `/wasm` (see web/public/wasm) rather than the jsdelivr CDN the library
 * defaults to, so the built site makes zero external requests. GPU
 * inference is tried first (fast, but unsupported in some headless/CI
 * environments); construction falls back to a CPU delegate on failure so a
 * single build works everywhere.
 */

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Only the SIMD wasm variant ships (no CDN fallback) -- WASM SIMD is
// universal in browsers meeting our WebGL2 baseline, so pre-2021 browsers
// (which would otherwise fall back to the nosimd variant) are unsupported.
const WASM_BASE_PATH = "./wasm";

export interface Landmarker {
  /**
   * Detect a face in a video frame (or any drawable source) at `timestampMs`.
   * Returns a Float32Array of 478 * 2 pixel coordinates ([x0, y0, x1, y1,
   * ...]) in the source's own pixel space, or null when no face is found.
   */
  detect(source: TexImageSource, timestampMs: number): Float32Array | null;
}

async function createFaceLandmarker(
  modelUrl: string,
  delegate: "GPU" | "CPU",
): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_PATH);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: modelUrl,
      delegate,
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

/**
 * Creates a landmarker backed by the model at `modelUrl`. Tries the GPU
 * delegate first and falls back to CPU if GPU construction throws (e.g. no
 * WebGL context available, as in some headless/software-rendered browsers).
 */
export async function createLandmarker(modelUrl: string): Promise<Landmarker> {
  let faceLandmarker: FaceLandmarker;
  try {
    faceLandmarker = await createFaceLandmarker(modelUrl, "GPU");
  } catch {
    faceLandmarker = await createFaceLandmarker(modelUrl, "CPU");
  }

  // detectForVideo rejects a timestamp that does not advance, and one
  // instance is shared by every caller on the page (see pipeline.ts). Two
  // callers each deriving timestamps from their own clock -- a render loop
  // from requestAnimationFrame, a batch job counting frames of its own --
  // can hand it a value behind one it has already seen, and the detector
  // then silently returns no face. Clamping here makes that impossible for
  // any caller, while leaving the timestamp each caller reasons about (the
  // landmark smoothing filter's, for one) untouched.
  let lastTimestamp = Number.NEGATIVE_INFINITY;

  return {
    detect(source: TexImageSource, timestampMs: number): Float32Array | null {
      const width = "videoWidth" in source ? source.videoWidth : (source as HTMLCanvasElement).width;
      const height = "videoHeight" in source ? source.videoHeight : (source as HTMLCanvasElement).height;

      const monotonic = timestampMs > lastTimestamp ? timestampMs : lastTimestamp + 1;
      lastTimestamp = monotonic;
      const result = faceLandmarker.detectForVideo(source, monotonic);
      const face = result.faceLandmarks[0];
      if (!face) {
        return null;
      }

      const pixels = new Float32Array(face.length * 2);
      for (let i = 0; i < face.length; i++) {
        pixels[i * 2] = face[i].x * width;
        pixels[i * 2 + 1] = face[i].y * height;
      }
      return pixels;
    },
  };
}
