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

  return {
    detect(source: TexImageSource, timestampMs: number): Float32Array | null {
      const width = "videoWidth" in source ? source.videoWidth : (source as HTMLCanvasElement).width;
      const height = "videoHeight" in source ? source.videoHeight : (source as HTMLCanvasElement).height;

      const result = faceLandmarker.detectForVideo(source, timestampMs);
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
