/**
 * Per-stage frame-cost measurement for the live path.
 *
 * A "60fps" claim is worth nothing without saying which stage costs what, on
 * what hardware, at what resolution. This module runs the three stages a
 * live frame actually pays for -- landmark detection, mask construction, and
 * the GL draw -- over a fixed number of iterations at 720p and reports the
 * median of each.
 *
 * **Median, not mean.** The first few iterations pay for shader compilation,
 * texture allocation and JIT warm-up, and a garbage collection can land on
 * any single frame. A mean folds all of that into the headline number; the
 * median reports what a typical frame costs, which is what a viewer of a
 * live preview experiences. The min and max are recorded alongside so the
 * spread is visible rather than hidden.
 *
 * **The draw is fenced with `gl.finish()`.** GL calls return as soon as the
 * commands are queued, so timing `render()` alone measures how fast this
 * code can talk to the driver, not how long the frame takes. `finish()`
 * blocks until the GPU is actually done, which costs a little synchronization
 * overhead but is the only number that means anything.
 *
 * The stubbed still frame stands in for a camera: `getUserMedia` must never
 * be called here, since a permission prompt would wedge a headless run.
 */

import { activeProducts, type LookConfig } from "../engine/look";
import { buildMasks, PROC_MAX_SIDE_LIVE } from "../engine/masks";
import { createRenderer, type GlossInputs } from "../engine/renderer";
import {
  computeGloss,
  loadImageElement,
  sharedLandmarker,
  toFloatPixels,
  toProcessingCanvas,
} from "../ui/pipeline";

/** Frames run and discarded before measurement starts. */
const WARMUP_FRAMES = 10;

/** Median/min/max of one stage, in milliseconds. */
export interface StageTiming {
  median: number;
  min: number;
  max: number;
  samples: number;
}

export interface TimingResults {
  /** Drawing-buffer size every stage was measured at. */
  width: number;
  height: number;
  frames: number;
  /** Frames discarded before measurement, so the medians are of a warm
   * pipeline rather than of shader compilation and detector startup. */
  warmupFrames: number;
  look: string;
  /**
   * Interocular distance in processing-resolution pixels.
   *
   * Reported because the mask stage's cost is mostly this number, not the
   * frame's: every feather radius is a fraction of it, and a Gaussian's cost
   * grows with its kernel. A frame filled by a face costs several times what
   * the same frame with a face at conversational distance costs, so a mask
   * timing without this figure attached cannot be compared to anything.
   */
  interocularPx: number;
  glRenderer: string;
  /** True when the draw was fenced with `gl.finish()`; false would mean the
   * draw number is submission time only. */
  fenced: boolean;
  detect: StageTiming;
  /** Mask construction plus, for looks that need it, the gloss percentile
   * reduction the shader cannot do for itself -- everything between having
   * landmarks and being ready to draw. */
  buildMasks: StageTiming;
  draw: StageTiming;
  /** Sum of the three medians -- the per-frame budget, if the stages ran
   * back to back with nothing overlapped. */
  totalMedian: number;
  /**
   * The same frame through the live path's mask construction: half
   * processing resolution, box-approximated feathers. The detect stage is
   * shared with the reference numbers above (it is the same detection,
   * measured once and used by both), so `totalMedian` here includes it.
   *
   * This is the number that describes the shipped mirror. The reference
   * numbers describe what a still render costs.
   */
  livePath: {
    buildMasks: StageTiming;
    draw: StageTiming;
    totalMedian: number;
    /** Mask resolution the live path built at, for the record. */
    maskWidth: number;
    maskHeight: number;
  };
  /** Mask resolution the reference path built at. */
  maskWidth: number;
  maskHeight: number;
}

function summarize(samples: number[]): StageTiming {
  const sorted = samples.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const median =
    sorted.length === 0
      ? 0
      : sorted.length % 2 === 1
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
  return {
    median,
    min: sorted.length > 0 ? sorted[0] : 0,
    max: sorted.length > 0 ? sorted[sorted.length - 1] : 0,
    samples: sorted.length,
  };
}

/**
 * Run `frames` iterations of the live pipeline over a still source.
 *
 * @param sourceUrl Image to stand in for a camera frame.
 * @param look The look to render; its active products decide how many masks
 *   are built, which is the dominant cost of the mask stage.
 * @param lookName Recorded alongside the numbers, since they only mean
 *   something in the context of a specific look.
 */
export async function measureTiming(
  sourceUrl: string,
  look: LookConfig,
  lookName: string,
  frames = 120,
  width = 1280,
  height = 720,
): Promise<TimingResults> {
  const image = await loadImageElement(sourceUrl);
  const landmarker = await sharedLandmarker();

  const stage = document.createElement("canvas");
  stage.width = width;
  stage.height = height;
  const ctx = stage.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  ctx.fillStyle = "#101010";
  ctx.fillRect(0, 0, width, height);

  // Frame the source the way a webcam would: cover-fit, with the face
  // centered. Neither naive fit works on a portrait-orientation source --
  // letterboxing shrinks the face until the detector stops finding it at
  // all, and a centered cover crop can slice the head off. Detecting once at
  // native size to find the face, then composing around it, gives a frame
  // whose face occupies roughly what a real preview's does, which is what
  // makes the per-stage numbers comparable to a live session.
  const native = document.createElement("canvas");
  native.width = image.naturalWidth;
  native.height = image.naturalHeight;
  const nativeCtx = native.getContext("2d", { willReadFrequently: true });
  if (!nativeCtx) {
    throw new Error("2D canvas context unavailable");
  }
  nativeCtx.drawImage(image, 0, 0);
  const located = landmarker.detect(native, performance.now());
  if (located === null) {
    throw new Error("no face detected in the timing source image at native size");
  }
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < located.length; i += 2) {
    sumX += located[i];
    sumY += located[i + 1];
  }
  const faceX = sumX / (located.length / 2);
  const faceY = sumY / (located.length / 2);

  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
  const drawWidth = image.naturalWidth * scale;
  const drawHeight = image.naturalHeight * scale;
  // Center on the face, then clamp so the scaled image still covers the
  // frame rather than exposing a bar at an edge.
  const offsetX = Math.min(0, Math.max(width - drawWidth, width / 2 - faceX * scale));
  const offsetY = Math.min(0, Math.max(height - drawHeight, height / 2 - faceY * scale));
  ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
  const target = document.createElement("canvas");
  const renderer = createRenderer(target);
  if (!renderer) {
    throw new Error("createRenderer returned null for the timing loop");
  }
  const gl = target.getContext("webgl2");
  const info = gl?.getExtension("WEBGL_debug_renderer_info");
  const glRenderer = gl
    ? String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER))
    : "none";

  const detectSamples: number[] = [];
  const maskSamples: number[] = [];
  const drawSamples: number[] = [];
  const liveMaskSamples: number[] = [];
  const liveDrawSamples: number[] = [];
  const products = activeProducts(look);
  let interocularPx = 0;
  let maskWidth = 0;
  let maskHeight = 0;
  let liveMaskWidth = 0;
  let liveMaskHeight = 0;

  try {
    renderer.resize(width, height);
    const wantsGloss =
      look.highlighter.intensity > 0 ||
      (look.lipstick.intensity > 0 && look.lipstick.finish === "gloss");
    const procCanvas = document.createElement("canvas");
    const liveProcCanvas = document.createElement("canvas");
    let timestamp = performance.now();

    // Warm-up, and not only for the JIT and the shader cache. The landmarker
    // runs in VIDEO mode, where it tracks a face from the previous frame's
    // region and only falls back to a full detection sweep periodically --
    // so the first frame after the source changes (as it just did, from
    // whatever the parity checks were looking at to this composed 720p
    // frame) can legitimately come back empty. Timing a cold pipeline would
    // be wrong regardless; discarding these frames fixes both problems at
    // once.
    let warmed = false;
    for (let attempt = 0; attempt < WARMUP_FRAMES; attempt++) {
      timestamp += 33;
      if (landmarker.detect(stage, timestamp) !== null) {
        warmed = true;
      }
    }
    if (!warmed) {
      throw new Error(
        `no face detected in the composed ${width}x${height} timing frame ` +
          `after ${WARMUP_FRAMES} warm-up attempts`,
      );
    }

    for (let frame = 0; frame < frames; frame++) {
      timestamp += 33;
      const t0 = performance.now();
      const landmarks = landmarker.detect(stage, timestamp);
      const t1 = performance.now();
      if (landmarks === null) {
        throw new Error(
          `no face detected in the timing source frame (iteration ${frame} of ${frames}, ` +
            `${width}x${height})`,
        );
      }
      // --- reference path ---
      const masks = buildMasks(landmarks, width, height, products);
      interocularPx = masks.interocular;
      maskWidth = masks.width;
      maskHeight = masks.height;
      let gloss: GlossInputs = {};
      if (wantsGloss) {
        // Mirrors the live loop: only looks with a gloss finish pay for the
        // readback and the percentile sort.
        const proc = toProcessingCanvas(stage, width, height, procCanvas);
        const procCtx = proc.getContext("2d", { willReadFrequently: true });
        if (procCtx) {
          const data = procCtx.getImageData(0, 0, proc.width, proc.height);
          gloss = computeGloss(toFloatPixels(data), masks, look);
        }
      }
      const t2 = performance.now();
      renderer.render(stage, masks, look, gloss);
      gl?.finish();
      const t3 = performance.now();

      // --- live path, same frame, same landmarks ---
      const liveMasks = buildMasks(landmarks, width, height, products, "live");
      liveMaskWidth = liveMasks.width;
      liveMaskHeight = liveMasks.height;
      let liveGloss: GlossInputs = {};
      if (wantsGloss) {
        const proc = toProcessingCanvas(
          stage,
          width,
          height,
          liveProcCanvas,
          PROC_MAX_SIDE_LIVE,
        );
        const procCtx = proc.getContext("2d", { willReadFrequently: true });
        if (procCtx) {
          const data = procCtx.getImageData(0, 0, proc.width, proc.height);
          liveGloss = computeGloss(toFloatPixels(data), liveMasks, look);
        }
      }
      const t4 = performance.now();
      renderer.render(stage, liveMasks, look, liveGloss);
      gl?.finish();
      const t5 = performance.now();

      detectSamples.push(t1 - t0);
      maskSamples.push(t2 - t1);
      drawSamples.push(t3 - t2);
      liveMaskSamples.push(t4 - t3);
      liveDrawSamples.push(t5 - t4);
    }
  } finally {
    renderer.dispose();
  }

  const detect = summarize(detectSamples);
  const masksStage = summarize(maskSamples);
  const draw = summarize(drawSamples);
  const liveMasks = summarize(liveMaskSamples);
  const liveDraw = summarize(liveDrawSamples);

  return {
    width,
    height,
    frames,
    warmupFrames: WARMUP_FRAMES,
    look: lookName,
    interocularPx,
    glRenderer,
    fenced: gl !== null,
    detect,
    buildMasks: masksStage,
    draw,
    totalMedian: detect.median + masksStage.median + draw.median,
    maskWidth,
    maskHeight,
    livePath: {
      buildMasks: liveMasks,
      draw: liveDraw,
      totalMedian: detect.median + liveMasks.median + liveDraw.median,
      maskWidth: liveMaskWidth,
      maskHeight: liveMaskHeight,
    },
  };
}
