/**
 * Cross-surface parity measurement: this engine against the Python one.
 *
 * `carmine` (NumPy + OpenCV) and this engine (TypeScript, plus a WebGL2
 * fragment shader for the live path) are independent implementations of the
 * same pigment math. "They agree" is only worth saying if it is measured, so
 * `scripts/export_parity_fixtures.py` renders a handful of frames on the
 * Python side and this module re-renders them here and reports the
 * difference in CIE76 ΔE.
 *
 * **The fixtures are not part of the site.** They contain dataset faces, so
 * they are git-ignored and mounted at `./parity/` only by
 * `scripts/verify_site.py --with-parity`, over localhost. When that mount is
 * absent -- which is every load of the deployed site -- the manifest fetch
 * 404s and the parity checks report themselves skipped instead of failing.
 * That is why `loadFixtures` distinguishes "not served" from "served but
 * broken": the first is the normal deployed case, the second is a bug.
 *
 * **What the numbers mean.** ΔE is reported over the union of this engine's
 * product-mask support and every pixel the Python render changed, so neither
 * side can be scored only where it chose to paint. Landmarks come from the
 * fixture, not from this page's landmarker, so the CPU number isolates
 * rendering math from the two sides' face detectors. The detector gap is
 * measured separately as `endToEnd` (re-detect on the same PNG, render
 * again, compare to the same reference) and is reported without a threshold,
 * since it is a property of two model runtimes rather than of this code.
 */

import { rgbToLab } from "../engine/color";
import type { Finish, LookConfig } from "../engine/look";
import type { MaskSet } from "../engine/masks";
import { createRenderer } from "../engine/renderer";
import {
  applyLookCpu,
  computeGloss,
  loadImageElement,
  masksFor,
  sharedLandmarker,
} from "../ui/pipeline";

/** Where `verify_site.py --with-parity` mounts `reports/parity_fixtures/`. */
const PARITY_BASE = "./parity";

interface RawProduct {
  color: string;
  intensity: number;
  finish: string;
}

interface ManifestFrame {
  name: string;
  input: string;
  landmarks: string;
  width: number;
  height: number;
  expected: Record<string, string>;
}

interface Manifest {
  version: number;
  proc_max_side: number;
  looks: Record<string, Record<string, RawProduct | number>>;
  frames: ManifestFrame[];
  sha256: Record<string, string>;
}

/** One frame-and-look measurement. */
export interface ParityCase {
  frame: string;
  look: string;
  width: number;
  height: number;
  /**
   * Pixels the comparison covers: the union of this engine's mask support
   * and every pixel the Python render actually changed.
   *
   * Taking only this engine's masks would be marking its own homework. The
   * two rasterizers disagree by a pixel here and there along mask
   * boundaries, so Python paints a thin ring this engine's support does not
   * cover -- and those are precisely the pixels where the engines differ
   * most. Scoring only where *we* painted would drop them from the mean and
   * hide them from the containment count at the same time.
   */
  comparedPixels: number;
  /**
   * Pixels inside the compared region that Python changed and this engine's
   * masks do not cover -- the boundary ring described above, reported so its
   * size is visible rather than inferred.
   */
  pythonOnlyPixels: number;
  meanDeltaE: number;
  maxDeltaE: number;
  /** ΔE below which 99% of compared pixels sit. A single rasterization
   * disagreement on a mask boundary can dominate the max while being
   * invisible; the percentile says whether that is what happened. */
  p99DeltaE: number;
  /** Pixels outside the compared region that this engine altered. Neither
   * engine touched them, so anything above zero is a containment bug rather
   * than a precision difference. */
  changedOutsideSupport: number;
}

export interface ParityResults {
  skipped?: boolean;
  reason?: string;
  /** Fixed-landmark comparison: rendering math only. */
  cpu?: ParityCase[];
  /** Same, through the WebGL2 live path. Report-only. */
  gpu?: ParityCase[];
  /** Each side running its own landmarker on the same PNG. Report-only. */
  endToEnd?: ParityCase[];
  /** The GL renderer string, so a software-rasterizer number is never
   * mistaken for a real GPU's. */
  glRenderer?: string;
  fixtures?: { frames: number; looks: number };
}

// --- fixture loading -----------------------------------------------------

/** A frame's inputs, decoded and ready to render from. */
interface Fixture {
  frame: ManifestFrame;
  image: HTMLImageElement;
  /** Flat `[x0, y0, ...]` pixel coordinates, as `buildMasks` wants them. */
  landmarks: Float32Array;
  pixels: Float32Array;
  /** Reference render per look name, as byte-valued floats. */
  expected: Map<string, Float32Array>;
}

export interface FixtureSet {
  manifest: Manifest;
  looks: Map<string, LookConfig>;
  fixtures: Fixture[];
}

function asFinish(value: string): Finish {
  return value === "matte" || value === "gloss" ? value : "satin";
}

/** Rehydrate a `Look.to_dict()` payload into this engine's `LookConfig`. */
function toLookConfig(raw: Record<string, RawProduct | number>): LookConfig {
  const product = (name: string): RawProduct => {
    const value = raw[name];
    if (typeof value !== "object" || value === null) {
      throw new Error(`fixture look is missing product ${name}`);
    }
    return value;
  };
  const config = (name: string) => {
    const p = product(name);
    return { color: p.color, intensity: p.intensity, finish: asFinish(p.finish) };
  };
  return {
    lipstick: config("lipstick"),
    eyeshadow: config("eyeshadow"),
    eyeliner: config("eyeliner"),
    brows: config("brows"),
    blush: config("blush"),
    highlighter: config("highlighter"),
    smoothing: typeof raw.smoothing === "number" ? raw.smoothing : 0,
  };
}

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  return ctx;
}

function canvasOfSize(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Decode an image element into the byte-valued RGBA floats pigment.ts uses. */
export function pixelsOf(source: CanvasImageSource, width: number, height: number): Float32Array {
  const canvas = canvasOfSize(width, height);
  const ctx = context2d(canvas);
  ctx.drawImage(source, 0, 0);
  return Float32Array.from(ctx.getImageData(0, 0, width, height).data);
}

/**
 * Load the fixture set, or null when it is not being served.
 *
 * A 404 on the manifest is the deployed site's normal state and returns
 * null. Anything else -- a manifest that parses but whose frames do not load
 * -- throws, because that is a broken verification run rather than an absent
 * one.
 */
export async function loadFixtures(): Promise<FixtureSet | null> {
  let response: Response;
  try {
    response = await fetch(`${PARITY_BASE}/manifest.json`, { cache: "no-store" });
  } catch {
    return null;
  }
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`parity manifest fetch failed with ${response.status}`);
  }
  const manifest = (await response.json()) as Manifest;

  const looks = new Map<string, LookConfig>();
  for (const [name, raw] of Object.entries(manifest.looks)) {
    looks.set(name, toLookConfig(raw));
  }

  const fixtures: Fixture[] = [];
  for (const frame of manifest.frames) {
    const image = await loadImageElement(`${PARITY_BASE}/${frame.input}`);
    const landmarkResponse = await fetch(`${PARITY_BASE}/${frame.landmarks}`, {
      cache: "no-store",
    });
    if (!landmarkResponse.ok) {
      throw new Error(`missing fixture landmarks for ${frame.name}`);
    }
    const pairs = (await landmarkResponse.json()) as number[][];
    const landmarks = new Float32Array(pairs.length * 2);
    for (let i = 0; i < pairs.length; i++) {
      landmarks[i * 2] = pairs[i][0];
      landmarks[i * 2 + 1] = pairs[i][1];
    }

    const expected = new Map<string, Float32Array>();
    for (const [lookName, file] of Object.entries(frame.expected)) {
      const reference = await loadImageElement(`${PARITY_BASE}/${file}`);
      expected.set(lookName, pixelsOf(reference, frame.width, frame.height));
    }

    fixtures.push({
      frame,
      image,
      landmarks,
      pixels: pixelsOf(image, frame.width, frame.height),
      expected,
    });
  }

  return { manifest, looks, fixtures };
}

// --- measurement ---------------------------------------------------------

/** Pixels touched by any of a look's masks. `compare` unions this with the
 * pixels the Python render moved to get the region it actually scores. */
function supportOf(masks: MaskSet): Uint8Array {
  const support = new Uint8Array(masks.width * masks.height);
  for (const mask of Object.values(masks.masks)) {
    if (!mask) {
      continue;
    }
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0) {
        support[i] = 1;
      }
    }
  }
  return support;
}

/**
 * Compare one render against its reference.
 *
 * ΔE is CIE76 -- a plain Euclidean distance in the same Lab space both
 * engines already work in. A perceptually-uniform metric (CIEDE2000) would
 * flatter the result by discounting exactly the chroma differences a makeup
 * engine exists to produce, so the blunter metric is the honest one here.
 *
 * The compared region is the union of `support` (where *this* engine's masks
 * are non-zero) and every pixel the Python render moved off the source. That
 * second half matters more than it sounds: scoring only our own support
 * would silently exclude the boundary ring where the two rasterizers
 * disagree -- the pixels most likely to be wrong -- from the mean, the
 * percentile, the max, *and* the containment count simultaneously. A bug
 * that painted nothing at all would score a perfect zero under that rule.
 */
export function compare(
  actual: Float32Array,
  expected: Float32Array,
  source: Float32Array,
  support: Uint8Array,
  meta: { frame: string; look: string; width: number; height: number },
): ParityCase {
  const deltas: number[] = [];
  let sum = 0;
  let max = 0;
  let changedOutside = 0;
  let pythonOnly = 0;

  for (let i = 0; i < support.length; i++) {
    const p = i * 4;
    const pythonChanged =
      expected[p] !== source[p] ||
      expected[p + 1] !== source[p + 1] ||
      expected[p + 2] !== source[p + 2];
    const covered = support[i] !== 0;

    if (!covered && !pythonChanged) {
      if (
        actual[p] !== source[p] ||
        actual[p + 1] !== source[p + 1] ||
        actual[p + 2] !== source[p + 2]
      ) {
        changedOutside++;
      }
      continue;
    }
    if (!covered) {
      pythonOnly++;
    }

    const a = rgbToLab([actual[p], actual[p + 1], actual[p + 2]]);
    const b = rgbToLab([expected[p], expected[p + 1], expected[p + 2]]);
    const dl = a[0] - b[0];
    const da = a[1] - b[1];
    const db = a[2] - b[2];
    const deltaE = Math.sqrt(dl * dl + da * da + db * db);
    deltas.push(deltaE);
    sum += deltaE;
    if (deltaE > max) {
      max = deltaE;
    }
  }

  deltas.sort((x, y) => x - y);
  const p99 =
    deltas.length > 0 ? deltas[Math.min(deltas.length - 1, Math.floor(deltas.length * 0.99))] : 0;

  return {
    frame: meta.frame,
    look: meta.look,
    width: meta.width,
    height: meta.height,
    comparedPixels: deltas.length,
    pythonOnlyPixels: pythonOnly,
    meanDeltaE: deltas.length > 0 ? sum / deltas.length : 0,
    maxDeltaE: max,
    p99DeltaE: p99,
    changedOutsideSupport: changedOutside,
  };
}

/** Every fixture through the CPU reference path, from the fixture landmarks. */
export function measureCpu(set: FixtureSet): ParityCase[] {
  const cases: ParityCase[] = [];
  for (const fixture of set.fixtures) {
    const { frame } = fixture;
    for (const [lookName, look] of set.looks) {
      const expected = fixture.expected.get(lookName);
      if (!expected) {
        throw new Error(`fixture ${frame.name} has no reference for look ${lookName}`);
      }
      const masks = masksFor(fixture.landmarks, frame.width, frame.height, look);
      const actual = applyLookCpu(fixture.pixels, masks, look);
      cases.push(
        compare(actual, expected, fixture.pixels, supportOf(masks), {
          frame: frame.name,
          look: lookName,
          width: frame.width,
          height: frame.height,
        }),
      );
    }
  }
  return cases;
}

/**
 * Every fixture through the WebGL2 live path.
 *
 * Two documented approximations make this path's numbers worse than the CPU
 * path's by construction, and neither is a defect: the shader takes its
 * gloss percentiles from the incoming frame rather than from the tinted
 * image (a whole-region reduction cannot be done mid-shader), and mask
 * weights reach it quantized to 8 bits. The measurement exists to size those
 * approximations, not to gate on them.
 */
export function measureGpu(set: FixtureSet): { cases: ParityCase[]; glRenderer: string } {
  const cases: ParityCase[] = [];
  let glRenderer = "unknown";

  for (const fixture of set.fixtures) {
    const { frame } = fixture;
    for (const [lookName, look] of set.looks) {
      const expected = fixture.expected.get(lookName);
      if (!expected) {
        throw new Error(`fixture ${frame.name} has no reference for look ${lookName}`);
      }
      const masks = masksFor(fixture.landmarks, frame.width, frame.height, look);
      const target = canvasOfSize(frame.width, frame.height);
      const renderer = createRenderer(target);
      if (!renderer) {
        throw new Error("createRenderer returned null for a parity fixture");
      }
      try {
        const gl = target.getContext("webgl2");
        if (gl && glRenderer === "unknown") {
          const info = gl.getExtension("WEBGL_debug_renderer_info");
          glRenderer = info
            ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL))
            : String(gl.getParameter(gl.RENDERER));
        }
        renderer.resize(frame.width, frame.height);
        renderer.render(fixture.image, masks, look, computeGloss(fixture.pixels, masks, look));
        // Read back in the same turn: the drawing buffer is not preserved,
        // so anything that yields first would read a cleared canvas.
        const actual = pixelsOf(target, frame.width, frame.height);
        cases.push(
          compare(actual, expected, fixture.pixels, supportOf(masks), {
            frame: frame.name,
            look: lookName,
            width: frame.width,
            height: frame.height,
          }),
        );
      } finally {
        renderer.dispose();
      }
    }
  }
  return { cases, glRenderer };
}

/**
 * The landmarker-inclusive number: this page detects its own landmarks on
 * the same PNG, renders through the CPU reference path, and is compared
 * against the reference that used Python's landmarks.
 *
 * The difference between this and `measureCpu` is entirely attributable to
 * the two landmarkers, which is the point of reporting both.
 */
export async function measureEndToEnd(set: FixtureSet): Promise<ParityCase[]> {
  const landmarker = await sharedLandmarker();
  const cases: ParityCase[] = [];
  let timestamp = performance.now();

  for (const fixture of set.fixtures) {
    const { frame } = fixture;
    const canvas = canvasOfSize(frame.width, frame.height);
    context2d(canvas).drawImage(fixture.image, 0, 0);
    // detectForVideo demands strictly increasing timestamps across calls.
    timestamp += 100;
    const detected = landmarker.detect(canvas, timestamp);
    if (detected === null) {
      throw new Error(`browser landmarker found no face in fixture ${frame.name}`);
    }
    for (const [lookName, look] of set.looks) {
      const expected = fixture.expected.get(lookName);
      if (!expected) {
        continue;
      }
      const masks = masksFor(detected, frame.width, frame.height, look);
      const actual = applyLookCpu(fixture.pixels, masks, look);
      // Masks come from the browser's own landmarks here; `compare` unions
      // them with what Python painted, so the region still covers both
      // renders even though the two disagree about where the face is.
      cases.push(
        compare(actual, expected, fixture.pixels, supportOf(masks), {
          frame: frame.name,
          look: lookName,
          width: frame.width,
          height: frame.height,
        }),
      );
    }
  }
  return cases;
}

/** The worst value of each statistic across a set of cases -- what a
 * threshold applies to. Taken per-statistic rather than from a single worst
 * case, so a fixture that is worst on one measure cannot mask another that
 * is worst on a different one. */
export function worstOf(cases: ParityCase[]): {
  mean: number;
  p99: number;
  max: number;
  outside: number;
} {
  let mean = 0;
  let p99 = 0;
  let max = 0;
  let outside = 0;
  for (const item of cases) {
    mean = Math.max(mean, item.meanDeltaE);
    p99 = Math.max(p99, item.p99DeltaE);
    max = Math.max(max, item.maxDeltaE);
    outside = Math.max(outside, item.changedOutsideSupport);
  }
  return { mean, p99, max, outside };
}
