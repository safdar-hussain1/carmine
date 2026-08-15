/**
 * App bootstrap and the `?selftest=1` check registry.
 *
 * The checks are the build's acceptance test: `scripts/verify_site.py` loads
 * the built site headlessly with that flag and waits for the aggregate
 * result in `document.title`. They deliberately exercise the parts that only
 * a real browser can exercise -- a driver actually compiling the fragment
 * shader, the wasm landmarker actually finding a face -- since those are the
 * failures the Node test suite structurally cannot catch.
 *
 * One hard rule: nothing here may call `getUserMedia`. A permission prompt in
 * a headless run would hang the verification, so every check that needs a
 * frame uses the bundled demo portrait instead of a camera.
 */

import "./styles.css";

import { createRenderer } from "./engine/renderer";
import { PRESETS, PRODUCT_ORDER } from "./engine/look";
import type { MaskSet } from "./engine/masks";
import {
  loadFixtures,
  measureCpu,
  measureEndToEnd,
  measureGpu,
  worstOf,
  type FixtureSet,
  type ParityResults,
} from "./lib/parity";
import { registerCheck, runSelftest, skip } from "./lib/selftest";
import { measureTiming } from "./lib/timing";
import { mountApp } from "./ui/app";
import { DEMO_PORTRAIT_URL, loadImageElement, masksFor, sharedLandmarker } from "./ui/pipeline";
import { PRODUCTS } from "./ui/shades";

window.__carmine_results = {};

const HEX = /^#[0-9a-f]{6}$/i;

function canvasOfSize(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  return ctx;
}

registerCheck("constants-loaded", async () => {
  const constants = await import("./gen/constants.json");
  const regions = constants.default.regions as Record<string, unknown>;
  const lipsOuter = regions.LIPS_OUTER;
  if (!Array.isArray(lipsOuter) || lipsOuter.length !== 20) {
    throw new Error(`expected LIPS_OUTER to have length 20, got ${JSON.stringify(lipsOuter)}`);
  }

  const presets = constants.default.presets as Record<string, unknown>;
  if (!presets || Object.keys(presets).length === 0) {
    throw new Error("expected constants.presets to be a non-empty object");
  }
});

registerCheck("landmarker-init", async () => {
  await sharedLandmarker();
});

/**
 * The first time any driver compiles the live shader.
 *
 * The Node test suite can check the shader's *source*, but only a GL driver
 * can reject it, so this check is the one that turns a typo in the fragment
 * program into a red build instead of a blank mirror.
 */
registerCheck("renderer-compiles", async () => {
  const canvas = canvasOfSize(64, 64);
  const renderer = createRenderer(canvas);
  if (!renderer) {
    throw new Error("createRenderer returned null on a scratch canvas");
  }

  const source = canvasOfSize(64, 64);
  const ctx = context2d(source);
  // A checkerboard with a colour block: flat grey would hide a shader that
  // ignores its input entirely.
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? "#2b2b2b" : "#d8c4bb";
      ctx.fillRect(x * 8, y * 8, 8, 8);
    }
  }
  ctx.fillStyle = "#7a5a52";
  ctx.fillRect(16, 16, 32, 32);

  const masks: MaskSet = {
    width: 8,
    height: 8,
    scale: 1 / 8,
    interocular: 4,
    masks: { lipstick: new Float32Array(64).fill(1) },
  };

  try {
    // The velvet preset's matte lipstick exercises the mip path too.
    renderer.render(source, masks, PRESETS.velvet, {}, { mirror: true, splitX: 32 });
    const gl = canvas.getContext("webgl2");
    const error = gl?.getError() ?? 0;
    if (error !== 0) {
      throw new Error(`GL error 0x${error.toString(16)} after one frame`);
    }
  } finally {
    renderer.dispose();
  }
});

registerCheck("presets-valid", async () => {
  const names = Object.keys(PRESETS);
  if (names.length !== 4) {
    throw new Error(`expected 4 presets, got ${names.length}: ${names.join(", ")}`);
  }
  for (const [name, look] of Object.entries(PRESETS)) {
    for (const product of PRODUCT_ORDER) {
      const config = look[product];
      if (!HEX.test(config.color)) {
        throw new Error(`${name}.${product} has a bad colour: ${config.color}`);
      }
      if (!(config.intensity >= 0 && config.intensity <= 1)) {
        throw new Error(`${name}.${product} has intensity ${config.intensity}`);
      }
    }
  }
});

registerCheck("ui-mounts", async () => {
  const products = document.querySelectorAll(".product");
  if (products.length !== PRODUCTS.length) {
    throw new Error(`expected ${PRODUCTS.length} products in the rail, found ${products.length}`);
  }
  const presets = document.querySelectorAll(".preset");
  if (presets.length !== 4) {
    throw new Error(`expected 4 preset chips, found ${presets.length}`);
  }
  if (!document.querySelector(".stage canvas")) {
    throw new Error("the mirror stage has no canvas");
  }
});

/**
 * The whole pipeline over a frame the build controls: detect, mask, render,
 * and then assert that the render changed the face and left the corners
 * alone. Both halves matter -- a shader that painted nothing and a shader
 * that tinted the entire frame would each pass only one of them.
 */
registerCheck("pipeline-canned-frame", async () => {
  const image = await loadImageElement(DEMO_PORTRAIT_URL);
  const width = image.naturalWidth;
  const height = image.naturalHeight;

  const reference = canvasOfSize(width, height);
  const referenceCtx = context2d(reference);
  referenceCtx.drawImage(image, 0, 0);
  const before = referenceCtx.getImageData(0, 0, width, height);

  const landmarker = await sharedLandmarker();
  const landmarks = landmarker.detect(reference, performance.now());
  if (landmarks === null) {
    throw new Error("no face detected in the bundled demo portrait");
  }

  const look = PRESETS.velvet;
  const masks = masksFor(landmarks, width, height, look);
  const target = canvasOfSize(width, height);
  const renderer = createRenderer(target);
  if (!renderer) {
    throw new Error("createRenderer returned null for the canned frame");
  }

  try {
    renderer.resize(width, height);
    renderer.render(image, masks, look, {});

    const readback = canvasOfSize(width, height);
    const readbackCtx = context2d(readback);
    readbackCtx.drawImage(target, 0, 0);
    const after = readbackCtx.getImageData(0, 0, width, height);

    let changed = 0;
    for (let i = 0; i < before.data.length; i += 4) {
      if (
        Math.abs(before.data[i] - after.data[i]) > 2 ||
        Math.abs(before.data[i + 1] - after.data[i + 1]) > 2 ||
        Math.abs(before.data[i + 2] - after.data[i + 2]) > 2
      ) {
        changed++;
      }
    }
    const fraction = changed / (width * height);
    if (fraction < 0.01) {
      throw new Error(`only ${(fraction * 100).toFixed(3)}% of pixels changed, expected at least 1%`);
    }

    const corners: Array<[number, number]> = [
      [2, 2],
      [width - 3, 2],
      [2, height - 3],
      [width - 3, height - 3],
    ];
    for (const [x, y] of corners) {
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        const delta = Math.abs(before.data[offset + channel] - after.data[offset + channel]);
        if (delta > 2) {
          throw new Error(`background corner (${x}, ${y}) moved by ${delta}`);
        }
      }
    }
  } finally {
    renderer.dispose();
  }
});

// --- parity and performance -----------------------------------------------

/**
 * Thresholds the CPU parity check enforces, and what each is for.
 *
 * The permitted sources of difference on this path are narrow: the two Lab
 * implementations (OpenCV interpolates its gamma and cube-root through
 * lookup tables, color.ts evaluates them directly, worth up to ~0.43 Lab
 * units) and single-level quantization on each side's byte conversion.
 * Anything larger means something structural diverged.
 *
 * `mean` and `p99` are the gates that mean something. The worst measured
 * values are 0.75 and 2.8, so both hold with real margin, and a mask
 * rasterized differently or a finish applied in the wrong order would blow
 * through them immediately.
 *
 * `max` is deliberately the loosest, because a single-pixel maximum over
 * ~390k pixels is a measure of the worst rasterization tie, not of whether
 * the engines agree. The measured worst is 11.4, and it is understood: 23
 * pixels out of 388,800 exceed ΔE 6, every one of them on the boundary of
 * the eyeliner stroke, where Python's feathered mask reads ~0.67 and this
 * engine's ~0.32. Eyeliner is two pixels wide under a sigma-1 feather -- the
 * one place in the engine where a one-pixel rasterization disagreement has
 * nothing to hide behind. Two real bugs were found and fixed chasing this
 * (see `strokePolyline`), which took the worst case from 23.1 to 11.4; the
 * remainder is OpenCV's fixed-point convex-polygon scanline, which this
 * engine does not reimplement. The limit is therefore set just above the
 * measured value rather than at a round number that would silently pass a
 * regression.
 */
const CPU_MEAN_DELTA_E_LIMIT = 2.0;
const CPU_P99_DELTA_E_LIMIT = 5.0;
const CPU_MAX_DELTA_E_LIMIT = 12.0;

/** The fixture set, loaded once and shared by the parity checks. */
let fixturePromise: Promise<FixtureSet | null> | null = null;

function parityFixtures(): Promise<FixtureSet | null> {
  if (fixturePromise === null) {
    fixturePromise = loadFixtures();
  }
  return fixturePromise;
}

function parityResults(): ParityResults {
  const existing = window.__carmine_results.parity;
  if (existing) {
    return existing;
  }
  const created: ParityResults = {};
  window.__carmine_results.parity = created;
  return created;
}

/**
 * The reference path against the Python engine, from fixed landmarks.
 *
 * This is the check that can fail. Everything it compares is deterministic
 * on both sides: same input pixels, same landmarks, same look, same product
 * order (`applyLookCpu` follows `engine.apply_look`, gloss percentiles
 * included, which are measured *after* the tint on both sides).
 */
registerCheck("parity-cpu", async () => {
  const set = await parityFixtures();
  const results = parityResults();
  if (set === null) {
    results.skipped = true;
    results.reason = "fixtures are not mounted at ./parity/";
    return skip("parity fixtures are not served");
  }
  results.skipped = false;
  results.fixtures = { frames: set.fixtures.length, looks: set.looks.size };

  const cases = measureCpu(set);
  results.cpu = cases;
  results.endToEnd = await measureEndToEnd(set);

  const worst = worstOf(cases);
  if (worst.mean >= CPU_MEAN_DELTA_E_LIMIT) {
    throw new Error(
      `worst mean ΔE ${worst.mean.toFixed(3)} exceeds ${CPU_MEAN_DELTA_E_LIMIT}`,
    );
  }
  if (worst.p99 >= CPU_P99_DELTA_E_LIMIT) {
    throw new Error(`worst p99 ΔE ${worst.p99.toFixed(3)} exceeds ${CPU_P99_DELTA_E_LIMIT}`);
  }
  if (worst.max >= CPU_MAX_DELTA_E_LIMIT) {
    throw new Error(`worst max ΔE ${worst.max.toFixed(3)} exceeds ${CPU_MAX_DELTA_E_LIMIT}`);
  }
  if (worst.outside > 0) {
    throw new Error(`${worst.outside} pixels changed outside the mask support`);
  }
});

/**
 * The live shader path against the same references. Report-only.
 *
 * No threshold is enforced, for two reasons that are both about honesty
 * rather than leniency. The shader takes documented approximations the CPU
 * reference does not (pre-tint gloss percentiles, 8-bit mask weights, a mip
 * level standing in for the matte blur), so it is expected to differ; and
 * headless verification runs on a software rasterizer, whose float behavior
 * is not the hardware this ships to. Gating on that number would be gating
 * on the wrong machine. The number is recorded and published with the
 * renderer string attached.
 */
registerCheck("parity-gpu", async () => {
  const set = await parityFixtures();
  const results = parityResults();
  if (set === null) {
    return skip("parity fixtures are not served");
  }
  const { cases, glRenderer } = measureGpu(set);
  results.gpu = cases;
  results.glRenderer = glRenderer;
});

/** Per-stage frame cost. Report-only: the machine running it is the
 * variable, so there is no threshold that would mean the same thing twice. */
registerCheck("timing", async () => {
  window.__carmine_results.timing = await measureTiming(
    DEMO_PORTRAIT_URL,
    PRESETS.velvet,
    "velvet",
  );
});

function bootstrap(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (app) {
    mountApp(app);
  }

  const params = new URLSearchParams(window.location.search);
  if (params.get("selftest") === "1") {
    void runSelftest();
  }
}

bootstrap();
