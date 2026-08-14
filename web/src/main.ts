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
import { registerCheck, runSelftest } from "./lib/selftest";
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
