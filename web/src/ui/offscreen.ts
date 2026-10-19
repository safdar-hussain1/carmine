/**
 * One off-screen engine for every still picture the page shows.
 *
 * The look thumbnails, the lookbook and the how-it-works story all paint the
 * sample in some look. Each WebGL2 context costs memory and browsers cap how
 * many a page may hold, so they share this one renderer and copy each result
 * out before the next draw.
 */

import { createRenderer, type Renderer } from "../engine/renderer";
import type { LookConfig } from "../engine/look";
import { applyLookCpu, masksFor, toFloatPixels, toImageData, toProcessingCanvas } from "./pipeline";
import { glossFor, type PhotoScene } from "./scene";

let renderer: Renderer | null = null;
let tried = false;
const surface = document.createElement("canvas");

function engine(): Renderer | null {
  if (!tried) {
    tried = true;
    renderer = createRenderer(surface);
  }
  return renderer;
}

function canvasOfSize(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

/**
 * The scene painted in `look`, at the scene's full resolution. Without
 * WebGL2 the CPU reference is used at processing size and scaled up.
 */
export function renderLook(scene: PhotoScene, look: LookConfig): HTMLCanvasElement {
  const active = engine();
  const out = canvasOfSize(scene.width, scene.height);
  const ctx = out.getContext("2d");
  if (!ctx) {
    return out;
  }
  if (active) {
    active.resize(scene.width, scene.height);
    active.render(scene.source, scene.masks, look, glossFor(scene, look), {});
    // Copied straight away: without preserveDrawingBuffer the drawing
    // buffer is only kept until the browser next composites the page.
    ctx.drawImage(surface, 0, 0);
    return out;
  }
  const proc = toProcessingCanvas(scene.source, scene.width, scene.height);
  const procCtx = proc.getContext("2d", { willReadFrequently: true });
  if (!procCtx) {
    ctx.drawImage(scene.source, 0, 0);
    return out;
  }
  const data = procCtx.getImageData(0, 0, proc.width, proc.height);
  const painted = applyLookCpu(
    toFloatPixels(data),
    masksFor(scene.landmarks, scene.width, scene.height, look),
    look,
  );
  procCtx.putImageData(toImageData(painted, proc.width, proc.height), 0, 0);
  ctx.drawImage(proc, 0, 0, scene.width, scene.height);
  return out;
}

export interface Crop {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A crop of `aspect` (width over height) centred on (cx, cy), `w` wide,
 * kept inside a picture of the given size. */
export function cropAround(
  cx: number,
  cy: number,
  w: number,
  aspect: number,
  width: number,
  height: number,
): Crop {
  let cw = Math.min(w, width, height * aspect);
  let ch = cw / aspect;
  if (ch > height) {
    ch = height;
    cw = ch * aspect;
  }
  return {
    x: Math.min(Math.max(cx - cw / 2, 0), width - cw),
    y: Math.min(Math.max(cy - ch / 2, 0), height - ch),
    w: cw,
    h: ch,
  };
}

/** The face, framed for a portrait of the given aspect. */
export function faceCrop(scene: PhotoScene, aspect: number, margin = 1.32): Crop {
  const box = scene.face;
  return cropAround(
    box.cx,
    box.cy,
    Math.max(box.width * margin, box.height * margin * aspect),
    aspect,
    scene.width,
    scene.height,
  );
}

/** Draws `area` of `source` into a new canvas of the given size. */
export function cropTo(
  source: CanvasImageSource,
  area: Crop,
  width: number,
  height: number,
): HTMLCanvasElement {
  const out = canvasOfSize(width, height);
  const ctx = out.getContext("2d");
  if (ctx) {
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(source, area.x, area.y, area.w, area.h, 0, 0, out.width, out.height);
  }
  return out;
}
