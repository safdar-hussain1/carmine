/**
 * The look taken apart: one sheet per product, floating over the face.
 *
 * Each product's soft mask is drawn on its own transparent sheet in its own
 * shade, and the sheets are stacked above the photo in 3D, in the order the
 * shader paints them. Pulled apart, they show what the engine builds; pushed
 * back together and turned to face you, they become the finished face, which
 * is the engine's real render, faded in as the sheets meet.
 *
 * It is plain DOM with CSS 3D transforms, so the browser composites it on
 * the graphics card and moving it costs nothing per frame but a few style
 * updates.
 */

import { hexToRgb, rgbToLab } from "../engine/color";
import { PRODUCT_ORDER, type LookConfig, type ProductName } from "../engine/look";
import { cropTo, faceCrop, renderLook } from "./offscreen";
import type { PhotoScene } from "./scene";
import { PRODUCTS, shadeName } from "./shades";

export interface LayerStack {
  element: HTMLElement;
  /**
   * @param explode How far apart the sheets are, 0 (flat) to 1.
   * @param tilt How far the stack is turned away, 0 (face-on) to 1.
   * @param finished How much of the finished render shows, 0 to 1.
   */
  set(explode: number, tilt: number, finished: number): void;
  /** Draws the sheets again at the stack's current size. */
  layout(): void;
}

const LABELS: Record<ProductName, string> = {
  blush: "Blush",
  highlighter: "Highlighter",
  eyeshadow: "Eyeshadow",
  brows: "Brows",
  lipstick: "Lipstick",
  eyeliner: "Liner",
};

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

/** A shade as it should glow on a dark sheet: very dark shades (a black
 * liner, a deep brow) are lifted toward white, or they would vanish. */
function sheetColour(hex: string): [number, number, number] {
  const rgb = hexToRgb(hex);
  const lightness = rgbToLab(rgb)[0];
  if (lightness >= 34) {
    return rgb;
  }
  const lift = Math.min(0.55, (34 - lightness) / 40 + 0.25);
  return [rgb[0] + (255 - rgb[0]) * lift, rgb[1] + (255 - rgb[1]) * lift, rgb[2] + (255 - rgb[2]) * lift];
}

export function createLayerStack(scene: PhotoScene, look: LookConfig): LayerStack {
  const root = el("div", "layers");
  const stack = el("div", "layers__stack");
  // The labels live in a flat layer above the 3D stack, so a higher sheet
  // can never hide a lower sheet's label. Each follows an invisible anchor
  // inside its sheet, whose on-screen position the browser works out.
  const labels = el("div", "layers__labels");
  labels.setAttribute("aria-hidden", "true");
  root.append(stack, labels);

  const base = el("div", "layer layer--base");
  const baseCanvas = el("canvas", "layer__canvas");
  base.append(baseCanvas);
  stack.append(base);

  interface Sheet {
    name: ProductName;
    node: HTMLElement;
    canvas: HTMLCanvasElement;
    label: HTMLElement;
    anchor: HTMLElement;
  }
  const sheets: Sheet[] = [];
  PRODUCT_ORDER.forEach((name, index) => {
    const node = el("div", "layer layer--sheet");
    node.dataset.product = name;
    const canvas = el("canvas", "layer__canvas");
    const label = el("span", "layer__label");
    const meta = PRODUCTS.find((product) => product.name === name);
    const shade = shadeName(name, look[name].color) ?? look[name].color;
    label.append(
      el("span", "layer__num", String(index + 1)),
      el("span", "layer__name", LABELS[name]),
      el("span", "layer__shade", shade),
    );
    label.style.setProperty("--swatch", look[name].color);
    const anchor = el("span", "layer__anchor");
    node.append(canvas, anchor);
    labels.append(label);
    node.setAttribute("aria-label", `${index + 1}: ${meta?.label ?? name}, ${shade}`);
    stack.append(node);
    sheets.push({ name, node, canvas, label, anchor });
  });

  const finish = el("div", "layer layer--final");
  const finishCanvas = el("canvas", "layer__canvas");
  finish.append(finishCanvas);
  stack.append(finish);

  let size = 0;
  let painted: HTMLCanvasElement | null = null;

  function layout(): void {
    const rect = root.getBoundingClientRect();
    const next = Math.round(Math.min(rect.width * 0.5, rect.height * 0.5));
    if (next <= 10 || next === size) {
      return;
    }
    size = next;
    root.style.setProperty("--layer", `${size}px`);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixels = Math.round(size * dpr);
    const crop = faceCrop(scene, 1, 1.18);

    const photo = cropTo(scene.source, crop, pixels, pixels);
    baseCanvas.width = pixels;
    baseCanvas.height = pixels;
    baseCanvas.getContext("2d")?.drawImage(photo, 0, 0);

    painted ??= renderLook(scene, look);
    finishCanvas.width = pixels;
    finishCanvas.height = pixels;
    finishCanvas.getContext("2d")?.drawImage(cropTo(painted, crop, pixels, pixels), 0, 0);

    // Each mask, cropped to the same square, coloured, then drawn twice:
    // once blurred underneath for a glow, once sharp on top.
    const { masks } = scene;
    const k = masks.scale;
    const mx = Math.max(0, Math.floor(crop.x * k));
    const my = Math.max(0, Math.floor(crop.y * k));
    const mw = Math.min(masks.width - mx, Math.ceil(crop.w * k));
    const mh = Math.min(masks.height - my, Math.ceil(crop.h * k));
    for (const sheet of sheets) {
      const mask = masks.masks[sheet.name];
      sheet.canvas.width = pixels;
      sheet.canvas.height = pixels;
      const ctx = sheet.canvas.getContext("2d");
      if (!mask || !ctx || mw <= 0 || mh <= 0) {
        continue;
      }
      const tile = document.createElement("canvas");
      tile.width = mw;
      tile.height = mh;
      const tileCtx = tile.getContext("2d");
      if (!tileCtx) {
        continue;
      }
      const image = tileCtx.createImageData(mw, mh);
      const [r, g, b] = sheetColour(look[sheet.name].color);
      for (let y = 0; y < mh; y++) {
        for (let x = 0; x < mw; x++) {
          const weight = mask[(my + y) * masks.width + (mx + x)];
          const p = (y * mw + x) * 4;
          image.data[p] = r;
          image.data[p + 1] = g;
          image.data[p + 2] = b;
          image.data[p + 3] = Math.min(255, weight * 245);
        }
      }
      tileCtx.putImageData(image, 0, 0);
      const scale = pixels / (crop.w * k);
      const dx = (mx - crop.x * k) * scale;
      const dy = (my - crop.y * k) * scale;
      ctx.imageSmoothingQuality = "high";
      ctx.filter = `blur(${Math.round(pixels / 60)}px)`;
      ctx.globalAlpha = 0.7;
      ctx.drawImage(tile, dx, dy, mw * scale, mh * scale);
      ctx.filter = "none";
      ctx.globalAlpha = 1;
      ctx.drawImage(tile, dx, dy, mw * scale, mh * scale);
    }
  }

  function placeLabels(): void {
    const host = root.getBoundingClientRect();
    for (const sheet of sheets) {
      const point = sheet.anchor.getBoundingClientRect();
      sheet.label.style.transform = `translate(${(point.left - host.left).toFixed(1)}px, ${(point.top - host.top).toFixed(1)}px)`;
    }
  }

  function set(explode: number, tilt: number, finished: number): void {
    root.style.setProperty("--explode", explode.toFixed(4));
    root.style.setProperty("--tilt", tilt.toFixed(4));
    root.style.setProperty("--finished", finished.toFixed(4));
    if (explode > 0.02) {
      placeLabels();
    }
  }

  sheets.forEach((sheet, i) => sheet.node.style.setProperty("--i", String(i + 1)));
  finish.style.setProperty("--i", String(sheets.length + 1));
  set(0, 0, 0);

  return { element: root, set, layout };
}
