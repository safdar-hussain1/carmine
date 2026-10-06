/**
 * The four steps of the pipeline, drawn by the engine itself.
 *
 * Each picture in the "How it works" section is made live, in the browser,
 * from the sample portrait the mirror has already analysed: the landmark
 * points it found, the masks it built from them, the colour change beside a
 * flat fill of the same lipstick, and the finished look. Nothing here is a
 * prepared image, so the section cannot describe a pipeline the page does
 * not actually run.
 *
 * The flat fill is the photo benchmark's opaque-fill baseline, for the lips:
 * the lip polygon filled hard and added on top of the picture, which is
 * what `cv2.fillPoly` plus `cv2.addWeighted` do in `carmine.baselines`, and
 * what a canvas "lighter" composite does here. It is painted with the same
 * lipstick the benchmark used -- the velvet look's, at satin -- so the
 * picture and the measured lightness shift quoted under it describe the
 * same comparison.
 */

import { createRenderer, type Renderer } from "../engine/renderer";
import { hexToRgb } from "../engine/color";
import { PRESETS, PRODUCT_ORDER, type LookConfig, type ProductName } from "../engine/look";
import constants from "../gen/constants.json";
import { applyLookCpu, masksFor, toFloatPixels, toImageData, toProcessingCanvas } from "./pipeline";
import { glossFor, type PhotoScene } from "./scene";
import { mixLook, withoutMakeup } from "./tween";

const R = constants.regions;

/** One colour per mask in the masks picture, picked to tell the regions
 * apart on a greyscale face rather than to be wearable. */
const MASK_COLOURS: Record<ProductName, string> = {
  lipstick: "#d0284f",
  eyeshadow: "#8e5bc4",
  eyeliner: "#2b3f8f",
  brows: "#b06a2c",
  blush: "#ff8a73",
  highlighter: "#fff2cc",
};

/** The benchmark's lipstick: the velvet look's, with the finish at satin. */
function benchmarkLips(): LookConfig {
  const look = withoutMakeup(PRESETS.velvet);
  look.lipstick = { ...PRESETS.velvet.lipstick, finish: "satin" };
  return look;
}

interface Crop {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * A crop of the frame's shape (`aspect`, width over height) centred on
 * (cx, cy), `w` wide, kept inside the picture. The shape is read from the
 * frame on screen, so the CSS can change it per breakpoint without
 * stretching the picture.
 */
function cropAround(cx: number, cy: number, w: number, aspect: number, scene: PhotoScene): Crop {
  let width = Math.min(w, scene.width, scene.height * aspect);
  let height = width / aspect;
  if (height > scene.height) {
    height = scene.height;
    width = height * aspect;
  }
  const x = Math.min(Math.max(cx - width / 2, 0), scene.width - width);
  const y = Math.min(Math.max(cy - height / 2, 0), scene.height - height);
  return { x, y, w: width, h: height };
}

function point(scene: PhotoScene, index: number): [number, number] {
  return [scene.landmarks[index * 2], scene.landmarks[index * 2 + 1]];
}

function bounds(scene: PhotoScene, indices: number[]): Crop {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const index of indices) {
    const [x, y] = point(scene, index);
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function canvasOfSize(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

export interface Explainer {
  setScene(scene: PhotoScene): void;
  /** The last frame shows whatever look is on the mirror. */
  setLook(look: LookConfig): void;
}

export function createExplainer(section: HTMLElement, look: LookConfig): Explainer {
  const frames = {
    landmarks: section.querySelector<HTMLCanvasElement>('[data-step="landmarks"] canvas'),
    masks: section.querySelector<HTMLCanvasElement>('[data-step="masks"] canvas'),
    texture: section.querySelector<HTMLCanvasElement>('[data-step="texture"] canvas'),
    final: section.querySelector<HTMLCanvasElement>('[data-step="final"] canvas'),
  };

  let scene: PhotoScene | null = null;
  let current = mixLook(look, look, 1);
  let visible = false;
  let drawnOnce = false;
  /** The frames' width when they were last drawn. */
  let drawnWidth = 0;
  let finalDirty = true;
  let renderer: Renderer | null = null;
  let rendererTried = false;
  const glCanvas = document.createElement("canvas");

  function engine(): Renderer | null {
    if (!rendererTried) {
      rendererTried = true;
      renderer = createRenderer(glCanvas);
    }
    return renderer;
  }

  /** Sizes a frame's canvas to its box at the screen's pixel density. */
  function prepare(canvas: HTMLCanvasElement | null): CanvasRenderingContext2D | null {
    if (!canvas) {
      return null;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return canvas.getContext("2d");
  }

  function aspectOf(canvas: HTMLCanvasElement): number {
    return canvas.width / Math.max(canvas.height, 1);
  }

  function faceCrop(s: PhotoScene, aspect: number): Crop {
    const box = s.face;
    const width = Math.max(box.width * 1.3, box.height * 1.3 * aspect);
    return cropAround(box.cx, box.cy, width, aspect, s);
  }

  function lipsCrop(s: PhotoScene, aspect: number): Crop {
    const lips = bounds(s, R.LIPS_OUTER);
    return cropAround(lips.x + lips.w / 2, lips.y + lips.h / 2, lips.w * 1.3, aspect, s);
  }

  function toFrame(crop: Crop, canvas: HTMLCanvasElement, x: number, y: number): [number, number] {
    return [((x - crop.x) / crop.w) * canvas.width, ((y - crop.y) / crop.h) * canvas.height];
  }

  /** The look rendered over the whole picture, at full resolution. */
  function rendered(s: PhotoScene, which: LookConfig): CanvasImageSource | null {
    const active = engine();
    if (active) {
      active.resize(s.width, s.height);
      active.render(s.source, s.masks, which, glossFor(s, which), {});
      // Read straight away: without preserveDrawingBuffer the drawing
      // buffer is only kept until the browser next composites the page.
      const copy = canvasOfSize(s.width, s.height);
      copy.getContext("2d")?.drawImage(glCanvas, 0, 0);
      return copy;
    }
    // No WebGL2: the CPU reference at processing size, scaled back up.
    const proc = toProcessingCanvas(s.source, s.width, s.height);
    const ctx = proc.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return null;
    }
    const data = ctx.getImageData(0, 0, proc.width, proc.height);
    const painted = applyLookCpu(toFloatPixels(data), masksFor(s.landmarks, s.width, s.height, which), which);
    ctx.putImageData(toImageData(painted, proc.width, proc.height), 0, 0);
    const full = canvasOfSize(s.width, s.height);
    full.getContext("2d")?.drawImage(proc, 0, 0, s.width, s.height);
    return full;
  }

  function drawLandmarks(s: PhotoScene): void {
    const canvas = frames.landmarks;
    const ctx = prepare(canvas);
    if (!canvas || !ctx) {
      return;
    }
    const crop = faceCrop(s, aspectOf(canvas));
    ctx.drawImage(s.source, crop.x, crop.y, crop.w, crop.h, 0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(24, 12, 15, 0.32)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const unit = canvas.width / 300;
    const outline = (indices: number[], closed: boolean) => {
      ctx.beginPath();
      indices.forEach((index, i) => {
        const [x, y] = toFrame(crop, canvas, ...point(s, index));
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      if (closed) {
        ctx.closePath();
      }
      ctx.stroke();
    };
    ctx.strokeStyle = "rgba(255, 255, 255, 0.42)";
    ctx.lineWidth = Math.max(1, unit * 0.9);
    ctx.lineJoin = "round";
    outline(R.FACE_OVAL, true);
    outline(R.LIPS_OUTER, true);
    outline(R.LIPS_INNER, true);
    outline(R.LEFT_EYE, true);
    outline(R.RIGHT_EYE, true);
    outline(R.LEFT_BROW_UPPER, false);
    outline(R.RIGHT_BROW_UPPER, false);

    const radius = Math.max(1, unit * 1.05);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    for (let i = 0; i + 1 < s.landmarks.length; i += 2) {
      const [x, y] = toFrame(crop, canvas, s.landmarks[i], s.landmarks[i + 1]);
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  function drawMasks(s: PhotoScene): void {
    const canvas = frames.masks;
    const ctx = prepare(canvas);
    if (!canvas || !ctx) {
      return;
    }
    const crop = faceCrop(s, aspectOf(canvas));
    ctx.drawImage(s.source, crop.x, crop.y, crop.w, crop.h, 0, 0, canvas.width, canvas.height);

    // Grey the face so the masks are the only colour in the frame.
    const face = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const px = face.data;
    for (let i = 0; i < px.length; i += 4) {
      const grey = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
      const lifted = 40 + grey * 0.78;
      px[i] = px[i + 1] = px[i + 2] = lifted;
    }
    ctx.putImageData(face, 0, 0);

    const { masks } = s;
    const art = canvasOfSize(masks.width, masks.height);
    const artCtx = art.getContext("2d");
    if (!artCtx) {
      return;
    }
    const image = artCtx.createImageData(masks.width, masks.height);
    const out = image.data;
    for (const name of PRODUCT_ORDER) {
      const mask = masks.masks[name];
      if (!mask) {
        continue;
      }
      const [r, g, b] = hexToRgb(MASK_COLOURS[name]);
      for (let i = 0; i < mask.length; i++) {
        const a = mask[i] * 0.9;
        if (a < 0.01) {
          continue;
        }
        const p = i * 4;
        const under = out[p + 3] / 255;
        const alpha = a + under * (1 - a);
        out[p] = (r * a + out[p] * under * (1 - a)) / alpha;
        out[p + 1] = (g * a + out[p + 1] * under * (1 - a)) / alpha;
        out[p + 2] = (b * a + out[p + 2] * under * (1 - a)) / alpha;
        out[p + 3] = alpha * 255;
      }
    }
    artCtx.putImageData(image, 0, 0);
    const k = masks.scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(art, crop.x * k, crop.y * k, crop.w * k, crop.h * k, 0, 0, canvas.width, canvas.height);
  }

  function drawTexture(s: PhotoScene): void {
    const canvas = frames.texture;
    const ctx = prepare(canvas);
    if (!canvas || !ctx) {
      return;
    }
    const crop = lipsCrop(s, aspectOf(canvas));
    const lips = benchmarkLips();
    const half = Math.round(canvas.width / 2);

    // Left: the flat fill, added on top of the picture.
    const flat = canvasOfSize(canvas.width, canvas.height);
    const flatCtx = flat.getContext("2d");
    if (!flatCtx) {
      return;
    }
    flatCtx.drawImage(s.source, crop.x, crop.y, crop.w, crop.h, 0, 0, flat.width, flat.height);
    flatCtx.globalCompositeOperation = "lighter";
    flatCtx.globalAlpha = Math.min(1, lips.lipstick.intensity);
    flatCtx.fillStyle = lips.lipstick.color;
    flatCtx.beginPath();
    R.LIPS_OUTER.forEach((index, i) => {
      const [x, y] = toFrame(crop, flat, ...point(s, index));
      if (i === 0) {
        flatCtx.moveTo(x, y);
      } else {
        flatCtx.lineTo(x, y);
      }
    });
    flatCtx.closePath();
    flatCtx.fill();

    // Right: the same lipstick through the engine.
    const tinted = rendered(s, lips);
    ctx.drawImage(flat, 0, 0, half, canvas.height, 0, 0, half, canvas.height);
    if (tinted) {
      const cut = crop.x + (crop.w * half) / canvas.width;
      ctx.drawImage(
        tinted,
        cut,
        crop.y,
        crop.x + crop.w - cut,
        crop.h,
        half,
        0,
        canvas.width - half,
        canvas.height,
      );
    }
    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.fillRect(half - Math.max(1, canvas.width / 300), 0, Math.max(2, canvas.width / 150), canvas.height);
  }

  function drawFinal(s: PhotoScene): void {
    const canvas = frames.final;
    const ctx = prepare(canvas);
    if (!canvas || !ctx) {
      return;
    }
    const crop = faceCrop(s, aspectOf(canvas));
    const picture = rendered(s, current) ?? s.source;
    ctx.drawImage(picture, crop.x, crop.y, crop.w, crop.h, 0, 0, canvas.width, canvas.height);
    finalDirty = false;
  }

  function drawAll(): void {
    if (!scene || !visible) {
      return;
    }
    drawLandmarks(scene);
    drawMasks(scene);
    drawTexture(scene);
    drawFinal(scene);
    drawnWidth = frames.landmarks?.clientWidth ?? 0;
    if (!drawnOnce) {
      drawnOnce = true;
      section.dataset.drawn = "true";
    }
  }

  // Drawn once the section is close to the viewport: the work is a few
  // full-resolution renders, and there is no reason to do it for a reader
  // who never scrolls this far.
  if (typeof IntersectionObserver === "function") {
    new IntersectionObserver(
      (entries) => {
        visible = entries.some((entry) => entry.isIntersecting);
        if (visible && (!drawnOnce || finalDirty)) {
          if (drawnOnce) {
            if (scene) {
              drawFinal(scene);
            }
          } else {
            drawAll();
          }
        }
      },
      { rootMargin: "400px 0px" },
    ).observe(section);
  } else {
    visible = true;
  }

  // Redrawn only when the frames change width. A phone fires resize every
  // time its address bar slides away, which changes nothing here, and four
  // full-resolution redraws in the middle of a scroll would be felt.
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      const width = frames.landmarks?.clientWidth ?? 0;
      if (drawnOnce && width !== drawnWidth) {
        drawnOnce = false;
        drawAll();
      }
    }, 200);
  });

  let lookTimer = 0;
  return {
    setScene(next) {
      scene = next;
      drawAll();
    },
    setLook(next) {
      current = mixLook(next, next, 1);
      finalDirty = true;
      // A slider drag sends a look per pixel moved; the final frame only
      // needs the one the reader lets go on.
      window.clearTimeout(lookTimer);
      lookTimer = window.setTimeout(() => {
        if (scene && visible && drawnOnce) {
          drawFinal(scene);
        }
      }, 120);
    },
  };
}
