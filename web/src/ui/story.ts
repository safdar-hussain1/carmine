/**
 * How it works, told by the engine as you scroll.
 *
 * The section pins one picture while four short chapters scroll past it.
 * Each chapter swaps the picture for its step and plays it: the landmark
 * mesh drawing itself across the face, the masks blooming in, a flat fill
 * and Carmine's tint of the same lipstick split down the lips (and
 * draggable), and the finished look fading in over the bare face.
 *
 * Every picture is made live, in the browser, from the sample portrait the
 * mirror has already analysed -- nothing here is a prepared image, so the
 * section cannot describe a pipeline the page does not actually run.
 *
 * The flat fill is the photo benchmark's opaque-fill baseline, for the
 * lips: the lip polygon filled hard and added on top of the picture, which
 * is what `cv2.fillPoly` plus `cv2.addWeighted` do in `carmine.baselines`,
 * and what a canvas "lighter" composite does here. It is painted with the
 * lipstick the benchmark used -- the velvet look's, at satin -- so the
 * picture and the measured lightness shift quoted beside it describe the
 * same comparison.
 */

import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { createRenderer, type Renderer } from "../engine/renderer";
import { hexToRgb } from "../engine/color";
import { PRESETS, PRODUCT_ORDER, type LookConfig, type ProductName } from "../engine/look";
import constants from "../gen/constants.json";
import { applyLookCpu, masksFor, toFloatPixels, toImageData, toProcessingCanvas } from "./pipeline";
import { glossFor, type PhotoScene } from "./scene";
import { easeInOut, mixLook, withoutMakeup } from "./tween";

const R = constants.regions;

type Step = "landmarks" | "masks" | "texture" | "final";
const STEPS: Step[] = ["landmarks", "masks", "texture", "final"];

/** How long each chapter's picture takes to play in. */
const PLAY_MS: Record<Step, number> = { landmarks: 1600, masks: 1000, texture: 1100, final: 1100 };

/** Longest side of a chapter canvas, in pixels: sharp on a retina screen
 * without four full-screen buffers' worth of memory. */
const MAX_CANVAS_SIDE = 1400;

/** One colour per mask in the masks picture, picked to tell the regions
 * apart on a greyscale face rather than to be wearable. */
const MASK_COLOURS: Record<ProductName, string> = {
  lipstick: "#e3345c",
  eyeshadow: "#9a62d6",
  eyeliner: "#3d55c4",
  brows: "#c27a35",
  blush: "#ff8f78",
  highlighter: "#fff1cf",
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

/** A crop of the frame's shape (`aspect`, width over height) centred on
 * (cx, cy), `w` wide, kept inside the picture. */
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

function context(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2D canvas context unavailable");
  }
  return ctx;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/** Resolves on the next animation frame: a place to hand the main thread
 * back, so a long build is several short tasks instead of one long one. */
function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/** Runs `work` when the browser is idle, or soon if it never is. */
function whenIdle(work: () => void): void {
  const idle = (window as Window & {
    requestIdleCallback?: (cb: () => void, options?: { timeout: number }) => number;
  }).requestIdleCallback;
  if (idle) {
    idle(work, { timeout: 1200 });
  } else {
    window.setTimeout(work, 1);
  }
}

/** Everything a chapter draws from, at the frame's pixel size. */
interface Layers {
  width: number;
  height: number;
  /** Landmarks: the face, darkened, and the mesh in frame coordinates. */
  dim: HTMLCanvasElement;
  points: Float32Array;
  /** Mesh edges as point-index pairs, bucketed by distance from the face's
   * middle so they can ripple outward. */
  meshBuckets: Uint16Array[];
  /** Masks: the face in grey, and every mask in its colour. */
  grey: HTMLCanvasElement;
  maskArt: HTMLCanvasElement;
  /** Texture: the lips, flat-filled and tinted. */
  flat: HTMLCanvasElement;
  tinted: HTMLCanvasElement;
  /** Final: the face bare and with the current look. */
  bare: HTMLCanvasElement;
  made: HTMLCanvasElement;
}

export interface Story {
  setScene(scene: PhotoScene): void;
  /** The last chapter shows whatever look is on the mirror. */
  setLook(look: LookConfig): void;
}

export function createStory(section: HTMLElement, look: LookConfig): Story {
  const frame = section.querySelector<HTMLElement>(".story__frame");
  const chapters = [...section.querySelectorAll<HTMLElement>(".chapter")];
  const progress = [...section.querySelectorAll<HTMLElement>(".story__progress span")];
  const canvases = new Map<Step, HTMLCanvasElement>();
  for (const step of STEPS) {
    const canvas = section.querySelector<HTMLCanvasElement>(`.story__canvas[data-step="${step}"]`);
    if (canvas) {
      canvases.set(step, canvas);
    }
  }

  let scene: PhotoScene | null = null;
  let current = mixLook(look, look, 1);
  let layers: Layers | null = null;
  let active: Step = "landmarks";
  /** Whether the active chapter has played since it became active. */
  let played = false;
  /** A chapter became active before its picture was built. */
  let pendingPlay = false;
  let near = false;
  let buildToken = 0;
  let builtSize = { width: 0, height: 0 };
  let split = 0.5;
  let animation = 0;
  let renderer: Renderer | null = null;
  let rendererTried = false;
  const glCanvas = document.createElement("canvas");

  // The comparison in the third chapter can be dragged, with a pointer or
  // with the arrow keys.
  const drag = document.createElement("div");
  drag.className = "story__drag";
  drag.tabIndex = 0;
  drag.setAttribute("role", "slider");
  drag.setAttribute("aria-label", "Flat fill and Carmine split");
  drag.setAttribute("aria-valuemin", "0");
  drag.setAttribute("aria-valuemax", "100");
  frame?.append(drag);

  function engine(): Renderer | null {
    if (!rendererTried) {
      rendererTried = true;
      renderer = createRenderer(glCanvas);
    }
    return renderer;
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
      context(copy).drawImage(glCanvas, 0, 0);
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
    context(full).drawImage(proc, 0, 0, s.width, s.height);
    return full;
  }

  function frameSize(): { width: number; height: number } | null {
    if (!frame) {
      return null;
    }
    const rect = frame.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) {
      return null;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = Math.min(dpr, MAX_CANVAS_SIDE / Math.max(rect.width, rect.height));
    return { width: Math.round(rect.width * scale), height: Math.round(rect.height * scale) };
  }

  function crop(source: CanvasImageSource, area: Crop, width: number, height: number): HTMLCanvasElement {
    const out = canvasOfSize(width, height);
    context(out).drawImage(source, area.x, area.y, area.w, area.h, 0, 0, width, height);
    return out;
  }

  /** Builds every chapter's raw material once, at the frame's size, in
   * steps with a frame between them so scrolling stays smooth meanwhile. */
  async function prepare(s: PhotoScene): Promise<Layers | null> {
    const size = frameSize();
    if (!size) {
      return null;
    }
    const { width, height } = size;
    const aspect = width / height;
    const box = s.face;
    const face = cropAround(
      box.cx,
      box.cy,
      Math.max(box.width * 1.32, box.height * 1.32 * aspect),
      aspect,
      s,
    );
    const lipsBox = bounds(s, R.LIPS_OUTER);
    const lips = cropAround(lipsBox.x + lipsBox.w / 2, lipsBox.y + lipsBox.h / 2, lipsBox.w * 1.35, aspect, s);
    const toFrame = (area: Crop, x: number, y: number): [number, number] => [
      ((x - area.x) / area.w) * width,
      ((y - area.y) / area.h) * height,
    ];

    // Landmarks.
    const dim = crop(s.source, face, width, height);
    const dimCtx = context(dim);
    dimCtx.fillStyle = "rgba(10, 4, 7, 0.55)";
    dimCtx.fillRect(0, 0, width, height);
    const count = s.landmarks.length / 2;
    const points = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const [x, y] = toFrame(face, s.landmarks[i * 2], s.landmarks[i * 2 + 1]);
      points[i * 2] = x;
      points[i * 2 + 1] = y;
    }
    const BUCKETS = 18;
    const reach = Math.hypot(box.width, box.height) / 2 || 1;
    const lists: number[][] = Array.from({ length: BUCKETS }, () => []);
    for (const edge of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
      const [ax, ay] = point(s, edge.start);
      const [bx, by] = point(s, edge.end);
      const distance = Math.min(Math.hypot((ax + bx) / 2 - box.cx, (ay + by) / 2 - box.cy) / reach, 0.999);
      lists[Math.floor(distance * BUCKETS)].push(edge.start, edge.end);
    }
    const meshBuckets = lists.map((list) => Uint16Array.from(list));

    await nextFrame();

    // Masks.
    const grey = crop(s.source, face, width, height);
    const greyCtx = context(grey);
    const pixels = greyCtx.getImageData(0, 0, width, height);
    const px = pixels.data;
    for (let i = 0; i < px.length; i += 4) {
      const luma = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
      const value = 12 + luma * 0.62;
      px[i] = px[i + 1] = px[i + 2] = value;
    }
    greyCtx.putImageData(pixels, 0, 0);

    const { masks } = s;
    const art = canvasOfSize(masks.width, masks.height);
    const artCtx = context(art);
    const image = artCtx.createImageData(masks.width, masks.height);
    const out = image.data;
    for (const name of PRODUCT_ORDER) {
      const mask = masks.masks[name];
      if (!mask) {
        continue;
      }
      const [r, g, b] = hexToRgb(MASK_COLOURS[name]);
      for (let i = 0; i < mask.length; i++) {
        const a = mask[i] * 0.92;
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
    const maskArt = canvasOfSize(width, height);
    const maskCtx = context(maskArt);
    maskCtx.imageSmoothingQuality = "high";
    maskCtx.drawImage(art, face.x * k, face.y * k, face.w * k, face.h * k, 0, 0, width, height);

    await nextFrame();

    // Texture: the flat fill added on top of the lips, and the engine's tint.
    const lipsLook = benchmarkLips();
    const flat = crop(s.source, lips, width, height);
    const flatCtx = context(flat);
    flatCtx.globalCompositeOperation = "lighter";
    flatCtx.globalAlpha = Math.min(1, lipsLook.lipstick.intensity);
    flatCtx.fillStyle = lipsLook.lipstick.color;
    flatCtx.beginPath();
    R.LIPS_OUTER.forEach((index, i) => {
      const [x, y] = toFrame(lips, ...point(s, index));
      if (i === 0) {
        flatCtx.moveTo(x, y);
      } else {
        flatCtx.lineTo(x, y);
      }
    });
    flatCtx.closePath();
    flatCtx.fill();
    const tintedFull = rendered(s, lipsLook) ?? s.source;
    const tinted = crop(tintedFull, lips, width, height);

    await nextFrame();

    // Final.
    const bare = crop(s.source, face, width, height);
    const madeFull = rendered(s, current) ?? s.source;
    const made = crop(madeFull, face, width, height);

    return { width, height, dim, points, meshBuckets, grey, maskArt, flat, tinted, bare, made };
  }

  function sized(step: Step, l: Layers): CanvasRenderingContext2D | null {
    const canvas = canvases.get(step);
    if (!canvas) {
      return null;
    }
    if (canvas.width !== l.width || canvas.height !== l.height) {
      canvas.width = l.width;
      canvas.height = l.height;
    }
    return canvas.getContext("2d");
  }

  /** Draws `step` at progress `t` (0 to 1) of its play-in. */
  function draw(step: Step, t: number): void {
    const l = layers;
    if (!l) {
      return;
    }
    const ctx = sized(step, l);
    if (!ctx) {
      return;
    }
    const { width, height } = l;
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";

    if (step === "landmarks") {
      ctx.drawImage(l.dim, 0, 0);
      const unit = width / 600;
      const buckets = l.meshBuckets.length;
      ctx.globalCompositeOperation = "lighter";
      ctx.lineWidth = Math.max(0.6, unit * 0.9);
      ctx.strokeStyle = "#f0d3ae";
      for (let b = 0; b < buckets; b++) {
        const local = Math.min(Math.max((t - (b / buckets) * 0.6) / 0.4, 0), 1);
        if (local <= 0) {
          continue;
        }
        ctx.globalAlpha = 0.5 * local;
        ctx.beginPath();
        const edges = l.meshBuckets[b];
        for (let i = 0; i < edges.length; i += 2) {
          const a = edges[i] * 2;
          const c = edges[i + 1] * 2;
          ctx.moveTo(l.points[a], l.points[a + 1]);
          ctx.lineTo(l.points[c], l.points[c + 1]);
        }
        ctx.stroke();
      }
      const radius = Math.max(1, unit * 1.6);
      ctx.globalAlpha = Math.min(1, t * 1.4);
      ctx.fillStyle = "#fff8f1";
      ctx.beginPath();
      for (let i = 0; i + 1 < l.points.length; i += 2) {
        ctx.moveTo(l.points[i] + radius, l.points[i + 1]);
        ctx.arc(l.points[i], l.points[i + 1], radius, 0, Math.PI * 2);
      }
      ctx.fill();
    } else if (step === "masks") {
      ctx.drawImage(l.grey, 0, 0);
      ctx.globalAlpha = easeInOut(t);
      ctx.drawImage(l.maskArt, 0, 0);
    } else if (step === "texture") {
      // The divider sweeps in from the right edge to wherever it rests.
      const at = 1 - (1 - split) * easeInOut(t);
      const cut = Math.round(at * width);
      ctx.drawImage(l.flat, 0, 0);
      if (cut < width) {
        ctx.drawImage(l.tinted, cut, 0, width - cut, height, cut, 0, width - cut, height);
      }
      ctx.fillStyle = "rgba(255, 248, 241, 0.95)";
      ctx.fillRect(cut - Math.max(1, width / 600), 0, Math.max(2, width / 300), height);
      drag.style.setProperty("--split", `${at * 100}%`);
      drag.setAttribute("aria-valuenow", String(Math.round(at * 100)));
    } else {
      ctx.drawImage(l.bare, 0, 0);
      ctx.globalAlpha = easeInOut(t);
      ctx.drawImage(l.made, 0, 0);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  /** Plays the active chapter's picture in from the start. */
  function play(step: Step): void {
    cancelAnimationFrame(animation);
    if (!layers) {
      pendingPlay = true;
      return;
    }
    pendingPlay = false;
    if (prefersReducedMotion()) {
      draw(step, 1);
      return;
    }
    const begin = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - begin) / PLAY_MS[step], 1);
      draw(step, t);
      if (t < 1) {
        animation = requestAnimationFrame(tick);
      }
    };
    animation = requestAnimationFrame(tick);
  }

  function activate(step: Step): void {
    if (step === active && played) {
      return;
    }
    active = step;
    played = true;
    if (frame) {
      frame.dataset.active = step;
    }
    const index = STEPS.indexOf(step);
    progress.forEach((bar, i) => {
      bar.dataset.state = i < index ? "done" : i === index ? "active" : "";
    });
    for (const chapter of chapters) {
      chapter.dataset.active = String(chapter.dataset.step === step);
    }
    play(step);
  }

  async function build(): Promise<void> {
    if (!scene || !near) {
      return;
    }
    const token = ++buildToken;
    const built = await prepare(scene);
    // A newer build (a resize, a new scene) superseded this one meanwhile.
    if (token !== buildToken || !built) {
      return;
    }
    layers = built;
    const rect = frame?.getBoundingClientRect();
    builtSize = { width: rect?.width ?? 0, height: rect?.height ?? 0 };
    section.dataset.drawn = "true";
    // Every frame starts drawn in its finished state, so a chapter that is
    // reached by a jump (a link, a fast scroll) never shows a blank frame.
    for (const step of STEPS) {
      draw(step, 1);
    }
    // Played only if its chapter is already on screen; otherwise it plays
    // when the reader gets there.
    if (pendingPlay) {
      play(active);
    }
  }

  // ---- chapters follow the scroll ----------------------------------------

  let chapterObserver: IntersectionObserver | null = null;
  const wide = window.matchMedia?.("(min-width: 64rem)");
  function observeChapters(): void {
    chapterObserver?.disconnect();
    if (typeof IntersectionObserver !== "function") {
      return;
    }
    // A thin band across the screen decides the chapter: in the middle when
    // the picture sits beside the text, below the picture when it is pinned
    // above the text on a phone.
    const band = wide?.matches ? "-48% 0px -48% 0px" : "-64% 0px -30% 0px";
    chapterObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            activate((entry.target as HTMLElement).dataset.step as Step);
          }
        }
      },
      { rootMargin: band },
    );
    for (const chapter of chapters) {
      chapterObserver.observe(chapter);
    }
  }
  observeChapters();
  wide?.addEventListener("change", observeChapters);
  progress.forEach((bar, i) => {
    bar.dataset.state = i === 0 ? "active" : "";
  });
  chapters.forEach((chapter, i) => {
    chapter.dataset.active = String(i === 0);
  });

  // The pictures are built once the section is in sight of the viewport,
  // when the browser is idle: a few full-resolution renders that a reader
  // who never scrolls here should not pay for.
  if (typeof IntersectionObserver === "function") {
    new IntersectionObserver(
      (entries) => {
        near = entries.some((entry) => entry.isIntersecting);
        if (near && !layers) {
          whenIdle(() => void build());
        }
      },
      { rootMargin: "1200px 0px" },
    ).observe(section);
  } else {
    near = true;
  }

  // Drawn again only when the frame changes size, in either direction. A
  // phone fires resize every time its address bar slides away, which (the
  // frame being sized in svh) changes nothing here.
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      const rect = frame?.getBoundingClientRect();
      if (
        layers &&
        rect &&
        (Math.abs(rect.width - builtSize.width) > 1 || Math.abs(rect.height - builtSize.height) > 1)
      ) {
        void build();
      }
    }, 200);
  });

  // ---- the draggable comparison --------------------------------------------

  function setSplit(fraction: number): void {
    split = Math.min(Math.max(fraction, 0.02), 0.98);
    cancelAnimationFrame(animation);
    draw("texture", 1);
  }

  let dragging = false;
  let touchStart: { x: number; y: number } | null = null;
  const moveTo = (clientX: number) => {
    const rect = drag.getBoundingClientRect();
    setSplit((clientX - rect.left) / rect.width);
  };
  drag.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse") {
      dragging = true;
      drag.setPointerCapture(event.pointerId);
      moveTo(event.clientX);
    } else {
      // A finger may be starting a scroll: wait to see which way it goes.
      touchStart = { x: event.clientX, y: event.clientY };
    }
  });
  drag.addEventListener("pointermove", (event) => {
    if (dragging) {
      moveTo(event.clientX);
      return;
    }
    if (!touchStart) {
      return;
    }
    const dx = Math.abs(event.clientX - touchStart.x);
    const dy = Math.abs(event.clientY - touchStart.y);
    if (dx > 8 && dx > dy) {
      dragging = true;
      touchStart = null;
      drag.setPointerCapture(event.pointerId);
      moveTo(event.clientX);
    } else if (dy > 8) {
      touchStart = null;
    }
  });
  drag.addEventListener("pointerup", (event) => {
    // A tap without a swipe moves the divider to where it landed.
    if (touchStart) {
      moveTo(event.clientX);
    }
    dragging = false;
    touchStart = null;
  });
  drag.addEventListener("pointercancel", () => {
    dragging = false;
    touchStart = null;
  });
  drag.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 0.1 : 0.04;
    if (event.key === "ArrowLeft") {
      setSplit(split - step);
    } else if (event.key === "ArrowRight") {
      setSplit(split + step);
    } else if (event.key === "Home") {
      setSplit(0);
    } else if (event.key === "End") {
      setSplit(1);
    } else {
      return;
    }
    event.preventDefault();
  });

  let lookTimer = 0;
  return {
    setScene(next) {
      scene = next;
      layers = null;
      whenIdle(() => void build());
    },
    setLook(next) {
      current = mixLook(next, next, 1);
      // A slider drag sends a look per pixel moved; the last chapter only
      // needs the one the reader lets go on.
      window.clearTimeout(lookTimer);
      lookTimer = window.setTimeout(() => {
        if (!scene || !layers) {
          return;
        }
        const madeFull = rendered(scene, current) ?? scene.source;
        const l = layers;
        const box = scene.face;
        const aspect = l.width / l.height;
        const face = cropAround(
          box.cx,
          box.cy,
          Math.max(box.width * 1.32, box.height * 1.32 * aspect),
          aspect,
          scene,
        );
        l.made = crop(madeFull, face, l.width, l.height);
        if (active === "final") {
          cancelAnimationFrame(animation);
        }
        draw("final", 1);
      }, 150);
    },
  };
}
