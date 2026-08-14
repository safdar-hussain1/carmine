/**
 * The mirror: camera in, look on your face, one frame at a time.
 *
 * Per live frame the work is detect -> (optionally) smooth -> build masks ->
 * one WebGL2 draw. The frame never leaves the page: it goes from the video
 * element straight into a texture, and the only things that cross back to
 * the CPU are 478 landmark coordinates and, when a gloss finish is on, two
 * percentiles.
 *
 * The loop is driven by `requestVideoFrameCallback` where the browser has
 * it, because that fires once per *decoded camera frame* rather than once
 * per display refresh: on a 30fps camera and a 120Hz screen, rAF would run
 * the whole detect-and-mask pipeline four times over the same picture.
 * `rAF` is the fallback.
 */

import { startCamera, stopCamera, loadPhoto } from "../lib/camera";
import { createRenderer, type Renderer } from "../engine/renderer";
import { OneEuroFilter } from "../engine/oneEuro";
import type { LookConfig } from "../engine/look";
import type { MaskSet } from "../engine/masks";
import {
  applyLookCpu,
  computeGloss,
  DEMO_PORTRAIT_URL,
  loadImageElement,
  masksFor,
  sharedLandmarker,
  toFloatPixels,
  toImageData,
  toProcessingCanvas,
} from "./pipeline";
import { ICONS } from "./icons";

/** Long side the drawing buffer is capped at, matching the camera request. */
const MAX_OUTPUT_SIDE = 1280;

const EMPTY_MASKS: MaskSet = { width: 1, height: 1, scale: 1, interocular: 0, masks: {} };

type Mode = "idle" | "live" | "photo" | "denied" | "unsupported";

export interface Mirror {
  element: HTMLElement;
  /** Push a new look; live picks it up next frame, photo re-renders now. */
  setLook(look: LookConfig): void;
}

interface MirrorOptions {
  getLook: () => LookConfig;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  html?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (html !== undefined) {
    node.innerHTML = html;
  }
  return node;
}

function button(className: string, label: string, icon?: string): HTMLButtonElement {
  const node = el("button", className);
  node.type = "button";
  node.innerHTML = `${icon ?? ""}<span>${label}</span>`;
  return node;
}

export function createMirror(options: MirrorOptions): Mirror {
  const root = el("div");

  const stage = el("div", "stage");
  stage.dataset.mode = "idle";
  stage.dataset.wipe = "off";

  const video = el("video", "stage__video");
  video.playsInline = true;
  video.muted = true;

  const canvas = el("canvas", "stage__canvas");

  const wipe = el("div", "wipe");
  const wipeLine = el("div", "wipe__line");
  const wipeGrip = el("div", "wipe__grip", ICONS.grip);
  const tagBefore = el("span", "wipe__tag wipe__tag--before", "Before");
  const tagAfter = el("span", "wipe__tag wipe__tag--after", "After");
  wipe.append(tagBefore, tagAfter, wipeLine, wipeGrip);

  const hud = el("div", "hud");
  const hudDot = el("span", "hud__dot");
  const hudText = el("span", "hud__text", "&mdash;");
  hud.append(hudDot, hudText);

  const panel = el("div", "stage__panel");
  const panelMark = el("p", "stage__mark", "Carmine");
  const panelClaim = el(
    "p",
    "stage__claim",
    "Every frame stays on this device. Nothing is uploaded &mdash; once the page has loaded it makes no network requests at all.",
  );
  const panelActions = el("div", "stage__actions");
  const openBtn = button("btn btn--primary", "Open the mirror", ICONS.camera);
  const sampleBtn = button("btn btn--ghost", "Sample portrait", ICONS.sample);
  const uploadBtn = button("btn btn--ghost", "Upload a photo", ICONS.photo);
  panelActions.append(openBtn, sampleBtn, uploadBtn);
  const panelNote = el("p", "stage__note", "Your camera is only asked for when you tap Open the mirror.");
  panel.append(panelMark, panelClaim, panelActions, panelNote);

  const fileInput = el("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.className = "visually-hidden";

  stage.append(video, canvas, wipe, hud, panel);

  const bar = el("div", "stage-bar");
  const steadyBtn = button("btn", "Steady");
  steadyBtn.setAttribute("aria-pressed", "true");
  steadyBtn.title = "One-Euro smoothing on the landmark stream";
  const wipeBtn = button("btn", "Before / after", ICONS.compare);
  wipeBtn.setAttribute("aria-pressed", "false");
  const spacer = el("div", "stage-bar__spacer");
  const captureBtn = button("btn", "Capture PNG", ICONS.download);
  const stopBtn = button("btn", "Stop camera", ICONS.stop);
  bar.append(steadyBtn, wipeBtn, spacer, captureBtn, stopBtn);

  root.append(stage, bar, fileInput);

  // ---- state ------------------------------------------------------------

  let mode: Mode = "idle";
  let look = options.getLook();
  let renderer: Renderer | null = null;
  let rendererTried = false;
  let filter = new OneEuroFilter();
  let steady = true;
  let splitFraction: number | null = null;
  let photoSource: CanvasImageSource | null = null;
  let photoSize = { width: 0, height: 0 };
  let running = false;
  let loopHandle = 0;
  let usingVideoFrames = false;
  let pendingCapture = false;
  let fps = 0;
  let lastFrameTime = 0;
  const procCanvas = document.createElement("canvas");

  function setMode(next: Mode): void {
    mode = next;
    stage.dataset.mode = next;
    const interactive = next === "live" || next === "photo";
    captureBtn.disabled = !interactive;
    wipeBtn.disabled = !interactive;
    steadyBtn.disabled = next !== "live";
    stopBtn.disabled = next !== "live";
    stopBtn.hidden = next !== "live";
  }

  function ensureRenderer(): Renderer | null {
    if (renderer || rendererTried) {
      return renderer;
    }
    rendererTried = true;
    renderer = createRenderer(canvas);
    return renderer;
  }

  /**
   * Where the wipe handle sits along the *canvas*, as a fraction of its
   * width. The handle is dragged in stage coordinates, and the canvas is
   * drawn with `object-fit: cover`, so a canvas that is wider than the stage
   * is cropped left and right -- without undoing that crop the seam would
   * drift away from the handle the reader is holding.
   */
  function splitCanvasFraction(): number | null {
    if (splitFraction === null) {
      return null;
    }
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || canvas.width <= 0 || canvas.height <= 0) {
      return splitFraction;
    }
    const scale = Math.max(rect.width / canvas.width, rect.height / canvas.height);
    const offsetX = (rect.width - canvas.width * scale) / 2;
    return Math.min(Math.max((splitFraction * rect.width - offsetX) / scale / canvas.width, 0), 1);
  }

  function splitPixels(): number | null {
    const fraction = splitCanvasFraction();
    return fraction === null ? null : fraction * canvas.width;
  }

  function sizeCanvas(width: number, height: number): void {
    const longSide = Math.max(width, height);
    const scale = longSide > MAX_OUTPUT_SIDE ? MAX_OUTPUT_SIDE / longSide : 1;
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    if (canvas.width === w && canvas.height === h) {
      return;
    }
    const active = ensureRenderer();
    if (active) {
      active.resize(w, h);
    } else {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function finishCapture(): void {
    if (!pendingCapture) {
      return;
    }
    pendingCapture = false;
    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `carmine-${Date.now()}.png`;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  /** One pass of the pipeline over `source`, drawn into the canvas. */
  function renderSource(
    source: HTMLVideoElement | CanvasImageSource,
    width: number,
    height: number,
    timestampMs: number,
    landmarker: { detect: (s: TexImageSource, t: number) => Float32Array | null },
    mirrored: boolean,
  ): boolean {
    const active = ensureRenderer();
    if (!active) {
      return false;
    }
    sizeCanvas(width, height);

    let landmarks = landmarker.detect(source as TexImageSource, timestampMs);
    if (landmarks === null) {
      filter.reset();
      active.render(source as TexImageSource, EMPTY_MASKS, look, {}, {
        mirror: mirrored,
        splitX: splitPixels(),
      });
      return false;
    }

    if (steady && mirrored) {
      // Smoothing is a preference, not a fix: our own stability measurements
      // (see the Measured section) found no benefit on a held-still clip.
      // It is only applied to the live stream -- a still photo has nothing
      // to smooth across.
      landmarks = Float32Array.from(filter.filter(landmarks, timestampMs / 1000));
    }

    const masks = masksFor(landmarks, width, height, look);

    let gloss = {};
    const needsGloss =
      (look.highlighter.intensity > 0 && masks.masks.highlighter) ||
      (look.lipstick.intensity > 0 && look.lipstick.finish === "gloss" && masks.masks.lipstick);
    if (needsGloss) {
      const proc = toProcessingCanvas(source, width, height, procCanvas);
      const ctx = proc.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        const data = ctx.getImageData(0, 0, proc.width, proc.height);
        gloss = computeGloss(toFloatPixels(data), masks, look);
      }
    }

    active.render(source as TexImageSource, masks, look, gloss, {
      mirror: mirrored,
      splitX: splitPixels(),
    });
    return true;
  }

  // ---- live loop --------------------------------------------------------

  async function frame(nowMs: number): Promise<void> {
    if (!running || mode !== "live") {
      return;
    }
    const landmarker = await sharedLandmarker();
    if (!running || mode !== "live") {
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width > 0 && height > 0) {
      const found = renderSource(video, width, height, nowMs, landmarker, true);
      hudDot.style.background = found ? "" : "#e0a24d";
      if (lastFrameTime > 0) {
        const delta = nowMs - lastFrameTime;
        if (delta > 0) {
          // Exponential moving average: an instantaneous reading flickers
          // by several frames per second and is unreadable.
          fps = fps === 0 ? 1000 / delta : fps * 0.9 + (1000 / delta) * 0.1;
        }
      }
      lastFrameTime = nowMs;
      hudText.textContent = `${Math.round(fps)} fps · ${canvas.width}×${canvas.height}${
        found ? "" : " · no face"
      }`;
      finishCapture();
    }
    schedule();
  }

  type VideoFrameCapable = HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number) => void) => number;
    cancelVideoFrameCallback?: (handle: number) => void;
  };

  function schedule(): void {
    if (!running) {
      return;
    }
    const withVideoFrame = video as VideoFrameCapable;
    if (typeof withVideoFrame.requestVideoFrameCallback === "function") {
      usingVideoFrames = true;
      loopHandle = withVideoFrame.requestVideoFrameCallback((now) => {
        void frame(now);
      });
    } else {
      usingVideoFrames = false;
      loopHandle = requestAnimationFrame((now) => {
        void frame(now);
      });
    }
  }

  function stopLoop(): void {
    running = false;
    if (loopHandle) {
      // The two schedulers hand out handles from separate pools, so
      // cancelling with the wrong one would leave this loop running and
      // abort an unrelated animation frame.
      const withVideoFrame = video as VideoFrameCapable;
      if (usingVideoFrames) {
        withVideoFrame.cancelVideoFrameCallback?.(loopHandle);
      } else {
        cancelAnimationFrame(loopHandle);
      }
      loopHandle = 0;
    }
  }

  // ---- entry points -----------------------------------------------------

  function showUnsupported(): void {
    setMode("unsupported");
    panelMark.textContent = "No WebGL2 here";
    panelClaim.innerHTML =
      "The live mirror needs WebGL2 to paint a camera frame in real time. Still photos work anyway &mdash; they run the same colour maths on the CPU instead.";
    openBtn.hidden = true;
    panelNote.textContent = "Photo mode runs the CPU reference path";
  }

  async function openCamera(): Promise<void> {
    if (!ensureRenderer()) {
      showUnsupported();
      return;
    }
    openBtn.disabled = true;
    panelNote.textContent = "Waiting for camera permission…";
    try {
      const size = await startCamera(video);
      photoSource = null;
      filter.reset();
      fps = 0;
      lastFrameTime = 0;
      sizeCanvas(size.width, size.height);
      setMode("live");
      running = true;
      schedule();
    } catch {
      setMode("denied");
      panelMark.textContent = "No camera";
      panelClaim.innerHTML =
        "The camera was blocked or is in use elsewhere. You can allow it in your browser's site settings and tap again, or try a shade on a photo instead.";
      panelNote.className = "stage__note stage__note--warn";
      panelNote.textContent = "Nothing was uploaded — the request never left this device.";
    } finally {
      openBtn.disabled = false;
    }
  }

  function closeCamera(): void {
    stopLoop();
    stopCamera(video);
    setMode("idle");
  }

  /** Renders a still, on the GPU when there is one and on the CPU when not. */
  async function showPhoto(source: CanvasImageSource, width: number, height: number): Promise<void> {
    stopLoop();
    stopCamera(video);
    photoSource = source;
    photoSize = { width, height };
    setMode("photo");
    await renderPhoto();
  }

  async function renderPhoto(): Promise<void> {
    if (!photoSource) {
      return;
    }
    const landmarker = await sharedLandmarker();
    const { width, height } = photoSize;
    const drawn = ensureRenderer()
      ? renderSource(photoSource, width, height, performance.now(), landmarker, false)
      : renderPhotoCpu(landmarker);
    hudText.textContent = `photo · ${canvas.width}×${canvas.height}${drawn ? "" : " · no face"}`;
    finishCapture();
  }

  /** The WebGL2-less path: same ops, on byte arrays, at processing size. */
  function renderPhotoCpu(landmarker: {
    detect: (s: TexImageSource, t: number) => Float32Array | null;
  }): boolean {
    if (!photoSource) {
      return false;
    }
    const { width, height } = photoSize;
    const proc = toProcessingCanvas(photoSource, width, height, procCanvas);
    const ctx = proc.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return false;
    }
    canvas.width = proc.width;
    canvas.height = proc.height;
    const out = canvas.getContext("2d");
    if (!out) {
      return false;
    }
    const landmarks = landmarker.detect(proc, performance.now());
    const data = ctx.getImageData(0, 0, proc.width, proc.height);
    if (landmarks === null) {
      out.putImageData(data, 0, 0);
      return false;
    }
    const masks = masksFor(landmarks, proc.width, proc.height, look);
    const painted = applyLookCpu(toFloatPixels(data), masks, look);
    const fraction = splitCanvasFraction();
    const split = fraction === null ? null : Math.round(fraction * proc.width);
    if (split !== null && split >= proc.width) {
      out.putImageData(data, 0, 0);
      return true;
    }
    out.putImageData(toImageData(painted, proc.width, proc.height), 0, 0);
    if (split !== null && split > 0) {
      // Same wipe, drawn by hand: the "before" columns come from the source.
      out.putImageData(data, 0, 0, 0, 0, split, proc.height);
    }
    return true;
  }

  async function useSample(): Promise<void> {
    sampleBtn.disabled = true;
    try {
      const img = await loadImageElement(DEMO_PORTRAIT_URL);
      await showPhoto(img, img.naturalWidth, img.naturalHeight);
    } finally {
      sampleBtn.disabled = false;
    }
  }

  // ---- wiring -----------------------------------------------------------

  openBtn.addEventListener("click", () => void openCamera());
  sampleBtn.addEventListener("click", () => void useSample());
  uploadBtn.addEventListener("click", () => fileInput.click());
  stopBtn.addEventListener("click", closeCamera);

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      void loadPhoto(file).then((photo) => showPhoto(photo, photo.width, photo.height));
    }
    fileInput.value = "";
  });

  // Drop a photo anywhere on the stage.
  stage.addEventListener("dragover", (event) => {
    event.preventDefault();
  });
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      void loadPhoto(file).then((photo) => showPhoto(photo, photo.width, photo.height));
    }
  });

  steadyBtn.addEventListener("click", () => {
    steady = !steady;
    steadyBtn.setAttribute("aria-pressed", String(steady));
    filter = new OneEuroFilter();
  });

  function setWipe(on: boolean): void {
    splitFraction = on ? 0.5 : null;
    stage.dataset.wipe = on ? "on" : "off";
    wipeBtn.setAttribute("aria-pressed", String(on));
    positionWipe();
    if (mode === "photo") {
      void renderPhoto();
    }
  }

  function positionWipe(): void {
    const fraction = splitFraction ?? 0.5;
    const pct = `${fraction * 100}%`;
    wipeLine.style.left = pct;
    wipeGrip.style.left = pct;
    wipe.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
  }

  wipeBtn.addEventListener("click", () => setWipe(splitFraction === null));

  let dragging = false;
  function moveWipe(clientX: number): void {
    const rect = stage.getBoundingClientRect();
    splitFraction = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    positionWipe();
    if (mode === "photo") {
      void renderPhoto();
    }
  }

  wipe.addEventListener("pointerdown", (event) => {
    dragging = true;
    wipe.setPointerCapture(event.pointerId);
    moveWipe(event.clientX);
  });
  wipe.addEventListener("pointermove", (event) => {
    if (dragging) {
      moveWipe(event.clientX);
    }
  });
  const endDrag = () => {
    dragging = false;
  };
  wipe.addEventListener("pointerup", endDrag);
  wipe.addEventListener("pointercancel", endDrag);

  // The handle is a slider, so it answers to arrow keys as well as to a
  // pointer -- a comparison you can only make with a mouse is a comparison
  // some people cannot make.
  wipe.tabIndex = 0;
  wipe.setAttribute("role", "slider");
  wipe.setAttribute("aria-label", "Before and after split");
  wipe.setAttribute("aria-valuemin", "0");
  wipe.setAttribute("aria-valuemax", "100");
  wipe.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 0.1 : 0.02;
    let next = splitFraction ?? 0.5;
    if (event.key === "ArrowLeft") {
      next -= step;
    } else if (event.key === "ArrowRight") {
      next += step;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = 1;
    } else {
      return;
    }
    event.preventDefault();
    splitFraction = Math.min(Math.max(next, 0), 1);
    positionWipe();
    if (mode === "photo") {
      void renderPhoto();
    }
  });

  captureBtn.addEventListener("click", () => {
    pendingCapture = true;
    if (mode === "photo") {
      void renderPhoto();
    }
  });

  positionWipe();
  setMode("idle");
  if (typeof document !== "undefined" && !ensureRenderer()) {
    showUnsupported();
  }

  return {
    element: root,
    setLook(next: LookConfig) {
      look = next;
      if (mode === "photo") {
        void renderPhoto();
      }
    },
  };
}
