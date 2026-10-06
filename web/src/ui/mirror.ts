/**
 * The mirror: a face, the look on it, and a handle to compare.
 *
 * It shows one of three sources in an arch-shaped stage -- the sample
 * portrait, the camera, or a photo of your own. Stills are analysed once
 * (see scene.ts) and redrawn on demand, which is cheap enough to blend from
 * one look to the next frame by frame. The camera runs the whole pipeline
 * on every frame.
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
 *
 * The first time the sample is ready, the mirror plays the pipeline back in
 * slow motion -- the landmark points, then the masks, then the colour -- so
 * the opening seconds explain what the engine does instead of only showing
 * the result. Anyone who asks for reduced motion gets the result directly.
 */

import { FaceLandmarker } from "@mediapipe/tasks-vision";
import { loadPhoto, startCamera, stopCamera } from "../lib/camera";
import { createRenderer, type Renderer } from "../engine/renderer";
import { OneEuroFilter } from "../engine/oneEuro";
import { hexToRgb } from "../engine/color";
import { PRODUCT_ORDER, type LookConfig, type ProductName } from "../engine/look";
import { PROC_MAX_SIDE_LIVE, type MaskSet } from "../engine/masks";
import {
  applyLookCpu,
  computeGloss,
  loadImageElement,
  masksFor,
  SAMPLE_PORTRAIT_URL,
  sharedLandmarker,
  toFloatPixels,
  toImageData,
  toProcessingCanvas,
} from "./pipeline";
import { analyzePhoto, glossFor, type PhotoScene } from "./scene";
import { boxToContentX, contentToBox, coverFit, focusOn, type CoverFit } from "./geometry";
import { easeInOut, mixLook, withoutMakeup } from "./tween";
import { FACTS } from "./facts";
import { ICONS } from "./icons";

/** Long side the drawing buffer is capped at: the photo cap in camera.ts,
 * so the sample and an uploaded photo render at their own resolution. */
const MAX_OUTPUT_SIDE = 1600;

/** How long a change of look or shade takes to blend in. Long enough to see
 * the colour turn, short enough that the mirror never feels behind you. */
const BLEND_MS = 420;

/** The opening reveal, phase by phase. */
const REVEAL_MS = { scan: 1500, masks: 800, tint: 1300, wipe: 750 };

const EMPTY_MASKS: MaskSet = {
  quality: "exact",
  width: 1,
  height: 1,
  scale: 1,
  interocular: 0,
  masks: {},
};

export type Source = "sample" | "camera" | "photo";

/** Landmarks that sit in the middle of each product's part of the face. */
const FEATURE_POINTS: Record<ProductName, number[]> = {
  lipstick: [0, 17],
  eyeshadow: [159, 386],
  eyeliner: [159, 386],
  brows: [105, 334],
  blush: [50, 280],
  highlighter: [116, 345],
};

type Tone = "busy" | "ready" | "live" | "warn";

export interface Mirror {
  /** The picture itself, sized by whatever it is placed in. */
  stage: HTMLElement;
  /** Its buttons -- back, compare, steady, save -- for the page to place. */
  tools: HTMLElement;
  /** Put a look on the face; `animate` blends to it instead of cutting. */
  setLook(look: LookConfig, animate: boolean): void;
  /** Load the face model and run the sample portrait. */
  start(): void;
  useCamera(): void;
  choosePhoto(): void;
  /**
   * Where a product's part of the face is on screen right now, in pixels
   * from the top of the stage, or null when no face has been found yet.
   */
  featureY(product: ProductName): number | null;
}

interface MirrorOptions {
  look: LookConfig;
  /** The sample already on the page, adopted so it neither reloads nor
   * replays its entrance when the script takes over. */
  poster?: HTMLImageElement | null;
  /** Start on the sample as soon as the page is up, without a click. */
  autostart: boolean;
  /** Called once, when the sample has been analysed. */
  onSampleReady(scene: PhotoScene): void;
  onSourceChange(source: Source): void;
}

interface NoticeAction {
  label: string;
  run: () => void;
  primary?: boolean;
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

function copyLook(look: LookConfig): LookConfig {
  return mixLook(look, look, 1);
}

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function createMirror(options: MirrorOptions): Mirror {
  // ---- markup ------------------------------------------------------------

  const stage = el("div", "stage");
  stage.dataset.source = "sample";
  stage.dataset.phase = "idle";
  stage.dataset.drawn = "false";
  stage.dataset.compare = "off";
  stage.setAttribute("role", "group");
  stage.setAttribute("aria-label", "Makeup mirror");

  // The sample is on screen from the first paint; the engine's own render
  // replaces it only once there is something to show.
  const poster = options.poster ?? el("img", "stage__poster");
  if (!options.poster) {
    poster.src = SAMPLE_PORTRAIT_URL;
    poster.alt = "The sample portrait the demo runs on";
    poster.width = 1600;
    poster.height = 1067;
    poster.decoding = "async";
  }

  const video = el("video", "stage__video");
  video.playsInline = true;
  video.muted = true;

  const canvas = el("canvas", "stage__canvas");
  const overlay = el("canvas", "stage__overlay");
  overlay.setAttribute("aria-hidden", "true");

  const wipe = el("div", "wipe");
  const wipeLine = el("span", "wipe__line");
  const wipeGrip = el("span", "wipe__grip", ICONS.grip);
  const tagBefore = el("span", "wipe__tag wipe__tag--before", "Before");
  const tagAfter = el("span", "wipe__tag wipe__tag--after", "After");
  wipe.append(wipeLine, tagBefore, tagAfter, wipeGrip);

  const status = el("p", "stage__status");
  status.setAttribute("role", "status");
  const statusDot = el("span", "stage__dot");
  const statusText = el("span", "stage__status-text");
  status.append(statusDot, statusText);
  status.hidden = true;

  const notice = el("div", "stage__notice");
  notice.setAttribute("role", "alert");
  const noticeText = el("p", "stage__notice-text");
  const noticeActions = el("div", "stage__notice-actions");
  notice.append(noticeText, noticeActions);
  notice.hidden = true;

  const startBtn = button("btn btn--primary stage__start", "Start the live demo", ICONS.play);
  startBtn.hidden = true;

  const fileInput = el("input", "visually-hidden");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.tabIndex = -1;
  fileInput.setAttribute("aria-hidden", "true");

  // Controls that only mean something for one source -- back to the sample,
  // smoothing for the camera -- sit on the picture beside its status, where
  // they appear when they apply; the always-useful ones go to the page.
  const context = el("div", "stage__context");

  stage.append(poster, video, canvas, overlay, wipe, status, context, notice, startBtn, fileInput);

  const bar = el("div", "tools");
  const backBtn = button("tool", "Back to the sample", ICONS.back);
  const compareBtn = button("tool", "Compare", ICONS.compare);
  compareBtn.setAttribute("aria-pressed", "false");
  compareBtn.title = "Show the face before and after, side by side";
  const steadyBtn = button("tool", "Steady", ICONS.steady);
  steadyBtn.setAttribute("aria-pressed", "true");
  steadyBtn.title = "Smooth the landmark points between camera frames";
  const saveBtn = button("tool", "Save photo", ICONS.download);
  saveBtn.title = "Download what the mirror shows as a PNG";
  saveBtn.disabled = true;
  context.append(backBtn, steadyBtn);
  bar.append(compareBtn, saveBtn);

  // ---- state -------------------------------------------------------------

  let source: Source = "sample";
  let sampleScene: PhotoScene | null = null;
  let photoScene: PhotoScene | null = null;
  let booting: Promise<void> | null = null;

  let renderer: Renderer | null = null;
  let rendererTried = false;

  /** What the controls say. */
  let target = copyLook(options.look);
  /** What is on the face this frame, which trails `target` while blending. */
  let shown = copyLook(options.look);
  let blend: { from: LookConfig; to: LookConfig; start: number; duration: number } | null = null;

  let compare = false;
  let splitPos = 0.5;
  let focus = { fx: 0.5, fy: 0.5 };

  let filter = new OneEuroFilter();
  let steady = true;
  let running = false;
  let loopHandle = 0;
  /** Bumped whenever the live loop starts or stops. A frame callback from an
   * older loop -- one still waiting on the model when the camera was closed
   * and opened again -- sees a stale number and ends its chain. */
  let loopId = 0;
  /** The camera request in flight, so a second click joins it instead of
   * opening a second stream nobody would ever stop. */
  let cameraStart: Promise<void> | null = null;
  /** Bumped when the reader moves on to another source, so a camera stream
   * that is only granted afterwards is closed instead of shown. */
  let cameraTicket = 0;
  let usingVideoFrames = false;
  let fps = 0;
  let lastFrameTime = 0;
  let lastStatusAt = 0;
  let pendingCapture = false;
  let drawHandle = 0;
  let revealing = false;
  let revealToken = 0;
  /** The camera's most recent landmarks, for `featureY`. */
  let liveLandmarks: Float32Array | null = null;
  const procCanvas = document.createElement("canvas");

  function ensureRenderer(): Renderer | null {
    if (renderer || rendererTried) {
      return renderer;
    }
    rendererTried = true;
    renderer = createRenderer(canvas);
    return renderer;
  }

  function currentScene(): PhotoScene | null {
    if (source === "sample") {
      return sampleScene;
    }
    return source === "photo" ? photoScene : null;
  }

  // ---- status and notices ------------------------------------------------

  function setStatus(text: string, tone: Tone): void {
    statusText.textContent = text;
    status.dataset.tone = tone;
    status.hidden = false;
  }

  function restingStatus(): void {
    if (source === "sample") {
      setStatus("Sample photo", "ready");
    } else if (source === "photo") {
      setStatus("Your photo", "ready");
    }
  }

  /** The status for whatever is showing, after something else failed. */
  function settleStatus(): void {
    if (source === "sample" && stage.dataset.phase === "loading") {
      setStatus("Loading the face model", "busy");
    } else if (source === "sample" && stage.dataset.phase === "idle") {
      status.hidden = true;
    } else {
      restingStatus();
    }
  }

  function showNotice(message: string, actions: NoticeAction[]): void {
    noticeText.textContent = message;
    noticeActions.textContent = "";
    for (const action of actions) {
      const node = button(action.primary ? "btn btn--primary" : "btn btn--quiet", action.label);
      node.addEventListener("click", () => {
        hideNotice();
        action.run();
      });
      noticeActions.append(node);
    }
    const dismiss = button("stage__notice-close", "Dismiss");
    dismiss.addEventListener("click", () => {
      hideNotice();
      // With the notice gone, the sample still needs a way to start.
      if (!sampleScene && !booting && source === "sample") {
        startBtn.hidden = false;
      }
    });
    noticeActions.append(dismiss);
    notice.hidden = false;
  }

  function hideNotice(): void {
    notice.hidden = true;
  }

  function showModelError(): void {
    status.hidden = true;
    stage.dataset.phase = sampleScene ? "ready" : "idle";
    showNotice("The face model didn't load. Check your connection and try again.", [
      { label: "Try again", run: () => void start(), primary: true },
    ]);
  }

  // ---- geometry ----------------------------------------------------------

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

  function fitNow(): CoverFit {
    const rect = stage.getBoundingClientRect();
    return coverFit(
      rect.width,
      rect.height,
      Math.max(canvas.width, 1),
      Math.max(canvas.height, 1),
      focus.fx,
      focus.fy,
    );
  }

  /**
   * Where the handle sits in drawing-buffer pixels. The handle is dragged
   * in stage coordinates while the canvas is cropped by `object-fit: cover`
   * and aimed at the face, so without undoing both the seam would drift
   * away from the handle the reader is holding.
   */
  function splitPixels(): number | null {
    if (!compare) {
      return null;
    }
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || canvas.width <= 0) {
      return null;
    }
    return boxToContentX(fitNow(), splitPos * rect.width, canvas.width);
  }

  /** Aims the crop at the face for stills; the camera stays centred, since
   * chasing a moving face with the crop would make the whole frame swim. */
  function aimAt(scene: PhotoScene | null): void {
    const rect = stage.getBoundingClientRect();
    focus =
      scene && rect.width > 0 && rect.height > 0
        ? focusOn(rect.width, rect.height, scene.width, scene.height, scene.face.cx, scene.face.cy)
        : { fx: 0.5, fy: 0.5 };
    const position = `${(focus.fx * 100).toFixed(2)}% ${(focus.fy * 100).toFixed(2)}%`;
    canvas.style.objectPosition = position;
    if (scene && scene === sampleScene) {
      poster.style.objectPosition = position;
    }
  }

  /**
   * The lowest the handle may go. On the wide layout the left of the picture
   * fades out under the text column, where the handle could neither be seen
   * nor grabbed again; everywhere else it has the whole width.
   */
  const wideLayout = window.matchMedia?.("(min-width: 75rem)");
  function splitMin(): number {
    return wideLayout?.matches ? 0.3 : 0;
  }

  function positionWipe(): void {
    splitPos = Math.max(splitPos, splitMin());
    const pct = `${splitPos * 100}%`;
    wipe.style.setProperty("--split", pct);
    wipe.setAttribute("aria-valuenow", String(Math.round(splitPos * 100)));
    // A label with no room beside the line would hang off the stage.
    wipe.dataset.edge = splitPos < splitMin() + 0.2 ? "left" : splitPos > 0.8 ? "right" : "";
  }

  function applyCompare(on: boolean): void {
    compare = on;
    stage.dataset.compare = on ? "on" : "off";
    compareBtn.setAttribute("aria-pressed", String(on));
    positionWipe();
    requestDraw();
  }

  // ---- drawing stills ----------------------------------------------------

  function lookAt(now: number): LookConfig {
    if (blend) {
      const t = (now - blend.start) / blend.duration;
      if (t >= 1) {
        shown = blend.to;
        blend = null;
      } else {
        shown = mixLook(blend.from, blend.to, easeInOut(t));
      }
    }
    return shown;
  }

  function applyLook(next: LookConfig, animate: boolean, duration = BLEND_MS): void {
    target = copyLook(next);
    // Blending redraws a still every frame; the CPU path takes hundreds of
    // milliseconds a draw, so without WebGL2 a change lands in one step.
    if (animate && renderer && !prefersReducedMotion()) {
      blend = { from: shown, to: target, start: performance.now(), duration };
    } else {
      blend = null;
      shown = target;
    }
    requestDraw();
  }

  function requestDraw(): void {
    if (source === "camera" || drawHandle) {
      return;
    }
    drawHandle = requestAnimationFrame(drawStill);
  }

  function drawStill(now: number): void {
    drawHandle = 0;
    const scene = currentScene();
    if (!scene) {
      return;
    }
    paintScene(scene, lookAt(now));
    if (blend) {
      requestDraw();
    }
  }

  function markDrawn(): void {
    if (stage.dataset.drawn !== "true") {
      stage.dataset.drawn = "true";
      saveBtn.disabled = false;
    }
  }

  function paintScene(scene: PhotoScene, look: LookConfig): void {
    const active = ensureRenderer();
    if (active) {
      sizeCanvas(scene.width, scene.height);
      active.render(scene.source, scene.masks, look, glossFor(scene, look), {
        mirror: false,
        splitX: splitPixels(),
      });
    } else {
      paintSceneCpu(scene, look);
    }
    markDrawn();
    finishCapture();
  }

  /** The WebGL2-less path: the exact CPU reference ops at processing size. */
  function paintSceneCpu(scene: PhotoScene, look: LookConfig): void {
    const proc = toProcessingCanvas(scene.source, scene.width, scene.height, procCanvas);
    const ctx = proc.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return;
    }
    if (canvas.width !== proc.width || canvas.height !== proc.height) {
      canvas.width = proc.width;
      canvas.height = proc.height;
    }
    const out = canvas.getContext("2d");
    if (!out) {
      return;
    }
    const data = ctx.getImageData(0, 0, proc.width, proc.height);
    const masks = masksFor(scene.landmarks, scene.width, scene.height, look);
    const painted = applyLookCpu(toFloatPixels(data), masks, look);
    out.putImageData(toImageData(painted, proc.width, proc.height), 0, 0);
    const split = splitPixels();
    if (split !== null && split > 0) {
      // Same wipe, drawn by hand: the "before" columns come from the source.
      out.putImageData(data, 0, 0, 0, 0, Math.round(split), proc.height);
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
      // Firefox and Safari can abort the download if the object URL is
      // revoked before the save dialog has read it; a synchronous revoke
      // right after click() races that. Defer well past any such window.
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    }, "image/png");
  }

  // ---- the live camera ---------------------------------------------------

  function renderLive(
    width: number,
    height: number,
    timestampMs: number,
    landmarker: { detect: (s: TexImageSource, t: number) => Float32Array | null },
    look: LookConfig,
  ): boolean {
    const active = ensureRenderer();
    if (!active) {
      return false;
    }
    sizeCanvas(width, height);

    let landmarks = landmarker.detect(video, timestampMs);
    liveLandmarks = landmarks;
    if (landmarks === null) {
      filter.reset();
      active.render(video, EMPTY_MASKS, look, {}, { mirror: true, splitX: splitPixels() });
      return false;
    }

    if (steady) {
      // Smoothing is a preference, not a fix: the stability benchmark found
      // no benefit on a slow, nearly still clip. A still photo has nothing
      // to smooth across, so it only ever applies here.
      landmarks = Float32Array.from(filter.filter(landmarks, timestampMs / 1000));
    }

    // Half-resolution masks with box-approximated feathers. Building them
    // the reference way costs hundreds of milliseconds on a face that fills
    // the frame -- the feather radii scale with the face, not with the
    // frame -- which is the difference between a mirror and a slideshow.
    const masks = masksFor(landmarks, width, height, look, "live");

    let gloss = {};
    const needsGloss =
      (look.highlighter.intensity > 0 && masks.masks.highlighter) ||
      (look.lipstick.intensity > 0 && look.lipstick.finish === "gloss" && masks.masks.lipstick);
    if (needsGloss) {
      // Sized to the mask set, not to the reference cap: the percentile
      // reduction walks masks and pixels with one index.
      const proc = toProcessingCanvas(video, width, height, procCanvas, PROC_MAX_SIDE_LIVE);
      const ctx = proc.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        gloss = computeGloss(toFloatPixels(ctx.getImageData(0, 0, proc.width, proc.height)), masks, look);
      }
    }

    active.render(video, masks, look, gloss, { mirror: true, splitX: splitPixels() });
    return true;
  }

  async function frame(nowMs: number, id: number): Promise<void> {
    if (!running || source !== "camera" || id !== loopId) {
      return;
    }
    let landmarker;
    try {
      landmarker = await sharedLandmarker();
    } catch {
      closeCamera();
      setSource("sample");
      requestDraw();
      showModelError();
      return;
    }
    if (!running || source !== "camera" || id !== loopId) {
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width > 0 && height > 0) {
      const found = renderLive(width, height, nowMs, landmarker, lookAt(nowMs));
      if (lastFrameTime > 0) {
        const delta = nowMs - lastFrameTime;
        if (delta > 0) {
          // Exponential moving average: an instantaneous reading flickers
          // by several frames per second and is unreadable.
          fps = fps === 0 ? 1000 / delta : fps * 0.9 + (1000 / delta) * 0.1;
        }
      }
      lastFrameTime = nowMs;
      // The label is text in the layout; rewriting it sixty times a second
      // would be sixty style recalculations for a number nobody can read.
      if (nowMs - lastStatusAt > 400) {
        lastStatusAt = nowMs;
        if (found) {
          setStatus(fps > 0 ? `Live at ${Math.round(fps)} fps` : "Live", "live");
        } else {
          setStatus("Looking for a face", "warn");
        }
      }
      markDrawn();
      finishCapture();
    }
    schedule(id);
  }

  type VideoFrameCapable = HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number) => void) => number;
    cancelVideoFrameCallback?: (handle: number) => void;
  };

  function schedule(id: number): void {
    if (!running || id !== loopId) {
      return;
    }
    const withVideoFrame = video as VideoFrameCapable;
    if (typeof withVideoFrame.requestVideoFrameCallback === "function") {
      usingVideoFrames = true;
      loopHandle = withVideoFrame.requestVideoFrameCallback((now) => {
        void frame(now, id);
      });
    } else {
      usingVideoFrames = false;
      loopHandle = requestAnimationFrame((now) => {
        void frame(now, id);
      });
    }
  }

  function startLoop(): void {
    loopId++;
    running = true;
    schedule(loopId);
  }

  function stopLoop(): void {
    running = false;
    loopId++;
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

  function closeCamera(): void {
    cameraTicket++;
    stopLoop();
    stopCamera(video);
  }

  // ---- sources -----------------------------------------------------------

  function syncBar(): void {
    backBtn.hidden = source === "sample";
    steadyBtn.hidden = source !== "camera";
  }

  function setSource(next: Source): void {
    source = next;
    stage.dataset.source = next;
    if (next !== "sample") {
      // The reveal and its start button belong to the sample: neither may
      // stay on top of a camera feed or someone's photo, whenever that
      // arrives.
      cancelReveal();
      startBtn.hidden = true;
    }
    syncBar();
    options.onSourceChange(next);
  }

  function showStill(which: "sample" | "photo"): void {
    const scene = which === "sample" ? sampleScene : photoScene;
    if (!scene) {
      return;
    }
    closeCamera();
    setSource(which);
    aimAt(scene);
    blend = null;
    shown = target;
    applyCompare(true);
    restingStatus();
    requestDraw();
  }

  function useCamera(): Promise<void> {
    if (!cameraStart) {
      cameraStart = openCamera().finally(() => {
        cameraStart = null;
      });
    }
    return cameraStart;
  }

  async function openCamera(): Promise<void> {
    cancelReveal();
    hideNotice();
    if (!ensureRenderer()) {
      showNotice(
        "The live camera needs WebGL2, which this browser doesn't offer. Photos still work.",
        [{ label: "Use a photo", run: choosePhoto, primary: true }],
      );
      return;
    }
    if (source === "camera" && running) {
      return;
    }
    const ticket = ++cameraTicket;
    setStatus("Waiting for the camera", "busy");
    try {
      const size = await startCamera(video);
      if (ticket !== cameraTicket) {
        // Granted after the reader had already moved on to a photo.
        stopCamera(video);
        return;
      }
      setSource("camera");
      filter.reset();
      fps = 0;
      lastFrameTime = 0;
      lastStatusAt = 0;
      aimAt(null);
      applyCompare(false);
      sizeCanvas(size.width, size.height);
      setStatus("Starting the camera", "busy");
      startLoop();
    } catch {
      if (ticket !== cameraTicket) {
        return;
      }
      settleStatus();
      showNotice(
        "The camera is blocked or busy in another app. Allow it in your browser's site settings and try again, or use a photo.",
        [
          { label: "Try again", run: () => void useCamera(), primary: true },
          { label: "Use a photo", run: choosePhoto },
        ],
      );
    }
  }

  function choosePhoto(): void {
    fileInput.click();
  }

  async function usePhotoFile(file: File): Promise<void> {
    if (!file.type.startsWith("image/")) {
      showNotice("That file isn't an image. Try a JPEG or PNG photo.", [
        { label: "Choose a photo", run: choosePhoto, primary: true },
      ]);
      return;
    }
    cancelReveal();
    hideNotice();
    setStatus("Reading your photo", "busy");
    try {
      await sharedLandmarker();
    } catch {
      showModelError();
      return;
    }
    try {
      const picture = await loadPhoto(file);
      const scene = await analyzePhoto(picture, picture.width, picture.height);
      if (!scene) {
        settleStatus();
        showNotice("No face found in that photo. A clear, front-facing photo works best.", [
          { label: "Try another photo", run: choosePhoto, primary: true },
        ]);
        return;
      }
      photoScene = scene;
      showStill("photo");
    } catch {
      settleStatus();
      showNotice("That photo couldn't be opened. Try a JPEG or PNG.", [
        { label: "Choose a photo", run: choosePhoto, primary: true },
      ]);
    }
  }

  // ---- the sample and its reveal -------------------------------------------

  function start(): Promise<void> {
    if (!booting) {
      booting = bootSample().finally(() => {
        booting = null;
      });
    }
    return booting;
  }

  async function bootSample(): Promise<void> {
    if (sampleScene) {
      showStill("sample");
      return;
    }
    startBtn.hidden = true;
    hideNotice();
    stage.dataset.phase = "loading";
    if (source === "sample") {
      setStatus("Loading the face model", "busy");
    }

    let image: HTMLImageElement;
    try {
      [image] = await Promise.all([loadImageElement(SAMPLE_PORTRAIT_URL), sharedLandmarker()]);
    } catch {
      showModelError();
      return;
    }

    let scene: PhotoScene | null;
    try {
      scene = await analyzePhoto(image, image.naturalWidth, image.naturalHeight);
    } catch {
      showModelError();
      return;
    }
    if (!scene) {
      stage.dataset.phase = "idle";
      status.hidden = true;
      showNotice("The face in the sample photo wasn't found. Try your camera or a photo instead.", []);
      return;
    }
    sampleScene = scene;
    stage.dataset.phase = "ready";
    options.onSampleReady(scene);
    // Someone who went straight for the camera or a photo while the model
    // loaded has already moved on; the reveal is only for the sample.
    if (source === "sample") {
      await reveal(scene);
    }
  }

  /** Runs `step(t)` for t from 0 to 1 over `duration`; false if cancelled. */
  function animate(duration: number, step: (t: number) => void, token: number): Promise<boolean> {
    return new Promise((resolve) => {
      const begin = performance.now();
      const tick = (now: number) => {
        if (token !== revealToken) {
          resolve(false);
          return;
        }
        const t = Math.min((now - begin) / duration, 1);
        step(t);
        if (t < 1) {
          requestAnimationFrame(tick);
        } else {
          resolve(true);
        }
      };
      requestAnimationFrame(tick);
    });
  }

  async function reveal(scene: PhotoScene): Promise<void> {
    const token = ++revealToken;
    setSource("sample");
    aimAt(scene);
    splitPos = 0.5;

    if (prefersReducedMotion() || !ensureRenderer()) {
      blend = null;
      shown = target;
      applyCompare(true);
      restingStatus();
      return;
    }

    revealing = true;
    // The canvas takes over from the poster showing the same bare picture,
    // so the swap cannot be seen and everything after it is the engine.
    applyCompare(false);
    blend = null;
    shown = withoutMakeup(target);
    paintScene(scene, shown);
    const art = revealArt(scene);

    setStatus(`Found the face: ${FACTS.landmarks} points`, "ready");
    if (!(await animate(REVEAL_MS.scan, (t) => art.draw(t, 1, 0), token))) {
      return;
    }
    setStatus("Drawing a soft mask for each product", "ready");
    if (!(await animate(REVEAL_MS.masks, (t) => art.draw(1, 1 - 0.55 * t, easeInOut(t)), token))) {
      return;
    }
    setStatus("Changing the colour, keeping the texture", "ready");
    applyLook(target, true, REVEAL_MS.tint);
    const tinted = await animate(
      REVEAL_MS.tint,
      (t) => art.draw(1, 0.45 * (1 - t), 1 - easeInOut(t)),
      token,
    );
    if (!tinted) {
      return;
    }
    art.clear();
    splitPos = 0;
    applyCompare(true);
    const wiped = await animate(
      REVEAL_MS.wipe,
      (t) => {
        splitPos = 0.5 * easeInOut(t);
        positionWipe();
        requestDraw();
      },
      token,
    );
    if (!wiped) {
      return;
    }
    revealing = false;
    restingStatus();
    wipe.classList.add("wipe--hint");
  }

  /** Stops the reveal where it is and settles on the resting state. */
  function cancelReveal(): void {
    if (!revealing) {
      return;
    }
    revealing = false;
    revealToken++;
    clearOverlay();
    // Stopped before the colour went on, the face would stay bare: blend
    // the look in from wherever the reveal left it.
    if (!blend) {
      applyLook(target, true);
    }
    if (source === "sample") {
      splitPos = 0.5;
      applyCompare(true);
      restingStatus();
    }
  }

  function clearOverlay(): void {
    overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  }

  /**
   * The reveal's drawing: the landmark mesh rippling out from the middle of
   * the face, and every mask the look will paint, each in its own shade.
   * Both are placed with the same cover fit the canvas is drawn with, so
   * they sit exactly on the features they belong to.
   */
  function revealArt(scene: PhotoScene): { draw(appear: number, dots: number, masks: number): void; clear(): void } {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = stage.getBoundingClientRect();
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = overlay.getContext("2d");
    if (!ctx) {
      return { draw() {}, clear() {} };
    }

    const toBuffer = canvas.width / scene.width;
    const fit = fitNow();

    const BUCKETS = 16;
    const buckets: number[][] = Array.from({ length: BUCKETS }, () => []);
    const edges: number[][] = Array.from({ length: BUCKETS }, () => []);
    const reach = Math.hypot(scene.face.width, scene.face.height) / 2 || 1;
    const onStage = new Float32Array(scene.landmarks.length);
    for (let i = 0; i + 1 < scene.landmarks.length; i += 2) {
      const x = scene.landmarks[i];
      const y = scene.landmarks[i + 1];
      const [bx, by] = contentToBox(fit, x * toBuffer, y * toBuffer);
      onStage[i] = bx * dpr;
      onStage[i + 1] = by * dpr;
      const distance = Math.min(Math.hypot(x - scene.face.cx, y - scene.face.cy) / reach, 0.999);
      buckets[Math.floor(distance * BUCKETS)].push(bx * dpr, by * dpr);
    }
    // The mesh the landmarker fits, edge by edge, sorted into the same
    // ripple so lines and points arrive together.
    for (const edge of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
      const mx = (scene.landmarks[edge.start * 2] + scene.landmarks[edge.end * 2]) / 2;
      const my = (scene.landmarks[edge.start * 2 + 1] + scene.landmarks[edge.end * 2 + 1]) / 2;
      const distance = Math.min(Math.hypot(mx - scene.face.cx, my - scene.face.cy) / reach, 0.999);
      edges[Math.floor(distance * BUCKETS)].push(edge.start * 2, edge.end * 2);
    }

    const { masks } = scene;
    const maskArt = document.createElement("canvas");
    maskArt.width = masks.width;
    maskArt.height = masks.height;
    const maskCtx = maskArt.getContext("2d");
    if (maskCtx) {
      const image = maskCtx.createImageData(masks.width, masks.height);
      const px = image.data;
      for (const name of PRODUCT_ORDER) {
        const mask = masks.masks[name];
        if (!mask || target[name].intensity <= 0) {
          continue;
        }
        const [r, g, b] = hexToRgb(target[name].color);
        for (let i = 0; i < mask.length; i++) {
          const a = mask[i] * 0.85;
          if (a < 0.01) {
            continue;
          }
          const p = i * 4;
          const under = px[p + 3] / 255;
          const outA = a + under * (1 - a);
          px[p] = (r * a + px[p] * under * (1 - a)) / outA;
          px[p + 1] = (g * a + px[p + 1] * under * (1 - a)) / outA;
          px[p + 2] = (b * a + px[p + 2] * under * (1 - a)) / outA;
          px[p + 3] = outA * 255;
        }
      }
      maskCtx.putImageData(image, 0, 0);
    }
    const maskScale = (toBuffer / masks.scale) * fit.scale * dpr;
    const radius = 1.2 * dpr;

    return {
      draw(appear, dots, maskAlpha) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (maskAlpha > 0) {
          ctx.globalAlpha = maskAlpha;
          ctx.imageSmoothingQuality = "high";
          ctx.drawImage(
            maskArt,
            fit.offsetX * dpr,
            fit.offsetY * dpr,
            masks.width * maskScale,
            masks.height * maskScale,
          );
        }
        if (dots > 0) {
          // Lit like a scan: the mesh adds light to the picture under it.
          ctx.globalCompositeOperation = "lighter";
          ctx.lineWidth = 0.75 * dpr;
          ctx.strokeStyle = "#f0d3ae";
          for (let b = 0; b < BUCKETS; b++) {
            const local = Math.min(Math.max((appear - (b / BUCKETS) * 0.6) / 0.4, 0), 1);
            const alpha = local * dots;
            if (alpha <= 0) {
              continue;
            }
            const list = edges[b];
            ctx.globalAlpha = alpha * 0.42;
            ctx.beginPath();
            for (let i = 0; i < list.length; i += 2) {
              ctx.moveTo(onStage[list[i]], onStage[list[i] + 1]);
              ctx.lineTo(onStage[list[i + 1]], onStage[list[i + 1] + 1]);
            }
            ctx.stroke();
            const points = buckets[b];
            const r = radius * (0.6 + 0.4 * local);
            ctx.globalAlpha = alpha;
            ctx.fillStyle = "#fff6ec";
            ctx.beginPath();
            for (let i = 0; i < points.length; i += 2) {
              ctx.moveTo(points[i] + r, points[i + 1]);
              ctx.arc(points[i], points[i + 1], r, 0, Math.PI * 2);
            }
            ctx.fill();
          }
          ctx.globalCompositeOperation = "source-over";
        }
        ctx.globalAlpha = 1;
      },
      clear: clearOverlay,
    };
  }

  // ---- wiring ------------------------------------------------------------

  startBtn.addEventListener("click", () => void start());
  backBtn.addEventListener("click", () => {
    hideNotice();
    if (sampleScene) {
      showStill("sample");
    } else {
      closeCamera();
      setSource("sample");
      void start();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) {
      void usePhotoFile(file);
    }
    fileInput.value = "";
  });

  // Drop a photo anywhere on the mirror.
  stage.addEventListener("dragover", (event) => {
    event.preventDefault();
    stage.dataset.dragging = "true";
  });
  stage.addEventListener("dragleave", () => {
    delete stage.dataset.dragging;
  });
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    delete stage.dataset.dragging;
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      void usePhotoFile(file);
    }
  });

  steadyBtn.addEventListener("click", () => {
    steady = !steady;
    steadyBtn.setAttribute("aria-pressed", String(steady));
    filter = new OneEuroFilter();
  });

  compareBtn.addEventListener("click", () => {
    // Read first: cancelling the reveal settles on the split, and the
    // click is about the state the reader saw.
    const wasOn = compare;
    cancelReveal();
    if (!wasOn) {
      splitPos = 0.5;
    }
    applyCompare(!wasOn);
  });

  saveBtn.addEventListener("click", () => {
    pendingCapture = true;
    requestDraw();
  });

  let dragging = false;
  function moveWipe(clientX: number): void {
    const rect = stage.getBoundingClientRect();
    splitPos = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    positionWipe();
    requestDraw();
  }

  wipe.addEventListener("pointerdown", (event) => {
    cancelReveal();
    wipe.classList.remove("wipe--hint");
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
    let next = splitPos;
    if (event.key === "ArrowLeft") {
      next -= step;
    } else if (event.key === "ArrowRight") {
      next += step;
    } else if (event.key === "Home") {
      next = splitMin();
    } else if (event.key === "End") {
      next = 1;
    } else {
      return;
    }
    event.preventDefault();
    cancelReveal();
    wipe.classList.remove("wipe--hint");
    splitPos = Math.min(Math.max(next, 0), 1);
    positionWipe();
    requestDraw();
  });

  // The crop follows the face, and the right crop depends on the stage's
  // shape, which changes with the window.
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      aimAt(currentScene());
      requestDraw();
    }).observe(stage);
  }

  positionWipe();
  syncBar();
  if (!ensureRenderer()) {
    // No WebGL2: photos still work through the CPU reference path.
    stage.dataset.gl = "none";
  }
  if (options.autostart) {
    setTimeout(() => void start(), 0);
  } else {
    startBtn.hidden = false;
  }

  return {
    stage,
    tools: bar,
    setLook(next, animate) {
      cancelReveal();
      applyLook(next, animate);
    },
    start: () => void start(),
    useCamera: () => void useCamera(),
    choosePhoto,
    featureY(product) {
      const scene = currentScene();
      const landmarks = source === "camera" ? liveLandmarks : (scene?.landmarks ?? null);
      const contentHeight = source === "camera" ? video.videoHeight : (scene?.height ?? 0);
      if (!landmarks || contentHeight <= 0 || canvas.height <= 0) {
        return null;
      }
      let sum = 0;
      for (const index of FEATURE_POINTS[product]) {
        sum += landmarks[index * 2 + 1];
      }
      const y = sum / FEATURE_POINTS[product].length;
      const toBuffer = canvas.height / contentHeight;
      return contentToBox(fitNow(), 0, y * toBuffer)[1];
    },
  };
}
