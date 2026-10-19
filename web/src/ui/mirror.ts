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
 * The sample's landmarks ship with the page (see sample.ts), so it can be
 * painted the moment the page loads; the face model is only downloaded for
 * a camera or a photo. Its first appearance plays the pipeline back in slow
 * motion -- the landmark points assembling into the face's mesh in 3D, then
 * the photo, then the masks, then the colour -- so the opening seconds
 * explain what the engine does instead of only showing the result. Anyone
 * who asks for reduced motion gets the result directly.
 */

import { loadPhoto, startCamera, stopCamera } from "../lib/camera";
import { createRenderer, type Renderer } from "../engine/renderer";
import { OneEuroFilter } from "../engine/oneEuro";
import { hexToRgb } from "../engine/color";
import { PRODUCT_ORDER, type LookConfig } from "../engine/look";
import type { Landmarker } from "../lib/landmarks";
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
import { analyzePhoto, glossFor, sceneFromLandmarks, type PhotoScene } from "./scene";
import { SAMPLE } from "./sample";
import { centroid, project } from "./mesh3d";
import { drawMesh } from "./meshArt";
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
const REVEAL_MS = { scan: 2400, masks: 900, tint: 1300, wipe: 800 };

/** How long a new shade takes to sweep across a still, like a brush. */
const SWEEP_MS = 950;

const EMPTY_MASKS: MaskSet = {
  quality: "exact",
  width: 1,
  height: 1,
  scale: 1,
  interocular: 0,
  masks: {},
};

export type Source = "sample" | "camera" | "photo";

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
  /** Starts downloading the face model, so a camera or a photo the visitor is
   * about to ask for opens without the wait. Safe to call repeatedly. */
  warmUp(): void;
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
  // The outgoing look, held over the new one while a shade change sweeps
  // across the face, and the soft light that travels with the sweep.
  const sweep = el("canvas", "stage__sweep");
  sweep.setAttribute("aria-hidden", "true");
  const brush = el("span", "stage__brush");
  brush.setAttribute("aria-hidden", "true");
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

  stage.append(poster, video, canvas, sweep, brush, overlay, wipe, status, notice, startBtn, fileInput);

  // The mirror's own buttons, for the page to place. The ones that only mean
  // something for one source -- back to the sample, smoothing for the camera
  // -- appear only when they apply.
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
  bar.append(backBtn, steadyBtn, compareBtn, saveBtn);

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
  /** The ticket `cameraStart` was opened with. */
  let cameraStartTicket = 0;
  /** Bumped by every request for a source -- the camera, a photo, the
   * sample -- so work that finishes after the reader has moved on (a camera
   * granted late, a photo still being read) is dropped instead of shown. */
  let sourceTicket = 0;
  let usingVideoFrames = false;
  let fps = 0;
  let lastFrameTime = 0;
  let lastStatusAt = 0;
  let pendingCapture = false;
  let drawHandle = 0;
  let revealing = false;
  let revealToken = 0;
  /** The face model, once it has loaded. */
  let landmarker: Landmarker | null = null;
  let sweepHandle = 0;
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

  /** @param retry Asks again for whatever needed the model. */
  function showModelError(retry: () => void): void {
    status.hidden = true;
    stage.dataset.phase = sampleScene ? "ready" : "idle";
    showNotice("The face model didn't load. Check your connection and try again.", [
      { label: "Try again", run: retry, primary: true },
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
    // Where in the stage the face should sit: the page sets it, because the
    // headline covers part of the picture on a wide screen.
    const styles = getComputedStyle(stage);
    const aimX = parseFloat(styles.getPropertyValue("--face-x")) || 0.5;
    const aimY = parseFloat(styles.getPropertyValue("--face-y")) || 0.5;
    focus =
      scene && rect.width > 0 && rect.height > 0
        ? focusOn(rect.width, rect.height, scene.width, scene.height, scene.face.cx, scene.face.cy, aimX, aimY)
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
  const wideLayout = window.matchMedia?.("(min-width: 62.5rem)");
  function splitMin(): number {
    // The picture's left edge fades into the dark on a wide screen.
    return wideLayout?.matches ? 0.16 : 0;
  }

  function positionWipe(): void {
    splitPos = Math.max(splitPos, splitMin());
    const pct = `${splitPos * 100}%`;
    wipe.style.setProperty("--split", pct);
    wipe.setAttribute("aria-valuenow", String(Math.round(splitPos * 100)));
    // A label with no room beside the line would hang off the stage.
    wipe.dataset.edge = splitPos - splitMin() < 0.12 ? "left" : splitPos > 0.88 ? "right" : "";
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

  /**
   * Puts a look on the face. A fade blends every frame from the old look to
   * the new; a sweep carries the new look across a still like a brush. The
   * CPU path takes hundreds of milliseconds a draw, so without WebGL2 a
   * change lands in one step, as it does for anyone who asks for less motion.
   */
  function applyLook(next: LookConfig, transition: "cut" | "fade" | "sweep", duration = BLEND_MS): void {
    target = copyLook(next);
    const moving = transition !== "cut" && renderer !== null && !prefersReducedMotion();
    if (moving && transition === "sweep" && source !== "camera" && currentScene()) {
      blend = null;
      sweepTo(target);
      return;
    }
    // A sweep still crossing the picture would hide this change under its
    // copy of an older look.
    stopSweep();
    if (moving) {
      blend = { from: shown, to: target, start: performance.now(), duration };
    } else {
      blend = null;
      shown = target;
    }
    requestDraw();
  }

  function stopSweep(): void {
    cancelAnimationFrame(sweepHandle);
    sweepHandle = 0;
    delete stage.dataset.sweeping;
  }

  /** Holds a copy of the outgoing look over the new one and wipes it away. */
  function sweepTo(next: LookConfig): void {
    const scene = currentScene();
    if (!scene) {
      return;
    }
    stopSweep();
    paintScene(scene, shown);
    if (sweep.width !== canvas.width || sweep.height !== canvas.height) {
      sweep.width = canvas.width;
      sweep.height = canvas.height;
    }
    const sweepCtx = sweep.getContext("2d");
    // Copied in the same task as the draw: without preserveDrawingBuffer the
    // drawing buffer is only kept until the browser next composites the page.
    sweepCtx?.clearRect(0, 0, sweep.width, sweep.height);
    sweepCtx?.drawImage(canvas, 0, 0);
    sweep.style.objectPosition = canvas.style.objectPosition;
    shown = next;
    paintScene(scene, shown);
    stage.style.setProperty("--sweep", "-12%");
    stage.dataset.sweeping = "true";
    const begin = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - begin) / SWEEP_MS, 1);
      stage.style.setProperty("--sweep", `${(-12 + easeInOut(t) * 124).toFixed(2)}%`);
      if (t < 1) {
        sweepHandle = requestAnimationFrame(tick);
      } else {
        stopSweep();
      }
    };
    sweepHandle = requestAnimationFrame(tick);
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
    faceModel: Landmarker,
    look: LookConfig,
  ): boolean {
    const active = ensureRenderer();
    if (!active) {
      return false;
    }
    sizeCanvas(width, height);

    let landmarks = faceModel.detect(video, timestampMs);
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
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!landmarker) {
      // The feed shows straight away; the makeup joins it once the face
      // model has arrived.
      const active = ensureRenderer();
      if (active && width > 0 && height > 0) {
        sizeCanvas(width, height);
        active.render(video, EMPTY_MASKS, lookAt(nowMs), {}, { mirror: true, splitX: splitPixels() });
        markDrawn();
      }
      setStatus("Loading the face model", "busy");
      schedule(id);
      return;
    }

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
    sourceTicket++;
    stopLoop();
    stopCamera(video);
  }

  // ---- sources -----------------------------------------------------------

  function syncBar(): void {
    backBtn.hidden = source === "sample";
    steadyBtn.hidden = source !== "camera";
  }

  function setSource(next: Source): void {
    // A sweep is a copy of the picture that was showing; it cannot cross
    // another one.
    stopSweep();
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

  /** Loads the face model once; failures surface where it was needed. */
  function loadModel(): Promise<Landmarker> {
    return sharedLandmarker().then((loaded) => {
      landmarker = loaded;
      return loaded;
    });
  }

  function useCamera(): Promise<void> {
    // A request overtaken while its permission prompt was still open no
    // longer stands in for a new one.
    if (!cameraStart || cameraStartTicket !== sourceTicket) {
      const request = openCamera().finally(() => {
        if (cameraStart === request) {
          cameraStart = null;
        }
      });
      cameraStart = request;
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
    const ticket = ++sourceTicket;
    cameraStartTicket = ticket;
    setStatus("Waiting for the camera", "busy");
    // The model downloads while the browser asks for the camera. If it
    // fails, this request is over, whether the camera is on yet or its
    // permission prompt is still open: closing bumps the ticket, so a stream
    // granted afterwards is stopped on arrival.
    loadModel().catch(() => {
      if (ticket !== sourceTicket) {
        return;
      }
      closeCamera();
      if (sampleScene) {
        showStill("sample");
      }
      showModelError(() => void useCamera());
    });
    try {
      const size = await startCamera(video);
      if (ticket !== sourceTicket) {
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
      if (ticket !== sourceTicket) {
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
    // A camera or the sample asked for after this photo wins over it.
    const ticket = ++sourceTicket;
    cancelReveal();
    hideNotice();
    setStatus("Reading your photo", "busy");
    try {
      await loadModel();
    } catch {
      if (ticket === sourceTicket) {
        if (source === "camera" && sampleScene) {
          showStill("sample");
        }
        showModelError(() => void usePhotoFile(file));
      }
      return;
    }
    if (ticket !== sourceTicket) {
      return;
    }
    try {
      const picture = await loadPhoto(file);
      const scene = await analyzePhoto(picture, picture.width, picture.height);
      if (ticket !== sourceTicket) {
        return;
      }
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
      if (ticket !== sourceTicket) {
        return;
      }
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

    let image: HTMLImageElement;
    try {
      image = await loadImageElement(SAMPLE_PORTRAIT_URL);
    } catch {
      stage.dataset.phase = "idle";
      stage.dataset.intro = "done";
      status.hidden = true;
      showNotice("The sample photo didn't load. Check your connection and try again.", [
        { label: "Try again", run: () => void start(), primary: true },
      ]);
      return;
    }

    // The landmarks were found on this exact file; were it ever served at
    // another size, they scale with it.
    const sx = image.naturalWidth / SAMPLE.width;
    const sy = image.naturalHeight / SAMPLE.height;
    const xy = Float32Array.from(SAMPLE.xy, (v, i) => v * (i % 2 === 0 ? sx : sy));
    const depth = Float32Array.from(SAMPLE.z, (v) => v * sx);
    const scene = sceneFromLandmarks(image, image.naturalWidth, image.naturalHeight, xy, depth);
    sampleScene = scene;
    stage.dataset.phase = "ready";
    options.onSampleReady(scene);
    // Someone who went straight for the camera or a photo has already moved
    // on; the reveal is only for the sample.
    if (source === "sample") {
      await reveal(scene);
    } else {
      stage.dataset.intro = "done";
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

    if (prefersReducedMotion() || !ensureRenderer() || !scene.depth) {
      blend = null;
      shown = target;
      stage.dataset.intro = "done";
      applyCompare(true);
      restingStatus();
      return;
    }

    revealing = true;
    applyCompare(false);
    blend = null;
    shown = withoutMakeup(target);
    paintScene(scene, shown);
    // The picture stays dark while the mesh assembles in front of it.
    stage.dataset.intro = "mesh";
    const art = revealArt(scene);

    setStatus(`Mapping the face: ${FACTS.landmarks} points`, "ready");
    if (!(await animate(REVEAL_MS.scan, (t) => art.draw(t, 1, 0), token))) {
      return;
    }
    stage.dataset.intro = "photo";
    setStatus("Shaping a soft mask for each product", "ready");
    if (!(await animate(REVEAL_MS.masks, (t) => art.draw(1, 1 - 0.45 * t, easeInOut(t)), token))) {
      return;
    }
    setStatus("Changing the colour, keeping the texture", "ready");
    applyLook(target, "fade", REVEAL_MS.tint);
    const tinted = await animate(
      REVEAL_MS.tint,
      (t) => art.draw(1, 0.55 * (1 - t), 1 - easeInOut(t)),
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
    stage.dataset.intro = "done";
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
    stage.dataset.intro = "done";
    // Stopped before the colour went on, the face would stay bare: blend
    // the look in from wherever the reveal left it.
    if (!blend) {
      applyLook(target, "fade");
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
   * The reveal's drawing. The face's 478 landmarks fly in from the dark, the
   * centre first, and assemble into its mesh while it turns from a
   * three-quarter view to face the camera -- the depth the model gives each
   * point is what makes the turn solid. As it settles, the perspective eases
   * away, so every point lands exactly on the photo underneath. Then every
   * mask the look will paint glows in its own shade. Everything is placed
   * with the same cover fit the canvas is drawn with.
   */
  function revealArt(scene: PhotoScene): { draw(build: number, mesh: number, masks: number): void; clear(): void } {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = stage.getBoundingClientRect();
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = overlay.getContext("2d");
    if (!ctx || !scene.depth) {
      return { draw() {}, clear() {} };
    }

    const toBuffer = canvas.width / scene.width;
    const fit = fitNow();
    const toOverlay = toBuffer * fit.scale * dpr;
    const count = scene.landmarks.length / 2;
    const finalXY = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const [bx, by] = contentToBox(fit, scene.landmarks[i * 2] * toBuffer, scene.landmarks[i * 2 + 1] * toBuffer);
      finalXY[i * 2] = bx * dpr;
      finalXY[i * 2 + 1] = by * dpr;
    }
    const center = centroid(finalXY);
    const faceSize = Math.max(scene.face.width, scene.face.height) * toOverlay;

    // Where each point starts, scattered through the dark around the face,
    // and when it sets off: a fixed seed, so every visit plays the same.
    let seed = 0x5eed;
    const random = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const fromX = new Float32Array(count);
    const fromY = new Float32Array(count);
    const fromZ = new Float32Array(count);
    const delay = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const angle = random() * Math.PI * 2;
      const lift = random() * 2 - 1;
      const radius = faceSize * (0.9 + random() * 1.1);
      fromX[i] = Math.cos(angle) * Math.sqrt(1 - lift * lift) * radius;
      fromY[i] = lift * radius * 0.8;
      fromZ[i] = Math.sin(angle) * radius;
      const dx = finalXY[i * 2] - center.cx;
      const dy = finalXY[i * 2 + 1] - center.cy;
      delay[i] = Math.min(Math.hypot(dx, dy) / (faceSize * 0.6), 1) * 0.32 + random() * 0.08;
    }

    const xy = new Float32Array(count * 2);
    const z = new Float32Array(count);
    const projected = new Float32Array(count * 2);
    const depthOut = new Float32Array(count);
    const arrival = new Float32Array(count);
    const unit = dpr * Math.min(Math.max(faceSize / dpr / 520, 0.8), 1.7);

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
    const easeOut = (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
    const ramp = (t: number, a: number, b: number) => Math.min(Math.max((t - a) / (b - a), 0), 1);

    return {
      draw(build, mesh, maskAlpha) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (maskAlpha > 0) {
          ctx.globalAlpha = maskAlpha;
          ctx.imageSmoothingQuality = "high";
          ctx.drawImage(maskArt, fit.offsetX * dpr, fit.offsetY * dpr, masks.width * maskScale, masks.height * maskScale);
          ctx.globalAlpha = 1;
        }
        if (mesh <= 0) {
          return;
        }
        for (let i = 0; i < count; i++) {
          const local = ramp(build, delay[i], delay[i] + 0.55);
          const e = easeOut(local);
          arrival[i] = local;
          const tx = finalXY[i * 2] - center.cx;
          const ty = finalXY[i * 2 + 1] - center.cy;
          const tz = scene.depth![i] * toOverlay;
          xy[i * 2] = center.cx + fromX[i] + (tx - fromX[i]) * e;
          xy[i * 2 + 1] = center.cy + fromY[i] + (ty - fromY[i]) * e;
          z[i] = fromZ[i] + (tz - fromZ[i]) * e;
        }
        const settle = easeInOut(ramp(build, 0.15, 1));
        const yaw = -0.8 * (1 - settle);
        const pitch = 0.12 * (1 - settle);
        const perspective = 1 - ramp(build, 0.72, 1);
        project(xy, z, center, yaw, pitch, faceSize * 2.4, perspective, projected, depthOut);
        drawMesh(ctx, {
          points: projected,
          depth: depthOut,
          arrival,
          edgeAlpha: mesh * ramp(build, 0.3, 0.75),
          pointAlpha: mesh,
          unit,
        });
      },
      clear: clearOverlay,
    };
  }

  // ---- wiring ------------------------------------------------------------

  startBtn.addEventListener("click", () => void start());
  // The intro states start dark only once the script is in charge; until
  // then the static page shows the sample as it is.
  stage.dataset.intro = options.autostart ? "pending" : "done";
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
      // A still gets the brush sweep; the camera blends, since its picture
      // changes every frame anyway.
      applyLook(next, animate ? (source === "camera" ? "fade" : "sweep") : "cut");
    },
    start: () => void start(),
    useCamera: () => void useCamera(),
    choosePhoto,
    warmUp() {
      if (!landmarker) {
        loadModel().catch(() => {
          // Reported when the model is actually needed.
        });
      }
    },
  };
}
