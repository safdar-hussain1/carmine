/**
 * How it works, told by the engine as you scroll.
 *
 * One picture stays pinned while four short chapters scroll past it, and
 * the picture is driven by how far through its chapter the reader is:
 *
 * 1. Find the face: the sample's 478 landmarks as a mesh in 3D, turning as
 *    you scroll (and under a drag), drawn from the depth the model gives
 *    every point.
 * 2. Shape soft masks: the look taken apart, one sheet per product floating
 *    over the face in the order the shader paints them.
 * 3. Change the colour, keep the texture: the lip's real pixels as a cloud
 *    in colour space, morphing from the untouched lip to a flat fill (the
 *    whole cloud lifted toward white) to Carmine (moved in colour, kept in
 *    lightness).
 * 4. Paint it in one pass: the sheets fall back together and turn to face
 *    you, and become the engine's finished render.
 *
 * Everything is made in the browser from the sample the mirror already
 * painted -- nothing here is a prepared image or video, so the section
 * cannot describe a pipeline the page does not actually run. The lipstick
 * in chapter 3 is the one the photo benchmark used, the velvet look's at a
 * satin finish, so the cloud and the measured lightness shift quoted beside
 * it describe the same comparison.
 */

import { animate } from "motion";
import { hexToRgb } from "../engine/color";
import { lightnessPullFor, PRESETS, type LookConfig } from "../engine/look";
import { buildMasks } from "../engine/masks";
import constants from "../gen/constants.json";
import { buildCloud } from "./colorCloud";
import { createCloudView, type CloudState, type CloudView, type Weights } from "./cloudArt";
import { createLayerStack, type LayerStack } from "./layers";
import { centroid, project } from "./mesh3d";
import { drawMesh } from "./meshArt";
import { cropAround, cropTo, renderLook } from "./offscreen";
import { toFloatPixels, toProcessingCanvas } from "./pipeline";
import type { PhotoScene } from "./scene";
import { easeInOut, withoutMakeup } from "./tween";

type Step = "landmarks" | "masks" | "texture" | "final";
const STEPS: Step[] = ["landmarks", "masks", "texture", "final"];

/** The look the sheets show: every product on, each in a named shade, so
 * all six sheets have something on them. */
const SHOWCASE: LookConfig = {
  blush: { color: "#D96C6C", intensity: 0.35, finish: "satin" },
  highlighter: { color: "#F5D9C8", intensity: 0.55, finish: "satin" },
  eyeshadow: { color: "#5C3A6E", intensity: 0.5, finish: "satin" },
  brows: { color: "#4A3728", intensity: 0.35, finish: "satin" },
  lipstick: { color: "#8E1B3A", intensity: 0.8, finish: "satin" },
  eyeliner: { color: "#1B1B1B", intensity: 0.75, finish: "satin" },
  smoothing: 0.25,
};

/** The photo benchmark's lipstick: velvet's, at a satin finish. */
function benchmarkLook(): LookConfig {
  const look = withoutMakeup(PRESETS.velvet);
  look.lipstick = { ...PRESETS.velvet.lipstick, finish: "satin" };
  return look;
}

function ramp(t: number, a: number, b: number): number {
  return Math.min(Math.max((t - a) / (b - a), 0), 1);
}

function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

function whenIdle(work: () => void): void {
  const idle = (window as Window & {
    requestIdleCallback?: (cb: () => void, options?: { timeout: number }) => number;
  }).requestIdleCallback;
  if (idle) {
    idle(work, { timeout: 1500 });
  } else {
    window.setTimeout(work, 1);
  }
}

export interface Story {
  setScene(scene: PhotoScene): void;
}

export function createStory(section: HTMLElement): Story {
  const stage = section.querySelector<HTMLElement>(".story__stage");
  const meshCanvas = section.querySelector<HTMLCanvasElement>(".scene--mesh");
  const layersHost = section.querySelector<HTMLElement>(".scene--layers");
  const cloudCanvas = section.querySelector<HTMLCanvasElement>(".cloud__canvas");
  const chips = [...section.querySelectorAll<HTMLButtonElement>(".cloud__chip")];
  const insets = new Map<CloudState, HTMLCanvasElement>();
  section.querySelectorAll<HTMLCanvasElement>(".cloud__inset canvas").forEach((canvas) => {
    insets.set(canvas.dataset.state as CloudState, canvas);
  });
  const chapters = [...section.querySelectorAll<HTMLElement>(".chapter")];
  const progress = [...section.querySelectorAll<HTMLElement>(".story__progress span")];
  const counter = section.querySelector<HTMLElement>("[data-count]");
  const reduced = prefersReducedMotion();
  const wide = window.matchMedia?.("(min-width: 62.5rem)");

  let scene: PhotoScene | null = null;
  let built = false;
  let near = false;
  let visible = false;
  let raf = 0;

  let active: Step | null = null;
  let progressNow = 0;
  let smooth = 0;

  // ---- 1: the mesh ----------------------------------------------------------

  let meshXY = new Float32Array(0);
  let meshZ = new Float32Array(0);
  let meshCenter = { cx: 0, cy: 0 };
  let meshProjected = new Float32Array(0);
  let meshDepth = new Float32Array(0);
  let meshFocal = 1;
  let meshUnit = 1;
  let dragYaw = 0;
  let dragging = false;
  let dragFrom = 0;
  let dragStartYaw = 0;

  function layoutMesh(): void {
    if (!meshCanvas || !scene?.depth) {
      return;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = meshCanvas.getBoundingClientRect();
    meshCanvas.width = Math.max(1, Math.round(rect.width * dpr));
    meshCanvas.height = Math.max(1, Math.round(rect.height * dpr));
    const count = scene.landmarks.length / 2;
    const face = centroid(scene.landmarks);
    const size = Math.max(scene.face.width, scene.face.height);
    const fit = (Math.min(meshCanvas.width, meshCanvas.height) * 0.72) / size;
    meshXY = new Float32Array(count * 2);
    meshZ = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      meshXY[i * 2] = meshCanvas.width / 2 + (scene.landmarks[i * 2] - face.cx) * fit;
      meshXY[i * 2 + 1] = meshCanvas.height / 2 + (scene.landmarks[i * 2 + 1] - face.cy) * fit;
      meshZ[i] = scene.depth[i] * fit;
    }
    meshCenter = { cx: meshCanvas.width / 2, cy: meshCanvas.height / 2 };
    meshProjected = new Float32Array(count * 2);
    meshDepth = new Float32Array(count);
    meshFocal = size * fit * 2.6;
    meshUnit = dpr * Math.min(Math.max(rect.width / 560, 0.85), 1.5);
  }

  function drawMeshScene(now: number, t: number): void {
    const ctx = meshCanvas?.getContext("2d");
    if (!meshCanvas || !ctx || meshZ.length === 0) {
      return;
    }
    if (!dragging) {
      dragYaw *= 0.92;
    }
    const sway = reduced ? 0 : Math.sin(now / 1400) * 0.05;
    const yaw = -0.85 + 1.7 * t + sway + dragYaw;
    const pitch = 0.1 + (reduced ? 0 : Math.sin(now / 2100) * 0.04);
    project(meshXY, meshZ, meshCenter, yaw, pitch, meshFocal, 1, meshProjected, meshDepth);
    ctx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
    drawMesh(ctx, { points: meshProjected, depth: meshDepth, edgeAlpha: 0.95, pointAlpha: 1, unit: meshUnit });
  }

  meshCanvas?.addEventListener("pointerdown", (event) => {
    dragging = true;
    dragFrom = event.clientX;
    dragStartYaw = dragYaw;
    meshCanvas.setPointerCapture(event.pointerId);
  });
  meshCanvas?.addEventListener("pointermove", (event) => {
    if (dragging) {
      dragYaw = dragStartYaw + (event.clientX - dragFrom) * 0.008;
    }
  });
  const endMeshDrag = () => {
    dragging = false;
  };
  meshCanvas?.addEventListener("pointerup", endMeshDrag);
  meshCanvas?.addEventListener("pointercancel", endMeshDrag);

  // ---- 2 and 4: the sheets ----------------------------------------------------

  let layers: LayerStack | null = null;
  let lastLayers = "";

  function setLayers(explode: number, tilt: number, finished: number): void {
    const key = `${explode.toFixed(3)}|${tilt.toFixed(3)}|${finished.toFixed(3)}`;
    if (key !== lastLayers) {
      lastLayers = key;
      layers?.set(explode, tilt, finished);
    }
  }

  // ---- 3: the colour cloud ---------------------------------------------------

  let cloudView: CloudView | null = null;
  let manual: { from: Weights; to: CloudState; start: number } | null = null;
  let shownWeights: Weights = { original: 1, flat: 0, carmine: 0 };

  function scrollWeights(t: number): Weights {
    const toFlat = easeInOut(ramp(t, 0.18, 0.42));
    const toCarmine = easeInOut(ramp(t, 0.58, 0.82));
    const flat = toFlat * (1 - toCarmine);
    return { original: 1 - flat - toCarmine, flat, carmine: toCarmine };
  }

  function weightsNow(now: number, t: number): Weights {
    if (!manual) {
      return scrollWeights(t);
    }
    const k = reduced ? 1 : easeInOut(Math.min((now - manual.start) / 700, 1));
    const goal: Weights = { original: 0, flat: 0, carmine: 0 };
    goal[manual.to] = 1;
    return {
      original: manual.from.original + (goal.original - manual.from.original) * k,
      flat: manual.from.flat + (goal.flat - manual.from.flat) * k,
      carmine: manual.from.carmine + (goal.carmine - manual.from.carmine) * k,
    };
  }

  function showWeights(weights: Weights, now: number): void {
    shownWeights = weights;
    const yaw = 0.75 + (reduced ? 0 : Math.sin(now / 3200) * 0.3);
    cloudView?.draw(weights, yaw, 0.38);
    const dominant = (Object.keys(weights) as CloudState[]).reduce((best, key) =>
      weights[key] > weights[best] ? key : best,
    );
    for (const chip of chips) {
      chip.setAttribute("aria-pressed", String(chip.dataset.state === dominant));
    }
    for (const [state, canvas] of insets) {
      canvas.style.opacity = String(weights[state]);
    }
  }

  for (const chip of chips) {
    chip.addEventListener("click", () => {
      manual = { from: shownWeights, to: chip.dataset.state as CloudState, start: performance.now() };
    });
  }

  // ---- building everything once ------------------------------------------------

  function buildCloudScene(s: PhotoScene): void {
    if (!cloudCanvas) {
      return;
    }
    const proc = toProcessingCanvas(s.source, s.width, s.height);
    const ctx = proc.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return;
    }
    const pixels = toFloatPixels(ctx.getImageData(0, 0, proc.width, proc.height));
    const lipMask = buildMasks(s.landmarks, s.width, s.height, new Set(["lipstick"]), "exact").masks.lipstick;
    if (!lipMask) {
      return;
    }
    const lips = benchmarkLook().lipstick;
    const cloud = buildCloud(
      pixels,
      lipMask,
      hexToRgb(lips.color),
      lips.intensity,
      lightnessPullFor("lipstick", lips.finish),
      1400,
    );
    cloudView = createCloudView(cloudCanvas, cloud);

    // The three lips, small, beside the cloud: what each state looks like.
    const outer = constants.regions.LIPS_OUTER;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const index of outer) {
      minX = Math.min(minX, s.landmarks[index * 2]);
      maxX = Math.max(maxX, s.landmarks[index * 2]);
      minY = Math.min(minY, s.landmarks[index * 2 + 1]);
      maxY = Math.max(maxY, s.landmarks[index * 2 + 1]);
    }
    const area = cropAround((minX + maxX) / 2, (minY + maxY) / 2, (maxX - minX) * 1.45, 1.6, s.width, s.height);
    const painted = renderLook(s, benchmarkLook());
    for (const [state, canvas] of insets) {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width * dpr));
      const height = Math.max(1, Math.round(rect.height * dpr));
      const picture = cropTo(state === "carmine" ? painted : s.source, area, width, height);
      if (state === "flat") {
        const flatCtx = picture.getContext("2d");
        if (flatCtx) {
          flatCtx.globalCompositeOperation = "lighter";
          flatCtx.globalAlpha = Math.min(1, lips.intensity);
          flatCtx.fillStyle = lips.color;
          flatCtx.beginPath();
          outer.forEach((index, i) => {
            const x = ((s.landmarks[index * 2] - area.x) / area.w) * width;
            const y = ((s.landmarks[index * 2 + 1] - area.y) / area.h) * height;
            if (i === 0) {
              flatCtx.moveTo(x, y);
            } else {
              flatCtx.lineTo(x, y);
            }
          });
          flatCtx.closePath();
          flatCtx.fill();
        }
      }
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d")?.drawImage(picture, 0, 0);
    }
  }

  function build(): void {
    if (built || !scene || !near) {
      return;
    }
    built = true;
    layoutMesh();
    if (layersHost) {
      layers = createLayerStack(scene, SHOWCASE);
      layersHost.append(layers.element);
      layers.layout();
    }
    buildCloudScene(scene);
    section.dataset.drawn = "true";
    start();
  }

  // ---- following the scroll ----------------------------------------------------

  function activate(step: Step): void {
    active = step;
    stage?.setAttribute("data-scene", step);
    const index = STEPS.indexOf(step);
    progress.forEach((bar, i) => {
      bar.dataset.state = i < index ? "done" : i === index ? "active" : "";
    });
    for (const chapter of chapters) {
      chapter.dataset.active = String(chapter.dataset.step === step);
    }
    manual = null;
    if (step === "landmarks" && counter) {
      const total = Number(counter.dataset.count);
      if (reduced) {
        counter.textContent = String(total);
      } else {
        animate(0, total, {
          duration: 1.4,
          ease: [0.22, 1, 0.36, 1],
          onUpdate: (value) => {
            counter.textContent = String(Math.round(value));
          },
        });
      }
    }
  }

  function readScroll(): void {
    const line = window.innerHeight * (wide?.matches ? 0.5 : 0.76);
    let step: Step = STEPS[0];
    let t = 0;
    for (const chapter of chapters) {
      const rect = chapter.getBoundingClientRect();
      if (rect.top <= line) {
        step = chapter.dataset.step as Step;
        t = Math.min(Math.max((line - rect.top) / rect.height, 0), 1);
      }
    }
    if (step !== active) {
      activate(step);
      smooth = t;
    }
    progressNow = t;
  }

  function frame(now: number): void {
    raf = 0;
    if (!visible || !built) {
      return;
    }
    readScroll();
    smooth = reduced ? progressNow : smooth + (progressNow - smooth) * 0.12;
    const t = smooth;
    switch (active) {
      case "landmarks":
        drawMeshScene(now, t);
        break;
      case "masks":
        setLayers(easeInOut(ramp(t, 0.05, 0.62)), easeInOut(ramp(t, 0, 0.35)), 0);
        break;
      case "texture":
        showWeights(weightsNow(now, t), now);
        break;
      case "final":
        // The sheets fall together and turn to face you; as they meet they
        // give way to the engine's real render, so the flat sheets are never
        // seen face-on as if they were the result.
        setLayers(1 - easeInOut(ramp(t, 0, 0.6)), 1 - easeInOut(ramp(t, 0.1, 0.7)), easeInOut(ramp(t, 0.3, 0.68)));
        break;
    }
    raf = requestAnimationFrame(frame);
  }

  function start(): void {
    if (!raf && visible && built) {
      raf = requestAnimationFrame(frame);
    }
  }

  if (typeof IntersectionObserver === "function") {
    // Built when the section is in sight of the viewport; animated only while
    // it is actually on screen.
    new IntersectionObserver(
      (entries) => {
        near = entries.some((entry) => entry.isIntersecting);
        if (near && !built) {
          whenIdle(build);
        }
      },
      { rootMargin: "1400px 0px" },
    ).observe(section);
    new IntersectionObserver((entries) => {
      visible = entries.some((entry) => entry.isIntersecting);
      start();
    }).observe(section);
  } else {
    near = true;
    visible = true;
  }

  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (!built) {
        return;
      }
      layoutMesh();
      layers?.layout();
      cloudView?.resize();
      lastLayers = "";
    }, 160);
  });

  return {
    setScene(next) {
      scene = next;
      whenIdle(build);
    },
  };
}
