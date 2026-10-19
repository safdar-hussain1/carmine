/**
 * Drawing a lip's pixels as a cloud in colour space.
 *
 * Each pixel is a dot in its own colour, placed by its CIELAB values:
 * lightness straight up, the two colour axes flat around it. A ring of hues
 * on the floor shows which way is red, yellow, green and blue, and a neutral
 * axis runs up the middle from dark to light. The cloud turns slowly so its
 * depth reads, and the untouched lip stays faintly in place behind the
 * others, so a cloud that has jumped upward is plain to see.
 */

import { labToRgb } from "../engine/color";
import type { Cloud } from "./colorCloud";

export type CloudState = "original" | "flat" | "carmine";

export interface Weights {
  original: number;
  flat: number;
  carmine: number;
}

export interface CloudView {
  draw(weights: Weights, yaw: number, pitch: number): void;
  resize(): void;
}

/** Lightness range the scene shows, and the radius of the hue ring. */
const L_MIN = 8;
const L_MAX = 96;
const RING = 54;

export function createCloudView(canvas: HTMLCanvasElement, cloud: Cloud): CloudView {
  const ctx = canvas.getContext("2d");
  const n = cloud.count;
  const lab = new Float32Array(n * 3);
  const rgb = new Float32Array(n * 3);
  const screen = new Float32Array(n * 2);
  const depth = new Float32Array(n);
  // Kept between frames: each frame re-sorts the last order, which is
  // nearly sorted already.
  const order = Uint16Array.from({ length: n }, (_, i) => i);
  const fills: string[] = new Array<string>(n).fill("");
  let blended: Weights | null = null;
  let dpr = 1;
  let scale = 1;
  let cx = 0;
  let floorY = 0;

  // Centre the view between where the three clouds sit in a*/b*.
  let centerA = 0;
  let centerB = 0;
  for (const state of ["original", "flat", "carmine"] as const) {
    for (let i = 0; i < n; i++) {
      centerA += cloud.lab[state][i * 3 + 1];
      centerB += cloud.lab[state][i * 3 + 2];
    }
  }
  centerA /= n * 3 || 1;
  centerB /= n * 3 || 1;
  // Pulled halfway back to neutral, so the ring and the axis stay in frame.
  centerA *= 0.5;
  centerB *= 0.5;

  const ringColours: string[] = [];
  for (let k = 0; k < 72; k++) {
    const angle = (k / 72) * Math.PI * 2;
    const c = labToRgb([62, Math.cos(angle) * 46, Math.sin(angle) * 46]);
    ringColours.push(
      `rgb(${c.map((v) => Math.round(Math.min(255, Math.max(0, v)))).join(",")})`,
    );
  }

  function resize(): void {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    cx = canvas.width / 2;
    floorY = canvas.height * 0.76;
    // The state chips sit along the top, so on a short frame the axis is
    // scaled down to end below them.
    const room = (floorY - 72 * dpr) / 84;
    scale = Math.min(canvas.width / 150, canvas.height / 150, room);
  }

  /** a*, b*, L* to the canvas, after turning the space by `yaw` and tilting
   * the viewer down by `pitch`. Returns the depth, larger further away: a
   * far point sits higher on the screen, as on a table seen from above. */
  function toScreen(a: number, b: number, l: number, yaw: number, pitch: number, out: Float32Array, k: number): number {
    const x = a - centerA;
    const z = b - centerB;
    const y = l - L_MIN;
    const xr = x * Math.cos(yaw) - z * Math.sin(yaw);
    const zr = x * Math.sin(yaw) + z * Math.cos(yaw);
    const yr = y * Math.cos(pitch) + zr * Math.sin(pitch) * 0.55;
    out[k * 2] = cx + xr * scale;
    out[k * 2 + 1] = floorY - yr * scale;
    return zr;
  }

  const ringPoint = new Float32Array(2);
  const axisPoint = new Float32Array(4);

  function draw(weights: Weights, yaw: number, pitch: number): void {
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // The hue ring on the floor, segment by segment, so it turns in 3D.
    ctx.lineWidth = 2.2 * dpr;
    ctx.lineCap = "round";
    for (let k = 0; k < 72; k++) {
      const a0 = (k / 72) * Math.PI * 2;
      const a1 = ((k + 1) / 72) * Math.PI * 2;
      toScreen(centerA + Math.cos(a0) * RING, centerB + Math.sin(a0) * RING, L_MIN, yaw, pitch, ringPoint, 0);
      const x0 = ringPoint[0];
      const y0 = ringPoint[1];
      const d = toScreen(centerA + Math.cos(a1) * RING, centerB + Math.sin(a1) * RING, L_MIN, yaw, pitch, ringPoint, 0);
      ctx.globalAlpha = 0.35 + 0.35 * (d < 0 ? 1 : 0.4);
      ctx.strokeStyle = ringColours[k];
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(ringPoint[0], ringPoint[1]);
      ctx.stroke();
    }

    // The neutral axis: grey from dark to light, labelled at its ends.
    toScreen(0, 0, L_MIN, yaw, pitch, axisPoint, 0);
    toScreen(0, 0, L_MAX, yaw, pitch, axisPoint, 1);
    const axis = ctx.createLinearGradient(axisPoint[0], axisPoint[1], axisPoint[2], axisPoint[3]);
    axis.addColorStop(0, "rgba(255,255,255,0.08)");
    axis.addColorStop(1, "rgba(255,255,255,0.55)");
    ctx.globalAlpha = 1;
    ctx.strokeStyle = axis;
    ctx.lineWidth = 1.4 * dpr;
    ctx.beginPath();
    ctx.moveTo(axisPoint[0], axisPoint[1]);
    ctx.lineTo(axisPoint[2], axisPoint[3]);
    ctx.stroke();
    ctx.fillStyle = "rgba(247, 237, 232, 0.72)";
    ctx.font = `${12 * dpr}px system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText("Lighter", axisPoint[2] + 8 * dpr, axisPoint[3] + 4 * dpr);
    ctx.fillText("Darker", axisPoint[0] + 8 * dpr, axisPoint[1] - 6 * dpr);

    // The untouched lip, faint, so movement away from it is visible.
    const ghost = 1 - weights.original;
    if (ghost > 0.01) {
      ctx.globalAlpha = 0.24 * ghost;
      ctx.fillStyle = "#f7ede8";
      for (let i = 0; i < n; i++) {
        toScreen(cloud.lab.original[i * 3 + 1], cloud.lab.original[i * 3 + 2], cloud.lab.original[i * 3], yaw, pitch, screen, i);
        ctx.fillRect(screen[i * 2] - dpr * 0.6, screen[i * 2 + 1] - dpr * 0.6, dpr * 1.2, dpr * 1.2);
      }
    }

    // The cloud itself: each pixel blended between the three states. The
    // blend only changes while the weights do; the turn changes every frame.
    if (
      !blended ||
      blended.original !== weights.original ||
      blended.flat !== weights.flat ||
      blended.carmine !== weights.carmine
    ) {
      blended = { ...weights };
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
          const j = i * 3 + c;
          lab[j] =
            cloud.lab.original[j] * weights.original + cloud.lab.flat[j] * weights.flat + cloud.lab.carmine[j] * weights.carmine;
          rgb[j] =
            cloud.rgb.original[j] * weights.original + cloud.rgb.flat[j] * weights.flat + cloud.rgb.carmine[j] * weights.carmine;
        }
        fills[i] = `rgb(${rgb[i * 3] | 0},${rgb[i * 3 + 1] | 0},${rgb[i * 3 + 2] | 0})`;
      }
    }
    for (let i = 0; i < n; i++) {
      depth[i] = toScreen(lab[i * 3 + 1], lab[i * 3 + 2], lab[i * 3], yaw, pitch, screen, i);
    }
    // Far dots first, so near ones sit on top.
    order.sort((p, q) => depth[q] - depth[p]);
    ctx.globalAlpha = 0.92;
    const radius = 2.3 * dpr;
    for (const i of order) {
      ctx.fillStyle = fills[i];
      ctx.beginPath();
      ctx.arc(screen[i * 2], screen[i * 2 + 1], radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  resize();
  return { draw, resize };
}
