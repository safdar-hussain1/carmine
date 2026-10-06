/**
 * The few measured figures the page quotes, read from the committed reports.
 *
 * The JSON is imported, so it is baked into the bundle at build time: what
 * the page says and what the repository measured cannot drift apart, and
 * quoting a number costs no network request. Only figures that explain
 * something to a visitor are here -- the full benchmark lives in the README.
 */

import { meta, photo } from "../../../reports/benchmark.json";
import { timing_hardware as hardware } from "../../../reports/browser_metrics.json";
import constants from "../gen/constants.json";

interface PhotoRow {
  method: string;
  lip_luminance_shift: number;
}

function row(method: string): PhotoRow {
  const found = (photo.rows as PhotoRow[]).find((entry) => entry.method === method);
  if (!found) {
    throw new Error(`benchmark.json has no ${method} row`);
  }
  return found;
}

/** "ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)"
 * -> "Apple M1 Pro". */
function gpuName(renderer: string): string {
  const match = /Renderer: ([^,)]+)/.exec(renderer);
  return (match ? match[1] : renderer).trim();
}

const frameMs = hardware.timing.livePath.totalMedian;

export const FACTS = {
  /** Points the face landmark model returns per frame. */
  landmarks: constants.regions.NUM_LANDMARKS as number,
  /** Portraits in the photo benchmark. */
  portraits: meta.n_images,
  /** Mean shift in lip lightness, 8-bit Lab L (0-255), for the flat fill. */
  lipShiftFlat: row("opaque_fill").lip_luminance_shift,
  /** The same measurement for carmine. */
  lipShiftCarmine: row("carmine").lip_luminance_shift,
  /** Median cost of one live frame on real hardware, in milliseconds. */
  frameMs,
  /** The frame rate that cost allows. */
  fps: 1000 / frameMs,
  /** The machine that measurement ran on. */
  gpu: gpuName(hardware.glRenderer),
} as const;

/** One decimal place, the precision the README quotes these at. */
export function oneDecimal(value: number): string {
  return value.toFixed(1);
}
