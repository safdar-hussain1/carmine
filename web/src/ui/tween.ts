/**
 * Blending one look into another.
 *
 * When a look or a shade changes, the mirror animates the change instead of
 * cutting to it: the lipstick visibly turns from one shade into the next.
 * That only reads as makeup if the in-between frames are plausible looks
 * themselves, so a product that is being switched on fades in at its *new*
 * colour, and one being switched off fades out at its *old* colour -- never
 * sliding through a shade that is on neither look.
 */

import { PRODUCT_ORDER, type LookConfig, type ProductConfig } from "../engine/look";

function clamp01(t: number): number {
  return t <= 0 ? 0 : t >= 1 ? 1 : t;
}

function channel(hex: string, index: number): number {
  return parseInt(hex.slice(1 + index * 2, 3 + index * 2), 16);
}

function copyLook(look: LookConfig): LookConfig {
  const out = { smoothing: look.smoothing } as LookConfig;
  for (const name of PRODUCT_ORDER) {
    out[name] = { ...look[name] };
  }
  return out;
}

/** sRGB blend of two `#rrggbb` colours, rounded to whole bytes. */
export function lerpHex(a: string, b: string, t: number): string {
  const f = clamp01(t);
  let out = "#";
  for (let i = 0; i < 3; i++) {
    const value = Math.round(channel(a, i) + (channel(b, i) - channel(a, i)) * f);
    out += value.toString(16).padStart(2, "0");
  }
  return out;
}

function mixProduct(a: ProductConfig, b: ProductConfig, t: number): ProductConfig {
  const intensity = a.intensity + (b.intensity - a.intensity) * t;
  if (a.intensity <= 0) {
    return { color: b.color, finish: b.finish, intensity };
  }
  if (b.intensity <= 0) {
    return { color: a.color, finish: a.finish, intensity };
  }
  return {
    color: lerpHex(a.color, b.color, t),
    finish: t < 0.5 ? a.finish : b.finish,
    intensity,
  };
}

/**
 * The look a fraction `t` of the way from `from` to `to`. Both ends are
 * exact copies, so an animation that finishes leaves precisely the target
 * look on the face, colour strings included.
 */
export function mixLook(from: LookConfig, to: LookConfig, t: number): LookConfig {
  const f = clamp01(t);
  if (f === 0) {
    return copyLook(from);
  }
  if (f === 1) {
    return copyLook(to);
  }
  const out = { smoothing: from.smoothing + (to.smoothing - from.smoothing) * f } as LookConfig;
  for (const name of PRODUCT_ORDER) {
    out[name] = mixProduct(from[name], to[name], f);
  }
  return out;
}

/** The same look with nothing applied: where the opening reveal starts. */
export function withoutMakeup(look: LookConfig): LookConfig {
  const out = copyLook(look);
  for (const name of PRODUCT_ORDER) {
    out[name].intensity = 0;
  }
  return out;
}

/** Cubic ease-in-out: slow at both ends, so a blend settles rather than stops. */
export function easeInOut(t: number): number {
  const f = clamp01(t);
  return f < 0.5 ? 4 * f * f * f : 1 - Math.pow(-2 * f + 2, 3) / 2;
}
