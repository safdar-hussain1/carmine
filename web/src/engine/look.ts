/**
 * Look configuration -- the browser mirror of `carmine.look.Look`.
 *
 * The shape here is exactly `Look.to_dict()`'s, so a look serialized by
 * either engine loads in the other without a translation step: that is what
 * lets a shareable look URL from the web app be replayed by the Python CLI
 * and produce the same render.
 *
 * The presets are not restated here -- they are read from the generated
 * constants, which serialize `carmine.look.PRESETS` directly.
 */

import constants from "../gen/constants.json";

/** Surface finish. Only lipstick actually branches on it; the other
 * products carry the field so the serialized shape matches Python's. */
export type Finish = "matte" | "satin" | "gloss";

export interface ProductConfig {
  /** `#RRGGBB`. */
  color: string;
  /** Opacity in [0, 1]. Zero means the product is skipped entirely. */
  intensity: number;
  finish: Finish;
}

export interface LookConfig {
  lipstick: ProductConfig;
  eyeshadow: ProductConfig;
  eyeliner: ProductConfig;
  brows: ProductConfig;
  blush: ProductConfig;
  highlighter: ProductConfig;
  /** Skin smoothing in [0, 1]. Not applied by the live shader path; see
   * renderer.ts. */
  smoothing: number;
}

/** The product fields of a look, in the order `engine.apply_look` paints
 * them. Later products layer over earlier ones. */
export const PRODUCT_ORDER = [
  "blush",
  "highlighter",
  "eyeshadow",
  "brows",
  "lipstick",
  "eyeliner",
] as const;

export type ProductName = (typeof PRODUCT_ORDER)[number];

/** Per-product lightness pulls, mirrored from `engine.apply_look`. Lipstick's
 * depends on its finish and is resolved by `lightnessPullFor`. */
const PULL = constants.pigment.lightness_pull;

/**
 * How much of the target color's own lightness is allowed to bleed into a
 * product's region.
 *
 * These are all well under 1 so the shading already in the frame survives:
 * blush at 0.15 barely touches L (cheeks must keep their own modeling),
 * while eyeshadow at 0.30 is allowed to darken the lid because a shadow
 * that does not darken does not read as shadow.
 */
export function lightnessPullFor(product: ProductName, finish: Finish): number {
  switch (product) {
    case "blush":
      return PULL.blush;
    case "highlighter":
      return PULL.highlighter;
    case "eyeshadow":
      return PULL.eyeshadow;
    case "brows":
      return PULL.brows;
    case "lipstick":
      return finish === "matte" ? PULL.lipstick_matte : PULL.lipstick_satin;
    case "eyeliner":
      // Eyeliner is painted flat rather than tinted, so it has no pull.
      return 0;
  }
}

function asFinish(value: string): Finish {
  return value === "matte" || value === "gloss" ? value : "satin";
}

function asProduct(raw: { color: string; intensity: number; finish: string }): ProductConfig {
  return { color: raw.color, intensity: raw.intensity, finish: asFinish(raw.finish) };
}

const RAW_PRESETS = constants.presets as Record<
  string,
  {
    lipstick: { color: string; intensity: number; finish: string };
    eyeshadow: { color: string; intensity: number; finish: string };
    eyeliner: { color: string; intensity: number; finish: string };
    brows: { color: string; intensity: number; finish: string };
    blush: { color: string; intensity: number; finish: string };
    highlighter: { color: string; intensity: number; finish: string };
    smoothing: number;
  }
>;

function toLook(raw: (typeof RAW_PRESETS)[string]): LookConfig {
  return {
    lipstick: asProduct(raw.lipstick),
    eyeshadow: asProduct(raw.eyeshadow),
    eyeliner: asProduct(raw.eyeliner),
    brows: asProduct(raw.brows),
    blush: asProduct(raw.blush),
    highlighter: asProduct(raw.highlighter),
    smoothing: raw.smoothing,
  };
}

/** Preset looks, keyed by name, deserialized from the generated constants. */
export const PRESETS: Record<string, LookConfig> = Object.fromEntries(
  Object.entries(RAW_PRESETS).map(([name, raw]) => [name, toLook(raw)]),
);

/** Names of the products a look actually paints -- the masks worth building
 * for it. A zero-intensity product contributes nothing, so its mask is not
 * worth the raster and the feather. */
export function activeProducts(look: LookConfig): Set<string> {
  const active = new Set<string>();
  for (const name of PRODUCT_ORDER) {
    if (look[name].intensity > 0) {
      active.add(name);
    }
  }
  return active;
}
