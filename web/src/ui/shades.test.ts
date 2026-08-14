import { describe, expect, it } from "vitest";

import { PRODUCTS } from "./shades";
import constants from "../gen/constants.json";
import type { ProductName } from "../engine/look";

/**
 * The rail can only show a preset as landing on a named shade if that
 * shade's hex actually lives in the product's ramp -- otherwise tapping a
 * preset selects a colour the swatch row has no swatch for. These tests
 * guard the two ways that promise breaks: a preset colour drifting out of
 * sync with the ramp, and a ramp with a duplicate hex (which would make
 * "the selected shade" ambiguous).
 */

const PRESETS = constants.presets as unknown as Record<
  string,
  Record<string, { color: string; finish: string; intensity: number }>
>;

describe("shade catalogue", () => {
  it("contains every preset product color in that product's ramp", () => {
    for (const [presetName, look] of Object.entries(PRESETS)) {
      for (const [productName, entry] of Object.entries(look)) {
        if (productName === "smoothing") {
          continue;
        }
        const meta = PRODUCTS.find((p) => p.name === (productName as ProductName));
        expect(meta, `preset ${presetName} references unknown product ${productName}`).toBeDefined();
        const hex = entry.color.toLowerCase();
        const found = meta!.shades.some((shade) => shade.hex.toLowerCase() === hex);
        expect(
          found,
          `preset ${presetName}'s ${productName} color ${entry.color} is not in the ${productName} ramp`,
        ).toBe(true);
      }
    }
  });

  it("has no duplicate hexes within any single ramp", () => {
    for (const product of PRODUCTS) {
      const hexes = product.shades.map((shade) => shade.hex.toLowerCase());
      const unique = new Set(hexes);
      expect(unique.size, `${product.name} ramp has duplicate hexes`).toBe(hexes.length);
    }
  });
});
