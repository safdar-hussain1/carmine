import { describe, expect, it } from "vitest";

import { activeProducts, lightnessPullFor, PRESETS, PRODUCT_ORDER } from "./look";
import constants from "../gen/constants.json";

describe("presets", () => {
  it("carries every preset the Python engine ships", () => {
    expect(Object.keys(PRESETS).sort()).toEqual(["bare", "everyday", "glass", "velvet"]);
  });

  it("round-trips each preset's serialized shape", () => {
    for (const [name, look] of Object.entries(PRESETS)) {
      const raw = (constants.presets as Record<string, Record<string, unknown>>)[name];
      expect(look.smoothing, name).toBe(raw.smoothing);
      for (const product of PRODUCT_ORDER) {
        expect(look[product], `${name}.${product}`).toEqual(raw[product]);
      }
    }
  });
});

describe("lightnessPullFor", () => {
  it("uses the per-product pulls from the Python engine", () => {
    const pulls = constants.pigment.lightness_pull;
    expect(lightnessPullFor("blush", "satin")).toBe(pulls.blush);
    expect(lightnessPullFor("highlighter", "satin")).toBe(pulls.highlighter);
    expect(lightnessPullFor("eyeshadow", "satin")).toBe(pulls.eyeshadow);
    expect(lightnessPullFor("brows", "satin")).toBe(pulls.brows);
  });

  it("gives matte lipstick a smaller pull than satin or gloss", () => {
    const pulls = constants.pigment.lightness_pull;
    expect(lightnessPullFor("lipstick", "matte")).toBe(pulls.lipstick_matte);
    expect(lightnessPullFor("lipstick", "satin")).toBe(pulls.lipstick_satin);
    expect(lightnessPullFor("lipstick", "gloss")).toBe(pulls.lipstick_gloss);
    expect(pulls.lipstick_matte).toBeLessThan(pulls.lipstick_satin);
  });

  it("gives eyeliner no pull, since it is painted flat rather than tinted", () => {
    expect(lightnessPullFor("eyeliner", "satin")).toBe(0);
  });
});

describe("activeProducts", () => {
  it("skips products a look leaves at zero intensity", () => {
    // "bare" is lipstick and brows only; building masks for the other four
    // would be four polygon rasters and four feathers thrown away.
    expect([...activeProducts(PRESETS.bare)].sort()).toEqual(["brows", "lipstick"]);
  });

  it("includes every product of a full look", () => {
    expect(activeProducts(PRESETS.velvet).has("eyeliner")).toBe(true);
    expect(activeProducts(PRESETS.velvet).has("brows")).toBe(false);
  });
});
