import { describe, expect, it } from "vitest";

import { PRESETS, PRODUCT_ORDER, type LookConfig } from "../engine/look";
import { easeInOut, lerpHex, mixLook, withoutMakeup } from "./tween";

function copy(look: LookConfig): LookConfig {
  return JSON.parse(JSON.stringify(look)) as LookConfig;
}

describe("lerpHex", () => {
  it("returns each end exactly", () => {
    expect(lerpHex("#000000", "#FFFFFF", 0)).toBe("#000000");
    expect(lerpHex("#000000", "#FFFFFF", 1)).toBe("#ffffff");
  });

  it("interpolates every channel and rounds to a byte", () => {
    expect(lerpHex("#000000", "#ffffff", 0.5)).toBe("#808080");
    expect(lerpHex("#ff0000", "#0000ff", 0.25)).toBe("#bf0040");
  });
});

describe("mixLook", () => {
  const from = PRESETS.bare;
  const to = PRESETS.velvet;

  it("is the starting look at t = 0 and the target look at t = 1", () => {
    expect(mixLook(from, to, 0)).toEqual(from);
    expect(mixLook(from, to, 1)).toEqual(to);
  });

  it("clamps t outside [0, 1]", () => {
    expect(mixLook(from, to, -2)).toEqual(from);
    expect(mixLook(from, to, 7)).toEqual(to);
  });

  it("fades a product in at its target colour rather than from the old one", () => {
    // Bare has no eyeliner; velvet has it at 0.8. Sliding the colour from
    // bare's unused value would flash a shade that is on neither look.
    const half = mixLook(from, to, 0.5);
    expect(half.eyeliner.color).toBe(to.eyeliner.color);
    expect(half.eyeliner.intensity).toBeCloseTo(to.eyeliner.intensity / 2, 9);
  });

  it("fades a product out at its old colour", () => {
    const half = mixLook(to, from, 0.5);
    expect(half.eyeliner.color).toBe(to.eyeliner.color);
    expect(half.eyeliner.intensity).toBeCloseTo(to.eyeliner.intensity / 2, 9);
  });

  it("blends colour and intensity when a product is on in both looks", () => {
    const a = copy(PRESETS.everyday);
    const b = copy(PRESETS.everyday);
    a.lipstick = { color: "#000000", intensity: 0.2, finish: "satin" };
    b.lipstick = { color: "#ffffff", intensity: 0.6, finish: "gloss" };
    const half = mixLook(a, b, 0.5);
    expect(half.lipstick.color).toBe("#808080");
    expect(half.lipstick.intensity).toBeCloseTo(0.4, 9);
  });

  it("switches finish at the midpoint, not at either end", () => {
    const a = copy(PRESETS.everyday);
    const b = copy(PRESETS.everyday);
    a.lipstick.finish = "matte";
    b.lipstick.finish = "gloss";
    expect(mixLook(a, b, 0.49).lipstick.finish).toBe("matte");
    expect(mixLook(a, b, 0.5).lipstick.finish).toBe("gloss");
  });

  it("never mutates its inputs", () => {
    const a = copy(from);
    const b = copy(to);
    mixLook(a, b, 0.3);
    expect(a).toEqual(from);
    expect(b).toEqual(to);
  });

  it("returns a fresh object, so a caller can edit the result safely", () => {
    const result = mixLook(from, to, 1);
    result.lipstick.intensity = 0;
    expect(to.lipstick.intensity).toBeGreaterThan(0);
  });
});

describe("withoutMakeup", () => {
  it("keeps every colour and finish and zeroes every intensity", () => {
    const bare = withoutMakeup(PRESETS.glass);
    for (const name of PRODUCT_ORDER) {
      expect(bare[name].intensity).toBe(0);
      expect(bare[name].color).toBe(PRESETS.glass[name].color);
      expect(bare[name].finish).toBe(PRESETS.glass[name].finish);
    }
  });
});

describe("easeInOut", () => {
  it("starts at 0, ends at 1 and passes through the middle", () => {
    expect(easeInOut(0)).toBe(0);
    expect(easeInOut(1)).toBe(1);
    expect(easeInOut(0.5)).toBeCloseTo(0.5, 9);
  });

  it("is monotonic", () => {
    let previous = 0;
    for (let i = 1; i <= 100; i++) {
      const value = easeInOut(i / 100);
      expect(value).toBeGreaterThanOrEqual(previous);
      previous = value;
    }
  });
});
