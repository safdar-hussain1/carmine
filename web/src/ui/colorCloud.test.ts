import { describe, expect, it } from "vitest";

import { buildCloud, lightnessStats } from "./colorCloud";

/** A small "lip": a column of reddish-brown pixels from shadow to highlight. */
function syntheticLip(): { pixels: Float32Array; mask: Float32Array } {
  const count = 64;
  const pixels = new Float32Array(count * 4);
  const mask = new Float32Array(count).fill(1);
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    pixels[i * 4] = 110 + 100 * t;
    pixels[i * 4 + 1] = 70 + 80 * t;
    pixels[i * 4 + 2] = 65 + 75 * t;
    pixels[i * 4 + 3] = 255;
  }
  return { pixels, mask };
}

const BERRY: [number, number, number] = [0x8e, 0x1b, 0x3a];

describe("buildCloud", () => {
  it("samples every masked pixel up to the limit", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    expect(cloud.count).toBe(64);
    expect(cloud.lab.original.length).toBe(64 * 3);
    expect(cloud.rgb.flat.length).toBe(64 * 3);
  });

  it("keeps to the limit by taking an even spread of the lip", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 16);
    expect(cloud.count).toBe(16);
  });

  it("ignores pixels outside the lip", () => {
    const { pixels, mask } = syntheticLip();
    mask.fill(0, 0, 32);
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    expect(cloud.count).toBe(32);
  });

  it("leaves the original colours as they were", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    expect(Array.from(cloud.rgb.original.slice(0, 3))).toEqual([110, 70, 65]);
  });

  it("adds the colour on top for the flat fill, clipped at white", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    // The darkest pixel plus 0.85 of the shade, per channel.
    expect(cloud.rgb.flat[0]).toBeCloseTo(110 + 0x8e * 0.85, 4);
    // The brightest pixel's red channel overflows and clips.
    expect(cloud.rgb.flat[63 * 3]).toBe(255);
  });

  it("lifts the lip's lightness far more with the flat fill than with the tint", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    const original = lightnessStats(cloud.lab.original);
    const flat = lightnessStats(cloud.lab.flat);
    const tinted = lightnessStats(cloud.lab.carmine);
    const flatShift = Math.abs(flat.mean - original.mean);
    const tintShift = Math.abs(tinted.mean - original.mean);
    expect(flatShift).toBeGreaterThan(3 * tintShift);
    expect(flat.mean).toBeGreaterThan(original.mean);
  });

  it("moves the tinted lip's colour toward the shade", () => {
    const { pixels, mask } = syntheticLip();
    const cloud = buildCloud(pixels, mask, BERRY, 0.85, 0.35, 1000);
    // a* (red-green) rises toward the berry's strong red.
    const meanA = (lab: Float32Array) => {
      let sum = 0;
      for (let i = 1; i < lab.length; i += 3) {
        sum += lab[i];
      }
      return sum / (lab.length / 3);
    };
    expect(meanA(cloud.lab.carmine)).toBeGreaterThan(meanA(cloud.lab.original) + 10);
  });
});

describe("lightnessStats", () => {
  it("measures the mean and spread of L", () => {
    const lab = new Float32Array([10, 0, 0, 20, 0, 0, 30, 0, 0]);
    const stats = lightnessStats(lab);
    expect(stats.mean).toBeCloseTo(20, 9);
    expect(stats.min).toBe(10);
    expect(stats.max).toBe(30);
  });
});
