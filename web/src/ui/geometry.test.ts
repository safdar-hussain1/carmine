import { describe, expect, it } from "vitest";

import { boxToContentX, contentToBox, coverFit, focusOn } from "./geometry";

describe("coverFit", () => {
  it("scales to fill the box and centres the overflow by default", () => {
    // 1600x1000 into a 400x500 portrait box: height decides, width overflows.
    const fit = coverFit(400, 500, 1600, 1000);
    expect(fit.scale).toBeCloseTo(0.5, 12);
    expect(fit.offsetX).toBeCloseTo(-200, 12);
    expect(fit.offsetY).toBeCloseTo(0, 12);
  });

  it("follows object-position at either end", () => {
    expect(coverFit(400, 500, 1600, 1000, 0, 0.5).offsetX).toBeCloseTo(0, 12);
    expect(coverFit(400, 500, 1600, 1000, 1, 0.5).offsetX).toBeCloseTo(-400, 12);
  });

  it("handles content that overflows vertically instead", () => {
    const fit = coverFit(800, 450, 720, 960, 0.5, 0);
    expect(fit.scale).toBeCloseTo(800 / 720, 12);
    expect(fit.offsetX).toBeCloseTo(0, 12);
    expect(fit.offsetY).toBeCloseTo(0, 12);
  });
});

describe("focusOn", () => {
  it("moves the crop so the point lands in the middle when it can", () => {
    // Drawn 800 wide in a 400 box; x=1000 of 1600 is 500 drawn, so the
    // offset that centres it is 200 - 500 = -300, i.e. 75% of the overflow.
    const focus = focusOn(400, 500, 1600, 1000, 1000, 500);
    expect(focus.fx).toBeCloseTo(0.75, 12);
    expect(focus.fy).toBe(0.5);
  });

  it("clamps at the edge rather than showing past the picture", () => {
    expect(focusOn(400, 500, 1600, 1000, 1590, 500).fx).toBe(1);
    expect(focusOn(400, 500, 1600, 1000, 10, 500).fx).toBe(0);
  });

  it("puts the centred point back at the box centre", () => {
    const focus = focusOn(400, 500, 1600, 1000, 900, 500);
    const fit = coverFit(400, 500, 1600, 1000, focus.fx, focus.fy);
    const [x] = contentToBox(fit, 900, 500);
    expect(x).toBeCloseTo(200, 9);
  });
});

describe("boxToContentX", () => {
  it("inverts contentToBox", () => {
    const fit = coverFit(400, 500, 1600, 1000, 0.3, 0.5);
    const [x] = contentToBox(fit, 1234, 0);
    expect(boxToContentX(fit, x, 1600)).toBeCloseTo(1234, 9);
  });

  it("clamps to the picture", () => {
    const fit = coverFit(400, 500, 1600, 1000);
    expect(boxToContentX(fit, -10_000, 1600)).toBe(0);
    expect(boxToContentX(fit, 10_000, 1600)).toBe(1600);
  });
});
