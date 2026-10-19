import { describe, expect, it } from "vitest";

import { centroid, project } from "./mesh3d";
import { SAMPLE } from "./sample";

describe("the sample's landmarks", () => {
  it("has 478 points with depth, in the sample photo's pixels", () => {
    expect(SAMPLE.width).toBe(1600);
    expect(SAMPLE.height).toBe(1067);
    expect(SAMPLE.xy.length).toBe(478 * 2);
    expect(SAMPLE.z.length).toBe(478);
  });

  it("puts the face where the photo has it", () => {
    const { cx, cy } = centroid(SAMPLE.xy);
    expect(cx).toBeGreaterThan(SAMPLE.width * 0.4);
    expect(cx).toBeLessThan(SAMPLE.width * 0.75);
    expect(cy).toBeGreaterThan(SAMPLE.height * 0.3);
    expect(cy).toBeLessThan(SAMPLE.height * 0.8);
  });
});

describe("project", () => {
  const xy = new Float32Array([100, 50, 140, 50, 120, 90]);
  const z = new Float32Array([0, 20, -20]);
  const center = { cx: 120, cy: 60 };

  it("returns the points unchanged when nothing is turned and perspective is off", () => {
    const out = new Float32Array(6);
    project(xy, z, center, 0, 0, 500, 0, out);
    expect(Array.from(out)).toEqual(Array.from(xy));
  });

  it("mirrors a flat point across the centre at half a turn", () => {
    const flatZ = new Float32Array([0, 0, 0]);
    const out = new Float32Array(6);
    project(xy, flatZ, center, Math.PI, 0, 500, 0, out);
    expect(out[0]).toBeCloseTo(140, 6);
    expect(out[1]).toBeCloseTo(50, 6);
    expect(out[4]).toBeCloseTo(120, 6);
  });

  it("swings points with depth sideways as the face turns", () => {
    const out = new Float32Array(6);
    project(xy, z, center, 0.5, 0, 500, 0, out);
    // The near point (negative z) and the far one (positive z) move in
    // opposite directions.
    const nearShift = out[4] - xy[4];
    const farShift = out[2] - (center.cx + (xy[2] - center.cx) * Math.cos(0.5));
    expect(Math.sign(nearShift)).not.toBe(Math.sign(farShift));
  });

  it("draws nearer points larger with perspective on", () => {
    const pair = new Float32Array([150, 60, 150, 60]);
    const depth = new Float32Array([-100, 100]);
    const out = new Float32Array(4);
    project(pair, depth, center, 0, 0, 500, 1, out);
    const near = Math.abs(out[0] - center.cx);
    const far = Math.abs(out[2] - center.cx);
    expect(near).toBeGreaterThan(far);
  });

  it("reports each point's depth after the turn", () => {
    const out = new Float32Array(6);
    const depth = new Float32Array(3);
    project(xy, z, center, 0, 0, 500, 0, out, depth);
    expect(Array.from(depth)).toEqual([0, 20, -20]);
  });
});
