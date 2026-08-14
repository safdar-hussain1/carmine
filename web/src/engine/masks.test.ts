import { describe, expect, it } from "vitest";

import { buildMasks, MASK_NAMES, processingSize, PROC_MAX_SIDE, type MaskName } from "./masks";
import constants from "../gen/constants.json";

/**
 * These are structural tests, not parity tests.
 *
 * Mask geometry is compared against the Python engine pixel-for-pixel in
 * Task 14, on real detected landmarks, with a tolerance that accounts for
 * the two rasterizers disagreeing at edges. What is worth pinning *here* is
 * the set of properties that make each mask the right shape at all -- the
 * lip mask leaving an open mouth alone, the crease gradient actually
 * fading toward the brow, the eyeliner wing extending past the eye corner.
 * Those are the things a refactor breaks silently, and they can be checked
 * on a synthetic face without a detector or a model file.
 */

const R = constants.regions;

const WIDTH = 480;
const HEIGHT = 480;

const FACE = { cx: 240, cy: 290, rx: 150, ry: 210 };
const RIGHT_EYE = { cx: 205, cy: 200, rx: 26, ry: 9 };
const LEFT_EYE = { cx: 275, cy: 200, rx: 26, ry: 9 };
const LIPS = { cx: 240, cy: 330, rx: 46, ry: 26, innerRx: 30, innerRy: 13 };

type Point = [number, number];

/**
 * A synthetic 478-point face.
 *
 * The eyes are built with linearly-spaced x and a sine-arched y rather than
 * as circles on purpose: on a circle, the landmark next to the outer corner
 * sits at a steep angle, which would send the eyeliner wing diving downward
 * and make the wing test measure the wrong thing. Linear spacing matches how
 * the real mesh distributes lid points -- densely across the lid, not evenly
 * around an arc.
 */
function syntheticFace(): Float32Array {
  const landmarks = new Float32Array(R.NUM_LANDMARKS * 2);
  // Anything not placed below sits at the face center; no mask reads it.
  for (let i = 0; i < R.NUM_LANDMARKS; i++) {
    landmarks[i * 2] = FACE.cx;
    landmarks[i * 2 + 1] = FACE.cy;
  }
  const put = (index: number, point: Point) => {
    landmarks[index * 2] = point[0];
    landmarks[index * 2 + 1] = point[1];
  };
  const spread = (indices: readonly number[], from: number, to: number) =>
    indices.map((_, i) => from + ((to - from) * i) / (indices.length - 1));

  // Face oval: index 0 is the forehead, index 18 is the chin.
  R.FACE_OVAL.forEach((index, i) => {
    const theta = Math.PI / 2 - (i * 2 * Math.PI) / R.FACE_OVAL.length;
    put(index, [FACE.cx + FACE.rx * Math.cos(theta), FACE.cy - FACE.ry * Math.sin(theta)]);
  });

  // Eyes: lower lid runs outer -> inner, upper lid runs inner -> outer.
  const eye = (
    geometry: typeof RIGHT_EYE,
    ring: readonly number[],
    upper: readonly number[],
    outerOnLeft: boolean,
  ) => {
    const outerX = geometry.cx + (outerOnLeft ? -geometry.rx : geometry.rx);
    const innerX = geometry.cx + (outerOnLeft ? geometry.rx : -geometry.rx);
    const lower = ring.slice(0, 9);
    lower.forEach((index, i) => {
      const u = i / (lower.length - 1);
      put(index, [outerX + (innerX - outerX) * u, geometry.cy + geometry.ry * Math.sin(Math.PI * u)]);
    });
    upper.forEach((index, i) => {
      const u = i / (upper.length - 1);
      put(index, [innerX + (outerX - innerX) * u, geometry.cy - geometry.ry * Math.sin(Math.PI * u)]);
    });
  };
  eye(RIGHT_EYE, R.RIGHT_EYE, R.RIGHT_EYE_UPPER, true);
  eye(LEFT_EYE, R.LEFT_EYE, R.LEFT_EYE_UPPER, false);

  // Brows. The outer junction landmark (70 / 300) belongs to both arcs, so
  // it is placed once, between the two edges.
  const brow = (
    lowerIndices: readonly number[],
    upperIndices: readonly number[],
    innerX: number,
    outerX: number,
    junctionX: number,
  ) => {
    const lowerBody = lowerIndices.slice(0, -1);
    spread(lowerBody, innerX, outerX).forEach((x, i) => {
      put(lowerBody[i], [x, 155 - 4 * Math.sin((Math.PI * i) / (lowerBody.length - 1))]);
    });
    const upperBody = upperIndices.slice(0, -1);
    spread(upperBody, innerX, outerX).forEach((x, i) => {
      put(upperBody[i], [x, 143 - 4 * Math.sin((Math.PI * i) / (upperBody.length - 1))]);
    });
    put(lowerIndices[lowerIndices.length - 1], [junctionX, 149]);
  };
  brow(R.RIGHT_BROW_LOWER, R.RIGHT_BROW_UPPER, 235, 191, 175);
  brow(R.LEFT_BROW_LOWER, R.LEFT_BROW_UPPER, 245, 289, 305);

  // Lips: index 0 of each ring is the right mouth corner, index 5 the top
  // of the ring, index 10 the left corner, index 15 the bottom.
  const lipRing = (indices: readonly number[], rx: number, ry: number) => {
    indices.forEach((index, i) => {
      const theta = Math.PI - (i * 2 * Math.PI) / indices.length;
      put(index, [LIPS.cx + rx * Math.cos(theta), LIPS.cy - ry * Math.sin(theta)]);
    });
  };
  lipRing(R.LIPS_OUTER, LIPS.rx, LIPS.ry);
  lipRing(R.LIPS_INNER, LIPS.innerRx, LIPS.innerRy);

  put(R.RIGHT_CHEEK, [185, 300]);
  put(R.LEFT_CHEEK, [295, 300]);
  R.RIGHT_CHEEKBONE.forEach((index, i) => put(index, [160 + i * 12, 285 - i * 4]));
  R.LEFT_CHEEKBONE.forEach((index, i) => put(index, [320 - i * 12, 285 - i * 4]));
  R.NOSE_BRIDGE.forEach((index, i) => put(index, [240, 205 + i * 22]));

  return landmarks;
}

const LANDMARKS = syntheticFace();
const ALL = new Set<string>(MASK_NAMES);

function build(active: Iterable<string> = ALL) {
  return buildMasks(LANDMARKS, WIDTH, HEIGHT, new Set(active));
}

function at(mask: Float32Array, width: number, x: number, y: number): number {
  return mask[Math.round(y) * width + Math.round(x)];
}

describe("processingSize", () => {
  it("leaves sources at or below the cap alone", () => {
    expect(processingSize(640, 480)).toEqual({ width: 640, height: 480, scale: 1 });
  });

  it("caps the long side and keeps the aspect ratio", () => {
    const size = processingSize(1920, 1080);
    expect(Math.max(size.width, size.height)).toBe(PROC_MAX_SIDE);
    expect(size.width / size.height).toBeCloseTo(1920 / 1080, 2);
  });

  it("caps portrait sources on their long side too", () => {
    const size = processingSize(1080, 1920);
    expect(size.height).toBe(PROC_MAX_SIDE);
    expect(size.width).toBe(405);
  });
});

describe("buildMasks", () => {
  it("builds only the requested products", () => {
    const set = build(["lipstick", "blush"]);
    expect(Object.keys(set.masks).sort()).toEqual(["blush", "lipstick"]);
  });

  it("sizes every mask to the processing resolution", () => {
    const set = build();
    expect(set.width).toBe(WIDTH);
    expect(set.height).toBe(HEIGHT);
    for (const name of MASK_NAMES) {
      expect(set.masks[name]!.length).toBe(WIDTH * HEIGHT);
    }
  });

  it("keeps every mask within [0, 1] and non-empty", () => {
    const set = build();
    for (const name of MASK_NAMES) {
      const mask = set.masks[name]!;
      let max = 0;
      let min = Infinity;
      for (const value of mask) {
        if (value > max) max = value;
        if (value < min) min = value;
      }
      expect(min, `${name} min`).toBeGreaterThanOrEqual(0);
      expect(max, `${name} max`).toBeLessThanOrEqual(1);
      // 0.3 rather than something near 1: eyeliner is a one-pixel stroke at
      // this face size, and its feather spreads that single pixel's weight
      // across the kernel, so its peak is legitimately well under 1.
      expect(max, `${name} is empty`).toBeGreaterThan(0.3);
    }
  });

  it("rejects landmarks whose eye corners have collapsed", () => {
    const degenerate = new Float32Array(LANDMARKS);
    degenerate[R.LEFT_EYE_OUTER * 2] = degenerate[R.RIGHT_EYE_OUTER * 2];
    degenerate[R.LEFT_EYE_OUTER * 2 + 1] = degenerate[R.RIGHT_EYE_OUTER * 2 + 1];
    expect(() => buildMasks(degenerate, WIDTH, HEIGHT, ALL)).toThrow(/interocular/);
  });

  it("scales masks down for a source above the resolution cap", () => {
    const big = new Float32Array(LANDMARKS.length);
    for (let i = 0; i < LANDMARKS.length; i++) {
      big[i] = LANDMARKS[i] * 4;
    }
    const set = buildMasks(big, WIDTH * 4, HEIGHT * 4, new Set(["lipstick"]));
    expect(set.width).toBe(PROC_MAX_SIDE);
    expect(set.scale).toBeCloseTo(PROC_MAX_SIDE / (WIDTH * 4), 6);
    // The same face at 4x lands in the same relative place.
    const lip = set.masks.lipstick!;
    const scaled = set.scale * 4;
    expect(at(lip, set.width, LIPS.cx * scaled, (LIPS.cy - LIPS.ry + 4) * scaled)).toBeGreaterThan(
      0.5,
    );
  });
});

describe("lip mask", () => {
  it("is zero at the centre of an open mouth", () => {
    const lip = build(["lipstick"]).masks.lipstick!;
    // The inner ring is subtracted before the feather, and the mouth's
    // centre is many feather radii from the nearest lip pixel.
    expect(at(lip, WIDTH, LIPS.cx, LIPS.cy)).toBeLessThan(0.02);
  });

  it("covers the upper and lower lip bodies", () => {
    const lip = build(["lipstick"]).masks.lipstick!;
    expect(at(lip, WIDTH, LIPS.cx, LIPS.cy - LIPS.ry + 6)).toBeGreaterThan(0.5);
    expect(at(lip, WIDTH, LIPS.cx, LIPS.cy + LIPS.ry - 6)).toBeGreaterThan(0.5);
  });

  it("feathers rather than cutting a hard edge", () => {
    const lip = build(["lipstick"]).masks.lipstick!;
    let solid = 0;
    let partial = 0;
    let support = 0;
    for (const value of lip) {
      if (value > 0) support++;
      if (value >= 0.99) solid++;
      else if (value > 0.01) partial++;
    }
    // A hard fill would have no partial pixels at all, and its support
    // would be exactly its solid core.
    expect(partial).toBeGreaterThan(0);
    expect(support).toBeGreaterThan(solid);
    // The blur spreads weight past the drawn outline.
    expect(at(lip, WIDTH, LIPS.cx, LIPS.cy - LIPS.ry - 2)).toBeGreaterThan(0);
  });
});

describe("eyeshadow mask", () => {
  it("decays from the lid toward the brow", () => {
    // The crease gradient runs 1.0 at the lash line down to 0.35 at the top
    // of the polygon, and the feather softens both ends. Right at the lash
    // line the eye cutout is also being subtracted, so the profile peaks a
    // few pixels up rather than at the lash line itself; from there it
    // falls away monotonically, which is what "heaviest at the lashes,
    // lightest at the brow bone" looks like once feathered. The exact
    // gradient values are pinned against Python by the Task 14 parity run.
    const shadow = build(["eyeshadow"]).masks.eyeshadow!;
    const lashY = RIGHT_EYE.cy - RIGHT_EYE.ry;
    const profile = [2, 8, 14, 22, 30].map((d) => at(shadow, WIDTH, RIGHT_EYE.cx, lashY - d));
    expect(profile[0]).toBeGreaterThan(0.3);
    for (let i = 2; i < profile.length; i++) {
      expect(profile[i], `sample ${i} did not fall below sample ${i - 1}`).toBeLessThan(
        profile[i - 1],
      );
    }
    // By the brow end of the band almost nothing is left.
    expect(profile[profile.length - 1]).toBeLessThan(profile[1] * 0.25);
  });

  it("stays off the eyeball itself", () => {
    const shadow = build(["eyeshadow"]).masks.eyeshadow!;
    expect(at(shadow, WIDTH, RIGHT_EYE.cx, RIGHT_EYE.cy)).toBeLessThan(0.25);
  });

  it("covers both eyes", () => {
    const shadow = build(["eyeshadow"]).masks.eyeshadow!;
    const lashY = RIGHT_EYE.cy - RIGHT_EYE.ry;
    expect(at(shadow, WIDTH, LEFT_EYE.cx, lashY - 2)).toBeGreaterThan(0.3);
  });
});

describe("eyeliner mask", () => {
  it("wings past the outer eye corner", () => {
    const set = build(["eyeliner"]);
    const liner = set.masks.eyeliner!;
    const cornerX = LANDMARKS[R.RIGHT_EYE_OUTER * 2];

    let minX = WIDTH;
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < cornerX; x++) {
        if (liner[y * WIDTH + x] > 0.05) {
          minX = Math.min(minX, x);
        }
      }
    }

    // Without a wing the stroke could only reach half its thickness plus
    // the feather radius past the corner landmark.
    const thickness = Math.max(1, Math.round(set.interocular * constants.masks.eyeliner.thickness_factor));
    const sigma = Math.max(
      constants.masks.eyeliner.feather_min,
      set.interocular * constants.masks.eyeliner.feather_factor,
    );
    const withoutWing = cornerX - thickness / 2 - Math.trunc(sigma * 3);
    expect(minX).toBeLessThan(withoutWing);
    // The wing is a flick, not a whole second eye: its length is bounded.
    expect(cornerX - minX).toBeLessThan(set.interocular * 0.15);
  });

  it("traces the upper lash line and not the lower one", () => {
    const liner = build(["eyeliner"]).masks.eyeliner!;
    expect(at(liner, WIDTH, RIGHT_EYE.cx, RIGHT_EYE.cy - RIGHT_EYE.ry)).toBeGreaterThan(0.3);
    expect(at(liner, WIDTH, RIGHT_EYE.cx, RIGHT_EYE.cy + RIGHT_EYE.ry)).toBeLessThan(0.05);
  });
});

describe("blush mask", () => {
  it("sits on the cheeks", () => {
    const blush = build(["blush"]).masks.blush!;
    expect(at(blush, WIDTH, 185, 300)).toBeGreaterThan(0.5);
    expect(at(blush, WIDTH, 295, 300)).toBeGreaterThan(0.5);
  });

  it("is clipped by the face oval so it cannot spill onto the background", () => {
    const blush = build(["blush"]).masks.blush!;
    // Outside the oval entirely: the feathered ellipse alone would still
    // have weight this far out, the oval multiply is what removes it.
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        const dx = (x - FACE.cx) / (FACE.rx + 4);
        const dy = (y - FACE.cy) / (FACE.ry + 4);
        if (dx * dx + dy * dy > 1.2) {
          expect(blush[y * WIDTH + x]).toBeLessThan(0.01);
        }
      }
    }
  });
});

describe("skin mask", () => {
  it("covers the cheeks but excludes eyes, lips and brows", () => {
    const skin = build(["skin"]).masks.skin!;
    expect(at(skin, WIDTH, 185, 300)).toBeGreaterThan(0.5);
    expect(at(skin, WIDTH, RIGHT_EYE.cx, RIGHT_EYE.cy)).toBeLessThan(0.2);
    expect(at(skin, WIDTH, LIPS.cx, LIPS.cy)).toBeLessThan(0.2);
    expect(at(skin, WIDTH, 213, 151)).toBeLessThan(0.5);
  });
});

describe("highlighter mask", () => {
  it("covers the cheekbones and the nose bridge", () => {
    const highlighter = build(["highlighter"]).masks.highlighter!;
    expect(at(highlighter, WIDTH, 178, 280)).toBeGreaterThan(0.3);
    expect(at(highlighter, WIDTH, 302, 280)).toBeGreaterThan(0.3);
    expect(at(highlighter, WIDTH, 240, 240)).toBeGreaterThan(0.3);
  });
});

describe("brow mask", () => {
  it("fills between each brow's lower and upper edges", () => {
    const brows = build(["brows"]).masks.brows!;
    expect(at(brows, WIDTH, 213, 149)).toBeGreaterThan(0.5);
    expect(at(brows, WIDTH, 267, 149)).toBeGreaterThan(0.5);
    expect(at(brows, WIDTH, 240, 149)).toBeLessThan(0.5);
  });
});

const NAMES: readonly MaskName[] = MASK_NAMES;
describe("mask registry", () => {
  it("covers every product plus the smoothing region", () => {
    expect([...NAMES].sort()).toEqual([
      "blush",
      "brows",
      "eyeliner",
      "eyeshadow",
      "highlighter",
      "lipstick",
      "skin",
    ]);
  });
});
