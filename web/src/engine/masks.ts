/**
 * Soft, face-scaled product masks -- a port of `carmine/masks.py`.
 *
 * Each mask is a Float32Array in [0, 1] where 1.0 means "apply the product
 * at full strength here" and 0.0 means "leave untouched". Every size that
 * matters -- feather radii, line thickness, ellipse axes -- is expressed as
 * a fraction of the interocular distance rather than a raw pixel count, so
 * a mask built for a 320px webcam frame and one built for a 4000px portrait
 * of the same face put makeup in the same relative place with the same
 * relative softness. Those fractions all come from `gen/constants.json`,
 * which is generated from the Python source, so neither side can drift.
 *
 * **Processing resolution.** Masks are built at a reduced resolution whose
 * long side is capped at `PROC_MAX_SIDE` (720) and then sampled by the
 * renderer as textures at whatever size the output actually is. Mask
 * construction is the expensive part of a frame -- seven polygon rasters
 * plus seven Gaussian feathers -- and it scales with pixel count, so a
 * 1080p camera feed would cost roughly four times what a 720p one does for
 * no visible benefit: every mask is heavily blurred before use, so the
 * detail lost to the downscale is detail the feather was about to destroy
 * anyway. Bilinear texture sampling handles the upscale, which is the same
 * smooth interpolation the feather is already producing.
 *
 * **Rasterization.** The Python side draws with OpenCV; here the polygon
 * fill, thick polyline and ellipse fill are implemented directly on the
 * Float32Array rather than going through a 2D canvas. That keeps the result
 * identical in a browser and in a Node test runner (no OffscreenCanvas
 * needed, so the structural tests below run in CI), and avoids depending on
 * a browser's anti-aliasing rules, which differ between engines. Each of the
 * three follows OpenCV's own construction rather than an idealized version
 * of the shape -- see `strokePolyline` for why that distinction is not
 * pedantic on a two-pixel-wide stroke.
 */

import { featherKernelSize, gaussianBlur } from "./blur";
import constants from "../gen/constants.json";

const R = constants.regions;
const M = constants.masks;

/** Long-side cap for the resolution masks are built at. */
export const PROC_MAX_SIDE = 720;

/** Products that have a mask. `skin` backs smoothing, which the live shader
 * path does not run (see renderer.ts). */
export type MaskName =
  | "lipstick"
  | "eyeshadow"
  | "eyeliner"
  | "brows"
  | "blush"
  | "highlighter"
  | "skin";

export const MASK_NAMES: readonly MaskName[] = [
  "lipstick",
  "eyeshadow",
  "eyeliner",
  "brows",
  "blush",
  "highlighter",
  "skin",
];

/** A frame's masks plus the resolution they were built at. */
export interface MaskSet {
  /** Processing-resolution width the masks are sized for. */
  width: number;
  /** Processing-resolution height the masks are sized for. */
  height: number;
  /** Factor applied to source-resolution landmarks to reach proc-res. */
  scale: number;
  /** Interocular distance in proc-res pixels -- every other size derives
   * from this, and the renderer reports it for diagnostics. */
  interocular: number;
  masks: Partial<Record<MaskName, Float32Array>>;
}

type Point = [number, number];

// --- rasterization -------------------------------------------------------

/**
 * Fill a closed polygon as a hard 0/1 region.
 *
 * Vertices are rounded to integers first, matching the Python side's
 * `np.round(points).astype(np.int32)` before `cv2.fillPoly`. Crossings use
 * the half-open rule `(y0 <= y) !== (y1 <= y)`, which drops horizontal edges
 * and counts each vertex exactly once, so the two edges meeting at a vertex
 * cannot both register and punch a hole in the row.
 */
function fillPolygon(
  mask: Float32Array,
  width: number,
  height: number,
  points: Point[],
  value = 1,
): void {
  if (points.length < 3) {
    return;
  }
  const xs = points.map((p) => Math.round(p[0]));
  const ys = points.map((p) => Math.round(p[1]));

  let yMin = Infinity;
  let yMax = -Infinity;
  for (const y of ys) {
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const rowStart = Math.max(0, Math.ceil(yMin));
  const rowEnd = Math.min(height - 1, Math.floor(yMax));

  const crossings: number[] = [];
  for (let y = rowStart; y <= rowEnd; y++) {
    crossings.length = 0;
    for (let i = 0; i < points.length; i++) {
      const j = (i + 1) % points.length;
      const y0 = ys[i];
      const y1 = ys[j];
      if (y0 <= y !== y1 <= y) {
        crossings.push(xs[i] + ((y - y0) / (y1 - y0)) * (xs[j] - xs[i]));
      }
    }
    if (crossings.length < 2) {
      continue;
    }
    crossings.sort((a, b) => a - b);
    const row = y * width;
    for (let k = 0; k + 1 < crossings.length; k += 2) {
      const from = Math.max(0, Math.ceil(crossings[k]));
      const to = Math.min(width - 1, Math.floor(crossings[k + 1]));
      for (let x = from; x <= to; x++) {
        mask[row + x] = value;
      }
    }
  }
}

/** Fill a disk of integer radius -- the round cap OpenCV stamps at each
 * polyline vertex, which is what rounds the joins. */
function fillDisk(
  mask: Float32Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  value: number,
): void {
  const radiusSq = radius * radius;
  const yFrom = Math.max(0, cy - radius);
  const yTo = Math.min(height - 1, cy + radius);
  const xFrom = Math.max(0, cx - radius);
  const xTo = Math.min(width - 1, cx + radius);
  for (let y = yFrom; y <= yTo; y++) {
    const dy = y - cy;
    const row = y * width;
    for (let x = xFrom; x <= xTo; x++) {
      const dx = x - cx;
      if (dx * dx + dy * dy <= radiusSq) {
        mask[row + x] = value;
      }
    }
  }
}

/** An 8-connected Bresenham line -- what `cv2.line` draws at thickness 1,
 * where OpenCV takes a thin-line path instead of building a quad. */
function drawLine8(
  mask: Float32Array,
  width: number,
  height: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  value: number,
): void {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let error = dx - dy;
  let x = x0;
  let y = y0;
  for (;;) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
      mask[y * width + x] = value;
    }
    if (x === x1 && y === y1) {
      break;
    }
    const doubled = 2 * error;
    if (doubled > -dy) {
      error -= dy;
      x += sx;
    }
    if (doubled < dx) {
      error += dx;
      y += sy;
    }
  }
}

/**
 * Stroke an open polyline of the given thickness, writing `value` (1 to
 * draw, 0 to cut a region back out -- the skin mask uses both).
 *
 * This mirrors how `cv2.polylines` actually builds a thick line, which is
 * not the capsule ("every pixel within thickness/2 of the segment") the
 * shape suggests. Two details are worth stating, because getting either
 * wrong moves a hairline stroke by a whole pixel:
 *
 * 1. **Vertices are rounded to integers first.** The Python side hands
 *    OpenCV an int32 array (`np.round(points).astype(np.int32)`), so it
 *    strokes between pixel centers rather than between the sub-pixel
 *    landmark positions.
 * 2. **The perpendicular offset is quantized to whole pixels.** OpenCV
 *    builds a quad from an offset vector whose components it rounds, so a
 *    45-degree segment of thickness 2 comes out about `sqrt(2)` times wider
 *    than its nominal width rather than exactly two pixels across. A capsule
 *    keeps the nominal width at every angle and therefore disagrees most on
 *    diagonals -- measured against `cv2.polylines` over 400 random segments,
 *    the capsule missed by ~22 pixels per segment and this construction by
 *    ~6. On the eyeliner mask (two pixels wide under a sigma-1 feather, the
 *    one place in the engine where a one-pixel disagreement has nothing to
 *    hide behind) the two fixes together took the worst-case parity ΔE from
 *    23.1 to 11.4.
 *
 * The residual is OpenCV's fixed-point scanline fill, which rasterizes the
 * quad slightly differently than the `cv2.fillPoly`-compatible rule
 * `fillPolygon` implements. Broad strokes (the highlighter's, under a
 * sigma-5 feather) were never sensitive to it either way.
 */
function strokePolyline(
  mask: Float32Array,
  width: number,
  height: number,
  points: Point[],
  thickness: number,
  value = 1,
): void {
  if (points.length < 2) {
    return;
  }
  const vertices: Point[] = points.map((p) => [Math.round(p[0]), Math.round(p[1])]);

  if (thickness <= 1) {
    for (let i = 0; i + 1 < vertices.length; i++) {
      const [x0, y0] = vertices[i];
      const [x1, y1] = vertices[i + 1];
      drawLine8(mask, width, height, x0, y0, x1, y1, value);
    }
    return;
  }

  // OpenCV widens odd thicknesses by half a pixel on each side before
  // rounding, so 3 behaves like 4; the cap radius rounds the same way.
  const halfWidth = (thickness + (thickness % 2)) / 2;
  const capRadius = (thickness + 1) >> 1;

  for (let i = 0; i + 1 < vertices.length; i++) {
    const [x0, y0] = vertices[i];
    const [x1, y1] = vertices[i + 1];
    // Note the sign convention: the offset is (dy, dx) with dx negated,
    // which is the perpendicular OpenCV uses.
    const dx = x0 - x1;
    const dy = y1 - y0;
    const length = Math.hypot(dx, dy);
    if (length > 0) {
      const scale = halfWidth / length;
      const offsetX = Math.round(dy * scale);
      const offsetY = Math.round(dx * scale);
      fillPolygon(
        mask,
        width,
        height,
        [
          [x0 + offsetX, y0 + offsetY],
          [x0 - offsetX, y0 - offsetY],
          [x1 - offsetX, y1 - offsetY],
          [x1 + offsetX, y1 + offsetY],
        ],
        value,
      );
    }
    fillDisk(mask, width, height, x0, y0, capRadius, value);
    fillDisk(mask, width, height, x1, y1, capRadius, value);
  }
}

/**
 * Fill a rotated ellipse, matching `cv2.ellipse`'s angle convention:
 * positive degrees rotate from the +x axis toward +y, which is clockwise on
 * screen because y grows downward.
 */
function fillEllipse(
  mask: Float32Array,
  width: number,
  height: number,
  center: Point,
  axisX: number,
  axisY: number,
  angleDegrees: number,
): void {
  if (axisX <= 0 || axisY <= 0) {
    return;
  }
  const radians = (angleDegrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const reach = Math.max(axisX, axisY) + 1;

  const xFrom = Math.max(0, Math.floor(center[0] - reach));
  const xTo = Math.min(width - 1, Math.ceil(center[0] + reach));
  const yFrom = Math.max(0, Math.floor(center[1] - reach));
  const yTo = Math.min(height - 1, Math.ceil(center[1] + reach));

  for (let y = yFrom; y <= yTo; y++) {
    for (let x = xFrom; x <= xTo; x++) {
      const u = x - center[0];
      const v = y - center[1];
      const ex = (u * cos + v * sin) / axisX;
      const ey = (-u * sin + v * cos) / axisY;
      if (ex * ex + ey * ey <= 1) {
        mask[y * width + x] = 1;
      }
    }
  }
}

// --- helpers -------------------------------------------------------------

function feather(
  mask: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  if (sigma <= 0) {
    return mask;
  }
  return gaussianBlur(mask, width, height, sigma, featherKernelSize(sigma));
}

function subtractClamped(target: Float32Array, other: Float32Array): void {
  for (let i = 0; i < target.length; i++) {
    const value = target[i] - other[i];
    target[i] = value < 0 ? 0 : value > 1 ? 1 : value;
  }
}

function maxInto(target: Float32Array, other: Float32Array): void {
  for (let i = 0; i < target.length; i++) {
    if (other[i] > target[i]) {
      target[i] = other[i];
    }
  }
}

/** Linear interpolation with clamped ends -- `np.interp` on a two-point
 * table, which is all the crease gradient needs. */
function interpolate(x: number, x0: number, x1: number, y0: number, y1: number): number {
  if (x <= x0) return y0;
  if (x >= x1) return y1;
  if (x1 === x0) return y1;
  return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
}

/**
 * Per-frame drawing context: proc-res dimensions, the scaled landmarks, and
 * the interocular distance every size is measured in.
 */
class MaskBuilder {
  readonly width: number;
  readonly height: number;
  readonly interocular: number;
  private readonly points: Float64Array;

  constructor(landmarks: ArrayLike<number>, width: number, height: number, scale: number) {
    this.width = width;
    this.height = height;
    this.points = new Float64Array(landmarks.length);
    for (let i = 0; i < landmarks.length; i++) {
      this.points[i] = landmarks[i] * scale;
    }
    const dx = this.point(R.LEFT_EYE_OUTER)[0] - this.point(R.RIGHT_EYE_OUTER)[0];
    const dy = this.point(R.LEFT_EYE_OUTER)[1] - this.point(R.RIGHT_EYE_OUTER)[1];
    const distance = Math.hypot(dx, dy);
    if (!(distance >= 1e-6)) {
      throw new Error(`Degenerate landmarks: interocular distance ${distance} is too small`);
    }
    this.interocular = distance;
  }

  point(index: number): Point {
    return [this.points[index * 2], this.points[index * 2 + 1]];
  }

  ring(indices: readonly number[]): Point[] {
    return indices.map((index) => this.point(index));
  }

  blank(): Float32Array {
    return new Float32Array(this.width * this.height);
  }

  polygon(points: Point[]): Float32Array {
    const mask = this.blank();
    fillPolygon(mask, this.width, this.height, points);
    return mask;
  }

  feather(mask: Float32Array, sigma: number): Float32Array {
    return feather(mask, this.width, this.height, sigma);
  }

  stroke(mask: Float32Array, points: Point[], thickness: number, value = 1): void {
    strokePolyline(mask, this.width, this.height, points, thickness, value);
  }
}

// --- individual masks ----------------------------------------------------

/**
 * Lipstick: the outer mouth contour minus the inner opening.
 *
 * Filling only the outer contour would paint straight across an open mouth;
 * subtracting the inner ring keeps color off teeth and tongue however wide
 * the mouth is.
 */
function lipMask(b: MaskBuilder): Float32Array {
  const mask = b.polygon(b.ring(R.LIPS_OUTER));
  subtractClamped(mask, b.polygon(b.ring(R.LIPS_INNER)));
  return b.feather(mask, b.interocular * M.lip.feather);
}

/**
 * Eyeshadow: the upper lid, swept 60% of the way toward the brow.
 *
 * Per eye, the brow arc is resampled so each lash-line point has a matching
 * brow point, and the polygon runs along the lash line and back across the
 * interpolated upper edge. On top of that sits a crease gradient -- full
 * strength at the lash line fading to 0.35 at the top -- because real
 * eyeshadow is heaviest at the lashes and lightest on the brow bone. The eye
 * opening is cut out before the final feather so the shadow does not wash
 * over the eyeball itself.
 */
function eyeshadowMask(b: MaskBuilder): Float32Array {
  const mask = b.blank();
  const pairs: Array<[readonly number[], readonly number[]]> = [
    [R.RIGHT_EYE_UPPER, R.RIGHT_BROW_LOWER],
    [R.LEFT_EYE_UPPER, R.LEFT_BROW_LOWER],
  ];

  for (const [lidIndices, browIndices] of pairs) {
    const lid = b.ring(lidIndices);
    const brow = b.ring(browIndices);

    // Resample the brow arc onto the lid's parameterization so the two
    // arcs have matching point counts regardless of how many landmarks
    // each contour carries.
    const upper: Point[] = lid.map((point, i) => {
      const t = lid.length > 1 ? i / (lid.length - 1) : 0;
      const source = t * (brow.length - 1);
      const lower = Math.min(Math.floor(source), brow.length - 1);
      const upperIndex = Math.min(lower + 1, brow.length - 1);
      const frac = source - lower;
      const bx = brow[lower][0] + (brow[upperIndex][0] - brow[lower][0]) * frac;
      const by = brow[lower][1] + (brow[upperIndex][1] - brow[lower][1]) * frac;
      return [
        point[0] + (bx - point[0]) * M.eyeshadow.lid_to_brow,
        point[1] + (by - point[1]) * M.eyeshadow.lid_to_brow,
      ];
    });

    const eyeMask = b.polygon([...lid, ...upper.slice().reverse()]);

    const lashRow = lid.reduce((sum, p) => sum + p[1], 0) / lid.length;
    const creaseRow = upper.reduce((sum, p) => sum + p[1], 0) / upper.length;
    const rowMin = Math.max(0, Math.floor(Math.min(lashRow, creaseRow)));
    const rowMax = Math.min(b.height - 1, Math.ceil(Math.max(lashRow, creaseRow)));
    for (let y = rowMin; y <= rowMax; y++) {
      // creaseRow is the smaller y (nearer the brow, higher on screen), so
      // it is the low end of the interpolation table.
      const factor = interpolate(
        y,
        creaseRow,
        lashRow,
        M.eyeshadow.crease_top_factor,
        M.eyeshadow.crease_lash_factor,
      );
      const row = y * b.width;
      for (let x = 0; x < b.width; x++) {
        eyeMask[row + x] *= factor;
      }
    }

    maxInto(mask, eyeMask);
  }

  for (const eyeIndices of [R.RIGHT_EYE, R.LEFT_EYE]) {
    subtractClamped(mask, b.polygon(b.ring(eyeIndices)));
  }
  return b.feather(mask, b.interocular * M.eyeshadow.feather);
}

/** Eyeliner: a narrow stroke along each upper lash line, winged at the
 * outer corner by extending the last segment's direction lifted upward. */
function eyelinerMask(b: MaskBuilder): Float32Array {
  const thickness = Math.max(
    M.eyeliner.thickness_min,
    Math.round(b.interocular * M.eyeliner.thickness_factor),
  );
  const mask = b.blank();
  for (const arcIndices of [R.RIGHT_EYE_UPPER, R.LEFT_EYE_UPPER]) {
    const arc = b.ring(arcIndices);
    const last = arc[arc.length - 1];
    const previous = arc[arc.length - 2];
    const dx = last[0] - previous[0];
    const dy = last[1] - previous[1];
    const norm = Math.hypot(dx, dy);
    if (norm > 1e-6) {
      const ux = dx / norm;
      const uy = dy / norm + M.eyeliner.wing_dy;
      const unit = Math.hypot(ux, uy);
      const length = b.interocular * M.eyeliner.wing_length_factor;
      arc.push([last[0] + (ux / unit) * length, last[1] + (uy / unit) * length]);
    }
    b.stroke(mask, arc, thickness);
  }
  return b.feather(
    mask,
    Math.max(M.eyeliner.feather_min, b.interocular * M.eyeliner.feather_factor),
  );
}

/**
 * Blush: a feathered ellipse on each cheek, multiplied by a feathered face
 * oval. Without the oval, the ellipse's soft edge spills onto hair and
 * background near the jawline, which a plain fill cannot prevent.
 */
function blushMask(b: MaskBuilder): Float32Array {
  const axisX = Math.round(b.interocular * M.blush.axis_x);
  const axisY = Math.round(b.interocular * M.blush.axis_y);
  const mask = b.blank();
  const cheeks: Array<[number, number]> = [
    [R.RIGHT_CHEEK, -M.blush.angle],
    [R.LEFT_CHEEK, M.blush.angle],
  ];
  for (const [index, angle] of cheeks) {
    const center = b.point(index);
    fillEllipse(
      mask,
      b.width,
      b.height,
      [Math.round(center[0]), Math.round(center[1])],
      axisX,
      axisY,
      angle,
    );
  }
  const feathered = b.feather(mask, b.interocular * M.blush.feather);
  const face = b.feather(
    b.polygon(b.ring(R.FACE_OVAL)),
    b.interocular * M.blush.oval_feather,
  );
  for (let i = 0; i < feathered.length; i++) {
    feathered[i] *= face[i];
  }
  return feathered;
}

/** Brows: each eyebrow's lower edge followed by its upper edge reversed,
 * which traces the brow's perimeter as one closed polygon. */
function browMask(b: MaskBuilder): Float32Array {
  const mask = b.blank();
  const pairs: Array<[readonly number[], readonly number[]]> = [
    [R.RIGHT_BROW_LOWER, R.RIGHT_BROW_UPPER],
    [R.LEFT_BROW_LOWER, R.LEFT_BROW_UPPER],
  ];
  for (const [lowerIndices, upperIndices] of pairs) {
    const polygon = [...b.ring(lowerIndices), ...b.ring(upperIndices).reverse()];
    maxInto(mask, b.polygon(polygon));
  }
  return b.feather(mask, b.interocular * M.brow.feather);
}

/**
 * Highlighter: the cheekbone crests plus the bridge of the nose, drawn as
 * thick feathered strips rather than filled polygons -- highlighter is
 * swept along these features, not packed into an enclosed area. The nose is
 * feathered separately because it is a much narrower stroke, and blurring
 * the two together would let the wide cheek strip dominate it.
 */
function highlighterMask(b: MaskBuilder): Float32Array {
  const cheekThickness = Math.max(
    1,
    Math.round(b.interocular * M.highlighter.cheek_thickness_factor),
  );
  const cheeks = b.blank();
  for (const arcIndices of [R.RIGHT_CHEEKBONE, R.LEFT_CHEEKBONE]) {
    b.stroke(cheeks, b.ring(arcIndices), cheekThickness);
  }
  const cheekMask = b.feather(cheeks, b.interocular * M.highlighter.cheek_feather);

  const noseThickness = Math.max(
    1,
    Math.round(b.interocular * M.highlighter.nose_thickness_factor),
  );
  const nose = b.blank();
  b.stroke(nose, b.ring(R.NOSE_BRIDGE), noseThickness);
  const noseMask = b.feather(nose, b.interocular * M.highlighter.nose_feather);

  for (let i = 0; i < cheekMask.length; i++) {
    const value = Math.max(cheekMask[i], noseMask[i]);
    cheekMask[i] = value > 1 ? 1 : value < 0 ? 0 : value;
  }
  return cheekMask;
}

/** Skin: the face oval minus eyes, lips and brows -- the smoothing region. */
function skinMask(b: MaskBuilder): Float32Array {
  const mask = b.polygon(b.ring(R.FACE_OVAL));
  for (const indices of [R.LIPS_OUTER, R.RIGHT_EYE, R.LEFT_EYE]) {
    subtractClamped(
      mask,
      b.feather(b.polygon(b.ring(indices)), b.interocular * M.skin.exclusion_feather),
    );
  }
  const browThickness = Math.max(
    M.skin.brow_cutout_thickness_min,
    Math.trunc(b.interocular * M.skin.brow_cutout_thickness_factor),
  );
  for (const indices of [R.RIGHT_BROW_LOWER, R.LEFT_BROW_LOWER]) {
    b.stroke(mask, b.ring(indices), browThickness, 0);
  }
  return b.feather(mask, b.interocular * M.skin.feather);
}

const BUILDERS: Record<MaskName, (b: MaskBuilder) => Float32Array> = {
  lipstick: lipMask,
  eyeshadow: eyeshadowMask,
  eyeliner: eyelinerMask,
  brows: browMask,
  blush: blushMask,
  highlighter: highlighterMask,
  skin: skinMask,
};

/** Processing-resolution dimensions for a source of the given size. */
export function processingSize(
  width: number,
  height: number,
): { width: number; height: number; scale: number } {
  const longSide = Math.max(width, height);
  const scale = longSide > PROC_MAX_SIDE ? PROC_MAX_SIDE / longSide : 1;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
    scale,
  };
}

/**
 * Build masks for exactly the requested products.
 *
 * Only the requested masks are built: a look with lipstick alone should not
 * pay for six unused polygon rasters and their feathers every frame, which
 * on a phone is the difference between a smooth preview and a stuttering
 * one.
 *
 * @param landmarks Flat `[x0, y0, x1, y1, ...]` pixel coordinates in the
 *   *source's* resolution; they are scaled to proc-res here.
 * @param width Source width in pixels.
 * @param height Source height in pixels.
 * @param activeProducts Mask names to build; unknown names are ignored.
 */
export function buildMasks(
  landmarks: ArrayLike<number>,
  width: number,
  height: number,
  activeProducts: Set<string>,
): MaskSet {
  const size = processingSize(width, height);
  const builder = new MaskBuilder(landmarks, size.width, size.height, size.scale);
  const masks: Partial<Record<MaskName, Float32Array>> = {};
  for (const name of MASK_NAMES) {
    if (activeProducts.has(name)) {
      masks[name] = BUILDERS[name](builder);
    }
  }
  return {
    width: size.width,
    height: size.height,
    scale: size.scale,
    interocular: builder.interocular,
    masks,
  };
}
