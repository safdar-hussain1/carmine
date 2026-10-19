/**
 * Drawing the face mesh as light.
 *
 * The face model fits a mesh of 478 points and 2,556 edges to the face.
 * This draws it the way a scanner's display would: thin lines and bright
 * points added onto the picture as light, nearer edges brighter than far
 * ones, in a soft rose-to-lilac-to-champagne gradient across the face.
 * Callers project the points first (see `mesh3d.ts`); this only paints.
 */

import { FaceLandmarker } from "@mediapipe/tasks-vision";

/** Every mesh edge as a pair of point indices. */
export const MESH_EDGES: Uint16Array = (() => {
  const edges = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  const out = new Uint16Array(edges.length * 2);
  edges.forEach((edge, i) => {
    out[i * 2] = edge.start;
    out[i * 2 + 1] = edge.end;
  });
  return out;
})();

/** Depth bands the edges are sorted into, far to near. */
const BANDS = 5;

/** Each band's edges as index pairs, reused from frame to frame. */
const bandEdges = Array.from({ length: BANDS }, () => new Uint16Array(MESH_EDGES.length));
const bandSizes = new Int32Array(BANDS);

/** A soft halo, then a bright core, both as added light. */
const POINT_PASSES = [
  { radius: 2.6, alpha: 0.16 },
  { radius: 0.95, alpha: 1 },
] as const;

export interface MeshPaint {
  /** Projected x/y pairs in canvas pixels. */
  points: Float32Array;
  /** Depth per point after the turn (see `project`). */
  depth: Float32Array;
  /** How much of each point has arrived, 0 to 1; omitted means all of it. */
  arrival?: Float32Array;
  /** Overall strength of the edges and of the points, 0 to 1. */
  edgeAlpha: number;
  pointAlpha: number;
  /** Size of one design pixel in canvas pixels (the device pixel ratio,
   * scaled by how big the face is drawn). */
  unit: number;
}

/** The gradient the mesh is drawn in, spanning the face's width. */
export function meshGradient(
  ctx: CanvasRenderingContext2D,
  left: number,
  right: number,
): CanvasGradient {
  const gradient = ctx.createLinearGradient(left, 0, right, 0);
  gradient.addColorStop(0, "#ff9fbf");
  gradient.addColorStop(0.5, "#c9b0ff");
  gradient.addColorStop(1, "#ffd9a8");
  return gradient;
}

/** The horizontal extent of a set of points. */
export function extent(points: Float32Array): { left: number; right: number } {
  let left = Infinity;
  let right = -Infinity;
  for (let i = 0; i < points.length; i += 2) {
    left = Math.min(left, points[i]);
    right = Math.max(right, points[i]);
  }
  return { left, right };
}

export function drawMesh(ctx: CanvasRenderingContext2D, paint: MeshPaint): void {
  const { points, depth, arrival, unit } = paint;
  const count = depth.length;
  if (count === 0) {
    return;
  }
  let near = Infinity;
  let far = -Infinity;
  for (let i = 0; i < count; i++) {
    near = Math.min(near, depth[i]);
    far = Math.max(far, depth[i]);
  }
  const range = far - near || 1;
  const { left, right } = extent(points);
  const gradient = meshGradient(ctx, left, right);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";

  if (paint.edgeAlpha > 0) {
    bandSizes.fill(0);
    for (let e = 0; e < MESH_EDGES.length; e += 2) {
      const a = MESH_EDGES[e];
      const b = MESH_EDGES[e + 1];
      if (arrival && (arrival[a] < 0.92 || arrival[b] < 0.92)) {
        continue;
      }
      const d = ((depth[a] + depth[b]) / 2 - near) / range;
      const band = Math.min(BANDS - 1, Math.floor((1 - d) * BANDS));
      bandEdges[band][bandSizes[band]++] = a;
      bandEdges[band][bandSizes[band]++] = b;
    }
    ctx.strokeStyle = gradient;
    for (let band = 0; band < BANDS; band++) {
      const list = bandEdges[band];
      const size = bandSizes[band];
      if (size === 0) {
        continue;
      }
      // Far edges thin and faint, near ones brighter: the mesh reads as a
      // solid shape rather than a flat net.
      const closeness = (band + 1) / BANDS;
      ctx.globalAlpha = paint.edgeAlpha * (0.18 + 0.5 * closeness);
      ctx.lineWidth = unit * (0.45 + 0.5 * closeness);
      ctx.beginPath();
      for (let i = 0; i < size; i += 2) {
        const a = list[i] * 2;
        const b = list[i + 1] * 2;
        ctx.moveTo(points[a], points[a + 1]);
        ctx.lineTo(points[b], points[b + 1]);
      }
      ctx.stroke();
    }
  }

  if (paint.pointAlpha > 0) {
    for (const pass of POINT_PASSES) {
      ctx.fillStyle = pass.radius > 1 ? gradient : "#fff6ee";
      ctx.beginPath();
      let drawn = 0;
      for (let i = 0; i < count; i++) {
        const arrived = arrival ? arrival[i] : 1;
        if (arrived <= 0) {
          continue;
        }
        const d = 1 - (depth[i] - near) / range;
        const r = unit * pass.radius * (0.7 + 0.5 * d) * (0.6 + 0.4 * arrived);
        const x = points[i * 2];
        const y = points[i * 2 + 1];
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, Math.PI * 2);
        drawn++;
      }
      if (drawn > 0) {
        ctx.globalAlpha = paint.pointAlpha * pass.alpha;
        ctx.fill();
      }
    }
  }
  ctx.restore();
}
