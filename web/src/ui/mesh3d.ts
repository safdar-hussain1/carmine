/**
 * Turning a face's landmarks in 3D.
 *
 * The face model returns a depth for each of its 478 points as well as a
 * position, so the mesh can be shown as the solid shape it is: turned side
 * to side and drawn in perspective. Rotation is about the vertical axis
 * through the face's centre (yaw), then about the horizontal one (pitch).
 * With both angles at zero and perspective off, every point lands exactly
 * where the photo has it -- which is what lets the opening animation settle
 * the turning mesh precisely onto the picture.
 */

export interface Center {
  cx: number;
  cy: number;
}

/** The mean position of a set of x/y pairs. */
export function centroid(xy: ArrayLike<number>): Center {
  let sx = 0;
  let sy = 0;
  const count = xy.length / 2;
  for (let i = 0; i < xy.length; i += 2) {
    sx += xy[i];
    sy += xy[i + 1];
  }
  return { cx: sx / count, cy: sy / count };
}

/**
 * Projects every point after turning the face.
 *
 * @param focal Distance from the eye to the face's centre, in the same
 *   pixels as the points: larger is a flatter, more telephoto look.
 * @param perspective 0 for a flat (orthographic) projection, 1 for full
 *   perspective; the opening animation eases it to 0 as the face settles.
 * @param out Receives the projected x/y pairs.
 * @param depthOut Optionally receives each point's depth after the turn.
 */
export function project(
  xy: ArrayLike<number>,
  z: ArrayLike<number>,
  center: Center,
  yaw: number,
  pitch: number,
  focal: number,
  perspective: number,
  out: Float32Array,
  depthOut?: Float32Array,
): void {
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);
  const count = z.length;
  for (let i = 0; i < count; i++) {
    const x0 = xy[i * 2] - center.cx;
    const y0 = xy[i * 2 + 1] - center.cy;
    const z0 = z[i];
    const x1 = x0 * cosY - z0 * sinY;
    const z1 = x0 * sinY + z0 * cosY;
    const y2 = y0 * cosP - z1 * sinP;
    const z2 = y0 * sinP + z1 * cosP;
    const scale = perspective > 0 ? focal / (focal + z2 * perspective) : 1;
    out[i * 2] = center.cx + x1 * scale;
    out[i * 2 + 1] = center.cy + y2 * scale;
    if (depthOut) {
      depthOut[i] = z2;
    }
  }
}
