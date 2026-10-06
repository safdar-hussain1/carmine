/**
 * Where a picture drawn with `object-fit: cover` actually lands.
 *
 * The mirror crops its frame to an arch, so a landscape photo or camera
 * feed loses its sides. Three things need the exact mapping between the
 * picture's pixels and the box on screen: the before/after handle (dragged
 * in box space, applied in picture space), the landmark points drawn over
 * the face during the opening reveal, and the crop itself, which follows
 * the face instead of the picture's centre.
 */

export interface CoverFit {
  /** Picture pixels to box pixels. */
  scale: number;
  /** Where the picture's left edge sits in the box (zero or negative). */
  offsetX: number;
  /** Where the picture's top edge sits in the box (zero or negative). */
  offsetY: number;
}

/**
 * `object-fit: cover` with `object-position: fx fy` (as fractions), the way
 * CSS lays it out: the picture's fx point lines up with the box's fx point.
 */
export function coverFit(
  boxW: number,
  boxH: number,
  contentW: number,
  contentH: number,
  fx = 0.5,
  fy = 0.5,
): CoverFit {
  const scale = Math.max(boxW / contentW, boxH / contentH);
  return {
    scale,
    offsetX: (boxW - contentW * scale) * fx,
    offsetY: (boxH - contentH * scale) * fy,
  };
}

function axisFocus(box: number, drawn: number, point: number): number {
  const overflow = box - drawn;
  if (overflow > -1e-9) {
    return 0.5;
  }
  const wanted = box / 2 - point;
  return Math.min(Math.max(wanted / overflow, 0), 1);
}

/**
 * The object-position that brings picture point (px, py) as close to the
 * middle of the box as the crop allows, without showing past an edge.
 */
export function focusOn(
  boxW: number,
  boxH: number,
  contentW: number,
  contentH: number,
  px: number,
  py: number,
): { fx: number; fy: number } {
  const scale = Math.max(boxW / contentW, boxH / contentH);
  return {
    fx: axisFocus(boxW, contentW * scale, px * scale),
    fy: axisFocus(boxH, contentH * scale, py * scale),
  };
}

/** A picture point in box coordinates. */
export function contentToBox(fit: CoverFit, x: number, y: number): [number, number] {
  return [fit.offsetX + x * fit.scale, fit.offsetY + y * fit.scale];
}

/** A box x coordinate back in picture pixels, clamped to the picture. */
export function boxToContentX(fit: CoverFit, boxX: number, contentW: number): number {
  const x = (boxX - fit.offsetX) / fit.scale;
  return Math.min(Math.max(x, 0), contentW);
}
