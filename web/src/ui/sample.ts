/**
 * The sample portrait's face, found ahead of time.
 *
 * `scripts/export_sample_landmarks.py` runs the face model over
 * `public/demo/model.jpg` once and writes the 478 landmarks, with depth, to
 * `gen/sample_landmarks.json`. Everything the page draws of the sample -- the
 * opening scan, the looks, the how-it-works animations -- starts from these,
 * so the page can move the moment it loads instead of first downloading the
 * face model and its wasm runtime, about 16 MB together. Those are fetched
 * only for a camera or a photo.
 */

import data from "../gen/sample_landmarks.json";

export interface SampleFace {
  /** The photo's own size in pixels. */
  width: number;
  height: number;
  /** x/y pairs in the photo's pixels, the layout every mask builder takes. */
  xy: Float32Array;
  /** Depth per point, in the same pixel scale; smaller is nearer the camera. */
  z: Float32Array;
}

function unpack(points: number[][]): { xy: Float32Array; z: Float32Array } {
  const xy = new Float32Array(points.length * 2);
  const z = new Float32Array(points.length);
  points.forEach((point, i) => {
    xy[i * 2] = point[0];
    xy[i * 2 + 1] = point[1];
    z[i] = point[2];
  });
  return { xy, z };
}

export const SAMPLE: SampleFace = {
  width: data.width,
  height: data.height,
  ...unpack(data.points),
};
