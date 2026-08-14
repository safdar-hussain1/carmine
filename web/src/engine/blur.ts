/**
 * Separable Gaussian blur over a single-channel Float32Array, matching
 * `cv2.GaussianBlur`'s conventions.
 *
 * The Python engine feathers every mask and flattens the matte finish with
 * `cv2.GaussianBlur`, so "close enough to a Gaussian" is not close enough
 * here -- a differently-normalized kernel or a different edge rule shifts
 * mask weights by several percent near the face boundary, which shows up as
 * a visible halo rather than as a rounding difference. Three things are
 * mirrored deliberately:
 *
 * 1. Kernel weights come from `cv::getGaussianKernel`'s formula
 *    (`exp(-x^2 / 2 sigma^2)` normalized to sum 1), not from a binomial
 *    approximation or a stack of box blurs.
 * 2. Edges use `BORDER_REFLECT_101` (OpenCV's default): the border pixel
 *    itself is not duplicated, so `abcd` extends as `cba|abcd|dcb`. The
 *    mapping folds repeatedly, because the masks' feather radii can easily
 *    exceed the width of a small processing-resolution image.
 * 3. When the caller does not supply a kernel size, OpenCV's own auto-size
 *    rule for float images is used (see `autoKernelSize`).
 *
 * The intermediate buffer is Float32Array rather than Float64Array on
 * purpose: OpenCV filters a CV_32F image through a float32 buffer, and
 * matching that keeps the accumulated rounding on the same order.
 */

/**
 * OpenCV's rule for `ksize=0`: `cvRound(sigma * k * 2 + 1) | 1`, where k is
 * 3 for 8-bit images and 4 for everything else. Every caller here works in
 * float, so the float constant is the one that applies -- `finish_matte`'s
 * sigma=5 blur of the L channel lands on a 41-tap kernel, not the 31 taps
 * the 8-bit rule would give.
 */
export function autoKernelSize(sigma: number): number {
  return Math.round(sigma * 4 * 2 + 1) | 1;
}

/**
 * The mask-feather kernel rule mirrored from `masks._feather`:
 * `k = int(sigma * 3) * 2 + 1`. Note the truncation -- `int()`, not
 * `round()` -- which is why this is not simply `autoKernelSize`.
 */
export function featherKernelSize(sigma: number): number {
  return Math.trunc(sigma * 3) * 2 + 1;
}

/** `cv::getGaussianKernel(n, sigma)` for sigma > 0. */
export function gaussianKernel(size: number, sigma: number): Float64Array {
  const kernel = new Float64Array(size);
  const scale = -0.5 / (sigma * sigma);
  const center = (size - 1) * 0.5;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - center;
    const value = Math.exp(scale * x * x);
    kernel[i] = value;
    sum += value;
  }
  const inv = 1 / sum;
  for (let i = 0; i < size; i++) {
    kernel[i] *= inv;
  }
  return kernel;
}

/**
 * BORDER_REFLECT_101 index folding, valid for indices arbitrarily far
 * outside `[0, n)`.
 *
 * The reflection has period `2 * (n - 1)`, so one modulo brings any index
 * into a single period and at most one fold finishes the job. A
 * single-pixel axis has no period to speak of and collapses to index 0.
 */
export function reflect101(index: number, n: number): number {
  if (n <= 1) {
    return 0;
  }
  const period = 2 * (n - 1);
  let i = index % period;
  if (i < 0) {
    i += period;
  }
  return i >= n ? period - i : i;
}

/**
 * Blur `src` (a `width * height` single-channel image) with the given sigma.
 *
 * @param ksize Kernel size. Omit (or pass 0) to use `autoKernelSize`.
 * @returns A new Float32Array; `src` is not modified. A non-positive sigma
 *   returns a copy, mirroring `masks._feather`'s early-out.
 */
export function gaussianBlur(
  src: Float32Array,
  width: number,
  height: number,
  sigma: number,
  ksize = 0,
): Float32Array {
  if (sigma <= 0) {
    return src.slice();
  }
  const size = ksize > 0 ? ksize | 1 : autoKernelSize(sigma);
  if (size <= 1) {
    return src.slice();
  }
  const kernel = gaussianKernel(size, sigma);
  const radius = (size - 1) >> 1;

  const horizontal = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < size; k++) {
        acc += kernel[k] * src[row + reflect101(x + k - radius, width)];
      }
      horizontal[row + x] = acc;
    }
  }

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < size; k++) {
        acc += kernel[k] * horizontal[reflect101(y + k - radius, height) * width + x];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}
