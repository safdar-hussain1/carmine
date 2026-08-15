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
 *
 * `approximateGaussianBlur` at the bottom of this file is the exception that
 * proves the rule: it *is* a stack of box blurs, it is used only by the live
 * camera path, and it is never what parity is measured against. It lives
 * beside the exact version so the two are read together.
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

// --- the live approximation ----------------------------------------------

/**
 * Box widths whose repeated application approximates a Gaussian of `sigma`.
 *
 * Three box blurs in succession converge on a Gaussian (the central limit
 * theorem, applied to the uniform kernel), and the standard construction
 * picks widths whose combined variance matches the target's: start from the
 * ideal width `sqrt(12 sigma^2 / n + 1)`, take the nearest odd width below
 * it, and use the next odd width up for however many of the passes are
 * needed to make up the remaining variance.
 *
 * Widths are odd so each box is symmetric about its pixel, and clamped to a
 * minimum of 1 (a width-1 box is the identity, which is the right answer for
 * a sigma too small to blur anything).
 *
 * Integer widths cannot hit an arbitrary variance exactly, and the residual
 * is not uniform: it is under 8% for the wide feathers that dominate the
 * mask stage's cost, but reaches ~35% at sigma 1, where the eyeliner's
 * feather sits. That is a large fraction of a very small blur -- the
 * difference between a one-pixel soft edge and a slightly tighter one -- and
 * it lands only on the live path, which is never what parity is measured
 * against. `blur.test.ts` pins both ends of that range so the trade stays
 * visible instead of becoming folklore.
 */
export function boxSizesForGaussian(sigma: number, passes = 3): number[] {
  const variance = 12 * sigma * sigma;
  const ideal = Math.sqrt(variance / passes + 1);
  let lower = Math.floor(ideal);
  if (lower % 2 === 0) {
    lower--;
  }
  if (lower < 1) {
    lower = 1;
  }
  const upper = lower + 2;
  const idealCount =
    (variance - passes * lower * lower - 4 * passes * lower - 3 * passes) / (-4 * lower - 4);
  const count = Math.round(idealCount);
  const sizes: number[] = [];
  for (let i = 0; i < passes; i++) {
    sizes.push(i < count ? lower : upper);
  }
  return sizes;
}

/**
 * Reflect101 index table covering `[-radius, n + radius)`.
 *
 * The box passes below walk a sliding window, so they need a border index
 * for every position the window can reach. Precomputing the fold once per
 * axis keeps the inner loop free of the modulo arithmetic `reflect101` does.
 */
function reflectTable(n: number, radius: number): Int32Array {
  const table = new Int32Array(n + 2 * radius);
  for (let i = 0; i < table.length; i++) {
    table[i] = reflect101(i - radius, n);
  }
  return table;
}

/**
 * One separable box blur of the given odd width, with reflected borders.
 *
 * Cost is independent of the width: each output pixel adds one sample and
 * drops one from a running sum. That is the entire reason this path exists
 * -- a true Gaussian at the radii the masks use costs one multiply per tap
 * per pixel, and the widest feather in a look is over a hundred taps.
 */
function boxBlurPass(
  src: Float32Array,
  dst: Float32Array,
  width: number,
  height: number,
  size: number,
  scratch: Float32Array,
): void {
  const radius = (size - 1) >> 1;
  if (radius <= 0) {
    dst.set(src);
    return;
  }
  const inverse = 1 / size;

  const columns = reflectTable(width, radius);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let k = 0; k < size; k++) {
      sum += src[row + columns[k]];
    }
    scratch[row] = sum * inverse;
    for (let x = 1; x < width; x++) {
      sum += src[row + columns[x + size - 1]] - src[row + columns[x - 1]];
      scratch[row + x] = sum * inverse;
    }
  }

  const rows = reflectTable(height, radius);
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let k = 0; k < size; k++) {
      sum += scratch[rows[k] * width + x];
    }
    dst[x] = sum * inverse;
    for (let y = 1; y < height; y++) {
      sum += scratch[rows[y + size - 1] * width + x] - scratch[rows[y - 1] * width + x];
      dst[y * width + x] = sum * inverse;
    }
  }
}

/**
 * A three-box approximation of `gaussianBlur`, for the live camera path.
 *
 * **This is not the reference.** `gaussianBlur` above is what the parity
 * checks measure and what every still render uses; this trades a small,
 * bounded shape error for a cost that no longer grows with the blur radius.
 * The approximation is good -- three boxes land within about 3% of a true
 * Gaussian's profile -- and it is applied to mask weights that are then
 * quantized to 8 bits for the GPU and sampled bilinearly, so the error is
 * comfortably beneath what the rest of the live path already tolerates.
 *
 * What it buys: the mask stage stops being the frame's bottleneck. At a
 * face filling a 720p frame the widest feather is a 121-tap Gaussian; the
 * boxes replace it with three constant-time passes.
 */
export function approximateGaussianBlur(
  src: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  if (sigma <= 0) {
    return src.slice();
  }
  const sizes = boxSizesForGaussian(sigma);
  const count = width * height;
  // Three buffers for the whole thing, ping-ponged: this runs several times
  // per live frame, and allocating a fresh output per pass would hand the
  // collector megabytes a second.
  const first = new Float32Array(count);
  const second = new Float32Array(count);
  const scratch = new Float32Array(count);
  let input: Float32Array = src;
  let output = first;
  for (const size of sizes) {
    boxBlurPass(input, output, width, height, size, scratch);
    input = output;
    output = output === first ? second : first;
  }
  return input;
}
