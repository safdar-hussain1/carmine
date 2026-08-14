/**
 * sRGB <-> CIELAB conversion matching OpenCV's float32 convention.
 *
 * `carmine.pigment` does all of its work in the Lab space OpenCV produces
 * for float32 input: L in [0, 100] and a/b centered on 0 (the uint8 path's
 * 0-255 rescaling never happens). The browser engine has to land in the
 * *same* space or every tuned constant on the Python side -- lightness
 * pulls, the 18.0 gloss factor, the matte blur -- would mean something
 * different here.
 *
 * The math below is the textbook definition: IEC 61966-2-1 sRGB decoding,
 * the D65 primaries matrix OpenCV ships, and the f(t) cube-root with
 * delta = 6/29. OpenCV's float path is *not* quite the textbook definition
 * -- it evaluates both the gamma decode and the cube root through
 * spline-interpolated 1024-entry lookup tables -- so the two agree to about
 * 0.43 Lab units in the worst corner of the 8-bit cube rather than
 * exactly. That residual is what the tolerances in color.test.ts are sized
 * against; it is far below a perceptible difference and well below the
 * 1/255 quantization that both sides land on when they convert back.
 */

/** [L, a, b] with L in [0, 100] and a/b roughly in [-128, 127]. */
export type Lab = [number, number, number];
/** [r, g, b] with channels in [0, 255]. */
export type Rgb = [number, number, number];

// OpenCV's sRGB (D65) primaries matrix -- cv::COLOR_BGR2XYZ's coefficients,
// listed row-major as X, Y, Z. Kept to OpenCV's 6-decimal values rather
// than the "more correct" full-precision ones so the two implementations
// start from the same numbers.
const RGB_TO_XYZ = [
  0.412453, 0.35758, 0.180423, 0.212671, 0.71516, 0.072169, 0.019334, 0.119193, 0.950227,
];

const XYZ_TO_RGB = [
  3.240479, -1.53715, -0.498535, -0.969256, 1.875991, 0.041556, 0.055648, -0.204043, 1.057311,
];

// D65 white point, OpenCV's values.
const WHITE_X = 0.950456;
const WHITE_Y = 1.0;
const WHITE_Z = 1.088754;

const DELTA = 6 / 29;
const DELTA_CUBED = DELTA * DELTA * DELTA;
const THREE_DELTA_SQ = 3 * DELTA * DELTA;

/** IEC 61966-2-1 sRGB decode: gamma-encoded [0,1] -> linear [0,1]. */
export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Inverse of `srgbToLinear`. */
export function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

function labF(t: number): number {
  return t > DELTA_CUBED ? Math.cbrt(t) : t / THREE_DELTA_SQ + 4 / 29;
}

function labFInverse(t: number): number {
  return t > DELTA ? t * t * t : THREE_DELTA_SQ * (t - 4 / 29);
}

/**
 * Convert one sRGB triple (channels in [0, 255]) to Lab.
 *
 * Note the L branch: OpenCV keeps the CIE standard's 903.3 linear segment
 * for very dark colors rather than deriving it from f(t), and near-black
 * pixels are common enough in eyeliner and lash-line regions that the
 * difference is worth mirroring.
 */
export function rgbToLab(rgb: Rgb | ArrayLike<number>): Lab {
  const r = srgbToLinear(rgb[0] / 255);
  const g = srgbToLinear(rgb[1] / 255);
  const b = srgbToLinear(rgb[2] / 255);

  const x = (RGB_TO_XYZ[0] * r + RGB_TO_XYZ[1] * g + RGB_TO_XYZ[2] * b) / WHITE_X;
  const y = (RGB_TO_XYZ[3] * r + RGB_TO_XYZ[4] * g + RGB_TO_XYZ[5] * b) / WHITE_Y;
  const z = (RGB_TO_XYZ[6] * r + RGB_TO_XYZ[7] * g + RGB_TO_XYZ[8] * b) / WHITE_Z;

  const fx = labF(x);
  const fy = labF(y);
  const fz = labF(z);

  const l = y > DELTA_CUBED ? 116 * fy - 16 : 903.3 * y;
  return [l, 500 * (fx - fy), 200 * (fy - fz)];
}

/**
 * Convert one Lab triple back to sRGB channels in [0, 255].
 *
 * The result is *not* clamped or rounded -- callers that need bytes do
 * their own clamp-and-truncate so they can match NumPy's
 * `np.clip(...).astype(np.uint8)` semantics exactly.
 */
export function labToRgb(lab: Lab | ArrayLike<number>): Rgb {
  const fy = (lab[0] + 16) / 116;
  const fx = fy + lab[1] / 500;
  const fz = fy - lab[2] / 200;

  const x = labFInverse(fx) * WHITE_X;
  const y = labFInverse(fy) * WHITE_Y;
  const z = labFInverse(fz) * WHITE_Z;

  const r = XYZ_TO_RGB[0] * x + XYZ_TO_RGB[1] * y + XYZ_TO_RGB[2] * z;
  const g = XYZ_TO_RGB[3] * x + XYZ_TO_RGB[4] * y + XYZ_TO_RGB[5] * z;
  const b = XYZ_TO_RGB[6] * x + XYZ_TO_RGB[7] * y + XYZ_TO_RGB[8] * z;

  return [linearToSrgb(r) * 255, linearToSrgb(g) * 255, linearToSrgb(b) * 255];
}

/**
 * Parse a `#RRGGBB` (or bare `RRGGBB`) string into [r, g, b].
 *
 * Mirrors `carmine.pigment.parse_hex_color`, including its rejection of
 * 3-digit shorthand -- Looks round-trip between the two engines, so a
 * string one side accepts and the other does not would be a silent
 * divergence.
 */
export function hexToRgb(value: string): Rgb {
  const match = /^#?([0-9a-fA-F]{6})$/.exec(value.trim());
  if (!match) {
    throw new Error(`invalid hex color ${JSON.stringify(value)}, expected '#RRGGBB'`);
  }
  const digits = match[1];
  return [
    parseInt(digits.slice(0, 2), 16),
    parseInt(digits.slice(2, 4), 16),
    parseInt(digits.slice(4, 6), 16),
  ];
}
