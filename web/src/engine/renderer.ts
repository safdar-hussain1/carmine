/**
 * WebGL2 pigment pass -- the live camera path.
 *
 * The whole look is one fragment shader invocation per output pixel: sample
 * the video frame, convert to Lab, apply every product in the same order
 * `carmine.engine.apply_look` does, convert back, write. No intermediate
 * framebuffers and no per-product passes, because on a phone the bandwidth
 * of writing and re-reading a 720p RGBA buffer six times costs more than all
 * of the arithmetic put together.
 *
 * Masks arrive as single-channel R8 textures at processing resolution and
 * are sampled bilinearly at output resolution. Quantizing a mask to 8 bits
 * costs at most 1/255 of weight, which is far below the tolerance the
 * feathered edges are built to; the upscale is invisible for the same reason
 * the downscale was safe (see masks.ts).
 *
 * Two deliberate departures from the CPU reference in `pigment.ts`:
 *
 * 1. **Skin smoothing is not in this path.** It is a bilateral filter --
 *    an edge-preserving neighborhood reduction that does not belong in a
 *    single-pass shader, and whose cost is the reason it is a separate
 *    quality mode rather than part of the live preview.
 * 2. **The matte finish's blur is approximated.** `finish_matte` pulls L
 *    toward a sigma=5 Gaussian of the L channel; here that blur is read
 *    from the frame texture's mip chain at a level chosen to match that
 *    radius. A mip level is a box-ish average of RGB rather than a Gaussian
 *    of L, so the damping is slightly different in character. `pigment.ts`
 *    remains the exact reference, and it is what the parity checks measure.
 *
 * Gloss cannot be done in the shader alone either: it needs the 75th and
 * 99th percentile of L *within the mask*, which is a whole-region
 * reduction. Those two numbers are computed on the CPU once per frame from
 * the downscaled masked image (`pigment.glossPercentiles`) and passed in as
 * uniforms, which keeps the per-pixel work a couple of arithmetic ops. Note
 * that the Python engine measures those percentiles on the image *after*
 * the product's tint, while the live path measures them on the incoming
 * frame -- the tint's own lightness pull is a small, near-uniform shift
 * inside the mask, so it moves both percentiles together and largely
 * cancels out of the spread the highlight map is built from.
 */

import { hexToRgb, rgbToLab } from "./color";
import { lightnessPullFor, PRODUCT_ORDER, type LookConfig, type ProductName } from "./look";
import type { MaskSet } from "./masks";
import type { GlossPercentiles } from "./pigment";
import constants from "../gen/constants.json";

const GLOSS_STRENGTH_FACTOR = constants.pigment.finish_gloss.strength_factor;
const HIGHLIGHTER_GLOSS_FACTOR = constants.pigment.finish_gloss.highlighter_strength_factor;
const MATTE_STRENGTH = constants.pigment.finish_matte.default_strength;
const MATTE_BLUR_SIGMA = constants.pigment.finish_matte.blur_sigma;

/** Percentiles for the products that have a gloss finish. */
export interface GlossInputs {
  highlighter?: GlossPercentiles | null;
  lipstick?: GlossPercentiles | null;
}

/** Per-frame draw options that do not change the look itself. */
export interface RenderOptions {
  /**
   * Flip horizontally, so a camera feed reads as a mirror. Done in the
   * shader rather than with a CSS transform on the canvas, so the pixels a
   * capture reads back are already the ones the viewer saw.
   */
  mirror?: boolean;
  /**
   * Before/after split, in drawing-buffer pixels from the left edge. Pixels
   * left of it are drawn untouched; pixels right of it get the full look.
   * Null or undefined draws the whole frame with the look.
   */
  splitX?: number | null;
}

export interface Renderer {
  /** Draw one frame. `source` is anything `texImage2D` accepts. */
  render(
    source: TexImageSource,
    masks: MaskSet,
    look: LookConfig,
    gloss: GlossInputs,
    options?: RenderOptions,
  ): void;
  /** Resize the drawing buffer and viewport. */
  resize(width: number, height: number): void;
  /** Release every GL object this renderer owns. */
  dispose(): void;
}

const VERTEX_SHADER = `#version 300 es
precision highp float;

out vec2 vUv;

void main() {
  // One oversized triangle covering the viewport: cheaper to set up than a
  // quad and it avoids the diagonal seam two triangles can produce.
  vec2 position = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = vec2(position.x, 1.0 - position.y);
  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uFrame;
uniform sampler2D uMaskBlush;
uniform sampler2D uMaskHighlighter;
uniform sampler2D uMaskEyeshadow;
uniform sampler2D uMaskBrows;
uniform sampler2D uMaskLipstick;
uniform sampler2D uMaskEyeliner;

// Per-product target color in Lab, its intensity, and its lightness pull.
uniform vec3 uLabBlush;
uniform vec3 uLabHighlighter;
uniform vec3 uLabEyeshadow;
uniform vec3 uLabBrows;
uniform vec3 uLabLipstick;
uniform vec3 uRgbEyeliner;

uniform float uIntensityBlush;
uniform float uIntensityHighlighter;
uniform float uIntensityEyeshadow;
uniform float uIntensityBrows;
uniform float uIntensityLipstick;
uniform float uIntensityEyeliner;

uniform float uPullBlush;
uniform float uPullHighlighter;
uniform float uPullEyeshadow;
uniform float uPullBrows;
uniform float uPullLipstick;

// Gloss: strength plus the CPU-measured percentile pair. A zero strength or
// a collapsed spread disables the pass, matching pigment.py's early-outs.
uniform float uGlossHighlighter;
uniform vec2 uGlossPercentilesHighlighter;
uniform float uGlossLipstick;
uniform vec2 uGlossPercentilesLipstick;

// Matte: strength plus the mip level standing in for the sigma=5 blur.
uniform float uMatteLipstick;
uniform float uMatteLod;

// 1.0 flips the frame horizontally (mirror mode); 1.0 on uBypass writes the
// source through untouched, which is how the before/after wipe draws its
// "before" side from the same program in a second scissored pass.
uniform float uMirror;
uniform float uBypass;

const float DELTA = 6.0 / 29.0;
const float DELTA_CUBED = DELTA * DELTA * DELTA;
const float THREE_DELTA_SQ = 3.0 * DELTA * DELTA;
const vec3 WHITE = vec3(0.950456, 1.0, 1.088754);
const float GLOSS_FACTOR = ${GLOSS_STRENGTH_FACTOR.toFixed(1)};

vec3 srgbToLinear(vec3 c) {
  vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
  vec3 low = c / 12.92;
  return mix(low, high, step(vec3(0.04045), c));
}

vec3 linearToSrgb(vec3 c) {
  c = max(c, vec3(0.0));
  vec3 high = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  vec3 low = c * 12.92;
  return mix(low, high, step(vec3(0.0031308), c));
}

vec3 labF(vec3 t) {
  vec3 high = pow(max(t, vec3(0.0)), vec3(1.0 / 3.0));
  vec3 low = t / THREE_DELTA_SQ + 4.0 / 29.0;
  return mix(low, high, step(vec3(DELTA_CUBED), t));
}

vec3 labFInverse(vec3 t) {
  vec3 high = t * t * t;
  vec3 low = THREE_DELTA_SQ * (t - 4.0 / 29.0);
  return mix(low, high, step(vec3(DELTA), t));
}

vec3 rgbToLab(vec3 rgb) {
  vec3 lin = srgbToLinear(rgb);
  // Column-major mat3 construction: each vec3 below is a column, so this is
  // the transpose of how the matrix reads in color.ts.
  mat3 toXyz = mat3(
    0.412453, 0.212671, 0.019334,
    0.357580, 0.715160, 0.119193,
    0.180423, 0.072169, 0.950227
  );
  vec3 xyz = (toXyz * lin) / WHITE;
  vec3 f = labF(xyz);
  float l = xyz.y > DELTA_CUBED ? 116.0 * f.y - 16.0 : 903.3 * xyz.y;
  return vec3(l, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}

vec3 labToRgb(vec3 lab) {
  float fy = (lab.x + 16.0) / 116.0;
  vec3 f = vec3(fy + lab.y / 500.0, fy, fy - lab.z / 200.0);
  vec3 xyz = labFInverse(f) * WHITE;
  mat3 toRgb = mat3(
     3.240479, -0.969256,  0.055648,
    -1.537150,  1.875991, -0.204043,
    -0.498535,  0.041556,  1.057311
  );
  return clamp(linearToSrgb(toRgb * xyz), 0.0, 1.0);
}

vec3 applyTint(vec3 lab, vec3 target, float weight, float pull) {
  lab.yz += (target.yz - lab.yz) * weight;
  lab.x += (target.x - lab.x) * weight * pull;
  return lab;
}

float applyGloss(float l, float weight, float strength, vec2 percentiles) {
  float spread = percentiles.y - percentiles.x;
  if (strength <= 0.0 || spread < 1e-6) {
    return l;
  }
  float highlight = clamp((l - percentiles.x) / spread, 0.0, 1.0);
  return clamp(l + highlight * weight * GLOSS_FACTOR * strength, 0.0, 100.0);
}

void main() {
  vec2 uv = vec2(mix(vUv.x, 1.0 - vUv.x, uMirror), vUv.y);
  vec4 frame = texture(uFrame, uv);

  if (uBypass > 0.5) {
    fragColor = frame;
    return;
  }

  float mBlush = texture(uMaskBlush, uv).r * uIntensityBlush;
  float mHighlighter = texture(uMaskHighlighter, uv).r * uIntensityHighlighter;
  float mEyeshadow = texture(uMaskEyeshadow, uv).r * uIntensityEyeshadow;
  float mBrows = texture(uMaskBrows, uv).r * uIntensityBrows;
  float mLipstick = texture(uMaskLipstick, uv).r * uIntensityLipstick;
  float mEyeliner = texture(uMaskEyeliner, uv).r * uIntensityEyeliner;

  float touched = max(max(max(mBlush, mHighlighter), max(mEyeshadow, mBrows)),
                      max(mLipstick, mEyeliner));
  if (touched <= 0.0) {
    // Untouched pixels stay bit-identical to the source, exactly as the CPU
    // path's restore step guarantees.
    fragColor = frame;
    return;
  }

  vec3 lab = rgbToLab(frame.rgb);

  lab = applyTint(lab, uLabBlush, mBlush, uPullBlush);

  lab = applyTint(lab, uLabHighlighter, mHighlighter, uPullHighlighter);
  lab.x = applyGloss(lab.x, mHighlighter / max(uIntensityHighlighter, 1e-6),
                     uGlossHighlighter, uGlossPercentilesHighlighter);

  lab = applyTint(lab, uLabEyeshadow, mEyeshadow, uPullEyeshadow);
  lab = applyTint(lab, uLabBrows, mBrows, uPullBrows);

  lab = applyTint(lab, uLabLipstick, mLipstick, uPullLipstick);
  float lipMask = mLipstick / max(uIntensityLipstick, 1e-6);
  if (uMatteLipstick > 0.0) {
    float blurredL = rgbToLab(textureLod(uFrame, uv, uMatteLod).rgb).x;
    lab.x += (blurredL - lab.x) * lipMask * uMatteLipstick;
  }
  lab.x = applyGloss(lab.x, lipMask, uGlossLipstick, uGlossPercentilesLipstick);

  vec3 rgb = labToRgb(lab);
  rgb = mix(rgb, uRgbEyeliner, mEyeliner);

  fragColor = vec4(rgb, frame.a);
}
`;

/** Mask uniform name for each product the live shader paints. */
const MASK_UNIFORMS: Record<ProductName, string> = {
  blush: "uMaskBlush",
  highlighter: "uMaskHighlighter",
  eyeshadow: "uMaskEyeshadow",
  brows: "uMaskBrows",
  lipstick: "uMaskLipstick",
  eyeliner: "uMaskEyeliner",
};

const CAPITALIZED: Record<ProductName, string> = {
  blush: "Blush",
  highlighter: "Highlighter",
  eyeshadow: "Eyeshadow",
  brows: "Brows",
  lipstick: "Lipstick",
  eyeliner: "Eyeliner",
};

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) {
    return null;
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error("carmine: shader compile failed", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function link(gl: WebGL2RenderingContext): WebGLProgram | null {
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  if (!vertex || !fragment) {
    return null;
  }
  const program = gl.createProgram();
  if (!program) {
    return null;
  }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  // The shaders are owned by the program once attached; deleting the
  // handles here frees them as soon as the program is deleted.
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("carmine: program link failed", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function createMaskTexture(gl: WebGL2RenderingContext): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) {
    return null;
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // A 1x1 zero mask so a product whose mask was not built this frame still
  // has something valid bound -- sampling an incomplete texture is
  // undefined behavior, not a no-op.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
  return texture;
}

/**
 * Convert a [0, 1] float mask to the R8 bytes the GPU wants.
 *
 * The scratch buffer is reused across frames: at 720p this is half a
 * megabyte per product per frame, and handing that much garbage to the
 * collector sixty times a second produces visible collection pauses.
 */
function toBytes(mask: Float32Array, scratch: Uint8Array): Uint8Array {
  const view = scratch.length === mask.length ? scratch : new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const value = mask[i];
    view[i] = value <= 0 ? 0 : value >= 1 ? 255 : Math.round(value * 255);
  }
  return view;
}

/**
 * Create a renderer drawing into `canvas`.
 *
 * Returns null when WebGL2 is unavailable or the program fails to build,
 * rather than throwing: the caller's job is to fall back to a CPU path, and
 * an old browser reaching this function is an expected outcome rather than
 * an error.
 */
export function createRenderer(canvas: HTMLCanvasElement): Renderer | null {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) {
    return null;
  }

  const program = link(gl);
  if (!program) {
    return null;
  }

  const frameTexture = gl.createTexture();
  const maskTextures = new Map<ProductName, WebGLTexture>();
  for (const name of PRODUCT_ORDER) {
    const texture = createMaskTexture(gl);
    if (texture) {
      maskTextures.set(name, texture);
    }
  }
  // A VAO is mandatory in WebGL2 even for an attribute-less draw.
  const vao = gl.createVertexArray();

  if (!frameTexture || !vao || maskTextures.size !== PRODUCT_ORDER.length) {
    gl.deleteProgram(program);
    return null;
  }

  gl.bindTexture(gl.TEXTURE_2D, frameTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

  const uniform = (name: string) => gl.getUniformLocation(program, name);
  const scratch = new Map<ProductName, Uint8Array>();

  gl.useProgram(program);
  gl.uniform1i(uniform("uFrame"), 0);
  PRODUCT_ORDER.forEach((name, index) => {
    gl.uniform1i(uniform(MASK_UNIFORMS[name]), index + 1);
  });

  let disposed = false;

  function uploadMask(name: ProductName, masks: MaskSet): boolean {
    const unit = PRODUCT_ORDER.indexOf(name) + 1;
    gl!.activeTexture(gl!.TEXTURE0 + unit);
    gl!.bindTexture(gl!.TEXTURE_2D, maskTextures.get(name)!);
    const mask = masks.masks[name];
    if (!mask) {
      return false;
    }
    const bytes = toBytes(mask, scratch.get(name) ?? new Uint8Array(0));
    scratch.set(name, bytes);
    gl!.texImage2D(
      gl!.TEXTURE_2D,
      0,
      gl!.R8,
      masks.width,
      masks.height,
      0,
      gl!.RED,
      gl!.UNSIGNED_BYTE,
      bytes,
    );
    return true;
  }

  return {
    render(source, masks, look, gloss, options) {
      if (disposed) {
        return;
      }
      const wantsMatte = look.lipstick.intensity > 0 && look.lipstick.finish === "matte";

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, frameTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
      // The mip chain only exists when the matte finish needs it, and a
      // LINEAR_MIPMAP_LINEAR filter on a texture without one is incomplete
      // (it samples black), so the filter follows the chain.
      gl.texParameteri(
        gl.TEXTURE_2D,
        gl.TEXTURE_MIN_FILTER,
        wantsMatte ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR,
      );
      if (wantsMatte) {
        gl.generateMipmap(gl.TEXTURE_2D);
      }

      gl.useProgram(program);

      for (const name of PRODUCT_ORDER) {
        const product = look[name];
        const uploaded = uploadMask(name, masks);
        const intensity = uploaded ? product.intensity : 0;
        const suffix = CAPITALIZED[name];
        gl.uniform1f(uniform(`uIntensity${suffix}`), intensity);
        if (name === "eyeliner") {
          const rgb = hexToRgb(product.color);
          gl.uniform3f(uniform("uRgbEyeliner"), rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
        } else {
          const lab = rgbToLab(hexToRgb(product.color));
          gl.uniform3f(uniform(`uLab${suffix}`), lab[0], lab[1], lab[2]);
          gl.uniform1f(
            uniform(`uPull${suffix}`),
            lightnessPullFor(name, look[name].finish),
          );
        }
      }

      const highlighterGloss =
        look.highlighter.intensity > 0 && gloss.highlighter
          ? look.highlighter.intensity * HIGHLIGHTER_GLOSS_FACTOR
          : 0;
      gl.uniform1f(uniform("uGlossHighlighter"), highlighterGloss);
      gl.uniform2f(
        uniform("uGlossPercentilesHighlighter"),
        gloss.highlighter?.p75 ?? 0,
        gloss.highlighter?.p99 ?? 0,
      );

      const lipstickGloss =
        look.lipstick.intensity > 0 && look.lipstick.finish === "gloss" && gloss.lipstick
          ? look.lipstick.intensity
          : 0;
      gl.uniform1f(uniform("uGlossLipstick"), lipstickGloss);
      gl.uniform2f(
        uniform("uGlossPercentilesLipstick"),
        gloss.lipstick?.p75 ?? 0,
        gloss.lipstick?.p99 ?? 0,
      );

      gl.uniform1f(uniform("uMatteLipstick"), wantsMatte ? MATTE_STRENGTH : 0);
      // A mip level halves resolution per step, so a level of log2(sigma)
      // covers roughly the sigma-pixel neighborhood the CPU blur averages.
      gl.uniform1f(uniform("uMatteLod"), Math.log2(Math.max(MATTE_BLUR_SIGMA, 1)));
      gl.uniform1f(uniform("uMirror"), options?.mirror ? 1 : 0);

      const bypass = uniform("uBypass");
      gl.bindVertexArray(vao);

      const splitX = options?.splitX ?? null;
      const cut = splitX === null ? null : Math.round(Math.min(Math.max(splitX, 0), canvas.width));
      if (cut === null || cut <= 0 || cut >= canvas.width) {
        // No wipe, or the handle is parked past an edge: one full-viewport
        // draw, entirely "after" (handle at the left) or entirely "before"
        // (handle at the right).
        gl.uniform1f(bypass, cut !== null && cut >= canvas.width ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      } else {
        // Two scissored passes of the same program: the left slice writes
        // the source through untouched, the right slice gets the look. A
        // second draw is cheaper here than a second framebuffer, and it
        // keeps the "before" side pixel-exact with the input.
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(0, 0, cut, canvas.height);
        gl.uniform1f(bypass, 1);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.scissor(cut, 0, canvas.width - cut, canvas.height);
        gl.uniform1f(bypass, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.disable(gl.SCISSOR_TEST);
      }

      gl.uniform1f(bypass, 0);
      gl.bindVertexArray(null);
    },

    resize(width, height) {
      if (disposed) {
        return;
      }
      canvas.width = Math.max(1, Math.round(width));
      canvas.height = Math.max(1, Math.round(height));
      gl.viewport(0, 0, canvas.width, canvas.height);
    },

    dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      gl.deleteProgram(program);
      gl.deleteTexture(frameTexture);
      for (const texture of maskTextures.values()) {
        gl.deleteTexture(texture);
      }
      gl.deleteVertexArray(vao);
      maskTextures.clear();
      scratch.clear();
    },
  };
}
