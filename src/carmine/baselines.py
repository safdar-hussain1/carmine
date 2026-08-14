"""Standard failure-mode baselines for the makeup-quality benchmark.

Landmark-based AR makeup breaks in a handful of well-known ways, and a
benchmark that only compares the engine against "no makeup" can't show
whether it actually solves those problems. Each function below reproduces
one of those failure modes faithfully -- nothing here is a straw man, and
nothing here is fixed on purpose -- so ``scripts/benchmark.py`` can score
the engine against the same failure modes shipped AR filters are known to
exhibit, on identical inputs:

1. ``mismatched_indices``: the classic index-topology mixup. A 68-point
   facial-landmark region table (lips = 48:60, eyes = 36:48, a handful of
   jaw points reused for blush) gets applied verbatim to a 478-point mesh.
   The numbers land nowhere near the regions they're named for, so
   "lipstick" ends up as a flat polygon across the lower face.

2. ``opaque_fill``: correct region indices, but every product is
   hard-filled with ``cv2.fillPoly`` and composited with additive
   ``cv2.addWeighted`` instead of a texture-aware blend. Lips are one
   opaque polygon over the whole mouth (teeth included, since there's no
   inner-contour subtraction), and eyeshadow is built from fixed pixel
   offsets that only look right at the image size they were tuned on.

3. ``channel_swap``: everything upstream is correct -- real masks, real
   texture-preserving pigment -- and the defect is a single line at the
   very end: an RGB<->BGR conversion applied to an image that's already in
   the target color order. It's the kind of bug that only shows up after
   the makeup already looks right, which is exactly why it keeps
   shipping.

4. ``untrained_gan``: what "GAN-based makeup" looks like when the training
   step is skipped. A three-layer convolutional network with freshly
   initialized (Glorot-uniform) weights, run once in inference. There is
   no training loop anywhere in this module -- the weights never see a
   gradient -- so the output is structured noise by construction.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import engine, pigment, regions
from .look import Look

__all__ = ["mismatched_indices", "opaque_fill", "channel_swap", "untrained_gan"]

# The region indices a 68-point facial-landmark scheme uses. Faithful to
# that scheme; wrong for anything else.
_SIXTY_EIGHT_LIPS = slice(48, 60)
_SIXTY_EIGHT_EYES = slice(36, 48)
_SIXTY_EIGHT_BLUSH_POINTS = (1, 2, 3, 4, 5, 6)


def mismatched_indices(
    image_bgr: np.ndarray, dlib_landmarks: np.ndarray, look: Look
) -> np.ndarray:
    """Apply 68-point region indices to a 478-point landmark array.

    Reproduces the index-topology mixup: the pipeline correctly runs a
    478-point mesh detector, but the code that consumes its output was
    written (or copy-pasted) against a 68-point scheme, so it slices the
    same fixed ranges -- lips at 48:60, eyes at 36:48, a run of jaw points
    for blush -- straight out of the 478-point array. Those slices are
    real indices into a real array, so nothing raises; they just point at
    the wrong landmarks, so pigment lands on the jaw and lower cheek
    instead of the lips and eyes.

    Args:
        image_bgr: BGR uint8 image of shape (H, W, 3).
        dlib_landmarks: The (478, 2) mesh landmark array -- named for the
            indexing scheme mistakenly applied to it, not for its own
            point count.
        look: The requested `Look`; each product's color is used where its
            intensity is above 0, and its intensity is otherwise ignored
            (this baseline paints at full opacity, another symptom of the
            same copy-pasted code).

    Returns:
        A new BGR uint8 array.
    """
    pts = np.round(dlib_landmarks).astype(np.int32)
    out = image_bgr.copy()

    if look.lipstick.intensity > 0:
        color = pigment.parse_hex_color(look.lipstick.color)[::-1]
        lip_mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(lip_mask, [pts[_SIXTY_EIGHT_LIPS]], 255)
        out[lip_mask == 255] = color

    if look.eyeshadow.intensity > 0:
        color = pigment.parse_hex_color(look.eyeshadow.color)[::-1]
        eye_mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(eye_mask, [pts[_SIXTY_EIGHT_EYES]], 255)
        out[eye_mask == 255] = color

    if look.blush.intensity > 0:
        color = tuple(int(c) for c in pigment.parse_hex_color(look.blush.color)[::-1])
        for i in _SIXTY_EIGHT_BLUSH_POINTS:
            center = (int(pts[i][0]), int(pts[i][1]))
            cv2.circle(out, center, 20, color, -1)

    return out


def opaque_fill(image_bgr: np.ndarray, landmarks: np.ndarray, look: Look) -> np.ndarray:
    """Fill the correct regions, but opaquely, with additive compositing.

    The region indices here are correct -- lips, eyes, cheeks all land
    where they should. The defect is how each region gets painted:
    ``cv2.fillPoly`` writes a flat color with no inner-contour
    subtraction, so an open mouth gets lipstick across the teeth; the
    eyeshadow polygon is built from fixed pixel offsets (10-20px) that
    were sized for one particular image and warp on any other; and every
    layer is composited with ``cv2.addWeighted`` instead of a blend that
    respects the pixels underneath, so the result brightens and flattens
    rather than tinting. None of this touches CIELAB, so no skin texture
    survives under the mask.

    Args:
        image_bgr: BGR uint8 image of shape (H, W, 3).
        landmarks: The (478, 2) mesh landmark array, correctly indexed.
        look: The requested `Look`; a product only paints when its
            intensity is above 0, and intensity sets the additive blend
            weight (clamped to 1.0).

    Returns:
        A new BGR uint8 array.
    """
    pts = np.round(landmarks).astype(np.int32)
    out = image_bgr.copy()

    if look.lipstick.intensity > 0:
        color = tuple(int(c) for c in pigment.parse_hex_color(look.lipstick.color)[::-1])
        mask = np.zeros_like(out)
        cv2.fillPoly(mask, [pts[regions.LIPS_OUTER]], color)
        out = cv2.addWeighted(out, 1, mask, min(1.0, look.lipstick.intensity), 0)

    if look.eyeshadow.intensity > 0:
        color = tuple(int(c) for c in pigment.parse_hex_color(look.eyeshadow.color)[::-1])
        eye_mask = np.zeros_like(out)
        for arc_idx in (regions.RIGHT_EYE_UPPER, regions.LEFT_EYE_UPPER):
            a, b, c, d = (pts[arc_idx[i]] for i in range(4))
            poly = np.array(
                [
                    (a[0], a[1] - 10),
                    (b[0], b[1] - 20),
                    (c[0], c[1] - 20),
                    (d[0], d[1] - 10),
                    (c[0], c[1] + 5),
                    (b[0], b[1] + 5),
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(eye_mask, [poly], color)
        eye_mask = cv2.GaussianBlur(eye_mask, (25, 25), 50)
        out = cv2.addWeighted(out, 1, eye_mask, min(1.0, look.eyeshadow.intensity), 0)

    if look.blush.intensity > 0:
        color = tuple(int(c) for c in pigment.parse_hex_color(look.blush.color)[::-1])
        blush_mask = np.zeros_like(out)
        for idx in (regions.RIGHT_CHEEK, regions.LEFT_CHEEK):
            center = (int(pts[idx][0]), int(pts[idx][1]))
            cv2.ellipse(blush_mask, center, (37, 45), 0, 0, 360, color, -1)
        blush_mask = cv2.GaussianBlur(blush_mask, (35, 35), 60)
        out = cv2.addWeighted(out, 1, blush_mask, min(1.0, look.blush.intensity), 0)

    return out


def channel_swap(image_bgr: np.ndarray, landmarks: np.ndarray, look: Look) -> np.ndarray:
    """Apply the look correctly, then swap color channels on the way out.

    Everything up to the return statement runs the real pipeline: real
    masks, real CIELAB pigment, the same code path `apply_look` uses. The
    defect is entirely in the last line -- an RGB<->BGR conversion run on
    an image that's already in the right color order, the classic
    save/export mixup. It reproduces the "blue face" failure mode users
    report: makeup placement and texture are both fine, and the whole
    photo is still wrong, uniformly, edge to edge.

    Args:
        image_bgr: BGR uint8 image of shape (H, W, 3).
        landmarks: The (478, 2) mesh landmark array.
        look: The `Look` to apply, painted correctly before the swap.

    Returns:
        A new BGR uint8 array with channels swapped.
    """
    out = engine.apply_look(image_bgr, look, landmarks=landmarks)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def untrained_gan(image_bgr: np.ndarray, look: Look, seed: int = 0) -> np.ndarray:
    """Run inference through a freshly initialized, never-trained conv net.

    Three convolution layers (64, 64, 3 channels; Glorot-uniform weights;
    ReLU, ReLU, tanh) with no training loop anywhere -- the weights are
    drawn once from their init distribution and used exactly as drawn.
    This is the failure mode behind "we tried a GAN and it made noise":
    a generator architecture guarantees nothing about its output until
    it's been trained, and an untrained one is, by construction,
    structured noise correlated with its random weights rather than with
    the input image.

    `look` is accepted for interface parity with the other baselines
    (the benchmark calls all of them the same way) but doesn't influence
    the output -- an untrained network has no mechanism to condition on
    it.

    Args:
        image_bgr: BGR uint8 image of shape (H, W, 3).
        look: Unused; present for interface parity.
        seed: Seeds the weight initialization so repeated calls with the
            same seed are bit-identical.

    Returns:
        A (256, 256, 3) BGR uint8 array.
    """
    del look  # unused: an untrained network can't condition on anything
    rng = np.random.default_rng(seed)
    x = cv2.resize(image_bgr, (256, 256)).astype(np.float32) / 255.0

    def conv(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        h, wd, _ = x.shape
        cout = w.shape[3]
        out = np.zeros((h, wd, cout), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patch = pad[i : i + h, j : j + wd, :]
                out += patch @ w[i, j]
        return out + b

    def glorot(shape: tuple[int, int, int, int]) -> np.ndarray:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[0] * shape[1] * shape[3]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    w1, b1 = glorot((3, 3, 3, 64)), np.zeros(64, np.float32)
    w2, b2 = glorot((3, 3, 64, 64)), np.zeros(64, np.float32)
    w3, b3 = glorot((3, 3, 64, 3)), np.zeros(3, np.float32)

    x = np.maximum(conv(x, w1, b1), 0)
    x = np.maximum(conv(x, w2, b2), 0)
    x = np.tanh(conv(x, w3, b3))
    return (x * 255).astype(np.uint8)  # tanh in [-1, 1]: negatives wrap on the uint8 cast
