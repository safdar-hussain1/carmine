"""Bug-for-bug reproduction of the 2024 course project pipelines.

This module exists so the benchmark can run the original methods against
the rewrite on identical inputs. Nothing here is fixed on purpose. The
reproduced defects, verified against the original notebooks:

1. ``legacy_mediapipe``: detects MediaPipe's 468-point face mesh but then
   indexes it with **dlib's 68-point region numbers** (lips = 48:60,
   eyes = 36:48). On the 468-point topology those indices land on the
   chin/jaw, so "lipstick" is a purple polygon across the lower face.
   Pigment is applied by hard pixel assignment (no blending).

2. ``legacy_dlib``: correct 68-point indices, but lips are filled as one
   opaque polygon covering the whole mouth (teeth included), eyeshadow
   uses fixed ±10/±20 *pixel* offsets that only fit one image size, and
   effects are composited with additive ``cv2.addWeighted`` which blows
   out brightness. The interactive notebook additionally saved results
   through ``cv2.cvtColor(image, cv2.COLOR_RGB2BGR)`` on an already-BGR
   image, swapping red and blue across the entire photo (the "blue face"
   outputs); pass ``channel_swap_bug=True`` to reproduce that.

3. ``legacy_gan``: an **untrained** 3-layer conv net (random weights, no
   training loop anywhere in the notebook) run in inference. Outputs are
   noise by construction; the notebook still reported SSIM and
   accuracy/precision/recall for it.

4. ``legacy_metrics``: the notebook's evaluation, in which ``y_true`` is
   all ones — so "precision" is 1.0 by definition and "accuracy" merely
   counts SSIM scores above an arbitrary 0.45 threshold. Reproduced so
   the benchmark can show *why* the original numbers were meaningless.
"""

from __future__ import annotations

import cv2
import numpy as np


# --- 1. MediaPipe pipeline with dlib indices (the purple-triangles bug) ---

def legacy_mediapipe(image_bgr: np.ndarray, mesh_landmarks: np.ndarray) -> np.ndarray:
    """``apply_virtual_makeup`` from FinalModelWithDataset.ipynb, verbatim
    logic: dlib region indices applied to 468 FaceMesh points."""
    landmarks = [tuple(np.round(p).astype(int)) for p in mesh_landmarks]
    lips = landmarks[48:60]
    eyes = landmarks[36:48]
    blush = [landmarks[i] for i in [1, 2, 3, 4, 5, 6]]

    out = image_bgr.copy()
    lip_mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillPoly(lip_mask, [np.array(lips, dtype=np.int32)], 255)
    out[lip_mask == 255] = (125, 0, 105)

    eye_mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.fillPoly(eye_mask, [np.array(eyes, dtype=np.int32)], 255)
    out[eye_mask == 255] = (128, 0, 128)

    for point in blush:
        cv2.circle(out, point, 20, (128, 0, 128), -1)
    return out


# --- 2. dlib pipeline: opaque fill + fixed offsets + additive blending ---

def legacy_dlib(
    image_bgr: np.ndarray,
    landmarks68: np.ndarray,
    channel_swap_bug: bool = False,
) -> np.ndarray:
    """The VirtualMakeup.ipynb interactive pipeline with its default
    slider colors (lips RGB 125,0,105; eyes/blush RGB 128,0,128)."""
    pts = np.round(landmarks68).astype(int)
    out = image_bgr.copy()

    # Lipstick: one polygon over landmarks 48:68, additive blend.
    mask = np.zeros_like(out)
    cv2.fillPoly(mask, [pts[48:68]], (105, 0, 125))
    out = cv2.addWeighted(out, 1, mask, 0.6, 0)

    # Eyeshadow: hand-built polygon with fixed pixel offsets.
    eye_mask = np.zeros_like(out)
    for a, b, c, d in ((36, 37, 38, 39), (42, 43, 44, 45)):
        poly = np.array([
            (pts[a][0], pts[a][1] - 10),
            (pts[b][0], pts[b][1] - 20),
            (pts[c][0], pts[c][1] - 20),
            (pts[d][0], pts[d][1] - 10),
            (pts[c][0], pts[c][1] + 5),
            (pts[b][0], pts[b][1] + 5),
        ])
        cv2.fillPoly(eye_mask, [poly], (128, 0, 128))
    eye_mask = cv2.GaussianBlur(eye_mask, (25, 25), 50)
    out = cv2.addWeighted(out, 1, eye_mask, 0.5, 0)

    # Blush: ellipses sized from face width, additive blend.
    blush_mask = np.zeros_like(out)
    face_width = pts[45][0] - pts[36][0]
    left_center = (
        int((pts[2][0] + pts[4][0] + pts[48][0]) // 3),
        int((pts[2][1] + pts[4][1] + pts[48][1]) // 3) - 30,
    )
    right_center = (
        int((pts[12][0] + pts[14][0] + pts[54][0]) // 3),
        int((pts[12][1] + pts[14][1] + pts[54][1]) // 3) - 30,
    )
    scale = (face_width / 100) * 0.5
    axes = (min(int(37 * scale), 60), min(int(45 * scale), 60))
    cv2.ellipse(blush_mask, left_center, axes, 0, 0, 360, (128, 0, 128), -1)
    cv2.ellipse(blush_mask, right_center, axes, 0, 0, 360, (128, 0, 128), -1)
    blush_mask = cv2.GaussianBlur(blush_mask, (35, 35), 60)
    out = cv2.addWeighted(out, 1, blush_mask, 0.5, 0)

    if channel_swap_bug:
        # save_final_image() ran RGB2BGR on an image that was already BGR.
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


# --- 3. The "GAN": an untrained conv net run in inference ---

def legacy_gan(image_bgr: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Numpy equivalent of the notebook's untrained Keras generator:
    Conv(64, relu) -> Conv(64, relu) -> Conv(3, tanh), Glorot-uniform
    random weights, never trained, output scaled by 255.

    The exact original output is irreproducible by construction (random
    weights, no seed saved), which is itself the finding; a seeded rng
    here makes *this* reproduction deterministic.
    """
    rng = rng or np.random.default_rng(0)
    x = cv2.resize(image_bgr, (256, 256)).astype(np.float32) / 255.0

    def conv(x, w, b):
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        h, wd, cin = x.shape
        cout = w.shape[3]
        out = np.zeros((h, wd, cout), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                patch = pad[i:i + h, j:j + wd, :]
                out += patch @ w[i, j]
        return out + b

    def glorot(shape, rng):
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[0] * shape[1] * shape[3]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    w1, b1 = glorot((3, 3, 3, 64), rng), np.zeros(64, np.float32)
    w2, b2 = glorot((3, 3, 64, 64), rng), np.zeros(64, np.float32)
    w3, b3 = glorot((3, 3, 64, 3), rng), np.zeros(3, np.float32)

    x = np.maximum(conv(x, w1, b1), 0)
    x = np.maximum(conv(x, w2, b2), 0)
    x = np.tanh(conv(x, w3, b3))
    return (x * 255).astype(np.uint8)  # tanh in [-1,1]: negatives wrap, as in the original


# --- 4. The notebook's evaluation protocol ---

def legacy_metrics(ssim_scores: list[float], threshold: float = 0.45) -> dict[str, float]:
    """Reproduces calculate_metrics_with_threshold: y_true is all ones,
    y_pred thresholds SSIM. With a single true class, precision is 1.0
    whenever anything is predicted positive, and accuracy == recall ==
    fraction above threshold. No negative class ever exists."""
    y_pred = np.array([1 if s >= threshold else 0 for s in ssim_scores])
    n = len(y_pred)
    if n == 0:
        raise ValueError("no scores")
    tp = int(y_pred.sum())
    accuracy = tp / n
    precision = 1.0  # zero_division=1 in the original masks the empty case
    recall = accuracy
    f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}
