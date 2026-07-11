"""Canonical MediaPipe FaceMesh landmark index sets for facial regions.

All indices refer to the 468-point FaceMesh topology (refine_landmarks adds
iris points 468-477, which this module does not use). Rings are ordered so
they can be filled directly as polygons with ``cv2.fillPoly``.

"Left"/"right" are in image coordinates (the subject's right eye is
``RIGHT_EYE`` and appears on the left side of the image).
"""

# Outer lip contour, ordered clockwise starting at the right mouth corner.
LIPS_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Inner lip contour (the mouth opening). Subtracting it from the outer
# contour is what keeps lipstick off the teeth.
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Eye contours, ordered around each eye.
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Upper eyelid arcs (subset of the eye rings, inner corner -> outer corner).
RIGHT_EYE_UPPER = [133, 173, 157, 158, 159, 160, 161, 246, 33]
LEFT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]

# Lower edge of each eyebrow, same direction as the matching upper-lid arc.
RIGHT_BROW_LOWER = [55, 65, 52, 53, 46, 70]
LEFT_BROW_LOWER = [285, 295, 282, 283, 276, 300]

# Face outline, used for the skin mask.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Mid-cheek anchor points for blush placement.
RIGHT_CHEEK = 50
LEFT_CHEEK = 280

# Interocular reference points (outer eye corners) used to scale every
# effect to the size of the face in the image.
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263

NUM_LANDMARKS = 468
