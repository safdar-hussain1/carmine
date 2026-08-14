"""Landmark index sets for facial regions in the MediaPipe FaceMesh topology.

All indices refer to the extended 478-point landmark space (468 base FaceMesh
points plus 10 iris-specific points at indices 468–477). Rings are ordered so
they can be rendered directly as polygons with cv2.fillPoly. In image
coordinates, "left" and "right" refer to the subject's sides (right eye appears
on the left side of the image).
"""

# Outer lip contour, ordered clockwise starting at the right mouth corner.
LIPS_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# Inner lip contour marking the mouth opening. Subtracting this from the outer
# contour prevents lipstick from appearing on teeth.
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]

# Complete eye contours, ordered around each eye.
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# Upper eyelid arcs for eye makeup (inner corner to outer corner).
RIGHT_EYE_UPPER = [133, 173, 157, 158, 159, 160, 161, 246, 33]
LEFT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]

# Lower eyebrow edges, ordered from inner point (55/285) to outer/junction point (70/300).
RIGHT_BROW_LOWER = [55, 65, 52, 53, 46, 70]
LEFT_BROW_LOWER = [285, 295, 282, 283, 276, 300]

# Upper eyebrow edges, ordered from inner point (107/336) to outer/junction point (70/300).
# When traversed as LOWER + reversed(UPPER), they trace the eyebrow perimeter as a closed
# polygon. The junction points (70 and 300) appear in both rings; polygon builders must
# handle the shared endpoints.
RIGHT_BROW_UPPER = [107, 66, 105, 63, 70]
LEFT_BROW_UPPER = [336, 296, 334, 293, 300]

# Face boundary contour, used to define the skin mask.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Bridge of the nose, ordered from top to bottom.
NOSE_BRIDGE = [168, 6, 197, 195]

# Cheekbone crests for highlighter placement, forming a horizontal arc.
RIGHT_CHEEKBONE = [116, 117, 118, 119]
LEFT_CHEEKBONE = [345, 346, 347, 348]

# Single-point landmark indices for blush placement.
RIGHT_CHEEK = 50
LEFT_CHEEK = 280

# Outer eye corners, used as reference points to scale effects to face size.
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263

# Total number of landmarks in the detection output.
NUM_LANDMARKS = 478
