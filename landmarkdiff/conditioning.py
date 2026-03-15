"""Conditioning signal generation: static adjacency wireframe + auto-Canny.

Uses a pre-defined anatomical adjacency matrix (NOT dynamic Delaunay) to prevent
triangle inversion on drastic landmark displacements. Auto-Canny adapts thresholds
to skin tone (Fitzpatrick I-VI safe).
"""

from __future__ import annotations

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

# Static anatomical adjacency for MediaPipe 478 landmarks.
# Connects landmarks along anatomically meaningful contours:
# jawline, nasal dorsum, orbital rim, lip vermilion, eyebrow arch.
# This is invariant to landmark displacement (unlike Delaunay).

JAWLINE_CONTOUR = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
]

LEFT_EYE_CONTOUR = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
    33,
]

RIGHT_EYE_CONTOUR = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    362,
]

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]
NOSE_BRIDGE_UPPER = [8, 9, 168, 6, 8]  # close the glabella/nasion gap
NOSE_TIP = [94, 2, 326, 327, 294, 278, 279, 275, 274, 460, 456, 363, 370]
NOSE_BOTTOM = [19, 1, 274, 275, 440, 344, 278, 294, 460, 305, 289, 392]

OUTER_LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    61,
]

INNER_LIPS = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
]

# Auto-Canny threshold factors (median-relative)
_CANNY_LOW_FACTOR = 0.66
_CANNY_HIGH_FACTOR = 1.33
_CANNY_DEFAULT_MEDIAN = 128.0  # fallback when no non-zero pixels exist

ALL_CONTOURS = [
    JAWLINE_CONTOUR,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
    NOSE_BRIDGE,
    NOSE_BRIDGE_UPPER,
    NOSE_TIP,
    NOSE_BOTTOM,
    OUTER_LIPS,
    INNER_LIPS,
]


def render_wireframe(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
    thickness: int = 1,
) -> np.ndarray:
    """Render static anatomical adjacency wireframe on black canvas.

    Args:
        face: Facial landmarks (normalized coordinates).
        width: Canvas width.
        height: Canvas height.
        thickness: Line thickness in pixels.

    Returns:
        Grayscale wireframe image.
    """
    w = width or face.image_width
    h = height or face.image_height
    canvas = np.zeros((h, w), dtype=np.uint8)

    coords = face.landmarks[:, :2].copy()
    coords[:, 0] *= w
    coords[:, 1] *= h
    pts = coords.astype(np.int32)

    for contour in ALL_CONTOURS:
        for i in range(len(contour) - 1):
            p1 = tuple(pts[contour[i]])
            p2 = tuple(pts[contour[i + 1]])
            cv2.line(canvas, p1, p2, 255, thickness)

    return canvas


def auto_canny(image: np.ndarray) -> np.ndarray:
    """Auto-Canny edge detection with adaptive thresholds.

    Uses median-based thresholds (0.66*median, 1.33*median) instead of
    hardcoded 50/150 to handle all Fitzpatrick skin types.
    Post-processes with morphological skeletonization for 1-pixel edges.

    Args:
        image: Grayscale input image.

    Returns:
        Binary edge map (uint8, 0 or 255).
    """
    median = np.median(image[image > 0]) if np.any(image > 0) else _CANNY_DEFAULT_MEDIAN
    low = int(max(0, _CANNY_LOW_FACTOR * median))
    high = int(min(255, _CANNY_HIGH_FACTOR * median))

    edges = cv2.Canny(image, low, high)

    # Morphological skeletonization for guaranteed 1-pixel thickness
    # ControlNet blurs on 2+ pixel edges
    skeleton = np.zeros_like(edges)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = edges.copy()

    max_iterations = max(edges.shape[0], edges.shape[1])
    for _ in range(max_iterations):
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


def generate_conditioning(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate full conditioning signal for ControlNet.

    Returns three channels per the spec:
    1. Rendered landmark dots (colored, BGR)
    2. Canny edge map from static wireframe (grayscale)
    3. Wireframe rendering (grayscale)

    Args:
        face: Extracted facial landmarks.
        width: Output width.
        height: Output height.

    Returns:
        Tuple of (landmark_image, canny_edges, wireframe).
    """
    from landmarkdiff.landmarks import render_landmark_image

    w = width or face.image_width
    h = height or face.image_height

    landmark_img = render_landmark_image(face, w, h)
    wireframe = render_wireframe(face, w, h)
    canny = auto_canny(wireframe)

    return landmark_img, canny, wireframe
