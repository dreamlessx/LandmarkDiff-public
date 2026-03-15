"""Before/after comparison utilities for surgical prediction visualization.

Provides side-by-side composites, vertical slider overlays, and difference
heatmaps for comparing original and predicted face images.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_slider_composite(
    original: np.ndarray,
    prediction: np.ndarray,
    position: float = 0.5,
    line_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 2,
) -> np.ndarray:
    """Create a before/after slider composite image.

    Splits the view at the given horizontal position, showing the original
    on the left and prediction on the right, with a divider line.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        position: Slider position 0-1 (0 = all prediction, 1 = all original).
        line_color: BGR color for the divider line.
        line_width: Width of the divider line in pixels.

    Returns:
        BGR composite image with slider divider.
    """
    h, w = original.shape[:2]
    pred = cv2.resize(prediction, (w, h)) if prediction.shape[:2] != (h, w) else prediction

    split_x = int(w * np.clip(position, 0.0, 1.0))
    result = pred.copy()
    result[:, :split_x] = original[:, :split_x]

    # Draw divider line
    if line_width > 0 and 0 < split_x < w:
        cv2.line(result, (split_x, 0), (split_x, h - 1), line_color, line_width)

    return result


def create_side_by_side(
    original: np.ndarray,
    prediction: np.ndarray,
    gap: int = 4,
    gap_color: tuple[int, int, int] = (255, 255, 255),
    add_labels: bool = True,
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Args:
        original: BGR original image.
        prediction: BGR predicted image.
        gap: Pixel width of the gap between images.
        gap_color: BGR color for the gap.
        add_labels: If True, overlay "Before"/"After" text.

    Returns:
        BGR image with original on left, prediction on right.
    """
    h, w = original.shape[:2]
    pred = cv2.resize(prediction, (w, h)) if prediction.shape[:2] != (h, w) else prediction

    canvas = np.full((h, w * 2 + gap, 3), gap_color, dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w + gap :] = pred

    if add_labels:
        font_scale = max(0.5, h / 512.0 * 0.7)
        thickness = max(1, int(h / 512.0 * 2))
        for label, x_offset in [("Before", 10), ("After", w + gap + 10)]:
            cv2.putText(
                canvas,
                label,
                (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return canvas


def create_difference_heatmap(
    original: np.ndarray,
    prediction: np.ndarray,
    amplify: float = 3.0,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a heatmap showing pixel-level differences.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        amplify: Amplification factor for subtle differences.
        colormap: OpenCV colormap for visualization.

    Returns:
        BGR heatmap image where brighter = more change.
    """
    h, w = original.shape[:2]
    pred = cv2.resize(prediction, (w, h)) if prediction.shape[:2] != (h, w) else prediction

    diff = cv2.absdiff(original, pred)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    amplified = np.clip(gray_diff.astype(np.float32) * amplify, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(amplified, colormap)


def create_checkerboard_blend(
    original: np.ndarray,
    prediction: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """Create a checkerboard pattern blending original and prediction.

    Alternating blocks show original and prediction for easy
    comparison of local differences.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        block_size: Size of each checkerboard square in pixels.

    Returns:
        BGR checkerboard composite.
    """
    h, w = original.shape[:2]
    pred = cv2.resize(prediction, (w, h)) if prediction.shape[:2] != (h, w) else prediction

    # Create checkerboard mask
    rows = (h + block_size - 1) // block_size
    cols = (w + block_size - 1) // block_size
    mask = np.zeros((h, w), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y1 = r * block_size
                y2 = min((r + 1) * block_size, h)
                x1 = c * block_size
                x2 = min((c + 1) * block_size, w)
                mask[y1:y2, x1:x2] = 1.0

    mask_3ch = np.stack([mask] * 3, axis=-1)
    result = original.astype(np.float32) * mask_3ch + pred.astype(np.float32) * (1.0 - mask_3ch)
    return np.clip(result, 0, 255).astype(np.uint8)
