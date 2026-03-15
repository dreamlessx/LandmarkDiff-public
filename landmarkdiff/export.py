"""Export utilities for LandmarkDiff outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def export_before_after_gif(
    original: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    duration_ms: int = 800,
    loop: int = 0,
    add_labels: bool = True,
) -> Path:
    """Export a before/after comparison as an animated GIF.

    Toggles between the original and predicted images at the given interval.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions as original).
        output_path: Path to save the GIF.
        duration_ms: Display time per frame in milliseconds.
        loop: Number of loops (0 = infinite).
        add_labels: If True, overlay "Before"/"After" text on frames.

    Returns:
        Path to the saved GIF.

    Raises:
        ImportError: If Pillow is not installed.
        ValueError: If images have different shapes.
    """
    from PIL import Image

    if original.shape != prediction.shape:
        raise ValueError(f"Image shapes must match: {original.shape} vs {prediction.shape}")

    frames = []
    for img, label in [(original, "Before"), (prediction, "After")]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if add_labels:
            canvas = img.copy()
            h = canvas.shape[0]
            font_scale = max(0.5, h / 512.0 * 0.8)
            thickness = max(1, int(h / 512.0 * 2))
            # Black outline + white text
            cv2.putText(
                canvas,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    logger.info("Saved animated GIF: %s", out)
    return out
