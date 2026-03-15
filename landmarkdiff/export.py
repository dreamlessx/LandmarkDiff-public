"""Export utilities for LandmarkDiff outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default frame count and duration for progressive preview
_DEFAULT_N_FRAMES = 20
_DEFAULT_FRAME_DURATION_MS = 100


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


def generate_progressive_frames(
    original: np.ndarray,
    prediction: np.ndarray,
    n_frames: int = _DEFAULT_N_FRAMES,
    add_labels: bool = True,
) -> list[np.ndarray]:
    """Generate frames that morph smoothly from original to prediction.

    Uses linear alpha blending between the original and prediction images
    to simulate a gradual intensity progression from 0 to the target.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        n_frames: Number of intermediate frames (including start and end).
        add_labels: If True, overlay intensity percentage on each frame.

    Returns:
        List of BGR frames from 0% to 100% intensity.

    Raises:
        ValueError: If images have different shapes or n_frames < 2.
    """
    if original.shape != prediction.shape:
        raise ValueError(f"Image shapes must match: {original.shape} vs {prediction.shape}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")

    frames = []
    orig_f = original.astype(np.float32)
    pred_f = prediction.astype(np.float32)

    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        blended = np.clip(orig_f * (1.0 - alpha) + pred_f * alpha, 0, 255).astype(np.uint8)

        if add_labels:
            h = blended.shape[0]
            font_scale = max(0.4, h / 512.0 * 0.6)
            thickness = max(1, int(h / 512.0 * 2))
            text = f"{int(alpha * 100)}%"
            cv2.putText(
                blended,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                blended,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        frames.append(blended)

    return frames


def export_progressive_gif(
    original: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    n_frames: int = _DEFAULT_N_FRAMES,
    frame_duration_ms: int = _DEFAULT_FRAME_DURATION_MS,
    loop: int = 0,
    add_labels: bool = True,
    boomerang: bool = True,
) -> Path:
    """Export a progressive intensity animation as a GIF.

    Creates a smooth morph from original to prediction, optionally
    with a boomerang (reverse) loop for continuous playback.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        output_path: Path to save the GIF.
        n_frames: Number of forward frames.
        frame_duration_ms: Duration per frame in milliseconds.
        loop: Number of loops (0 = infinite).
        add_labels: If True, overlay intensity percentage.
        boomerang: If True, append reversed frames for ping-pong effect.

    Returns:
        Path to the saved GIF.
    """
    from PIL import Image

    frames_bgr = generate_progressive_frames(
        original, prediction, n_frames=n_frames, add_labels=add_labels
    )

    if boomerang and len(frames_bgr) > 2:
        frames_bgr = frames_bgr + frames_bgr[-2:0:-1]

    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(out),
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=loop,
    )
    logger.info("Saved progressive GIF (%d frames): %s", len(pil_frames), out)
    return out
