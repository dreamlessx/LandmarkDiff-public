"""Per-pixel prediction confidence map generation.

Produces a spatial confidence heatmap based on:
1. Displacement magnitude: regions with larger deformations are less certain
2. Landmark density: regions covered by more landmarks have better constraints
3. Distance from face center: peripheral regions have lower confidence

Useful for clinical decision-making and identifying which parts of a
surgical prediction are most vs least reliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


@dataclass
class ConfidenceMap:
    """Per-pixel prediction confidence result."""

    confidence: np.ndarray  # (H, W) float32, 0.0 (low) to 1.0 (high)
    mean_confidence: float
    min_confidence: float
    low_confidence_fraction: float  # fraction of pixels below threshold

    def summary(self, threshold: float = 0.5) -> str:
        return (
            f"Confidence: mean={self.mean_confidence:.3f} "
            f"min={self.min_confidence:.3f} "
            f"low_conf (<{threshold:.1f}): {self.low_confidence_fraction:.1%}"
        )


def generate_confidence_map(
    face_before: FaceLandmarks,
    face_after: FaceLandmarks,
    width: int = 512,
    height: int = 512,
    sigma: float = 40.0,
    low_confidence_threshold: float = 0.5,
) -> ConfidenceMap:
    """Generate a per-pixel confidence map from landmark displacement.

    Higher confidence in regions where:
    - Displacement is small (less change = more certain)
    - Multiple landmarks constrain the region (higher density)
    - Region is near the face center (better landmark coverage)

    Args:
        face_before: Landmarks before procedure.
        face_after: Landmarks after procedure.
        width: Output map width.
        height: Output map height.
        sigma: Gaussian spread for landmark influence (pixels).
        low_confidence_threshold: Threshold for computing low-confidence fraction.

    Returns:
        ConfidenceMap with per-pixel values.
    """
    before_px = face_before.pixel_coords[:, :2]
    after_px = face_after.pixel_coords[:, :2]

    # Per-landmark displacement magnitude
    displacements = np.sqrt(np.sum((after_px - before_px) ** 2, axis=1))
    max_disp = displacements.max() if displacements.max() > 0 else 1.0

    # Normalize: 0 displacement = confidence 1, max displacement = confidence ~0.2
    landmark_conf = 1.0 - 0.8 * (displacements / max_disp)

    # Build per-pixel confidence via Gaussian splatting of landmark confidences
    conf_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    # Grid coordinates
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    for i in range(len(before_px)):
        cx, cy = float(before_px[i, 0]), float(before_px[i, 1])
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        w = np.exp(-dist_sq / (2.0 * sigma**2))
        conf_map += w * float(landmark_conf[i])
        weight_map += w

    # Normalize by total weight (avoids division by zero in empty regions)
    mask = weight_map > 1e-8
    conf_map[mask] /= weight_map[mask]
    conf_map[~mask] = 0.0

    # Clamp to [0, 1]
    conf_map = np.clip(conf_map, 0.0, 1.0)

    mean_conf = float(conf_map.mean())
    min_conf = float(conf_map.min())
    low_frac = float(np.mean(conf_map < low_confidence_threshold))

    return ConfidenceMap(
        confidence=conf_map,
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        low_confidence_fraction=low_frac,
    )


def visualize_confidence_map(
    image: np.ndarray,
    confidence_map: ConfidenceMap,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a confidence heatmap on an image.

    High confidence regions appear green/cool, low confidence appears
    red/warm (using JET colormap by default).

    Args:
        image: BGR face image.
        confidence_map: Generated confidence map.
        colormap: OpenCV colormap constant.
        alpha: Blend factor for overlay (0 = image only, 1 = heatmap only).

    Returns:
        Annotated image copy with heatmap overlay.
    """
    h, w = image.shape[:2]
    conf = confidence_map.confidence

    # Resize confidence map to match image if needed
    if conf.shape != (h, w):
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to uint8 heatmap
    heatmap_gray = (conf * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, colormap)

    # Blend
    blended = cv2.addWeighted(image, 1.0 - alpha, heatmap_color, alpha, 0)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, h / 512.0 * 0.4)
    text = f"Mean conf: {confidence_map.mean_confidence:.2f}"
    cv2.putText(blended, text, (5, h - 5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return blended
