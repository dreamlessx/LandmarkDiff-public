"""Fitzpatrick skin type auto-detection from face images.

Uses landmark-guided face region masking to sample skin pixels from
the forehead and cheeks (avoiding eyes, lips, hair), then classifies
via the Individual Typology Angle (ITA) method.

Also provides post-processing parameter recommendations per skin type
to minimize color artifacts in darker skin tones.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

# ITA thresholds (Chardon et al. 1991)
_ITA_THRESHOLDS: list[tuple[float, str]] = [
    (55.0, "I"),
    (41.0, "II"),
    (28.0, "III"),
    (10.0, "IV"),
    (-30.0, "V"),
]

# MediaPipe landmark indices for skin sampling regions
# Left cheek: landmarks forming a polygon on the left cheek
_LEFT_CHEEK = [117, 118, 119, 100, 126, 209, 49, 129, 203, 205, 36, 142]
# Right cheek: mirror of left cheek
_RIGHT_CHEEK = [346, 347, 348, 329, 355, 429, 279, 358, 423, 425, 266, 371]
# Forehead: upper face landmarks
_FOREHEAD = [
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
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


@dataclass
class SkinTypeResult:
    """Result of Fitzpatrick skin type classification."""

    fitzpatrick_type: str  # "I" through "VI"
    ita_angle: float  # ITA value in degrees
    confidence: float  # 0-1, based on pixel sample count and variance
    sampled_pixels: int  # number of skin pixels sampled

    @property
    def description(self) -> str:
        descriptions = {
            "I": "Very light, always burns",
            "II": "Light, usually burns",
            "III": "Medium, sometimes burns",
            "IV": "Olive, rarely burns",
            "V": "Brown, very rarely burns",
            "VI": "Dark brown/black, never burns",
        }
        return descriptions.get(self.fitzpatrick_type, "Unknown")


@dataclass
class PostProcessParams:
    """Recommended post-processing parameters based on skin type."""

    histogram_match_strength: float  # 0-1, how aggressively to match colors
    lab_blend_weight: float  # weight for LAB-space blending vs RGB
    sharpen_amount: float  # sharpening strength
    color_correction_strength: float  # strength of color correction step


# Per-type post-processing recommendations
_POSTPROCESS_PARAMS: dict[str, PostProcessParams] = {
    "I": PostProcessParams(
        histogram_match_strength=0.8,
        lab_blend_weight=0.5,
        sharpen_amount=0.4,
        color_correction_strength=0.6,
    ),
    "II": PostProcessParams(
        histogram_match_strength=0.8,
        lab_blend_weight=0.5,
        sharpen_amount=0.4,
        color_correction_strength=0.6,
    ),
    "III": PostProcessParams(
        histogram_match_strength=0.7,
        lab_blend_weight=0.6,
        sharpen_amount=0.35,
        color_correction_strength=0.7,
    ),
    "IV": PostProcessParams(
        histogram_match_strength=0.6,
        lab_blend_weight=0.7,
        sharpen_amount=0.3,
        color_correction_strength=0.8,
    ),
    "V": PostProcessParams(
        histogram_match_strength=0.5,
        lab_blend_weight=0.8,
        sharpen_amount=0.25,
        color_correction_strength=0.9,
    ),
    "VI": PostProcessParams(
        histogram_match_strength=0.4,
        lab_blend_weight=0.9,
        sharpen_amount=0.2,
        color_correction_strength=1.0,
    ),
}


def _build_region_mask(
    face: FaceLandmarks,
    indices: list[int],
    width: int,
    height: int,
) -> np.ndarray:
    """Build a filled polygon mask from landmark indices."""
    coords = face.pixel_coords[indices].astype(np.int32)
    hull = cv2.convexHull(coords)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def detect_skin_type(
    image: np.ndarray,
    face: FaceLandmarks,
    min_pixels: int = 100,
) -> SkinTypeResult:
    """Detect Fitzpatrick skin type using landmark-guided ITA analysis.

    Samples skin pixels from the cheeks and forehead regions (avoiding
    eyes, lips, eyebrows, and hair) for more accurate classification
    than a simple center-crop approach.

    Args:
        image: BGR face image.
        face: Extracted face landmarks.
        min_pixels: Minimum skin pixels required for reliable classification.

    Returns:
        SkinTypeResult with type, ITA angle, and confidence.
    """
    h, w = image.shape[:2]

    # Build combined skin region mask from cheeks
    left_mask = _build_region_mask(face, _LEFT_CHEEK, w, h)
    right_mask = _build_region_mask(face, _RIGHT_CHEEK, w, h)
    skin_mask = cv2.bitwise_or(left_mask, right_mask)

    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample skin pixels
    skin_pixels = lab[skin_mask > 0]
    n_pixels = len(skin_pixels)

    if n_pixels < min_pixels:
        # Fall back to center crop if landmarks give too few pixels
        cy, cx = h // 2, w // 2
        r = min(h, w) // 6
        skin_pixels = lab[cy - r : cy + r, cx - r : cx + r].reshape(-1, 3)
        n_pixels = len(skin_pixels)

    # Compute ITA
    l_mean = float(skin_pixels[:, 0].mean()) * 100.0 / 255.0  # scale to 0-100
    b_mean = float(skin_pixels[:, 2].mean()) - 128.0  # center around 0

    if abs(b_mean) < 1e-6:
        b_mean = 1e-6

    ita = float(np.arctan2(l_mean - 50.0, b_mean) * (180.0 / np.pi))

    # Classify
    fitz_type = "VI"
    for threshold, type_name in _ITA_THRESHOLDS:
        if ita > threshold:
            fitz_type = type_name
            break

    # Confidence based on sample size and luminance variance
    l_std = float(skin_pixels[:, 0].std())
    # High confidence when many pixels and low variance
    size_conf = min(1.0, n_pixels / 1000.0)
    var_conf = max(0.0, 1.0 - l_std / 50.0)
    confidence = size_conf * 0.6 + var_conf * 0.4

    return SkinTypeResult(
        fitzpatrick_type=fitz_type,
        ita_angle=ita,
        confidence=float(confidence),
        sampled_pixels=n_pixels,
    )


def get_postprocess_params(skin_type: str) -> PostProcessParams:
    """Get recommended post-processing parameters for a skin type.

    Darker skin types get stronger LAB-space blending, weaker histogram
    matching, and more aggressive color correction to minimize artifacts.

    Args:
        skin_type: Fitzpatrick type ("I" through "VI").

    Returns:
        PostProcessParams with recommended settings.
    """
    return _POSTPROCESS_PARAMS.get(
        skin_type,
        _POSTPROCESS_PARAMS["III"],  # default to type III
    )
