"""Surgical mask generation with morphological dilation and Gaussian feathering.

Procedural masks (not SAM2) -- deterministic, no model dependency.
Feathered boundaries prevent visible seams in ControlNet inpainting.
Supports clinical edge cases (vitiligo preservation, keloid softening).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

if TYPE_CHECKING:
    from landmarkdiff.clinical import ClinicalFlags

# Boundary noise parameters for seam prevention
_BOUNDARY_KERNEL_SIZE = 5  # px, morphological kernel for boundary extraction
_BOUNDARY_NOISE_MAX = 4  # max random noise offset in pixels
_BOUNDARY_NOISE_SCALE = 64  # intensity multiplier for noise mask
_GAUSSIAN_KERNEL_FACTOR = 6  # sigma multiplier for Gaussian kernel size


class MaskParams(TypedDict):
    """Typed schema for per-procedure mask configuration."""

    landmark_indices: list[int]
    dilation_px: int
    feather_sigma: float


# Procedure-specific mask parameters
MASK_CONFIG: dict[str, MaskParams] = {
    "rhinoplasty": {
        "landmark_indices": [
            1,
            2,
            4,
            5,
            6,
            19,
            94,
            141,
            168,
            195,
            197,
            236,
            240,
            274,
            275,
            278,
            279,
            294,
            326,
            327,
            360,
            363,
            370,
            456,
            460,
        ],
        "dilation_px": 30,
        "feather_sigma": 15.0,
    },
    "blepharoplasty": {
        "landmark_indices": [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            157,
            158,
            159,
            160,
            161,
            246,
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
        ],
        "dilation_px": 15,
        "feather_sigma": 10.0,
    },
    "rhytidectomy": {
        "landmark_indices": [
            10,
            21,
            54,
            58,
            67,
            93,
            103,
            109,
            127,
            132,
            136,
            150,
            162,
            172,
            176,
            187,
            207,
            213,
            234,
            284,
            297,
            323,
            332,
            338,
            356,
            361,
            365,
            379,
            389,
            397,
            400,
            427,
            454,
        ],
        "dilation_px": 40,
        "feather_sigma": 20.0,
    },
    "orthognathic": {
        "landmark_indices": [
            0,
            17,
            18,
            36,
            37,
            39,
            40,
            57,
            61,
            78,
            80,
            81,
            82,
            84,
            87,
            88,
            91,
            95,
            146,
            167,
            169,
            170,
            175,
            181,
            191,
            200,
            201,
            202,
            204,
            208,
            211,
            212,
            214,
        ],
        "dilation_px": 35,
        "feather_sigma": 18.0,
    },
    "brow_lift": {
        "landmark_indices": [
            70,
            63,
            105,
            66,
            107,  # left brow
            300,
            293,
            334,
            296,
            336,  # right brow
            9,
            8,
            10,  # forehead midline
            109,
            67,
            103,  # upper face left
            338,
            297,
            332,  # upper face right
        ],
        "dilation_px": 25,
        "feather_sigma": 15.0,
    },
    "mentoplasty": {
        "landmark_indices": [
            148,
            149,
            150,
            152,
            171,
            175,
            176,
            377,
        ],
        "dilation_px": 25,
        "feather_sigma": 12.0,
    },
    "alarplasty": {
        "landmark_indices": [
            94,
            141,
            236,
            240,
            274,
            275,
            278,
            279,
            360,
            363,
            370,
            456,
            460,
        ],
        "dilation_px": 20,
        "feather_sigma": 10.0,
    },
    "canthoplasty": {
        "landmark_indices": [
            33,
            133,
            155,
            160,
            161,
            173,
            246,
            263,
            362,
            384,
            385,
            390,
            398,
            466,
        ],
        "dilation_px": 15,
        "feather_sigma": 8.0,
    },
    "buccal_fat_removal": {
        "landmark_indices": [
            116,
            117,
            118,
            119,
            120,
            121,
            187,
            205,
            206,
            207,
            213,
            345,
            346,
            347,
            348,
            349,
            350,
            411,
            425,
            426,
            427,
            435,
        ],
        "dilation_px": 30,
        "feather_sigma": 15.0,
    },
    "dimpleplasty": {
        "landmark_indices": [
            205,
            206,
            425,
            426,
        ],
        "dilation_px": 10,
        "feather_sigma": 6.0,
    },
}


def generate_surgical_mask(
    face: FaceLandmarks,
    procedure: str,
    width: int | None = None,
    height: int | None = None,
    clinical_flags: ClinicalFlags | None = None,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a feathered surgical mask for a procedure.

    Pipeline:
    1. Create convex hull from procedure-specific landmarks
    2. Morphological dilation by N pixels
    3. Gaussian feathering for smooth alpha gradient
    4. Add Perlin-style noise at boundary to prevent visible seams

    Args:
        face: Extracted facial landmarks.
        procedure: Procedure name (e.g. "rhinoplasty").
        width: Mask width (defaults to face.image_width).
        height: Mask height (defaults to face.image_height).
        clinical_flags: Optional clinical edge-case flags (vitiligo, keloid).
        image: Original BGR image, required when clinical_flags.vitiligo is set.

    Returns:
        Float32 mask array [0.0-1.0] with feathered boundaries.
    """
    if procedure not in MASK_CONFIG:
        raise ValueError(f"Unknown procedure: {procedure}. Choose from {list(MASK_CONFIG)}")

    config = MASK_CONFIG[procedure]
    w = width or face.image_width
    h = height or face.image_height

    # Get pixel coordinates of procedure landmarks
    coords = face.landmarks[:, :2].copy()
    coords[:, 0] *= w
    coords[:, 1] *= h
    pts = coords[config["landmark_indices"]].astype(np.int32)

    # Create binary mask from convex hull
    binary = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(binary, hull, 255)

    # Morphological dilation
    dilation = config["dilation_px"]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation + 1, 2 * dilation + 1),
    )
    dilated = cv2.dilate(binary, kernel)

    # Add slight boundary noise to prevent clean-edge seams
    # (Spec: Perlin noise 2-4px on boundary before feathering)
    boundary = cv2.subtract(
        cv2.dilate(dilated, np.ones((_BOUNDARY_KERNEL_SIZE, _BOUNDARY_KERNEL_SIZE), np.uint8)),
        cv2.erode(dilated, np.ones((_BOUNDARY_KERNEL_SIZE, _BOUNDARY_KERNEL_SIZE), np.uint8)),
    )
    noise = np.random.default_rng().integers(0, _BOUNDARY_NOISE_MAX, size=(h, w), dtype=np.uint8)
    noise_boundary = cv2.bitwise_and(boundary, noise.astype(np.uint8) * _BOUNDARY_NOISE_SCALE)
    dilated = cv2.add(dilated, noise_boundary)
    dilated = np.clip(dilated, 0, 255).astype(np.uint8)

    # Gaussian feathering
    sigma = config["feather_sigma"]
    ksize = int(_GAUSSIAN_KERNEL_FACTOR * sigma) | 1  # ensure odd
    feathered = cv2.GaussianBlur(
        dilated.astype(np.float32) / 255.0,
        (ksize, ksize),
        sigma,
    )

    mask = np.clip(feathered, 0.0, 1.0)

    # Extra forehead fade: soften the top boundary to prevent visible
    # seams at the hairline (especially on receding hairlines).
    # Find the top edge of the mask and apply a gradual vertical fade.
    top_y = _find_mask_top_edge(mask)
    if top_y > 0:
        fade_height = max(int(sigma * 2), 10)
        fade_start = max(0, top_y - fade_height)
        if top_y > fade_start:
            rows = np.arange(fade_start, top_y)
            t = (rows - fade_start).astype(np.float32) / max(fade_height, 1)
            mask[fade_start:top_y, :] *= t[:, np.newaxis]

    # Clinical edge case adjustments
    if clinical_flags is not None:
        # Vitiligo: reduce mask over depigmented patches to preserve them
        if clinical_flags.vitiligo and image is not None:
            from landmarkdiff.clinical import adjust_mask_for_vitiligo, detect_vitiligo_patches

            patches = detect_vitiligo_patches(image, face)
            mask = adjust_mask_for_vitiligo(mask, patches)

        # Keloid: soften transitions in keloid-prone regions
        if clinical_flags.keloid_prone and clinical_flags.keloid_regions:
            from landmarkdiff.clinical import adjust_mask_for_keloid, get_keloid_exclusion_mask

            keloid_mask = get_keloid_exclusion_mask(
                face,
                clinical_flags.keloid_regions,
                w,
                h,
            )
            mask = adjust_mask_for_keloid(mask, keloid_mask)

    return mask


def _find_mask_top_edge(mask: np.ndarray, threshold: float = 0.05) -> int:
    """Find the topmost row where the mask exceeds threshold."""
    row_max = np.max(mask, axis=1)
    above = np.where(row_max > threshold)[0]
    return int(above[0]) if len(above) > 0 else 0


def mask_to_3channel(mask: np.ndarray) -> np.ndarray:
    """Convert single-channel mask to 3-channel for compositing."""
    return np.stack([mask, mask, mask], axis=-1)
