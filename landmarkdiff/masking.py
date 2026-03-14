"""Surgical mask generation with morphological dilation and Gaussian feathering.

Procedural masks (not SAM2) — deterministic, no model dependency.
Feathered boundaries prevent visible seams in ControlNet inpainting.
Supports clinical edge cases (vitiligo preservation, keloid softening).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

if TYPE_CHECKING:
    from landmarkdiff.clinical import ClinicalFlags

# Procedure-specific mask parameters
MASK_CONFIG: dict[str, dict] = {
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
        procedure: Procedure name.
        width: Mask width.
        height: Mask height.

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
        cv2.dilate(dilated, np.ones((5, 5), np.uint8)),
        cv2.erode(dilated, np.ones((5, 5), np.uint8)),
    )
    noise = np.random.default_rng().integers(0, 4, size=(h, w), dtype=np.uint8)
    noise_boundary = cv2.bitwise_and(boundary, noise.astype(np.uint8) * 64)
    dilated = cv2.add(dilated, noise_boundary)
    dilated = np.clip(dilated, 0, 255).astype(np.uint8)

    # Gaussian feathering
    sigma = config["feather_sigma"]
    ksize = int(6 * sigma) | 1  # ensure odd
    feathered = cv2.GaussianBlur(
        dilated.astype(np.float32) / 255.0,
        (ksize, ksize),
        sigma,
    )

    mask = np.clip(feathered, 0.0, 1.0)

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


def mask_to_3channel(mask: np.ndarray) -> np.ndarray:
    """Convert single-channel mask to 3-channel for compositing."""
    return np.stack([mask, mask, mask], axis=-1)
