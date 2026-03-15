"""Clinical edge case handling for pathological conditions.

Implements special-case logic for:
- Vitiligo: preserve depigmented patches (don't blend over them)
- Bell's palsy: disable bilateral symmetry in deformation vectors
- Keloid: flag keloid-prone areas to reduce aggressive compositing
- Ehlers-Danlos: wider influence radii for hypermobile tissue
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


@dataclass
class ClinicalFlags:
    """Clinical condition flags that modify pipeline behavior.

    Set flags to True to enable condition-specific handling.
    """

    vitiligo: bool = False
    bells_palsy: bool = False
    bells_palsy_side: str = "left"  # affected side: "left" or "right"
    keloid_prone: bool = False
    keloid_regions: list[str] = field(default_factory=list)  # e.g. ["jawline", "nose"]
    ehlers_danlos: bool = False

    def has_any(self) -> bool:
        return self.vitiligo or self.bells_palsy or self.keloid_prone or self.ehlers_danlos


def detect_vitiligo_patches(
    image: np.ndarray,
    face: FaceLandmarks,
    l_threshold: float = 85.0,
    min_patch_area: int = 200,
) -> np.ndarray:
    """Detect depigmented (vitiligo) patches on face using LAB luminance.

    Vitiligo patches appear as high-L, low-saturation regions that deviate
    significantly from surrounding skin tone.

    Args:
        image: BGR face image.
        face: Extracted landmarks for face ROI.
        l_threshold: Luminance threshold (patches brighter than surrounding skin).
        min_patch_area: Minimum contour area in pixels to count as a patch.

    Returns:
        Binary mask (uint8, 0/255) of detected vitiligo patches.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Create face ROI mask from landmarks
    coords = face.pixel_coords.astype(np.int32)
    hull = cv2.convexHull(coords)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    # Get face-region luminance statistics
    l_channel = lab[:, :, 0]
    face_pixels = l_channel[face_mask > 0]
    if len(face_pixels) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    l_mean = np.mean(face_pixels)
    l_std = np.std(face_pixels)

    # Vitiligo patches: significantly brighter than mean skin
    threshold = min(l_threshold, l_mean + 2.0 * l_std)
    bright_mask = ((l_channel > threshold) & (face_mask > 0)).astype(np.uint8) * 255

    # Also check for low saturation (a,b channels close to 128)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    low_sat = ((np.abs(a_channel - 128) < 15) & (np.abs(b_channel - 128) < 15)).astype(
        np.uint8
    ) * 255

    # Combined: bright AND low-saturation within face
    vitiligo_raw = cv2.bitwise_and(bright_mask, low_sat)

    # Filter small noise patches
    contours, _ = cv2.findContours(vitiligo_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_patch_area:
            cv2.fillPoly(result, [cnt], 255)

    return result


def adjust_mask_for_vitiligo(
    mask: np.ndarray,
    vitiligo_patches: np.ndarray,
    preservation_factor: float = 0.3,
) -> np.ndarray:
    """Reduce mask intensity over vitiligo patches to preserve them.

    Instead of full blending over depigmented patches, we reduce the
    mask weight so the original vitiligo pattern shows through.

    Args:
        mask: Float32 surgical mask [0-1].
        vitiligo_patches: Binary mask of vitiligo regions (0/255 uint8).
        preservation_factor: How much to reduce blending (0=full blend, 1=fully preserve).

    Returns:
        Modified mask with reduced intensity over vitiligo patches.
    """
    patches_f = vitiligo_patches.astype(np.float32) / 255.0
    reduction = patches_f * preservation_factor
    return np.clip(mask - reduction, 0.0, 1.0)


def get_bells_palsy_side_indices(
    side: str,
) -> dict[str, list[int]]:
    """Get landmark indices for the affected side in Bell's palsy.

    In Bell's palsy, one side of the face is paralyzed. We should NOT
    apply bilateral symmetric deformations — only deform the healthy side.

    Returns:
        Dict mapping region names to landmark indices on the affected side.
    """
    if side == "left":
        return {
            "eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            "eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "mouth_corner": [61, 146, 91, 181, 84],
            "jawline": [132, 136, 172, 58, 150, 176, 148, 149],
        }
    else:
        return {
            "eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            "mouth_corner": [291, 308, 324, 318, 402],
            "jawline": [361, 365, 397, 288, 379, 400, 377, 378],
        }


def get_keloid_exclusion_mask(
    face: FaceLandmarks,
    regions: list[str],
    width: int,
    height: int,
    margin_px: int = 10,
) -> np.ndarray:
    """Generate mask of keloid-prone regions to exclude from aggressive compositing.

    Keloid patients should have reduced blending intensity and no sharp
    boundary transitions in prone areas (typically jawline, ears, chest).

    Args:
        face: Extracted landmarks.
        regions: List of region names prone to keloids.
        width: Image width.
        height: Image height.
        margin_px: Extra margin around keloid regions.

    Returns:
        Float32 mask [0-1] where 1 = keloid-prone area.
    """
    from landmarkdiff.landmarks import LANDMARK_REGIONS

    mask = np.zeros((height, width), dtype=np.float32)
    coords = face.pixel_coords.astype(np.int32)

    for region in regions:
        indices = LANDMARK_REGIONS.get(region, [])
        if not indices:
            continue
        pts = coords[indices]
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

    # Dilate by margin
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return np.clip(mask, 0.0, 1.0)


def adjust_mask_for_keloid(
    mask: np.ndarray,
    keloid_mask: np.ndarray,
    reduction_factor: float = 0.5,
) -> np.ndarray:
    """Soften mask transitions in keloid-prone areas.

    Reduces the mask gradient steepness to prevent hard boundaries
    that could trigger keloid formation in real surgical planning.

    Args:
        mask: Float32 surgical mask [0-1].
        keloid_mask: Float32 keloid region mask [0-1].
        reduction_factor: How much to reduce mask intensity in keloid areas.

    Returns:
        Modified mask with gentler transitions in keloid regions.
    """
    # Reduce mask intensity in keloid-prone areas
    keloid_reduction = keloid_mask * reduction_factor
    modified = mask * (1.0 - keloid_reduction)

    # Extra Gaussian blur in keloid regions for softer transitions
    blur_kernel = 31
    blurred = cv2.GaussianBlur(modified, (blur_kernel, blur_kernel), 10.0)

    # Use blurred version only in keloid regions
    result = modified * (1.0 - keloid_mask) + blurred * keloid_mask
    return np.clip(result, 0.0, 1.0)


# Bilateral landmark pairs (left_index, right_index) for asymmetry analysis.
# Covers jawline, eyes, eyebrows, nose, and lips.
BILATERAL_PAIRS: list[tuple[int, int]] = [
    # Jawline
    (234, 454),
    (132, 361),
    (58, 288),
    (172, 397),
    (136, 365),
    (150, 379),
    (176, 400),
    (93, 323),
    (127, 356),
    (162, 389),
    (21, 251),
    (54, 284),
    (103, 332),
    (67, 297),
    (109, 338),
    # Eyes
    (33, 263),
    (133, 362),
    (160, 385),
    (159, 386),
    (158, 387),
    (157, 388),
    (161, 384),
    (246, 466),
    (173, 398),
    (145, 374),
    (153, 380),
    (154, 381),
    (155, 390),
    (144, 373),
    (163, 382),
    (7, 249),
    # Eyebrows
    (70, 300),
    (63, 293),
    (105, 334),
    (66, 296),
    (107, 336),
    (55, 285),
    (65, 295),
    (52, 282),
    (53, 283),
    (46, 276),
    # Nose (bilateral parts)
    (240, 460),
    (236, 456),
    (141, 370),
    (195, 419),
    (197, 399),
    # Lips
    (61, 291),
    (146, 375),
    (91, 321),
    (181, 405),
    (84, 314),
    (78, 308),
    (95, 324),
    (88, 318),
    (178, 402),
]


@dataclass
class AsymmetryResult:
    """Facial asymmetry analysis result."""

    score: float  # overall asymmetry (0 = perfect symmetry, higher = more asymmetric)
    region_scores: dict[str, float]  # per-region asymmetry scores
    pair_deviations: np.ndarray  # per-pair deviation magnitudes

    def summary(self) -> str:
        lines = [f"Asymmetry score: {self.score:.4f}"]
        for region, s in sorted(self.region_scores.items()):
            lines.append(f"  {region}: {s:.4f}")
        return "\n".join(lines)


def quantify_asymmetry(face: FaceLandmarks) -> AsymmetryResult:
    """Compute facial asymmetry by comparing bilateral landmark positions.

    For each left-right pair, reflects the left landmark across the face
    midline and measures its distance to the corresponding right landmark.
    Distances are normalized by inter-ocular distance for scale invariance.

    Args:
        face: Extracted face landmarks.

    Returns:
        AsymmetryResult with overall score, per-region scores, and
        per-pair deviation magnitudes.
    """
    coords = face.landmarks[:, :2].copy()

    # Midline x-coordinate: average of nose tip (1) and nasion (168)
    midline_x = (coords[1, 0] + coords[168, 0]) / 2.0

    # Inter-ocular distance for normalization
    iod = np.linalg.norm(coords[33] - coords[263])
    if iod < 1e-6:
        iod = 1.0

    # Reflect left landmarks across midline and compare to right
    deviations = np.zeros(len(BILATERAL_PAIRS))
    for i, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        left = coords[left_idx].copy()
        right = coords[right_idx].copy()
        # Reflect left across midline: mirror x coordinate
        left[0] = 2.0 * midline_x - left[0]
        deviations[i] = np.linalg.norm(left - right) / iod

    # Per-region scores using pair index ranges
    region_ranges = {
        "jawline": (0, 15),
        "eyes": (15, 31),
        "eyebrows": (31, 41),
        "nose": (41, 46),
        "lips": (46, 55),
    }
    region_scores = {}
    for region, (start, end) in region_ranges.items():
        region_devs = deviations[start:end]
        region_scores[region] = float(np.mean(region_devs)) if len(region_devs) > 0 else 0.0

    return AsymmetryResult(
        score=float(np.mean(deviations)),
        region_scores=region_scores,
        pair_deviations=deviations,
    )


def visualize_asymmetry(
    image: np.ndarray,
    face: FaceLandmarks,
    result: AsymmetryResult,
    threshold: float = 0.05,
) -> np.ndarray:
    """Overlay asymmetry visualization on a face image.

    Draws circles on landmark pairs that exceed the asymmetry threshold,
    colored by severity (green=mild, yellow=moderate, red=severe).

    Args:
        image: BGR face image.
        face: Extracted face landmarks.
        result: Asymmetry analysis result.
        threshold: Minimum deviation to highlight.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    coords = face.pixel_coords

    for i, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        dev = result.pair_deviations[i]
        if dev < threshold:
            continue

        # Color by severity
        if dev < 0.08:
            color = (0, 255, 0)  # green - mild
        elif dev < 0.15:
            color = (0, 255, 255)  # yellow - moderate
        else:
            color = (0, 0, 255)  # red - severe

        radius = max(3, int(dev * 50))
        for idx in (left_idx, right_idx):
            pt = (int(coords[idx, 0]), int(coords[idx, 1]))
            cv2.circle(canvas, pt, radius, color, 2, cv2.LINE_AA)

    # Add score text
    h = canvas.shape[0]
    font_scale = max(0.4, h / 512.0 * 0.6)
    cv2.putText(
        canvas,
        f"Asymmetry: {result.score:.3f}",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas
