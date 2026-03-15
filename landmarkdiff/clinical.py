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


# ---------------------------------------------------------------------------
# Frankfort horizontal plane
# ---------------------------------------------------------------------------

# MediaPipe proxies for Frankfort plane landmarks:
# Porion (ear canal top): tragus area, use landmark 234 (left) / 454 (right)
# Infraorbital point: lower orbital rim, use landmark 145 (left) / 374 (right)
_PORION_LEFT = 234
_PORION_RIGHT = 454
_INFRAORBITAL_LEFT = 145
_INFRAORBITAL_RIGHT = 374


def compute_frankfort_angle(face: FaceLandmarks) -> float:
    """Compute the Frankfort horizontal plane angle in degrees.

    The Frankfort plane runs from the porion (top of ear canal) to the
    infraorbital point (bottom of eye orbit). In an ideally aligned face,
    this plane is horizontal (0 degrees).

    Uses MediaPipe landmark proxies:
    - Porion: tragus region (landmarks 234/454)
    - Infraorbital: lower orbital rim (landmarks 145/374)

    Returns:
        Angle in degrees (positive = tilted left-high, negative = right-high).
        0.0 means perfectly horizontal.
    """
    coords = face.landmarks[:, :2].copy()
    coords[:, 0] *= face.image_width
    coords[:, 1] *= face.image_height

    # Average left and right sides for robustness
    porion = (coords[_PORION_LEFT] + coords[_PORION_RIGHT]) / 2.0
    infraorbital = (coords[_INFRAORBITAL_LEFT] + coords[_INFRAORBITAL_RIGHT]) / 2.0

    dx = infraorbital[0] - porion[0]
    dy = infraorbital[1] - porion[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def align_to_frankfort(face: FaceLandmarks) -> FaceLandmarks:
    """Rotate landmarks so the Frankfort horizontal plane is level.

    Computes the Frankfort angle and applies a counter-rotation around
    the face center to align the plane horizontally. Landmarks remain
    in normalized [0, 1] space.

    Args:
        face: Input face landmarks.

    Returns:
        New FaceLandmarks with rotated landmarks.
    """
    angle_deg = compute_frankfort_angle(face)
    angle_rad = -np.radians(angle_deg)  # negate to counter-rotate

    coords = face.landmarks.copy()
    # Compute rotation center (face centroid in normalized space)
    cx = np.mean(coords[:, 0])
    cy = np.mean(coords[:, 1])

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy
    coords[:, 0] = cx + dx * cos_a - dy * sin_a
    coords[:, 1] = cy + dx * sin_a + dy * cos_a

    # Clamp to [0, 1]
    coords[:, :2] = np.clip(coords[:, :2], 0.0, 1.0)

    return FaceLandmarks(
        landmarks=coords,
        image_width=face.image_width,
        image_height=face.image_height,
        confidence=face.confidence,
    )


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


# ---------------------------------------------------------------------------
# Golden ratio and neoclassical facial proportions
# ---------------------------------------------------------------------------

PHI = 1.618033988749895  # golden ratio


@dataclass
class FacialProportions:
    """Neoclassical facial proportion analysis."""

    # Facial thirds (ideal: each ~1/3)
    upper_third: float  # hairline to brow
    middle_third: float  # brow to nose base
    lower_third: float  # nose base to chin

    # Lower face subdivisions (ideal: upper lip 1/3, lower 2/3)
    upper_lip_ratio: float
    lower_lip_ratio: float

    # Width ratios
    nose_to_face_width: float  # ideal ~0.25 (nose = 1/4 face width)
    eye_spacing_ratio: float  # intercanthal / biocular, ideal ~0.3

    # Golden ratio comparisons
    face_height_to_width: float  # ideal ~PHI
    nose_to_chin_over_lips_to_chin: float  # ideal ~PHI

    def summary(self) -> str:
        lines = [
            "Facial Proportions (neoclassical ideals):",
            f"  Thirds: upper={self.upper_third:.2f} mid={self.middle_third:.2f} "
            f"lower={self.lower_third:.2f} (ideal: 0.33 each)",
            f"  Lower face: upper_lip={self.upper_lip_ratio:.2f} "
            f"lower_lip={self.lower_lip_ratio:.2f} (ideal: 0.33/0.67)",
            f"  Nose/face width: {self.nose_to_face_width:.3f} (ideal: 0.25)",
            f"  Eye spacing: {self.eye_spacing_ratio:.3f} (ideal: 0.30)",
            f"  Height/width: {self.face_height_to_width:.3f} (ideal: {PHI:.3f})",
        ]
        return "\n".join(lines)


def analyze_proportions(face: FaceLandmarks) -> FacialProportions:
    """Compute neoclassical facial proportions from landmarks.

    Uses MediaPipe landmarks to approximate classical measurement points:
    - Trichion (hairline): top of face (landmark 10)
    - Glabella (brow bridge): landmark 9
    - Subnasale (nose base): landmark 94
    - Menton (chin bottom): landmark 152
    - Stomion (lip junction): landmark 14

    Args:
        face: Extracted face landmarks.

    Returns:
        FacialProportions with ratio measurements.
    """
    coords = face.pixel_coords

    # Vertical reference points
    trichion_y = coords[10, 1]  # forehead/hairline
    glabella_y = coords[9, 1]  # brow bridge
    subnasale_y = coords[94, 1]  # nose base
    menton_y = coords[152, 1]  # chin bottom
    stomion_y = coords[14, 1]  # lip junction

    face_height = max(menton_y - trichion_y, 1.0)

    # Facial thirds
    upper = (glabella_y - trichion_y) / face_height
    middle = (subnasale_y - glabella_y) / face_height
    lower = (menton_y - subnasale_y) / face_height

    # Lower face: lip proportions
    lower_face_h = max(menton_y - subnasale_y, 1.0)
    upper_lip_r = (stomion_y - subnasale_y) / lower_face_h
    lower_lip_r = (menton_y - stomion_y) / lower_face_h

    # Width measurements
    face_left_x = coords[234, 0]  # left jaw
    face_right_x = coords[454, 0]  # right jaw
    face_width = max(face_right_x - face_left_x, 1.0)

    nose_left_x = coords[240, 0]  # left alar
    nose_right_x = coords[460, 0]  # right alar
    nose_width = nose_right_x - nose_left_x

    # Eye spacing
    left_inner = coords[133, 0]  # left medial canthus
    right_inner = coords[362, 0]  # right medial canthus
    left_outer = coords[33, 0]  # left lateral canthus
    right_outer = coords[263, 0]  # right lateral canthus
    intercanthal = right_inner - left_inner
    biocular = right_outer - left_outer
    eye_spacing = intercanthal / max(biocular, 1.0)

    return FacialProportions(
        upper_third=float(upper),
        middle_third=float(middle),
        lower_third=float(lower),
        upper_lip_ratio=float(upper_lip_r),
        lower_lip_ratio=float(lower_lip_r),
        nose_to_face_width=float(nose_width / face_width),
        eye_spacing_ratio=float(eye_spacing),
        face_height_to_width=float(face_height / face_width),
        nose_to_chin_over_lips_to_chin=float(
            (menton_y - subnasale_y) / max(menton_y - stomion_y, 1.0)
        ),
    )


def visualize_proportions(
    image: np.ndarray,
    face: FaceLandmarks,
    proportions: FacialProportions,
) -> np.ndarray:
    """Overlay facial proportion guidelines on an image.

    Draws horizontal lines for facial thirds and vertical lines for
    facial fifths, with deviation scores from ideal proportions.

    Args:
        image: BGR face image.
        face: Extracted face landmarks.
        proportions: Computed facial proportions.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    coords = face.pixel_coords

    # Reference points
    trichion_y = int(coords[10, 1])
    glabella_y = int(coords[9, 1])
    subnasale_y = int(coords[94, 1])
    menton_y = int(coords[152, 1])

    # Draw facial thirds lines
    thirds_color = (0, 200, 200)  # yellow-ish
    for y, label in [
        (trichion_y, "Trichion"),
        (glabella_y, "Glabella"),
        (subnasale_y, "Subnasale"),
        (menton_y, "Menton"),
    ]:
        cv2.line(canvas, (0, y), (w, y), thirds_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (w - 90, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            thirds_color,
            1,
            cv2.LINE_AA,
        )

    # Draw facial fifths (vertical lines at eye corners + nose)
    fifths_color = (200, 200, 0)  # cyan-ish
    x_points = [
        int(coords[234, 0]),  # left jaw
        int(coords[33, 0]),  # left outer eye
        int(coords[133, 0]),  # left inner eye
        int(coords[362, 0]),  # right inner eye
        int(coords[263, 0]),  # right outer eye
        int(coords[454, 0]),  # right jaw
    ]
    for x in x_points:
        cv2.line(canvas, (x, 0), (x, h), fifths_color, 1, cv2.LINE_AA)

    # Add proportion text at bottom
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, h / 512.0 * 0.4)
    text = (
        f"Thirds: {proportions.upper_third:.2f}/{proportions.middle_third:.2f}/"
        f"{proportions.lower_third:.2f}  H/W: {proportions.face_height_to_width:.2f}"
    )
    cv2.putText(canvas, text, (5, h - 5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Deviated septum detection
# ---------------------------------------------------------------------------

# Nasal midline landmarks (dorsum, tip, columella base)
_NASAL_MIDLINE = [6, 197, 195, 5, 4, 1, 2, 94, 19]
# Alar (nostril) landmarks for bilateral comparison
_LEFT_ALAR = [240, 236, 141, 363, 370]
_RIGHT_ALAR = [460, 456, 274, 275, 278]


@dataclass
class SeptumAnalysis:
    """Results of deviated septum analysis."""

    deviation_angle: float  # degrees from vertical midline
    deviation_direction: str  # "left", "right", or "centered"
    midline_rmse: float  # RMS deviation of nasal midline from face midline (px)
    alar_asymmetry: float  # absolute difference in left vs right nostril width (px)
    severity: str  # "none", "mild", "moderate", "severe"

    def summary(self) -> str:
        lines = [
            f"Septum deviation: {self.deviation_angle:.1f} deg {self.deviation_direction}",
            f"  Midline RMSE: {self.midline_rmse:.2f} px",
            f"  Alar asymmetry: {self.alar_asymmetry:.2f} px",
            f"  Severity: {self.severity}",
        ]
        return "\n".join(lines)


def detect_deviated_septum(
    face: FaceLandmarks,
    mild_threshold: float = 2.0,
    moderate_threshold: float = 5.0,
    severe_threshold: float = 8.0,
) -> SeptumAnalysis:
    """Detect nasal septum deviation from facial landmarks.

    Measures deviation of the nasal midline from the facial vertical
    midline. Compares left and right alar (nostril) widths to assess
    bilateral asymmetry.

    Args:
        face: Extracted face landmarks.
        mild_threshold: Deviation angle (degrees) for mild classification.
        moderate_threshold: Deviation angle for moderate classification.
        severe_threshold: Deviation angle for severe classification.

    Returns:
        SeptumAnalysis with deviation measurements.
    """
    coords = face.pixel_coords

    # Facial midline: average x of left and right jaw reference points
    face_mid_x = (coords[234, 0] + coords[454, 0]) / 2.0

    # Nasal midline points
    nasal_pts = coords[_NASAL_MIDLINE]
    nasal_x = nasal_pts[:, 0]
    nasal_y = nasal_pts[:, 1]

    # Fit line to nasal midline: x = a*y + b
    # Use least squares to find the angle of deviation
    y_centered = nasal_y - nasal_y.mean()
    x_centered = nasal_x - nasal_x.mean()

    denom = float(np.sum(y_centered**2))
    slope = float(np.sum(x_centered * y_centered)) / denom if denom > 0 else 0.0

    # Angle from vertical (positive = deviated right)
    deviation_angle = float(np.degrees(np.arctan(slope)))

    # RMS deviation of nasal midline from face midline
    midline_rmse = float(np.sqrt(np.mean((nasal_x - face_mid_x) ** 2)))

    # Alar asymmetry: compare left and right nostril spread
    left_alar_x = coords[_LEFT_ALAR, 0]
    right_alar_x = coords[_RIGHT_ALAR, 0]
    left_width = float(face_mid_x - left_alar_x.mean())
    right_width = float(right_alar_x.mean() - face_mid_x)
    alar_asymmetry = abs(left_width - right_width)

    # Direction
    if abs(deviation_angle) < mild_threshold:
        direction = "centered"
    elif deviation_angle > 0:
        direction = "right"
    else:
        direction = "left"

    # Severity classification
    abs_angle = abs(deviation_angle)
    if abs_angle < mild_threshold:
        severity = "none"
    elif abs_angle < moderate_threshold:
        severity = "mild"
    elif abs_angle < severe_threshold:
        severity = "moderate"
    else:
        severity = "severe"

    return SeptumAnalysis(
        deviation_angle=deviation_angle,
        deviation_direction=direction,
        midline_rmse=midline_rmse,
        alar_asymmetry=alar_asymmetry,
        severity=severity,
    )


def visualize_septum_deviation(
    image: np.ndarray,
    face: FaceLandmarks,
    analysis: SeptumAnalysis,
) -> np.ndarray:
    """Overlay nasal midline and deviation indicator on an image.

    Args:
        image: BGR face image.
        face: Extracted face landmarks.
        analysis: Septum analysis results.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    coords = face.pixel_coords

    # Draw facial midline (green dashed approximation)
    face_mid_x = int((coords[234, 0] + coords[454, 0]) / 2.0)
    for y in range(0, h, 8):
        cv2.line(canvas, (face_mid_x, y), (face_mid_x, min(y + 4, h)), (0, 180, 0), 1)

    # Draw nasal midline (red)
    nasal_pts = coords[_NASAL_MIDLINE]
    for i in range(len(nasal_pts) - 1):
        pt1 = (int(nasal_pts[i, 0]), int(nasal_pts[i, 1]))
        pt2 = (int(nasal_pts[i + 1, 0]), int(nasal_pts[i + 1, 1]))
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw alar points
    for idx in _LEFT_ALAR:
        pt = (int(coords[idx, 0]), int(coords[idx, 1]))
        cv2.circle(canvas, pt, 3, (255, 200, 0), -1)
    for idx in _RIGHT_ALAR:
        pt = (int(coords[idx, 0]), int(coords[idx, 1]))
        cv2.circle(canvas, pt, 3, (0, 200, 255), -1)

    # Text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, h / 512.0 * 0.4)
    text = f"Deviation: {analysis.deviation_angle:.1f} deg ({analysis.severity})"
    cv2.putText(canvas, text, (5, h - 5), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Age-aware deformation scaling
# ---------------------------------------------------------------------------

# Elasticity scaling factors by age decade.
# Based on skin biomechanics: collagen density decreases ~1% per year after 30,
# tissue becomes less compliant and recovers slower from deformation.
# Values represent multiplicative scaling on displacement magnitudes.
AGE_ELASTICITY_SCALE: dict[str, float] = {
    "pediatric": 1.3,  # <18: very elastic, larger natural range
    "young_adult": 1.15,  # 18-29: high elasticity
    "adult": 1.0,  # 30-44: baseline calibration
    "middle_age": 0.85,  # 45-59: reduced elasticity
    "senior": 0.7,  # 60+: significantly reduced compliance
}

# Mapping from numeric age to bracket name
_AGE_BRACKETS = [
    (18, "pediatric"),
    (30, "young_adult"),
    (45, "adult"),
    (60, "middle_age"),
]


def classify_age_bracket(age: int) -> str:
    """Map a numeric age to an elasticity bracket.

    Args:
        age: Patient age in years.

    Returns:
        Age bracket name (e.g. "adult", "senior").
    """
    for threshold, bracket in _AGE_BRACKETS:
        if age < threshold:
            return bracket
    return "senior"


def get_age_scale_factor(age: int) -> float:
    """Get the deformation scaling factor for a given age.

    Younger patients get higher scaling (more elastic tissue allows
    larger realistic deformations), while older patients get lower
    scaling (less compliant tissue = more conservative predictions).

    Args:
        age: Patient age in years.

    Returns:
        Multiplicative scale factor for displacement magnitudes.
    """
    bracket = classify_age_bracket(age)
    return AGE_ELASTICITY_SCALE[bracket]


def scale_intensity_for_age(
    intensity: float,
    age: int,
    clamp: bool = True,
) -> float:
    """Adjust procedure intensity based on patient age.

    Modulates the requested intensity by an age-dependent elasticity
    factor. This produces more conservative predictions for older
    patients and allows slightly more aggressive predictions for
    younger patients with more elastic tissue.

    Args:
        intensity: Requested intensity (0-100).
        age: Patient age in years.
        clamp: If True, clamp result to [0, 100].

    Returns:
        Age-adjusted intensity value.
    """
    factor = get_age_scale_factor(age)
    scaled = intensity * factor
    if clamp:
        scaled = max(0.0, min(100.0, scaled))
    return scaled
