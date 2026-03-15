"""Clinical facial measurements from MediaPipe 478-point landmarks.

Implements standard anthropometric measurements used in surgical planning:
- Goode ratio (nasal projection / nasal length) for rhinoplasty
- Nasofrontal angle for rhinoplasty planning
- Canthal tilt for blepharoplasty
- Cervico-mental angle for rhytidectomy
- Lip-chin relationship (Holdaway H-line) for mentoplasty
- Scleral show detection for blepharoplasty safety
- Dental show analysis for orthognathic planning
- Mandibular angle classification for orthognathic
- Facial thirds and fifths proportion analysis
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pixel(face: FaceLandmarks, idx: int) -> np.ndarray:
    """Get pixel coordinates for a single landmark index."""
    lm = face.landmarks[idx]
    return np.array([lm[0] * face.image_width, lm[1] * face.image_height])


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Compute angle at vertex in degrees (0-180)."""
    v1 = a - vertex
    v2 = b - vertex
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# MediaPipe landmark indices for measurements
# ---------------------------------------------------------------------------

# Nose landmarks
_NASION = 168  # bridge of nose (glabella-nasion junction)
_NOSE_TIP = 1  # pronasale
_NOSE_BASE = 94  # subnasale
_COLUMELLA = 2  # columella base
_LEFT_ALAR = 240  # left alar crease
_RIGHT_ALAR = 460  # right alar crease

# Eye landmarks
_LEFT_MEDIAL_CANTHUS = 133
_LEFT_LATERAL_CANTHUS = 33
_RIGHT_MEDIAL_CANTHUS = 362
_RIGHT_LATERAL_CANTHUS = 263

# Lower eyelid (for scleral show)
_LEFT_LOWER_LID = 145  # inferior orbital rim
_RIGHT_LOWER_LID = 374
_LEFT_IRIS_BOTTOM = 145  # lower lid center
_RIGHT_IRIS_BOTTOM = 374

# Lip landmarks
_UPPER_LIP = 0  # vermilion border upper lip center
_LOWER_LIP = 17  # vermilion border lower lip center
_STOMION = 14  # lip junction

# Chin and jaw
_CHIN = 152  # menton (lowest chin point)
_POGONION = 199  # most anterior chin point
_GLABELLA = 9  # forehead/brow bridge

# Jaw angle landmarks (approximations using MediaPipe)
_LEFT_GONION = 172  # left jaw angle
_RIGHT_GONION = 397  # right jaw angle

# Forehead
_TRICHION = 10  # hairline/top of forehead

# Neck/submental
_SUBMENTALE = 152  # chin bottom (same as menton for frontal)
_HYOID_APPROX = 152  # approximation; true hyoid not visible in photos


# ---------------------------------------------------------------------------
# Goode ratio (rhinoplasty)
# ---------------------------------------------------------------------------


@dataclass
class GoodeRatio:
    """Nasal projection analysis using the Goode method."""

    ratio: float  # projection / length; ideal 0.55-0.60
    projection_px: float  # nasal tip to alar crease line (pixels)
    length_px: float  # nasion to tip (pixels)
    classification: str  # "underprojected", "normal", "overprojected"


def compute_goode_ratio(face: FaceLandmarks) -> GoodeRatio:
    """Compute Goode ratio (nasal projection / nasal length).

    Ideal Goode ratio is 0.55-0.60. Values below suggest underprojection
    (common rhinoplasty indication), values above suggest overprojection.

    Args:
        face: Extracted face landmarks.

    Returns:
        GoodeRatio with measurement and classification.
    """
    nasion = _pixel(face, _NASION)
    tip = _pixel(face, _NOSE_TIP)
    base = _pixel(face, _NOSE_BASE)

    # Nasal length: nasion to tip
    length_px = float(np.linalg.norm(tip - nasion))

    # Nasal projection: perpendicular distance from tip to alar crease line
    # Approximate as distance from tip to base (subnasale) projected forward
    projection_px = float(np.linalg.norm(tip - base))

    ratio = projection_px / max(length_px, 1.0)

    if ratio < 0.50:
        classification = "underprojected"
    elif ratio > 0.65:
        classification = "overprojected"
    else:
        classification = "normal"

    return GoodeRatio(
        ratio=ratio,
        projection_px=projection_px,
        length_px=length_px,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Nasofrontal angle (rhinoplasty)
# ---------------------------------------------------------------------------


@dataclass
class NasofrontalAngle:
    """Nasofrontal angle measurement."""

    angle: float  # degrees; ideal 115-135
    classification: str  # "acute", "normal", "obtuse"


def compute_nasofrontal_angle(face: FaceLandmarks) -> NasofrontalAngle:
    """Compute the nasofrontal angle (glabella-nasion-tip).

    The nasofrontal angle is formed at the nasion between the glabella
    and the nasal dorsum. Ideal range is 115-135 degrees.

    Args:
        face: Extracted face landmarks.

    Returns:
        NasofrontalAngle with measurement and classification.
    """
    glabella = _pixel(face, _GLABELLA)
    nasion = _pixel(face, _NASION)
    tip = _pixel(face, _NOSE_TIP)

    angle = _angle_between(glabella, nasion, tip)

    if angle < 115.0:
        classification = "acute"
    elif angle > 135.0:
        classification = "obtuse"
    else:
        classification = "normal"

    return NasofrontalAngle(angle=angle, classification=classification)


# ---------------------------------------------------------------------------
# Canthal tilt (blepharoplasty)
# ---------------------------------------------------------------------------


@dataclass
class CanthalTilt:
    """Canthal tilt measurement for both eyes."""

    left_angle: float  # degrees from horizontal (positive = upward tilt)
    right_angle: float  # degrees from horizontal
    mean_angle: float
    classification: str  # "negative", "neutral", "positive"


def compute_canthal_tilt(face: FaceLandmarks) -> CanthalTilt:
    """Measure canthal tilt from medial to lateral canthus.

    Positive canthal tilt (lateral canthus higher than medial) is considered
    aesthetically desirable. Negative canthal tilt may be an indication for
    canthoplasty or lateral canthopexy.

    Normal range: +2 to +5 degrees.

    Args:
        face: Extracted face landmarks.

    Returns:
        CanthalTilt with per-eye and mean measurements.
    """
    l_med = _pixel(face, _LEFT_MEDIAL_CANTHUS)
    l_lat = _pixel(face, _LEFT_LATERAL_CANTHUS)
    r_med = _pixel(face, _RIGHT_MEDIAL_CANTHUS)
    r_lat = _pixel(face, _RIGHT_LATERAL_CANTHUS)

    # Angle from horizontal: positive means lateral is higher (lower y)
    # In image coords, y increases downward, so negative dy = lateral higher
    left_dy = l_lat[1] - l_med[1]
    left_dx = l_lat[0] - l_med[0]
    # Negate because image y is inverted
    left_angle = float(-np.degrees(np.arctan2(left_dy, abs(left_dx))))

    right_dy = r_lat[1] - r_med[1]
    right_dx = r_lat[0] - r_med[0]
    right_angle = float(-np.degrees(np.arctan2(right_dy, abs(right_dx))))

    mean_angle = (left_angle + right_angle) / 2.0

    if mean_angle < -1.0:
        classification = "negative"
    elif mean_angle > 6.0:
        classification = "positive"
    else:
        classification = "neutral"

    return CanthalTilt(
        left_angle=left_angle,
        right_angle=right_angle,
        mean_angle=mean_angle,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Cervico-mental angle (rhytidectomy / facelift)
# ---------------------------------------------------------------------------


@dataclass
class CervicoMentalAngle:
    """Cervico-mental angle measurement."""

    angle: float  # degrees; ideal 105-120
    classification: str  # "acute", "ideal", "obtuse"


def compute_cervicomental_angle(face: FaceLandmarks) -> CervicoMentalAngle:
    """Compute the cervico-mental angle (chin-neck junction).

    This angle is formed between a line from the chin (menton) to the
    submentale and the anterior neck plane. A more acute angle indicates
    a well-defined jawline; an obtuse angle suggests submental fullness.

    Ideal range: 105-120 degrees. Values >130 suggest candidate for
    rhytidectomy or submental liposuction.

    Note: MediaPipe does not capture neck landmarks, so we approximate
    using the chin and jaw angle points.

    Args:
        face: Extracted face landmarks.

    Returns:
        CervicoMentalAngle with measurement and classification.
    """
    chin = _pixel(face, _CHIN)
    pogonion = _pixel(face, _POGONION)
    # Approximate neck point: extrapolate below chin
    # Use jaw angles to estimate the neck plane direction
    left_gonion = _pixel(face, _LEFT_GONION)
    right_gonion = _pixel(face, _RIGHT_GONION)
    gonion_mid = (left_gonion + right_gonion) / 2.0

    # Vector from gonion to chin (jaw line)
    jaw_vec = chin - gonion_mid
    # Approximate neck point: below chin along a steeper angle
    neck_point = chin + np.array([0.0, abs(jaw_vec[1]) * 0.5])

    angle = _angle_between(pogonion, chin, neck_point)

    if angle < 105.0:
        classification = "acute"
    elif angle > 130.0:
        classification = "obtuse"
    else:
        classification = "ideal"

    return CervicoMentalAngle(angle=angle, classification=classification)


# ---------------------------------------------------------------------------
# Lip-chin relationship / Holdaway H-line (mentoplasty)
# ---------------------------------------------------------------------------


@dataclass
class LipChinRelation:
    """Lip-chin relationship analysis (Holdaway H-line)."""

    h_line_angle: float  # angle of H-line from vertical
    lip_to_chin_distance_px: float  # distance from lower lip to chin
    lower_face_height_px: float  # subnasale to menton
    ratio: float  # lip_to_chin / lower_face_height
    classification: str  # "retruded_chin", "normal", "prominent_chin"


def compute_lip_chin_relation(face: FaceLandmarks) -> LipChinRelation:
    """Compute the Holdaway H-line lip-chin relationship.

    The H-line connects the most prominent point of the chin (pogonion)
    to the upper lip. Ideally, the lower lip should touch or be within
    1-2mm of this line.

    Args:
        face: Extracted face landmarks.

    Returns:
        LipChinRelation with measurements and classification.
    """
    upper_lip = _pixel(face, _UPPER_LIP)
    lower_lip = _pixel(face, _LOWER_LIP)
    chin = _pixel(face, _CHIN)
    pogonion = _pixel(face, _POGONION)
    subnasale = _pixel(face, _NOSE_BASE)

    # H-line: upper lip to pogonion
    h_vec = pogonion - upper_lip
    h_line_angle = float(np.degrees(np.arctan2(h_vec[0], h_vec[1])))

    # Lower lip to chin distance
    lip_chin_dist = float(np.linalg.norm(chin - lower_lip))

    # Lower face height
    lower_face_h = float(np.linalg.norm(chin - subnasale))
    ratio = lip_chin_dist / max(lower_face_h, 1.0)

    # Classification based on ratio
    if ratio > 0.70:
        classification = "retruded_chin"
    elif ratio < 0.50:
        classification = "prominent_chin"
    else:
        classification = "normal"

    return LipChinRelation(
        h_line_angle=h_line_angle,
        lip_to_chin_distance_px=lip_chin_dist,
        lower_face_height_px=lower_face_h,
        ratio=ratio,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Scleral show detection (blepharoplasty safety)
# ---------------------------------------------------------------------------


@dataclass
class ScleralShow:
    """Inferior scleral show measurement."""

    left_show_px: float  # pixels of visible sclera below left iris
    right_show_px: float  # pixels below right iris
    mean_show_px: float
    has_scleral_show: bool  # True if show > threshold
    risk_level: str  # "none", "mild", "significant"


def detect_scleral_show(
    face: FaceLandmarks,
    threshold_px: float = 3.0,
) -> ScleralShow:
    """Detect inferior scleral show from landmark positions.

    Inferior scleral show (visible white below the iris) is a risk factor
    for blepharoplasty complications. Patients with existing scleral show
    may experience worsening after lower lid surgery.

    Uses the distance between the lower iris boundary and the lower lid
    margin as a proxy. In MediaPipe, the lower eyelid landmarks approximate
    the lid margin, while iris landmarks approximate the limbus.

    Args:
        face: Extracted face landmarks.
        threshold_px: Minimum pixels to classify as scleral show.

    Returns:
        ScleralShow with measurements and risk assessment.
    """
    # Lower lid midpoints
    left_lid = _pixel(face, _LEFT_LOWER_LID)
    right_lid = _pixel(face, _RIGHT_LOWER_LID)

    # Iris center approximation (MediaPipe iris landmarks 468-472 left, 473-477 right)
    # Lower iris = iris center y + iris radius
    # Use landmarks 468 (left iris center) and 473 (right iris center)
    left_iris = _pixel(face, 468)
    right_iris = _pixel(face, 473)

    # Iris radius approximation: distance from center to edge landmark
    left_iris_edge = _pixel(face, 469)
    right_iris_edge = _pixel(face, 474)
    left_radius = float(np.linalg.norm(left_iris - left_iris_edge))
    right_radius = float(np.linalg.norm(right_iris - right_iris_edge))

    # Scleral show = lower lid y - (iris center y + iris radius)
    # Positive means lid is below iris bottom = visible sclera
    left_show = float(left_lid[1] - (left_iris[1] + left_radius))
    right_show = float(right_lid[1] - (right_iris[1] + right_radius))

    # Clamp negative values to 0 (no show)
    left_show = max(0.0, left_show)
    right_show = max(0.0, right_show)
    mean_show = (left_show + right_show) / 2.0

    has_show = mean_show > threshold_px
    if mean_show < threshold_px:
        risk = "none"
    elif mean_show < threshold_px * 2:
        risk = "mild"
    else:
        risk = "significant"

    return ScleralShow(
        left_show_px=left_show,
        right_show_px=right_show,
        mean_show_px=mean_show,
        has_scleral_show=has_show,
        risk_level=risk,
    )


# ---------------------------------------------------------------------------
# Dental show (orthognathic)
# ---------------------------------------------------------------------------


@dataclass
class DentalShow:
    """Incisor show analysis."""

    show_px: float  # vertical distance of visible teeth (pixels)
    upper_lip_to_teeth_px: float  # upper lip vermilion to upper tooth edge
    classification: str  # "insufficient", "normal", "excessive"


def compute_dental_show(face: FaceLandmarks) -> DentalShow:
    """Analyze incisor show at rest from lip and dental landmarks.

    Normal upper incisor show at rest is 2-4mm. Excessive show (>4mm)
    may indicate vertical maxillary excess; insufficient show (<1mm)
    may indicate aging or lip ptosis.

    MediaPipe does not detect teeth directly. We approximate using
    the lip separation (stomion gap) as a proxy for dental visibility.

    Args:
        face: Extracted face landmarks.

    Returns:
        DentalShow with measurements and classification.
    """
    upper_lip = _pixel(face, _UPPER_LIP)
    stomion = _pixel(face, _STOMION)

    # Lip separation as proxy for dental show
    show_px = float(stomion[1] - upper_lip[1])
    show_px = max(0.0, show_px)

    # Upper lip to "teeth" (upper vermilion to stomion)
    lip_teeth = show_px  # same proxy

    # Classify based on approximate mm conversion
    # At 512px for typical face height (~200mm), 1mm ~ 2.5px
    if show_px < 2.5:
        classification = "insufficient"
    elif show_px > 10.0:
        classification = "excessive"
    else:
        classification = "normal"

    return DentalShow(
        show_px=show_px,
        upper_lip_to_teeth_px=lip_teeth,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Mandibular angle (orthognathic)
# ---------------------------------------------------------------------------


@dataclass
class MandibularAngle:
    """Mandibular angle classification."""

    left_angle: float  # degrees
    right_angle: float  # degrees
    mean_angle: float
    classification: str  # "low" (<120), "normal" (120-130), "high" (>130)


def compute_mandibular_angle(face: FaceLandmarks) -> MandibularAngle:
    """Classify mandibular angle from jaw landmarks.

    The mandibular angle is formed at the gonion between the ramus
    (ascending jaw) and the body (horizontal jaw). Normal is 120-130
    degrees.

    Low angle (<120): short face, strong jaw, deep bite tendency.
    High angle (>130): long face, weak chin projection, open bite tendency.

    Args:
        face: Extracted face landmarks.

    Returns:
        MandibularAngle with measurements and classification.
    """
    chin = _pixel(face, _CHIN)
    left_gonion = _pixel(face, _LEFT_GONION)
    right_gonion = _pixel(face, _RIGHT_GONION)

    # Approximate ramus direction using ear-adjacent landmarks
    left_ramus_top = _pixel(face, 234)  # near tragus
    right_ramus_top = _pixel(face, 454)

    left_angle = _angle_between(chin, left_gonion, left_ramus_top)
    right_angle = _angle_between(chin, right_gonion, right_ramus_top)
    mean_angle = (left_angle + right_angle) / 2.0

    if mean_angle < 120.0:
        classification = "low"
    elif mean_angle > 130.0:
        classification = "high"
    else:
        classification = "normal"

    return MandibularAngle(
        left_angle=left_angle,
        right_angle=right_angle,
        mean_angle=mean_angle,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Facial thirds and fifths (proportions)
# ---------------------------------------------------------------------------


@dataclass
class FacialThirds:
    """Facial vertical proportion analysis."""

    upper: float  # trichion to glabella / total height
    middle: float  # glabella to subnasale / total height
    lower: float  # subnasale to menton / total height
    deviation_from_ideal: float  # RMS deviation from 1/3 each


@dataclass
class FacialFifths:
    """Facial horizontal proportion analysis (five-eye rule)."""

    widths: list[float]  # 5 segment widths as fractions of total
    deviation_from_ideal: float  # RMS deviation from 1/5 each


def compute_facial_thirds(face: FaceLandmarks) -> FacialThirds:
    """Compute vertical facial third proportions.

    Ideal: each third (upper, middle, lower) is 1/3 of total face height.

    Args:
        face: Extracted face landmarks.

    Returns:
        FacialThirds with proportions and deviation from ideal.
    """
    trichion_y = _pixel(face, _TRICHION)[1]
    glabella_y = _pixel(face, _GLABELLA)[1]
    subnasale_y = _pixel(face, _NOSE_BASE)[1]
    menton_y = _pixel(face, _CHIN)[1]

    total = max(menton_y - trichion_y, 1.0)
    upper = (glabella_y - trichion_y) / total
    middle = (subnasale_y - glabella_y) / total
    lower = (menton_y - subnasale_y) / total

    ideal = 1.0 / 3.0
    diffs = [(upper - ideal) ** 2, (middle - ideal) ** 2, (lower - ideal) ** 2]
    deviation = float(np.sqrt(np.mean(diffs)))

    return FacialThirds(
        upper=float(upper),
        middle=float(middle),
        lower=float(lower),
        deviation_from_ideal=deviation,
    )


def compute_facial_fifths(face: FaceLandmarks) -> FacialFifths:
    """Compute horizontal facial fifth proportions (five-eye rule).

    The face is divided into 5 equal vertical segments, each ideally
    the width of one eye. Segments: temporal-to-outer-canthus,
    outer-to-inner-canthus (eye), inner-to-inner (intercanthal),
    inner-to-outer-canthus (eye), outer-canthus-to-temporal.

    Args:
        face: Extracted face landmarks.

    Returns:
        FacialFifths with segment widths and deviation from ideal.
    """
    left_temp = _pixel(face, 234)[0]  # left face edge
    left_outer = _pixel(face, _LEFT_LATERAL_CANTHUS)[0]
    left_inner = _pixel(face, _LEFT_MEDIAL_CANTHUS)[0]
    right_inner = _pixel(face, _RIGHT_MEDIAL_CANTHUS)[0]
    right_outer = _pixel(face, _RIGHT_LATERAL_CANTHUS)[0]
    right_temp = _pixel(face, 454)[0]  # right face edge

    total = max(right_temp - left_temp, 1.0)
    segments = [
        (left_outer - left_temp) / total,
        (left_inner - left_outer) / total,
        (right_inner - left_inner) / total,
        (right_outer - right_inner) / total,
        (right_temp - right_outer) / total,
    ]

    ideal = 0.2
    deviation = float(np.sqrt(np.mean([(s - ideal) ** 2 for s in segments])))

    return FacialFifths(widths=[float(s) for s in segments], deviation_from_ideal=deviation)


# ---------------------------------------------------------------------------
# Intensity calibration (literature-based)
# ---------------------------------------------------------------------------

# Clinically plausible displacement ranges per procedure, in mm.
# Based on published anthropometric data:
# - Farkas (1994) facial norms
# - Aesthetic Surgery Journal guidelines
# - Rhinoplasty: tip projection change 2-6mm typical
# - Blepharoplasty: lid show change 1-3mm
# - Orthognathic: advancement/setback 2-12mm
# Intensity 50 = median change, intensity 100 = 95th percentile
PROCEDURE_CALIBRATION: dict[str, dict[str, float]] = {
    "rhinoplasty": {"median_mm": 3.0, "p95_mm": 6.0},
    "blepharoplasty": {"median_mm": 1.5, "p95_mm": 3.5},
    "rhytidectomy": {"median_mm": 4.0, "p95_mm": 8.0},
    "orthognathic": {"median_mm": 5.0, "p95_mm": 12.0},
    "brow_lift": {"median_mm": 3.0, "p95_mm": 6.0},
    "mentoplasty": {"median_mm": 4.0, "p95_mm": 10.0},
    "alarplasty": {"median_mm": 2.0, "p95_mm": 4.0},
    "canthoplasty": {"median_mm": 1.5, "p95_mm": 3.0},
    "buccal_fat_removal": {"median_mm": 3.0, "p95_mm": 6.0},
    "dimpleplasty": {"median_mm": 1.0, "p95_mm": 2.5},
    "genioplasty": {"median_mm": 4.0, "p95_mm": 10.0},
    "malarplasty": {"median_mm": 3.0, "p95_mm": 7.0},
    "lip_lift": {"median_mm": 2.0, "p95_mm": 4.0},
    "lip_augmentation": {"median_mm": 2.0, "p95_mm": 5.0},
    "forehead_reduction": {"median_mm": 8.0, "p95_mm": 20.0},
    "submental_liposuction": {"median_mm": 3.0, "p95_mm": 6.0},
    "otoplasty": {"median_mm": 8.0, "p95_mm": 15.0},
}

# Average interpupillary distance for mm-to-pixel conversion
_AVERAGE_IPD_MM = 63.0


def calibrate_intensity(
    procedure: str,
    intensity: float,
    face: FaceLandmarks | None = None,
    use_sigmoid: bool = True,
) -> float:
    """Map user intensity (0-100) to calibrated displacement in mm.

    Applies a non-linear (sigmoid) mapping so that:
    - intensity=50 -> median published surgical change
    - intensity=100 -> 95th percentile published change
    - Small changes near 0 and 100 produce less dramatic effects

    Args:
        procedure: Procedure name.
        intensity: User-specified intensity (0-100).
        face: Optional face for pixel-scale calibration.
        use_sigmoid: If True, apply sigmoid non-linearity.

    Returns:
        Calibrated displacement in mm. If face is provided,
        returns displacement in pixels instead.
    """
    cal = PROCEDURE_CALIBRATION.get(procedure)
    if cal is None:
        # Fall back to linear scaling for unknown procedures
        return intensity / 100.0

    p95_mm = cal["p95_mm"]

    t = intensity / 100.0  # normalize to 0-1

    if use_sigmoid:
        # Sigmoid mapping: steeper in middle, flatter at extremes
        # Shift and scale so sigmoid(0.5) = median, sigmoid(1.0) ~ p95
        k = 6.0  # steepness
        s = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
        # Normalize sigmoid to 0-1 range
        s0 = 1.0 / (1.0 + np.exp(-k * (-0.5)))
        s1 = 1.0 / (1.0 + np.exp(-k * 0.5))
        s_norm = (s - s0) / (s1 - s0)
        displacement_mm = float(s_norm * p95_mm)
    else:
        # Linear: 50 -> median, 100 -> p95
        displacement_mm = float(t * p95_mm)

    if face is not None:
        # Convert mm to pixels using interpupillary distance
        left_eye = _pixel(face, _LEFT_LATERAL_CANTHUS)
        right_eye = _pixel(face, _RIGHT_LATERAL_CANTHUS)
        ipd_px = float(np.linalg.norm(right_eye - left_eye))
        px_per_mm = ipd_px / _AVERAGE_IPD_MM
        return displacement_mm * px_per_mm

    return displacement_mm
