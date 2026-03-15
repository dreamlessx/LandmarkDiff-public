"""Landmark manipulation via Gaussian RBF deformation.

v1/v2 uses relative sliders (0-100 intensity).
mm inputs only in v3+ with FLAME calibrated metric space.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

if TYPE_CHECKING:
    from landmarkdiff.clinical import ClinicalFlags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeformationHandle:
    """Single deformation control point."""

    landmark_index: int
    displacement: np.ndarray  # (2,) or (3,) pixel displacement
    influence_radius: float  # Gaussian RBF radius in pixels


@dataclass
class RegionalIntensity:
    """Per-sub-region intensity overrides for fine-grained control.

    Each field is a multiplier (0.0 to 2.0) applied to the base
    intensity for that anatomical sub-region. Default 1.0 means
    no override (use base intensity as-is).

    Available sub-regions vary by procedure:
    - rhinoplasty: tip, bridge, alar
    - blepharoplasty: upper_lid, lower_lid, corners
    - rhytidectomy: jowl, chin, temple
    - orthognathic: jaw, chin_projection, lateral_jaw
    - brow_lift: brow, forehead
    - mentoplasty: chin_tip, contour, jaw_angle
    """

    tip: float = 1.0
    bridge: float = 1.0
    alar: float = 1.0
    upper_lid: float = 1.0
    lower_lid: float = 1.0
    corners: float = 1.0
    jowl: float = 1.0
    chin: float = 1.0
    temple: float = 1.0
    jaw: float = 1.0
    chin_projection: float = 1.0
    lateral_jaw: float = 1.0
    brow: float = 1.0
    forehead: float = 1.0
    chin_tip: float = 1.0
    contour: float = 1.0
    jaw_angle: float = 1.0


# Procedure-specific landmark indices from the technical specification
PROCEDURE_LANDMARKS: dict[str, list[int]] = {
    "rhinoplasty": [
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
    "blepharoplasty": [
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
    "rhytidectomy": [
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
    "orthognathic": [
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
        269,
        270,
        291,
        311,
        312,
        317,
        321,
        324,
        325,
        375,
        396,
        405,
        407,
        415,
    ],
    "brow_lift": [
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
        10,
        109,
        67,
        103,
        338,
        297,
        332,  # forehead/upper face
    ],
    "mentoplasty": [
        148,
        149,
        150,
        152,
        171,
        175,
        176,
        377,
    ],
    "alarplasty": [
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
    "canthoplasty": [
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
    "buccal_fat_removal": [
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
    "dimpleplasty": [
        205,
        206,
        425,
        426,
    ],
    "genioplasty": [
        17,
        149,
        150,
        152,
        175,
        176,
        199,
        200,
        314,
        377,
        378,
        396,
        405,
    ],
    "malarplasty": [
        34,
        93,
        116,
        117,
        123,
        132,
        187,
        205,
        213,
        234,
        264,
        323,
        345,
        346,
        352,
        361,
        411,
        425,
        435,
        454,
    ],
    "lip_lift": [
        0,
        11,
        12,
        13,
        14,
        37,
        39,
        40,
        61,
        72,
        73,
        267,
        269,
        270,
        291,
        302,
        303,
    ],
    "lip_augmentation": [
        0,
        11,
        12,
        13,
        14,
        37,
        39,
        40,
        61,
        72,
        73,
        80,
        81,
        82,
        178,
        181,
        267,
        269,
        270,
        291,
        302,
        303,
        310,
        311,
        312,
        402,
        405,
    ],
    "forehead_reduction": [
        8,
        9,
        10,
        21,
        54,
        67,
        68,
        69,
        103,
        104,
        108,
        109,
        251,
        284,
        297,
        298,
        299,
        332,
        333,
        337,
        338,
    ],
    "submental_liposuction": [
        148,
        149,
        150,
        152,
        169,
        170,
        171,
        175,
        176,
        199,
        200,
        377,
        378,
        396,
        400,
    ],
    "otoplasty": [
        # Left ear-adjacent: preauricular, temple, upper jaw
        234,
        127,
        93,
        132,
        162,
        21,
        # Right ear-adjacent
        454,
        356,
        323,
        361,
        389,
        251,
    ],
}
# Default influence radii per procedure (in pixels at 512x512)
PROCEDURE_RADIUS: dict[str, float] = {
    "rhinoplasty": 30.0,
    "blepharoplasty": 15.0,
    "rhytidectomy": 40.0,
    "orthognathic": 35.0,
    "brow_lift": 25.0,
    "mentoplasty": 25.0,
    "alarplasty": 20.0,
    "canthoplasty": 15.0,
    "buccal_fat_removal": 30.0,
    "dimpleplasty": 10.0,
    "genioplasty": 30.0,
    "malarplasty": 35.0,
    "lip_lift": 15.0,
    "lip_augmentation": 15.0,
    "forehead_reduction": 30.0,
    "submental_liposuction": 30.0,
    "otoplasty": 35.0,
}


def gaussian_rbf_deform(
    landmarks: np.ndarray,
    handle: DeformationHandle,
) -> np.ndarray:
    """Gaussian RBF deform: delta * exp(-dist^2 / 2r^2). Returns copy."""
    result = landmarks.copy()
    center = landmarks[handle.landmark_index, :2]
    displacement = handle.displacement[:2]

    distances_sq = np.sum((landmarks[:, :2] - center) ** 2, axis=1)
    weights = np.exp(-distances_sq / (2.0 * handle.influence_radius**2))

    result[:, 0] += displacement[0] * weights
    result[:, 1] += displacement[1] * weights

    if landmarks.shape[1] > 2 and len(handle.displacement) > 2:
        result[:, 2] += handle.displacement[2] * weights

    return result


def gaussian_rbf_deform_batch(
    landmarks: np.ndarray,
    handles: list[DeformationHandle],
) -> np.ndarray:
    """Batch RBF deformation: apply all handles in one vectorized pass.

    Computes all N handle weights simultaneously via (N, 478) matrix ops
    instead of looping. Assumes handles are independent (additive
    displacements from original positions).
    """
    if not handles:
        return landmarks.copy()

    # Build (N, 2) arrays of centers, displacements, and radii
    centers = np.array(
        [landmarks[h.landmark_index, :2] for h in handles], dtype=np.float32
    )  # (N, 2)
    displacements = np.array([h.displacement[:2] for h in handles], dtype=np.float32)  # (N, 2)
    radii_sq = np.array([2.0 * h.influence_radius**2 for h in handles], dtype=np.float32)  # (N,)

    # Compute pairwise squared distances: (N, 478)
    # landmarks[:, :2] is (478, 2), centers is (N, 2)
    diff = landmarks[np.newaxis, :, :2] - centers[:, np.newaxis, :]  # (N, 478, 2)
    dist_sq = np.sum(diff**2, axis=2)  # (N, 478)

    # RBF weights: (N, 478)
    weights = np.exp(-dist_sq / radii_sq[:, np.newaxis])

    # Weighted displacements summed across all handles: (478, 2)
    total_dx = np.sum(displacements[:, 0:1] * weights, axis=0)  # (478,)
    total_dy = np.sum(displacements[:, 1:2] * weights, axis=0)  # (478,)

    result = landmarks.copy()
    result[:, 0] += total_dx
    result[:, 1] += total_dy

    # Handle Z displacement if present
    has_z = landmarks.shape[1] > 2 and any(len(h.displacement) > 2 for h in handles)
    if has_z:
        dz = np.array(
            [h.displacement[2] if len(h.displacement) > 2 else 0.0 for h in handles],
            dtype=np.float32,
        )
        total_dz = np.sum(dz[:, np.newaxis] * weights, axis=0)
        result[:, 2] += total_dz

    return result


def apply_procedure_preset(
    face: FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
    image_size: int = 512,
    clinical_flags: ClinicalFlags | None = None,
    displacement_model_path: str | None = None,
    noise_scale: float = 0.0,
    regional_intensity: RegionalIntensity | None = None,
) -> FaceLandmarks:
    """Apply a surgical procedure preset to landmarks.

    Args:
        face: Input face landmarks.
        procedure: Procedure name (rhinoplasty, blepharoplasty, etc.).
        intensity: Relative intensity 0-100 (mild=33, moderate=66, aggressive=100).
        image_size: Reference image size for displacement scaling.
        clinical_flags: Optional clinical condition flags.
        displacement_model_path: Path to a fitted DisplacementModel (.npz).
            When provided, uses data-driven displacements from real surgery pairs
            instead of hand-tuned RBF vectors.
        noise_scale: Variation noise scale for data-driven mode (0=deterministic).

    Returns:
        New FaceLandmarks with manipulated landmarks.
    """
    if procedure not in PROCEDURE_LANDMARKS:
        raise ValueError(f"Unknown procedure: {procedure}. Choose from {list(PROCEDURE_LANDMARKS)}")

    landmarks = face.landmarks.copy()
    scale = intensity / 100.0

    # Data-driven displacement mode (fall back to RBF if procedure not in model)
    # Map UI intensity (0-100) to displacement model intensity (0-2):
    # 50 -> 1.0x mean displacement, matching inference.py scaling
    if displacement_model_path is not None:
        dm_scale = intensity / 50.0
        try:
            return _apply_data_driven(
                face,
                procedure,
                dm_scale,
                displacement_model_path,
                noise_scale,
            )
        except KeyError:
            logger.warning(
                "Procedure '%s' not in displacement model, falling back to RBF preset",
                procedure,
            )
            # Fall through to RBF-based preset below

    indices = PROCEDURE_LANDMARKS[procedure]
    radius = PROCEDURE_RADIUS[procedure]

    # Ehlers-Danlos: wider influence radii for hypermobile tissue
    if clinical_flags and clinical_flags.ehlers_danlos:
        radius *= 1.5

    # Scale radius based on geometric mean of actual image dimensions.
    # Radii are calibrated for 512x512; using geometric mean handles
    # non-square inputs without asymmetric deformation.
    geo_mean = math.sqrt(face.image_width * face.image_height)
    pixel_scale = geo_mean / 512.0
    handles = _get_procedure_handles(
        procedure, indices, scale, radius * pixel_scale, regional_intensity
    )

    # Bell's palsy: remove handles on the affected (paralyzed) side
    if clinical_flags and clinical_flags.bells_palsy:
        from landmarkdiff.clinical import get_bells_palsy_side_indices

        affected = get_bells_palsy_side_indices(clinical_flags.bells_palsy_side)
        affected_indices = set()
        for region_indices in affected.values():
            affected_indices.update(region_indices)
        handles = [h for h in handles if h.landmark_index not in affected_indices]

    # Convert to pixel space for deformation
    pixel_landmarks = landmarks.copy()
    pixel_landmarks[:, 0] *= face.image_width
    pixel_landmarks[:, 1] *= face.image_height

    # Scale each handle's displacement by the confidence of its anchor
    # landmark. Low-confidence landmarks (e.g., near face boundary on
    # profile views) are deformed less aggressively.
    conf = face.landmark_confidence
    scaled_handles = []
    for handle in handles:
        c = float(conf[handle.landmark_index])
        if c < 1.0:
            scaled_handles.append(
                DeformationHandle(
                    landmark_index=handle.landmark_index,
                    displacement=handle.displacement * c,
                    influence_radius=handle.influence_radius,
                )
            )
        else:
            scaled_handles.append(handle)

    pixel_landmarks = gaussian_rbf_deform_batch(pixel_landmarks, scaled_handles)

    # Convert back to normalized
    result = pixel_landmarks.copy()
    result[:, 0] /= face.image_width
    result[:, 1] /= face.image_height

    return FaceLandmarks(
        landmarks=result,
        image_width=face.image_width,
        image_height=face.image_height,
        confidence=face.confidence,
    )


def _apply_data_driven(
    face: FaceLandmarks,
    procedure: str,
    scale: float,
    model_path: str,
    noise_scale: float = 0.0,
) -> FaceLandmarks:
    """Apply data-driven displacements from a fitted DisplacementModel.

    The model provides mean displacement vectors learned from real surgery pairs,
    applied directly to all 478 landmarks (not just procedure-specific subset).
    """
    from landmarkdiff.displacement_model import DisplacementModel

    model = DisplacementModel.load(model_path)
    field = model.get_displacement_field(
        procedure=procedure,
        intensity=scale,
        noise_scale=noise_scale,
    )

    # field is (478, 2) in normalized coordinates
    landmarks = face.landmarks.copy()
    n_lm = min(landmarks.shape[0], field.shape[0])
    landmarks[:n_lm, :2] += field[:n_lm]

    # Clamp to [0, 1]
    landmarks = np.clip(landmarks, 0.0, 1.0)

    return FaceLandmarks(
        landmarks=landmarks,
        image_width=face.image_width,
        image_height=face.image_height,
        confidence=face.confidence,
    )


def _get_procedure_handles(
    procedure: str,
    indices: list[int],
    scale: float,
    radius: float,
    regional: RegionalIntensity | None = None,
) -> list[DeformationHandle]:
    """Build deformation handles per procedure. 2D pixel displacements, calibrated at 512x512.

    When ``regional`` is provided, per-sub-region multipliers are applied
    to the displacement magnitudes, allowing fine-grained control over
    which anatomical areas receive more or less deformation.
    """
    handles = []

    def _r(region_name: str) -> float:
        """Get regional multiplier (1.0 if no regional overrides)."""
        if regional is None:
            return 1.0
        return getattr(regional, region_name, 1.0)

    if procedure == "rhinoplasty":
        # --- Alar base narrowing: move nostrils inward (toward midline) ---
        alar_m = _r("alar")
        # left nostril -> move RIGHT (+X)
        left_alar = [240, 236, 141, 363, 370]
        for idx in left_alar:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.5 * scale * alar_m, 0.0]),
                        influence_radius=radius * 0.6,
                    )
                )
        # right nostril -> move LEFT (-X)
        right_alar = [460, 456, 274, 275, 278, 279]
        for idx in right_alar:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-2.5 * scale * alar_m, 0.0]),
                        influence_radius=radius * 0.6,
                    )
                )

        # --- Tip refinement: subtle upward rotation + narrowing ---
        tip_m = _r("tip")
        tip_indices = [1, 2, 94, 19]
        for idx in tip_indices:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale * tip_m]),
                        influence_radius=radius * 0.5,
                    )
                )

        # --- Dorsum narrowing: bilateral squeeze of nasal bridge ---
        bridge_m = _r("bridge")
        dorsum_left = [195, 197, 236]
        for idx in dorsum_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.5 * scale * bridge_m, 0.0]),
                        influence_radius=radius * 0.5,
                    )
                )
        dorsum_right = [326, 327, 456]
        for idx in dorsum_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.5 * scale * bridge_m, 0.0]),
                        influence_radius=radius * 0.5,
                    )
                )

    elif procedure == "blepharoplasty":
        # --- Upper lid elevation (primary effect) ---
        ul_m = _r("upper_lid")
        upper_lid_left = [159, 160, 161]  # central upper lid
        upper_lid_right = [386, 385, 384]
        for idx in upper_lid_left + upper_lid_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale * ul_m]),
                        influence_radius=radius,
                    )
                )
        # --- Medial/lateral lid corners: less displacement (tapered) ---
        co_m = _r("corners")
        corner_left = [158, 157, 133, 33]
        corner_right = [387, 388, 362, 263]
        for idx in corner_left + corner_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -0.8 * scale * co_m]),
                        influence_radius=radius * 0.7,
                    )
                )
        # --- Subtle lower lid tightening ---
        ll_m = _r("lower_lid")
        lower_lid_left = [145, 153, 154]
        lower_lid_right = [374, 380, 381]
        for idx in lower_lid_left + lower_lid_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 0.5 * scale * ll_m]),
                        influence_radius=radius * 0.5,
                    )
                )

    elif procedure == "rhytidectomy":
        # Different displacement vectors by anatomical sub-region.
        # Jowl area: strongest lift (upward + toward ear)
        jw_m = _r("jowl")
        jowl_left = [132, 136, 172, 58, 150, 176]
        for idx in jowl_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-2.5 * scale * jw_m, -3.0 * scale * jw_m]),
                        influence_radius=radius,
                    )
                )
        jowl_right = [361, 365, 397, 288, 379, 400]
        for idx in jowl_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.5 * scale * jw_m, -3.0 * scale * jw_m]),
                        influence_radius=radius,
                    )
                )
        # Chin/submental: upward only (no lateral)
        ch_m = _r("chin")
        chin = [152, 148, 377, 378]
        for idx in chin:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale * ch_m]),
                        influence_radius=radius * 0.8,
                    )
                )
        # Temple/upper face: very mild lift
        tm_m = _r("temple")
        temple_left = [10, 21, 54, 67, 103, 109, 162, 127]
        temple_right = [284, 297, 332, 338, 323, 356, 389, 454]
        for idx in temple_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-0.5 * scale * tm_m, -1.0 * scale * tm_m]),
                        influence_radius=radius * 0.6,
                    )
                )
        for idx in temple_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.5 * scale * tm_m, -1.0 * scale * tm_m]),
                        influence_radius=radius * 0.6,
                    )
                )

    elif procedure == "orthognathic":
        # --- Mandible repositioning: move jaw up and forward (visible as upward in 2D) ---
        lower_jaw = [17, 18, 200, 201, 202, 204, 208, 211, 212, 214]
        for idx in lower_jaw:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -3.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # --- Chin projection: move chin point forward/upward ---
        chin_pts = [175, 170, 169, 167, 396]
        for idx in chin_pts:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )
        # --- Lateral jaw: bilateral symmetric inward pull for narrowing ---
        jaw_left = [57, 61, 78, 91, 95, 146, 181]
        for idx in jaw_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.5 * scale, -1.0 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )
        jaw_right = [291, 311, 312, 321, 324, 325, 375, 405]
        for idx in jaw_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.5 * scale, -1.0 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )

    elif procedure == "brow_lift":
        # --- Brow elevation ---
        brow_left = [70, 63, 105, 66, 107]
        brow_right = [300, 293, 334, 296, 336]

        # Lateral brow often lifted more than medial
        left_weights = [0.7, 0.8, 0.9, 1.0, 1.1]
        for i, idx in enumerate(brow_left):
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -4.0 * left_weights[i] * scale]),
                        influence_radius=radius,
                    )
                )

        right_weights = [0.7, 0.8, 0.9, 1.0, 1.1]
        for i, idx in enumerate(brow_right):
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -4.0 * right_weights[i] * scale]),
                        influence_radius=radius,
                    )
                )

        # --- Forehead smoothing / subtle lift ---
        forehead = [9, 8, 10, 109, 67, 103, 338, 297, 332]
        for idx in forehead:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.5 * scale]),
                        influence_radius=radius * 1.2,
                    )
                )
    elif procedure == "mentoplasty":
        # --- Chin tip advancement: move chin forward (upward in 2D) ---
        chin_tip = [152, 175]
        for idx in chin_tip:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -4.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # --- Lower chin contour: follow tip with softer displacement ---
        lower_contour = [148, 149, 150, 176, 377]
        for idx in lower_contour:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.5 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )
        # --- Jaw angles: minimal upward pull for natural transition ---
        jaw_angles = [171, 396]
        for idx in jaw_angles:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.0 * scale]),
                        influence_radius=radius * 0.6,
                    )
                )

    elif procedure == "alarplasty":
        # Alar base reduction: narrow nostrils without modifying bridge or tip.
        # Left alar -> move right (toward midline)
        left_alar = [240, 236, 141, 363, 370]
        for idx in left_alar:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([3.0 * scale, 0.0]),
                        influence_radius=radius,
                    )
                )
        # Right alar -> move left (toward midline)
        right_alar = [460, 456, 274, 275, 278, 279]
        for idx in right_alar:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-3.0 * scale, 0.0]),
                        influence_radius=radius,
                    )
                )
        # Nose base anchor: very mild inward pull
        base = [94, 360]
        for idx in base:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 0.3 * scale]),
                        influence_radius=radius * 0.5,
                    )
                )

    elif procedure == "canthoplasty":
        # Lateral canthoplasty: lift outer eye corners for almond shape.
        # Left lateral canthus -> move up and slightly outward
        left_lateral = [33, 246, 161, 160]
        for idx in left_lateral:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.0 * scale, -2.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Right lateral canthus -> move up and slightly outward
        right_lateral = [263, 466, 384, 385]
        for idx in right_lateral:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.0 * scale, -2.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Medial corners: subtle elongation pull outward
        left_medial = [133, 173, 155]
        for idx in left_medial:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-0.5 * scale, -0.5 * scale]),
                        influence_radius=radius * 0.6,
                    )
                )
        right_medial = [362, 398, 390]
        for idx in right_medial:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.5 * scale, -0.5 * scale]),
                        influence_radius=radius * 0.6,
                    )
                )

    elif procedure == "buccal_fat_removal":
        # Move lower cheek landmarks medially to simulate reduced buccal volume.
        # Left cheek -> move right (inward)
        left_cheek = [116, 117, 118, 119, 120, 121, 205, 206, 207, 213, 187]
        for idx in left_cheek:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.0 * scale, 0.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Right cheek -> move left (inward)
        right_cheek = [345, 346, 347, 348, 349, 350, 425, 426, 427, 435, 411]
        for idx in right_cheek:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-2.0 * scale, 0.5 * scale]),
                        influence_radius=radius,
                    )
                )

    elif procedure == "dimpleplasty":
        # Localized concavity at typical dimple positions on each cheek.
        # Left dimple: slight inward pull
        left_dimple = [205, 206]
        for idx in left_dimple:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.5 * scale, 0.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Right dimple: slight inward pull
        right_dimple = [425, 426]
        for idx in right_dimple:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.5 * scale, 0.5 * scale]),
                        influence_radius=radius,
                    )
                )

    elif procedure == "genioplasty":
        # Osseous chin repositioning: horizontal sliding advancement + vertical change.
        # Differs from mentoplasty (implant) in affecting wider mandibular region.
        # Primary chin segment: forward advancement (upward in 2D frontal view)
        chin_core = [152, 175, 199, 200]
        for idx in chin_core:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -3.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Lower lip margin: follow chin with softer displacement
        lip_margin = [17, 314, 405]
        for idx in lip_margin:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.5 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )
        # Jawline transition: blend into natural contour
        jaw_blend = [149, 150, 176, 377, 378, 396]
        for idx in jaw_blend:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.0 * scale]),
                        influence_radius=radius * 0.6,
                    )
                )

    elif procedure == "malarplasty":
        # Cheekbone augmentation: outward displacement of zygomatic region.
        # Left cheek -> move outward (left)
        left_zygoma = [34, 93, 116, 117, 123, 132, 187, 205, 213, 234]
        for idx in left_zygoma:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-2.0 * scale, -0.5 * scale]),
                        influence_radius=radius,
                    )
                )
        # Right cheek -> move outward (right)
        right_zygoma = [264, 323, 345, 346, 352, 361, 411, 425, 435, 454]
        for idx in right_zygoma:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.0 * scale, -0.5 * scale]),
                        influence_radius=radius,
                    )
                )

    elif procedure == "lip_lift":
        # Subnasal bullhorn lip lift: shorten philtrum, increase upper lip show.
        # Upper lip border moves upward toward nose base
        upper_lip_central = [0, 11, 12, 13, 14]
        for idx in upper_lip_central:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -3.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # Lateral upper lip: tapered lift
        upper_lip_left = [37, 39, 40, 61, 72, 73]
        for idx in upper_lip_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.5 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )
        upper_lip_right = [267, 269, 270, 291, 302, 303]
        for idx in upper_lip_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -1.5 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )

    elif procedure == "lip_augmentation":
        # Volumetric lip augmentation: expand both lips outward from midline.
        # Upper lip: move upward (away from mouth center)
        upper_lip = [0, 11, 12, 13, 37, 39, 40, 61, 72, 73, 267, 269, 270, 291, 302, 303]
        for idx in upper_lip:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # Lower lip: move downward
        lower_lip = [14, 80, 81, 82, 178, 181, 310, 311, 312, 402, 405]
        for idx in lower_lip:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 2.0 * scale]),
                        influence_radius=radius,
                    )
                )

    elif procedure == "forehead_reduction":
        # Hairline lowering: move forehead/hairline landmarks downward.
        # Central hairline: strongest downward displacement
        hairline_central = [10, 8, 9]
        for idx in hairline_central:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 3.0 * scale]),
                        influence_radius=radius * 1.2,
                    )
                )
        # Left lateral forehead
        forehead_left = [21, 54, 67, 68, 69, 103, 104, 108, 109, 251]
        for idx in forehead_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 2.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # Right lateral forehead
        forehead_right = [284, 297, 298, 299, 332, 333, 337, 338]
        for idx in forehead_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 2.0 * scale]),
                        influence_radius=radius,
                    )
                )

    elif procedure == "submental_liposuction":
        # Double chin reduction: lift submental tissue upward and inward.
        # Central submental: strong upward pull
        submental_core = [152, 175, 199, 200]
        for idx in submental_core:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -3.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # Lateral submental: upward + inward pull
        submental_left = [149, 150, 169, 170, 171, 176]
        for idx in submental_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.0 * scale, -2.0 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )
        submental_right = [377, 378, 396, 400]
        for idx in submental_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.0 * scale, -2.0 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )

    elif procedure == "otoplasty":
        # Ear pinning: move ear-adjacent landmarks medially (toward face center).
        # Left ear region -> move right (toward midline)
        left_ear = [234, 127, 93, 132, 162, 21]
        # Graduated displacement: preauricular strongest, temple/jaw softer
        left_weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.4]
        for idx, w in zip(left_ear, left_weights):
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([3.0 * scale * w, 0.0]),
                        influence_radius=radius,
                    )
                )
        # Right ear region -> move left (toward midline)
        right_ear = [454, 356, 323, 361, 389, 251]
        right_weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.4]
        for idx, w in zip(right_ear, right_weights):
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-3.0 * scale * w, 0.0]),
                        influence_radius=radius,
                    )
                )

    return handles
