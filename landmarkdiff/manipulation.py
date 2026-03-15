"""Landmark manipulation via Gaussian RBF deformation.

v1/v2 uses relative sliders (0-100 intensity).
mm inputs only in v3+ with FLAME calibrated metric space.
"""

from __future__ import annotations

import logging
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
}
# Default influence radii per procedure (in pixels at 512x512)
PROCEDURE_RADIUS: dict[str, float] = {
    "rhinoplasty": 30.0,
    "blepharoplasty": 15.0,
    "rhytidectomy": 40.0,
    "orthognathic": 35.0,
    "brow_lift": 25.0,
    "mentoplasty": 25.0,
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


def apply_procedure_preset(
    face: FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
    image_size: int = 512,
    clinical_flags: ClinicalFlags | None = None,
    displacement_model_path: str | None = None,
    noise_scale: float = 0.0,
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

    # Procedure-specific displacement vectors (normalized to image_size)
    pixel_scale = image_size / 512.0
    handles = _get_procedure_handles(procedure, indices, scale, radius * pixel_scale)

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

    for handle in handles:
        pixel_landmarks = gaussian_rbf_deform(pixel_landmarks, handle)

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
) -> list[DeformationHandle]:
    """Build deformation handles per procedure. 2D pixel displacements, calibrated at 512x512."""
    handles = []

    if procedure == "rhinoplasty":
        # --- Alar base narrowing: move nostrils inward (toward midline) ---
        # left nostril -> move RIGHT (+X)
        left_alar = [240, 236, 141, 363, 370]
        for idx in left_alar:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.5 * scale, 0.0]),
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
                        displacement=np.array([-2.5 * scale, 0.0]),
                        influence_radius=radius * 0.6,
                    )
                )

        # --- Tip refinement: subtle upward rotation + narrowing ---
        tip_indices = [1, 2, 94, 19]
        for idx in tip_indices:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius * 0.5,
                    )
                )

        # --- Dorsum narrowing: bilateral squeeze of nasal bridge ---
        dorsum_left = [195, 197, 236]
        for idx in dorsum_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([1.5 * scale, 0.0]),
                        influence_radius=radius * 0.5,
                    )
                )
        dorsum_right = [326, 327, 456]
        for idx in dorsum_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-1.5 * scale, 0.0]),
                        influence_radius=radius * 0.5,
                    )
                )

    elif procedure == "blepharoplasty":
        # --- Upper lid elevation (primary effect) ---
        upper_lid_left = [159, 160, 161]  # central upper lid
        upper_lid_right = [386, 385, 384]
        for idx in upper_lid_left + upper_lid_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # --- Medial/lateral lid corners: less displacement (tapered) ---
        corner_left = [158, 157, 133, 33]
        corner_right = [387, 388, 362, 263]
        for idx in corner_left + corner_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -0.8 * scale]),
                        influence_radius=radius * 0.7,
                    )
                )
        # --- Subtle lower lid tightening ---
        lower_lid_left = [145, 153, 154]
        lower_lid_right = [374, 380, 381]
        for idx in lower_lid_left + lower_lid_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, 0.5 * scale]),
                        influence_radius=radius * 0.5,
                    )
                )

    elif procedure == "rhytidectomy":
        # Different displacement vectors by anatomical sub-region.
        # Jowl area: strongest lift (upward + toward ear)
        jowl_left = [132, 136, 172, 58, 150, 176]
        for idx in jowl_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-2.5 * scale, -3.0 * scale]),
                        influence_radius=radius,
                    )
                )
        jowl_right = [361, 365, 397, 288, 379, 400]
        for idx in jowl_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([2.5 * scale, -3.0 * scale]),
                        influence_radius=radius,
                    )
                )
        # Chin/submental: upward only (no lateral)
        chin = [152, 148, 377, 378]
        for idx in chin:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.0, -2.0 * scale]),
                        influence_radius=radius * 0.8,
                    )
                )
        # Temple/upper face: very mild lift
        temple_left = [10, 21, 54, 67, 103, 109, 162, 127]
        temple_right = [284, 297, 332, 338, 323, 356, 389, 454]
        for idx in temple_left:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([-0.5 * scale, -1.0 * scale]),
                        influence_radius=radius * 0.6,
                    )
                )
        for idx in temple_right:
            if idx in indices:
                handles.append(
                    DeformationHandle(
                        landmark_index=idx,
                        displacement=np.array([0.5 * scale, -1.0 * scale]),
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
    return handles
