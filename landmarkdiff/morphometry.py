"""Nasal morphometry and facial symmetry evaluation.

Geometric evaluation metrics derived from Varghaei et al. (2025),
adapted for evaluating surgical prediction outputs.

Computes five nasal ratios plus bilateral facial symmetry from
MediaPipe 478-point landmarks, enabling interpretable clinical
quality assessment beyond perceptual metrics (LPIPS, FID).

Usage::

    from landmarkdiff.morphometry import NasalMorphometry, FacialSymmetry

    morph = NasalMorphometry()
    ratios = morph.compute(landmarks_478)

    sym = FacialSymmetry()
    score = sym.compute(landmarks_478)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark indices (478-point mesh)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
NOSE_TIP = 1
LEFT_NOSTRIL = 98
RIGHT_NOSTRIL = 327
LEFT_INNER_EYE = 133
RIGHT_INNER_EYE = 362
LEFT_OUTER_EYE = 33
RIGHT_OUTER_EYE = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10
GLABELLA = 168


@dataclass
class NasalRatios:
    """Five nasal morphometric ratios from Varghaei et al. (2025).

    Attributes:
        alar_intercanthal: Alar width / intercanthal distance.
            Ideal ~1.0 (nose width equals eye spacing).
        alar_face_width: Alar width / face width.
            Ideal ~0.20 (nose is 1/5 of face width).
        nose_length_face_height: Nose length / face height.
            Proportional measure of nose vertical extent.
        tip_midline_deviation: Horizontal offset of nose tip from
            facial midline, normalized by face width. Lower is better.
        nostril_vertical_asymmetry: Vertical height difference between
            nostrils, normalized by face height. Lower is better.
    """

    alar_intercanthal: float = 0.0
    alar_face_width: float = 0.0
    nose_length_face_height: float = 0.0
    tip_midline_deviation: float = 0.0
    nostril_vertical_asymmetry: float = 0.0

    def improvement_score(self, reference: NasalRatios) -> dict[str, bool]:
        """Check which ratios improved relative to reference (pre-op).

        A ratio 'improved' if the prediction moved it closer to the
        anthropometric ideal compared to the reference.
        """
        ideals = {
            "alar_intercanthal": 1.0,
            "alar_face_width": 0.20,
        }
        results = {}
        for name, ideal in ideals.items():
            pred_val = getattr(self, name)
            ref_val = getattr(reference, name)
            results[name] = abs(pred_val - ideal) < abs(ref_val - ideal)

        # For deviation/asymmetry, lower is always better
        results["tip_midline_deviation"] = (
            self.tip_midline_deviation < reference.tip_midline_deviation
        )
        results["nostril_vertical_asymmetry"] = (
            self.nostril_vertical_asymmetry < reference.nostril_vertical_asymmetry
        )
        return results

    def to_dict(self) -> dict[str, float]:
        return {
            "alar_intercanthal": self.alar_intercanthal,
            "alar_face_width": self.alar_face_width,
            "nose_length_face_height": self.nose_length_face_height,
            "tip_midline_deviation": self.tip_midline_deviation,
            "nostril_vertical_asymmetry": self.nostril_vertical_asymmetry,
        }


class NasalMorphometry:
    """Compute nasal morphometric ratios from MediaPipe landmarks.

    Five geometric features following Varghaei et al. (2025):
    1. Alar width / intercanthal distance (ideal ~1.0)
    2. Alar width / face width (ideal ~0.20)
    3. Nose length / face height
    4. Tip midline deviation (normalized)
    5. Nostril vertical asymmetry (normalized)
    """

    def compute(self, landmarks: np.ndarray) -> NasalRatios:
        """Compute all five nasal ratios.

        Args:
            landmarks: (N, 2) or (N, 3) array of MediaPipe landmarks.
                Must have at least 478 points. Uses only x, y.

        Returns:
            NasalRatios dataclass with computed values.
        """
        pts = landmarks[:, :2]  # use only x, y

        # Key points
        nose_tip = pts[NOSE_TIP]
        left_nostril = pts[LEFT_NOSTRIL]
        right_nostril = pts[RIGHT_NOSTRIL]
        left_inner_eye = pts[LEFT_INNER_EYE]
        right_inner_eye = pts[RIGHT_INNER_EYE]
        left_cheek = pts[LEFT_CHEEK]
        right_cheek = pts[RIGHT_CHEEK]
        forehead = pts[FOREHEAD]
        chin = pts[CHIN]
        glabella = pts[GLABELLA]

        # Distances (cast to float for mypy compatibility)
        alar_width: float = float(np.linalg.norm(left_nostril - right_nostril))
        intercanthal: float = max(float(np.linalg.norm(left_inner_eye - right_inner_eye)), 1e-6)
        face_width: float = max(float(np.linalg.norm(left_cheek - right_cheek)), 1e-6)
        face_height: float = max(float(np.linalg.norm(forehead - chin)), 1e-6)
        nose_length: float = float(np.linalg.norm(glabella - nose_tip))

        # Facial midline (between outer eye corners)
        midline_x = (pts[LEFT_OUTER_EYE][0] + pts[RIGHT_OUTER_EYE][0]) / 2

        # Ratios
        alar_intercanthal = float(alar_width / intercanthal)
        alar_face = float(alar_width / face_width)
        nose_face = float(nose_length / face_height)
        tip_deviation = float(abs(nose_tip[0] - midline_x) / face_width)
        nostril_asymmetry = float(abs(left_nostril[1] - right_nostril[1]) / face_height)

        return NasalRatios(
            alar_intercanthal=alar_intercanthal,
            alar_face_width=alar_face,
            nose_length_face_height=nose_face,
            tip_midline_deviation=tip_deviation,
            nostril_vertical_asymmetry=nostril_asymmetry,
        )

    def compute_from_image(self, image: np.ndarray) -> NasalRatios | None:
        """Extract landmarks from image and compute ratios.

        Args:
            image: BGR uint8 image (H, W, 3).

        Returns:
            NasalRatios or None if landmark detection fails.
        """
        try:
            import mediapipe as mp
        except ImportError:
            logger.warning("mediapipe required for landmark extraction")
            return None

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return None

            h, w = image.shape[:2]
            face = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])
            return self.compute(landmarks)


class FacialSymmetry:
    """Bilateral facial symmetry scoring.

    Measures deviation from perfect bilateral symmetry by reflecting
    left-side landmarks across the facial midline and computing
    distances to nearest right-side counterparts.

    Lower scores indicate greater symmetry.
    """

    def compute(
        self,
        landmarks: np.ndarray,
        left_eye_idx: int = LEFT_OUTER_EYE,
        right_eye_idx: int = RIGHT_OUTER_EYE,
    ) -> float:
        """Compute bilateral symmetry error.

        Args:
            landmarks: (N, 2) or (N, 3) array. Uses only x, y.
            left_eye_idx: Landmark index for left outer eye corner.
            right_eye_idx: Landmark index for right outer eye corner.

        Returns:
            Mean symmetry error (lower = more symmetric).
            Normalized by inter-ocular distance.
        """
        pts = landmarks[:, :2].copy()

        # Midline from eye corners
        midline_x = (pts[left_eye_idx][0] + pts[right_eye_idx][0]) / 2
        iod = abs(pts[left_eye_idx][0] - pts[right_eye_idx][0])
        if iod < 1e-6:
            return 0.0

        # Partition into left and right
        left_mask = pts[:, 0] < midline_x
        right_mask = pts[:, 0] > midline_x

        left_pts = pts[left_mask]
        right_pts = pts[right_mask]

        if len(left_pts) == 0 or len(right_pts) == 0:
            return 0.0

        # Reflect left across midline
        reflected = left_pts.copy()
        reflected[:, 0] = 2 * midline_x - reflected[:, 0]

        # KDTree nearest-neighbor matching
        try:
            from scipy.spatial import KDTree

            tree = KDTree(right_pts)
            distances, _ = tree.query(reflected)
            return float(np.mean(distances) / iod)
        except ImportError:
            # Fallback: brute force
            total = 0.0
            for pt in reflected:
                dists = np.linalg.norm(right_pts - pt, axis=1)
                total += np.min(dists)
            return float(total / (len(reflected) * iod))

    def compute_from_image(self, image: np.ndarray) -> float | None:
        """Extract landmarks from image and compute symmetry.

        Args:
            image: BGR uint8 image (H, W, 3).

        Returns:
            Symmetry error or None if detection fails.
        """
        try:
            import mediapipe as mp
        except ImportError:
            logger.warning("mediapipe required for landmark extraction")
            return None

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return None

            h, w = image.shape[:2]
            face = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])
            return self.compute(landmarks)


def compare_morphometry(
    pred_image: np.ndarray,
    input_image: np.ndarray,
    procedure: str = "rhinoplasty",
) -> dict:
    """Compare morphometric quality between prediction and input.

    Computes nasal ratios and symmetry for both images and reports
    which metrics improved. Useful for evaluating whether the predicted
    surgical output shows clinically meaningful improvement.

    Args:
        pred_image: Predicted output (BGR uint8).
        input_image: Original input (BGR uint8).
        procedure: Procedure type (affects which metrics are relevant).

    Returns:
        Dict with 'input_ratios', 'pred_ratios', 'improvements',
        'input_symmetry', 'pred_symmetry', 'symmetry_improved'.
    """
    morph = NasalMorphometry()
    sym = FacialSymmetry()

    input_ratios = morph.compute_from_image(input_image)
    pred_ratios = morph.compute_from_image(pred_image)
    input_sym = sym.compute_from_image(input_image)
    pred_sym = sym.compute_from_image(pred_image)

    result: dict = {
        "procedure": procedure,
        "input_ratios": input_ratios.to_dict() if input_ratios else None,
        "pred_ratios": pred_ratios.to_dict() if pred_ratios else None,
        "input_symmetry": input_sym,
        "pred_symmetry": pred_sym,
        "symmetry_improved": (
            pred_sym < input_sym if pred_sym is not None and input_sym is not None else None
        ),
    }

    if input_ratios and pred_ratios:
        result["improvements"] = pred_ratios.improvement_score(input_ratios)
    else:
        result["improvements"] = None

    return result
