"""Auto-detect optimal procedure and intensity from a reference image.

Given a before and after (reference) face, compute which procedure
and intensity best reproduce the observed landmark displacements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS, PROCEDURE_RADIUS

logger = logging.getLogger(__name__)


@dataclass
class IntensityEstimate:
    """Result of auto-intensity estimation."""

    procedure: str
    intensity: float  # estimated intensity 0-100
    score: float  # match score (higher = better match)

    def summary(self) -> str:
        return (
            f"Estimated: {self.procedure} at intensity {self.intensity:.1f} "
            f"(score={self.score:.4f})"
        )


def estimate_intensity(
    face_before: FaceLandmarks,
    face_after: FaceLandmarks,
    procedures: list[str] | None = None,
) -> IntensityEstimate:
    """Estimate procedure and intensity from before/after landmark pairs.

    Computes the mean displacement magnitude for each procedure's landmark
    subset, then estimates intensity by comparing to the expected displacement
    at intensity=100. The procedure with the highest relative displacement
    in its region is selected.

    Args:
        face_before: Landmarks from the original (before) image.
        face_after: Landmarks from the reference (after) image.
        procedures: Optional subset of procedures to consider.
            If None, all procedures are evaluated.

    Returns:
        IntensityEstimate with best-matching procedure and intensity.
    """
    before_px = face_before.pixel_coords[:, :2]
    after_px = face_after.pixel_coords[:, :2]
    displacements = after_px - before_px

    candidates = procedures or list(PROCEDURE_LANDMARKS.keys())
    best = IntensityEstimate(procedure=candidates[0], intensity=50.0, score=0.0)

    for proc in candidates:
        indices = PROCEDURE_LANDMARKS.get(proc)
        if not indices:
            continue

        # Mean displacement magnitude in this procedure's region
        proc_disp = displacements[indices]
        mean_mag = float(np.mean(np.sqrt(np.sum(proc_disp**2, axis=1))))

        # Compare to the procedure's influence radius as a reference scale.
        # At intensity=100, displacement magnitudes are on the order of
        # 2-4x the base scale values (calibrated for 512px).
        radius = PROCEDURE_RADIUS.get(proc, 30.0)
        # Normalize: radius * 0.08 is a rough "intensity=100" displacement per px
        # (from typical scale * 2.5 factors in handle generation)
        reference_disp = radius * 0.1
        estimated_intensity = (mean_mag / reference_disp) * 50.0 if reference_disp > 0 else 50.0
        estimated_intensity = min(100.0, max(0.0, estimated_intensity))

        # Score: ratio of procedure-region displacement to global displacement
        global_mag = float(np.mean(np.sqrt(np.sum(displacements**2, axis=1))))
        score = mean_mag / (global_mag + 1e-8)

        if score > best.score:
            best = IntensityEstimate(
                procedure=proc,
                intensity=estimated_intensity,
                score=score,
            )

    return best


def estimate_all_procedures(
    face_before: FaceLandmarks,
    face_after: FaceLandmarks,
) -> list[IntensityEstimate]:
    """Estimate intensity for all procedures, sorted by match score.

    Args:
        face_before: Landmarks from the original image.
        face_after: Landmarks from the reference image.

    Returns:
        List of IntensityEstimate sorted by descending score.
    """
    results = []
    before_px = face_before.pixel_coords[:, :2]
    after_px = face_after.pixel_coords[:, :2]
    displacements = after_px - before_px
    global_mag = float(np.mean(np.sqrt(np.sum(displacements**2, axis=1))))

    for proc, indices in PROCEDURE_LANDMARKS.items():
        proc_disp = displacements[indices]
        mean_mag = float(np.mean(np.sqrt(np.sum(proc_disp**2, axis=1))))

        radius = PROCEDURE_RADIUS.get(proc, 30.0)
        reference_disp = radius * 0.1
        estimated_intensity = (mean_mag / reference_disp) * 50.0 if reference_disp > 0 else 50.0
        estimated_intensity = min(100.0, max(0.0, estimated_intensity))

        score = mean_mag / (global_mag + 1e-8)

        results.append(
            IntensityEstimate(
                procedure=proc,
                intensity=estimated_intensity,
                score=score,
            )
        )

    return sorted(results, key=lambda x: x.score, reverse=True)
