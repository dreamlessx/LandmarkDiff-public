"""Reverse prediction: infer procedure and intensity from a desired outcome.

Given a before face and a desired after face, determine which procedure(s)
and intensities would produce the observed landmark displacement pattern.
This is the inverse of the forward prediction pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    apply_procedure_preset,
)

logger = logging.getLogger(__name__)


@dataclass
class ReversePrediction:
    """Result of reverse prediction analysis."""

    procedure: str
    intensity: float  # estimated intensity 0-100
    confidence: float  # match confidence 0-1
    displacement_error: float  # mean per-landmark error in pixels

    def summary(self) -> str:
        return (
            f"{self.procedure}: intensity={self.intensity:.1f}, "
            f"confidence={self.confidence:.3f}, error={self.displacement_error:.2f}px"
        )


@dataclass
class MultiReversePrediction:
    """Result of multi-procedure reverse prediction."""

    predictions: list[ReversePrediction] = field(default_factory=list)
    residual_error: float = 0.0  # unexplained displacement after all procedures

    @property
    def procedures(self) -> list[str]:
        return [p.procedure for p in self.predictions]

    @property
    def intensities(self) -> dict[str, float]:
        return {p.procedure: p.intensity for p in self.predictions}

    def summary(self) -> str:
        lines = [p.summary() for p in self.predictions]
        lines.append(f"Residual error: {self.residual_error:.2f}px")
        return "\n".join(lines)


def reverse_predict(
    face_before: FaceLandmarks,
    face_after: FaceLandmarks,
    procedures: list[str] | None = None,
    intensity_steps: int = 20,
) -> ReversePrediction:
    """Find the single best procedure and intensity to match the desired outcome.

    Sweeps intensity values for each candidate procedure, computes forward
    predictions, and selects the procedure + intensity combination that
    minimizes landmark displacement error relative to the desired outcome.

    Args:
        face_before: Landmarks from the original image.
        face_after: Landmarks from the desired outcome image.
        procedures: Subset of procedures to consider (default: all).
        intensity_steps: Number of intensity levels to test (higher = finer).

    Returns:
        ReversePrediction with the best matching procedure and intensity.
    """
    candidates = procedures or list(PROCEDURE_LANDMARKS.keys())
    target_px = face_after.pixel_coords[:, :2]

    best = ReversePrediction(
        procedure=candidates[0],
        intensity=50.0,
        confidence=0.0,
        displacement_error=float("inf"),
    )

    for proc in candidates:
        indices = PROCEDURE_LANDMARKS.get(proc)
        if not indices:
            continue

        for step in range(intensity_steps + 1):
            intensity = (step / intensity_steps) * 100.0
            predicted = apply_procedure_preset(face_before, proc, intensity)
            pred_px = predicted.pixel_coords[:, :2]

            # Error in the procedure's region only
            proc_error = float(
                np.mean(np.sqrt(np.sum((pred_px[indices] - target_px[indices]) ** 2, axis=1)))
            )

            if proc_error < best.displacement_error:
                # Confidence based on how well procedure region matches vs global
                global_error = float(np.mean(np.sqrt(np.sum((pred_px - target_px) ** 2, axis=1))))
                region_ratio = proc_error / (global_error + 1e-8)
                confidence = max(0.0, 1.0 - region_ratio)

                best = ReversePrediction(
                    procedure=proc,
                    intensity=intensity,
                    confidence=confidence,
                    displacement_error=proc_error,
                )

    return best


def reverse_predict_multi(
    face_before: FaceLandmarks,
    face_after: FaceLandmarks,
    max_procedures: int = 3,
    min_confidence: float = 0.1,
    intensity_steps: int = 20,
) -> MultiReversePrediction:
    """Detect multiple procedures from the desired outcome via greedy decomposition.

    Iteratively finds the best single procedure, subtracts its contribution
    from the target displacement, and repeats until the residual is small
    or max_procedures is reached.

    Args:
        face_before: Landmarks from the original image.
        face_after: Landmarks from the desired outcome image.
        max_procedures: Maximum number of procedures to detect.
        min_confidence: Minimum confidence to include a procedure.
        intensity_steps: Number of intensity levels to test per procedure.

    Returns:
        MultiReversePrediction with the list of detected procedures.
    """
    before_px = face_before.pixel_coords[:, :2]
    target_px = face_after.pixel_coords[:, :2]
    residual = target_px - before_px  # displacement to explain

    result = MultiReversePrediction()
    used_procedures: set[str] = set()

    for _ in range(max_procedures):
        remaining_procs = [p for p in PROCEDURE_LANDMARKS if p not in used_procedures]
        if not remaining_procs:
            break

        # Build a synthetic "target" face from current residual
        synthetic_target = before_px + residual
        synthetic_lm = face_before.landmarks.copy()
        synthetic_lm[:, 0] = synthetic_target[:, 0] / face_before.image_width
        synthetic_lm[:, 1] = synthetic_target[:, 1] / face_before.image_height
        face_target = FaceLandmarks(
            landmarks=synthetic_lm,
            image_width=face_before.image_width,
            image_height=face_before.image_height,
            confidence=face_before.confidence,
        )

        pred = reverse_predict(
            face_before,
            face_target,
            procedures=remaining_procs,
            intensity_steps=intensity_steps,
        )

        if pred.confidence < min_confidence or pred.intensity < 1.0:
            break

        # Subtract this procedure's contribution from the residual
        predicted = apply_procedure_preset(face_before, pred.procedure, pred.intensity)
        pred_px = predicted.pixel_coords[:, :2]
        procedure_displacement = pred_px - before_px
        residual = residual - procedure_displacement

        result.predictions.append(pred)
        used_procedures.add(pred.procedure)

    result.residual_error = float(np.mean(np.sqrt(np.sum(residual**2, axis=1))))
    return result


def invert_deformation(
    face: FaceLandmarks,
    procedure: str,
    intensity: float = 50.0,
) -> FaceLandmarks:
    """Apply a procedure in reverse (negative displacement).

    Computes the deformation at the given intensity and applies it
    in the opposite direction, effectively "undoing" the procedure.

    Args:
        face: Input face landmarks (post-operative).
        procedure: Procedure to invert.
        intensity: Intensity to invert (same scale as forward).

    Returns:
        FaceLandmarks with reversed deformation applied.
    """
    original_px = face.pixel_coords[:, :2]
    forward = apply_procedure_preset(face, procedure, intensity)
    forward_px = forward.pixel_coords[:, :2]

    # Displacement vector from original to forward
    displacement = forward_px - original_px

    # Apply in reverse direction
    inverted_px = original_px - displacement
    inverted_px[:, 0] = np.clip(inverted_px[:, 0], 0, face.image_width - 1)
    inverted_px[:, 1] = np.clip(inverted_px[:, 1], 0, face.image_height - 1)

    # Convert back to normalized coordinates
    inverted_norm = face.landmarks.copy()
    inverted_norm[:, 0] = inverted_px[:, 0] / face.image_width
    inverted_norm[:, 1] = inverted_px[:, 1] / face.image_height

    return FaceLandmarks(
        landmarks=inverted_norm,
        image_width=face.image_width,
        image_height=face.image_height,
        confidence=face.confidence,
    )
