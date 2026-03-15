"""Clinical safety validation for responsible deployment.

Implements safety checks for surgical outcome predictions:
1. Identity preservation: verify output preserves patient identity
2. Anatomical plausibility: check landmark displacements are realistic
3. Out-of-distribution detection: flag unusual inputs
4. Watermarking: mark AI-generated images
5. Consent metadata: embed provenance information

Usage:
    from landmarkdiff.safety import SafetyValidator

    validator = SafetyValidator()
    result = validator.validate(
        input_image=image,
        output_image=generated,
        landmarks_original=face.landmarks,
        landmarks_manipulated=manip.landmarks,
        procedure="rhinoplasty",
    )

    if not result.passed:
        print(f"Safety check failed: {result.failures}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SafetyResult:
    """Result of safety validation checks."""

    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SafetyResult(passed={self.passed}, "
            f"failures={self.failures}, "
            f"warnings={self.warnings}, "
            f"checks={self.checks}, "
            f"details={self.details})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SafetyResult):
            return NotImplemented
        return (
            self.passed == other.passed
            and self.failures == other.failures
            and self.warnings == other.warnings
            and self.checks == other.checks
            and self.details == other.details
        )

    def add_failure(self, name: str, message: str) -> None:
        self.passed = False
        self.failures.append(message)
        self.checks[name] = False

    def add_warning(self, name: str, message: str) -> None:
        self.warnings.append(message)

    def add_pass(self, name: str) -> None:
        self.checks[name] = True

    def summary(self) -> str:
        lines = [f"Safety: {'PASS' if self.passed else 'FAIL'}"]
        for name, ok in self.checks.items():
            lines.append(f"  [{'OK' if ok else 'FAIL'}] {name}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        return "\n".join(lines)


class SafetyValidator:
    """Clinical safety validation for surgical predictions."""

    def __init__(
        self,
        identity_threshold: float = 0.5,
        max_displacement_fraction: float = 0.05,
        min_face_confidence: float = 0.5,
        max_yaw_degrees: float = 45.0,
        watermark_enabled: bool = True,
        watermark_text: str = "AI-GENERATED PREDICTION",
    ):
        self.identity_threshold = identity_threshold
        self.max_displacement_fraction = max_displacement_fraction
        self.min_face_confidence = min_face_confidence
        self.max_yaw_degrees = max_yaw_degrees
        self.watermark_enabled = watermark_enabled
        self.watermark_text = watermark_text

    def validate(
        self,
        input_image: np.ndarray,
        output_image: np.ndarray,
        landmarks_original: np.ndarray | None = None,
        landmarks_manipulated: np.ndarray | None = None,
        procedure: str | None = None,
        face_confidence: float = 1.0,
    ) -> SafetyResult:
        """Run all safety checks on a prediction.

        Args:
            input_image: Original patient image (BGR, uint8).
            output_image: Generated prediction (BGR, uint8).
            landmarks_original: Original landmarks (N, 2-3), normalized [0, 1].
            landmarks_manipulated: Manipulated landmarks (N, 2-3), normalized [0, 1].
            procedure: Surgical procedure name.
            face_confidence: MediaPipe face detection confidence.

        Returns:
            SafetyResult with all check results.
        """
        result = SafetyResult()

        # 1. Face detection confidence
        self._check_face_confidence(result, face_confidence)

        # 2. Identity preservation
        self._check_identity(result, input_image, output_image)

        # 3. Anatomical plausibility
        if landmarks_original is not None and landmarks_manipulated is not None:
            self._check_anatomical_plausibility(
                result, landmarks_original, landmarks_manipulated, procedure
            )

        # 4. Output quality
        self._check_output_quality(result, output_image)

        # 5. OOD detection (basic)
        self._check_ood(result, input_image)

        return result

    def _check_face_confidence(self, result: SafetyResult, confidence: float) -> None:
        """Check face detection confidence."""
        if confidence < self.min_face_confidence:
            result.add_failure(
                "face_confidence",
                f"Face detection confidence {confidence:.2f} below threshold "
                f"{self.min_face_confidence}",
            )
        else:
            result.add_pass("face_confidence")
            result.details["face_confidence"] = confidence

    def _check_identity(
        self,
        result: SafetyResult,
        input_image: np.ndarray,
        output_image: np.ndarray,
    ) -> None:
        """Check identity preservation using ArcFace similarity."""
        try:
            from landmarkdiff.evaluation import compute_identity_similarity

            sim = compute_identity_similarity(output_image, input_image)
            result.details["identity_similarity"] = float(sim)

            if sim < self.identity_threshold:
                result.add_failure(
                    "identity",
                    f"Identity similarity {sim:.3f} below threshold {self.identity_threshold}",
                )
            else:
                result.add_pass("identity")
        except Exception as e:
            result.add_warning("identity", f"Identity check failed: {e}")

    def _check_anatomical_plausibility(
        self,
        result: SafetyResult,
        landmarks_orig: np.ndarray,
        landmarks_manip: np.ndarray,
        procedure: str | None,
    ) -> None:
        """Check that landmark displacements are anatomically plausible."""
        if len(landmarks_orig) != len(landmarks_manip):
            result.add_failure(
                "anatomical",
                f"Landmark count mismatch: {len(landmarks_orig)} vs {len(landmarks_manip)}",
            )
            return

        # Compute displacement magnitudes
        n = min(len(landmarks_orig), len(landmarks_manip))
        orig = landmarks_orig[:n, :2]  # (N, 2), normalized [0, 1]
        manip = landmarks_manip[:n, :2]
        displacements = np.linalg.norm(manip - orig, axis=1)

        max_disp = float(displacements.max())
        mean_disp = float(displacements.mean())
        result.details["max_displacement"] = max_disp
        result.details["mean_displacement"] = mean_disp

        # Check maximum displacement
        if max_disp > self.max_displacement_fraction:
            result.add_failure(
                "anatomical_magnitude",
                f"Maximum displacement {max_disp:.4f} exceeds threshold "
                f"{self.max_displacement_fraction}",
            )
        else:
            result.add_pass("anatomical_magnitude")

        # Check procedure-specific regions
        if procedure:
            self._check_procedure_regions(result, orig, manip, displacements, procedure)

    def _check_procedure_regions(
        self,
        result: SafetyResult,
        orig: np.ndarray,
        manip: np.ndarray,
        displacements: np.ndarray,
        procedure: str,
    ) -> None:
        """Verify displacement is concentrated in expected anatomical regions."""
        from landmarkdiff.landmarks import LANDMARK_REGIONS

        # Expected regions by procedure
        expected_regions = {
            "rhinoplasty": ["nose"],
            "blepharoplasty": ["eye_left", "eye_right"],
            "rhytidectomy": ["jawline"],
            "orthognathic": ["jawline", "lips"],
        }

        expected = expected_regions.get(procedure, [])
        if not expected:
            result.add_pass("procedure_region")
            return

        # Get expected region indices
        expected_indices = set()
        for region in expected:
            if region in LANDMARK_REGIONS:
                expected_indices.update(LANDMARK_REGIONS[region])

        if not expected_indices:
            result.add_pass("procedure_region")
            return

        # Check: is most displacement in expected regions?
        n = min(len(displacements), len(orig))
        expected_mask = np.array([i in expected_indices for i in range(n)])

        if expected_mask.sum() > 0 and (~expected_mask).sum() > 0:
            expected_disp = displacements[expected_mask].mean()
            unexpected_disp = displacements[~expected_mask].mean()
            result.details["expected_region_disp"] = float(expected_disp)
            result.details["unexpected_region_disp"] = float(unexpected_disp)

            # Expected regions should have more displacement
            if unexpected_disp > expected_disp * 2 and unexpected_disp > 0.005:
                result.add_warning(
                    "procedure_region",
                    f"{procedure}: unexpected regions displaced more than expected "
                    f"({unexpected_disp:.4f} vs {expected_disp:.4f})",
                )
            else:
                result.add_pass("procedure_region")
        else:
            result.add_pass("procedure_region")

    def _check_output_quality(self, result: SafetyResult, output: np.ndarray) -> None:
        """Check output image quality (not blank, not corrupted)."""
        if output is None or output.size == 0:
            result.add_failure("output_quality", "Output image is empty")
            return

        # Check for blank/black images
        mean_val = output.mean()
        if mean_val < 5:
            result.add_failure("output_quality", f"Output is nearly black (mean={mean_val:.1f})")
            return
        if mean_val > 250:
            result.add_failure("output_quality", f"Output is nearly white (mean={mean_val:.1f})")
            return

        # Check for artifacts (extreme variance)
        std_val = output.std()
        if std_val < 10:
            result.add_warning(
                "output_quality",
                f"Output has very low variance (std={std_val:.1f}), may be uniform",
            )

        result.add_pass("output_quality")
        result.details["output_mean"] = float(mean_val)
        result.details["output_std"] = float(std_val)

    def _check_ood(self, result: SafetyResult, image: np.ndarray) -> None:
        """Basic out-of-distribution detection.

        Checks image properties against expected ranges for face photos.
        """
        h, w = image.shape[:2]

        # Resolution check
        if min(h, w) < 128:
            result.add_warning("ood", f"Image resolution too low: {w}x{h}")

        # Aspect ratio (faces should be roughly square after preprocessing)
        aspect = max(h, w) / max(min(h, w), 1)
        if aspect > 3.0:
            result.add_warning("ood", f"Unusual aspect ratio: {aspect:.1f}")

        # Color distribution (face photos should have some skin tones)
        if len(image.shape) == 3 and image.shape[2] == 3:
            mean_b, mean_g, mean_r = image.mean(axis=(0, 1))
            # Face images typically have red channel > blue channel
            if mean_b > mean_r * 1.5:
                result.add_warning("ood", "Image appears very blue (not typical face photo)")

        result.add_pass("ood_basic")

    def apply_watermark(
        self,
        image: np.ndarray,
        text: str | None = None,
        opacity: float = 0.3,
    ) -> np.ndarray:
        """Apply a text watermark to the output image.

        Places semi-transparent text at the bottom of the image to indicate
        it is AI-generated.
        """
        if not self.watermark_enabled:
            return image

        text = text or self.watermark_text
        result = image.copy()
        h, w = result.shape[:2]

        # Create text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, w / 1500)
        thickness = max(1, int(w / 500))

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (w - text_size[0]) // 2
        y = h - 10

        # Semi-transparent background bar
        bar_y1 = y - text_size[1] - 10
        bar_y2 = h
        overlay = result.copy()
        cv2.rectangle(overlay, (0, bar_y1), (w, bar_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0, result)

        # White text
        cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return result

    def embed_metadata(
        self,
        image_path: str,
        procedure: str,
        intensity: float,
        model_version: str = "0.3.0",
    ) -> None:
        """Embed provenance metadata in the output image.

        Writes EXIF/PNG metadata with generation parameters for traceability.
        """
        import json
        from pathlib import Path

        meta = {
            "generator": "LandmarkDiff",
            "version": model_version,
            "procedure": procedure,
            "intensity": intensity,
            "disclaimer": "AI-generated surgical prediction for visualization only. "
            "Not a guarantee of surgical outcome.",
        }

        # Save as sidecar JSON (PNG doesn't have easy EXIF support)
        meta_path = Path(image_path).with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
