"""Tests for modular NumPy RBF pre-warp stage.

These tests lock parity with the legacy inline path from
`apply_procedure_preset` while validating the new explicit stage API.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from landmarkdiff.clinical import ClinicalFlags
from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    PROCEDURE_RADIUS,
    DeformationHandle,
    RegionalIntensity,
    _get_procedure_handles,
    apply_procedure_preset,
    apply_rbf_prewarp_stage,
    build_rbf_prewarp_handles,
    gaussian_rbf_deform_batch,
)


def _make_face(seed: int = 7, width: int = 512, height: int = 512) -> FaceLandmarks:
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=width,
        image_height=height,
        confidence=0.93,
    )


def _legacy_build_rbf_handles(
    face: FaceLandmarks,
    procedure: str,
    intensity: float,
    clinical_flags: ClinicalFlags | None = None,
    regional_intensity: RegionalIntensity | None = None,
) -> list[DeformationHandle]:
    scale = intensity / 100.0
    indices = PROCEDURE_LANDMARKS[procedure]
    radius = PROCEDURE_RADIUS[procedure]

    if clinical_flags and clinical_flags.ehlers_danlos:
        radius *= 1.5

    geo_mean = math.sqrt(face.image_width * face.image_height)
    pixel_scale = geo_mean / 512.0
    handles = _get_procedure_handles(
        procedure,
        indices,
        scale,
        radius * pixel_scale,
        regional_intensity,
    )

    if clinical_flags and clinical_flags.bells_palsy:
        from landmarkdiff.clinical import get_bells_palsy_side_indices

        affected = get_bells_palsy_side_indices(clinical_flags.bells_palsy_side)
        affected_indices = set()
        for region_indices in affected.values():
            affected_indices.update(region_indices)
        handles = [h for h in handles if h.landmark_index not in affected_indices]

    return handles


def _legacy_apply_rbf_prewarp(
    face: FaceLandmarks,
    procedure: str,
    intensity: float,
    clinical_flags: ClinicalFlags | None = None,
    regional_intensity: RegionalIntensity | None = None,
) -> FaceLandmarks:
    handles = _legacy_build_rbf_handles(
        face=face,
        procedure=procedure,
        intensity=intensity,
        clinical_flags=clinical_flags,
        regional_intensity=regional_intensity,
    )

    pixel_landmarks = face.landmarks.copy()
    pixel_landmarks[:, 0] *= face.image_width
    pixel_landmarks[:, 1] *= face.image_height

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

    result = pixel_landmarks.copy()
    result[:, 0] /= face.image_width
    result[:, 1] /= face.image_height

    return FaceLandmarks(
        landmarks=result,
        image_width=face.image_width,
        image_height=face.image_height,
        confidence=face.confidence,
    )


class TestRBFPrewarpStageParity:
    @pytest.mark.parametrize("procedure", sorted(PROCEDURE_LANDMARKS.keys()))
    @pytest.mark.parametrize("intensity", [0.0, 50.0, 100.0])
    def test_stage_matches_legacy_behavior(self, procedure: str, intensity: float):
        face = _make_face()
        stage_out = apply_rbf_prewarp_stage(face, procedure, intensity=intensity)
        legacy_out = _legacy_apply_rbf_prewarp(face, procedure, intensity=intensity)

        np.testing.assert_allclose(stage_out.landmarks, legacy_out.landmarks, atol=1e-7, rtol=0)
        assert stage_out.image_width == legacy_out.image_width
        assert stage_out.image_height == legacy_out.image_height
        assert stage_out.confidence == legacy_out.confidence

    def test_stage_matches_legacy_with_clinical_flags(self):
        face = _make_face()
        flags = ClinicalFlags(bells_palsy=True, bells_palsy_side="left", ehlers_danlos=True)

        stage_out = apply_rbf_prewarp_stage(
            face,
            "rhytidectomy",
            intensity=65.0,
            clinical_flags=flags,
        )
        legacy_out = _legacy_apply_rbf_prewarp(
            face,
            "rhytidectomy",
            intensity=65.0,
            clinical_flags=flags,
        )

        np.testing.assert_allclose(stage_out.landmarks, legacy_out.landmarks, atol=1e-7, rtol=0)

    def test_stage_matches_legacy_with_regional_intensity(self):
        face = _make_face()
        regional = RegionalIntensity(tip=1.2, bridge=0.85, alar=1.1)

        stage_out = apply_rbf_prewarp_stage(
            face,
            "rhinoplasty",
            intensity=70.0,
            regional_intensity=regional,
        )
        legacy_out = _legacy_apply_rbf_prewarp(
            face,
            "rhinoplasty",
            intensity=70.0,
            regional_intensity=regional,
        )

        np.testing.assert_allclose(stage_out.landmarks, legacy_out.landmarks, atol=1e-7, rtol=0)


class TestRBFHandleBuilderParity:
    def test_handle_builder_matches_legacy(self):
        face = _make_face(width=640, height=768)
        flags = ClinicalFlags(bells_palsy=True, bells_palsy_side="right", ehlers_danlos=True)
        regional = RegionalIntensity(tip=1.1, bridge=0.9, alar=1.05)

        built = build_rbf_prewarp_handles(
            face,
            "rhinoplasty",
            intensity=55.0,
            clinical_flags=flags,
            regional_intensity=regional,
        )
        legacy = _legacy_build_rbf_handles(
            face,
            "rhinoplasty",
            intensity=55.0,
            clinical_flags=flags,
            regional_intensity=regional,
        )

        assert len(built) == len(legacy)
        for current, previous in zip(built, legacy):
            assert current.landmark_index == previous.landmark_index
            np.testing.assert_allclose(
                current.displacement, previous.displacement, atol=1e-7, rtol=0
            )
            assert current.influence_radius == pytest.approx(previous.influence_radius)


class TestProcedurePresetRouting:
    def test_apply_procedure_uses_rbf_stage_without_model(self, monkeypatch):
        face = _make_face()
        calls: dict[str, object] = {}

        expected = FaceLandmarks(
            landmarks=np.zeros((478, 3), dtype=np.float32),
            image_width=face.image_width,
            image_height=face.image_height,
            confidence=face.confidence,
        )

        def fake_stage(
            face: FaceLandmarks,
            procedure: str,
            intensity: float = 50.0,
            clinical_flags: ClinicalFlags | None = None,
            regional_intensity: RegionalIntensity | None = None,
        ) -> FaceLandmarks:
            calls["face"] = face
            calls["procedure"] = procedure
            calls["intensity"] = intensity
            calls["clinical_flags"] = clinical_flags
            calls["regional_intensity"] = regional_intensity
            return expected

        monkeypatch.setattr("landmarkdiff.manipulation.apply_rbf_prewarp_stage", fake_stage)

        result = apply_procedure_preset(face, "rhinoplasty", intensity=42.0)

        assert result is expected
        assert calls["face"] is face
        assert calls["procedure"] == "rhinoplasty"
        assert calls["intensity"] == 42.0
        assert calls["clinical_flags"] is None
        assert calls["regional_intensity"] is None

    def test_apply_procedure_prefers_data_driven_when_model_path_present(self, monkeypatch):
        face = _make_face()
        expected = FaceLandmarks(
            landmarks=np.ones((478, 3), dtype=np.float32),
            image_width=face.image_width,
            image_height=face.image_height,
            confidence=face.confidence,
        )

        monkeypatch.setattr(
            "landmarkdiff.manipulation._apply_data_driven",
            lambda *_args, **_kwargs: expected,
        )
        monkeypatch.setattr(
            "landmarkdiff.manipulation.apply_rbf_prewarp_stage",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("RBF stage should not run when data-driven path succeeds")
            ),
        )

        result = apply_procedure_preset(
            face,
            "rhinoplasty",
            intensity=50.0,
            displacement_model_path="dummy.npz",
        )

        assert result is expected

    def test_image_size_is_compatibility_only_in_rbf_path(self):
        face = _make_face(width=640, height=640)
        out_512 = apply_procedure_preset(face, "rhinoplasty", intensity=60.0, image_size=512)
        out_1024 = apply_procedure_preset(face, "rhinoplasty", intensity=60.0, image_size=1024)
        np.testing.assert_allclose(out_512.landmarks, out_1024.landmarks, atol=1e-7, rtol=0)


class TestConfidenceSemantics:
    def test_builder_is_confidence_agnostic_stage_applies_confidence_weighting(self):
        face = _make_face(seed=21)
        handles = build_rbf_prewarp_handles(face, "rhinoplasty", intensity=70.0)

        # Builder returns raw policy handles; confidence scaling is applied
        # only by the explicit stage execution step.
        conf = face.landmark_confidence
        assert any(float(conf[h.landmark_index]) < 1.0 for h in handles)

        pixel_landmarks = face.landmarks.copy()
        pixel_landmarks[:, 0] *= face.image_width
        pixel_landmarks[:, 1] *= face.image_height
        raw_pixels = gaussian_rbf_deform_batch(pixel_landmarks, handles)
        raw_norm = raw_pixels.copy()
        raw_norm[:, 0] /= face.image_width
        raw_norm[:, 1] /= face.image_height

        stage = apply_rbf_prewarp_stage(face, "rhinoplasty", intensity=70.0)
        assert not np.allclose(raw_norm, stage.landmarks)
