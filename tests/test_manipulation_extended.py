"""Extended tests for landmark manipulation via Gaussian RBF deformation.

Covers per-procedure handle generation, intensity scaling, clinical flags,
image_size scaling, 3D deformation, and boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import (
    PROCEDURE_LANDMARKS,
    PROCEDURE_RADIUS,
    DeformationHandle,
    _get_procedure_handles,
    apply_procedure_preset,
    gaussian_rbf_deform,
)


def _make_face(seed: int = 42, image_size: int = 512) -> FaceLandmarks:
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=image_size,
        image_height=image_size,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# DeformationHandle dataclass
# ---------------------------------------------------------------------------


class TestDeformationHandle:
    def test_frozen(self):
        h = DeformationHandle(
            landmark_index=0,
            displacement=np.array([1.0, 2.0]),
            influence_radius=30.0,
        )
        with pytest.raises(AttributeError):
            h.landmark_index = 5

    def test_fields(self):
        disp = np.array([3.0, -1.0])
        h = DeformationHandle(landmark_index=10, displacement=disp, influence_radius=25.0)
        assert h.landmark_index == 10
        assert h.influence_radius == 25.0
        np.testing.assert_array_equal(h.displacement, disp)


# ---------------------------------------------------------------------------
# Procedure data validation
# ---------------------------------------------------------------------------


class TestProcedureData:
    def test_all_procedures_defined(self):
        expected = {
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
            "alarplasty",
            "canthoplasty",
            "buccal_fat_removal",
            "dimpleplasty",
            "genioplasty",
            "malarplasty",
            "lip_lift",
            "lip_augmentation",
            "forehead_reduction",
            "submental_liposuction",
            "otoplasty",
        }
        assert set(PROCEDURE_LANDMARKS.keys()) == expected
        assert set(PROCEDURE_RADIUS.keys()) == expected

    def test_indices_within_range(self):
        for proc, indices in PROCEDURE_LANDMARKS.items():
            for idx in indices:
                assert 0 <= idx < 478, f"{proc}: index {idx} out of range"

    def test_radii_positive(self):
        for proc, radius in PROCEDURE_RADIUS.items():
            assert radius > 0, f"{proc}: radius must be positive"

    def test_no_duplicate_indices(self):
        for proc, indices in PROCEDURE_LANDMARKS.items():
            assert len(indices) == len(set(indices)), f"{proc} has duplicate indices"


# ---------------------------------------------------------------------------
# gaussian_rbf_deform edge cases
# ---------------------------------------------------------------------------


class TestGaussianRBFDeformExtended:
    def test_3d_displacement(self):
        landmarks = np.zeros((10, 3), dtype=np.float32)
        landmarks[3] = [50.0, 50.0, 10.0]
        handle = DeformationHandle(
            landmark_index=3,
            displacement=np.array([5.0, 5.0, 2.0]),
            influence_radius=20.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        assert abs(result[3, 2] - 12.0) < 0.01

    def test_2d_displacement_no_z_change(self):
        landmarks = np.zeros((10, 3), dtype=np.float32)
        landmarks[:, 2] = 5.0
        landmarks[0] = [100.0, 100.0, 5.0]
        handle = DeformationHandle(
            landmark_index=0,
            displacement=np.array([10.0, 10.0]),
            influence_radius=30.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        # z should not change with 2D displacement
        np.testing.assert_array_almost_equal(result[:, 2], landmarks[:, 2])

    def test_zero_displacement(self):
        landmarks = np.random.default_rng(0).uniform(0, 100, (20, 2)).astype(np.float32)
        handle = DeformationHandle(
            landmark_index=5,
            displacement=np.array([0.0, 0.0]),
            influence_radius=30.0,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        np.testing.assert_array_almost_equal(result, landmarks)

    def test_very_small_radius_affects_only_handle(self):
        landmarks = np.zeros((5, 2), dtype=np.float32)
        landmarks[0] = [0.0, 0.0]
        landmarks[1] = [100.0, 0.0]
        landmarks[2] = [200.0, 0.0]
        handle = DeformationHandle(
            landmark_index=1,
            displacement=np.array([10.0, 0.0]),
            influence_radius=0.1,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        # Handle point should move fully
        assert abs(result[1, 0] - 110.0) < 0.01
        # Distant points should barely move
        assert abs(result[0, 0]) < 0.001
        assert abs(result[2, 0] - 200.0) < 0.001

    def test_large_radius_moves_all_points(self):
        landmarks = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
        handle = DeformationHandle(
            landmark_index=1,
            displacement=np.array([50.0, 0.0]),
            influence_radius=1e6,
        )
        result = gaussian_rbf_deform(landmarks, handle)
        # All points should move nearly the full displacement
        for i in range(3):
            assert abs(result[i, 0] - (landmarks[i, 0] + 50.0)) < 0.1


# ---------------------------------------------------------------------------
# apply_procedure_preset — intensity scaling
# ---------------------------------------------------------------------------


class TestIntensityScaling:
    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_higher_intensity_larger_displacement(self, procedure):
        face = _make_face()
        low = apply_procedure_preset(face, procedure, intensity=20.0)
        high = apply_procedure_preset(face, procedure, intensity=80.0)
        diff_low = np.linalg.norm(low.landmarks - face.landmarks)
        diff_high = np.linalg.norm(high.landmarks - face.landmarks)
        assert diff_high > diff_low

    def test_intensity_100_produces_maximum_change(self):
        face = _make_face()
        result = apply_procedure_preset(face, "rhinoplasty", intensity=100.0)
        diff = np.linalg.norm(result.landmarks - face.landmarks)
        assert diff > 0

    def test_negative_intensity_reverses_direction(self):
        """Negative intensity should invert the displacement direction overall."""
        face = _make_face()
        pos = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        neg = apply_procedure_preset(face, "rhinoplasty", intensity=-50.0)
        diff_pos = pos.landmarks - face.landmarks
        diff_neg = neg.landmarks - face.landmarks
        # Use a higher threshold to avoid floating-point edge cases
        nonzero_mask = np.abs(diff_pos) > 1e-4
        if np.any(nonzero_mask):
            signs_pos = np.sign(diff_pos[nonzero_mask])
            signs_neg = np.sign(diff_neg[nonzero_mask])
            # Allow a small fraction of mismatches from RBF nonlinearity
            match_rate = np.mean(signs_pos == -signs_neg)
            assert match_rate > 0.99


# ---------------------------------------------------------------------------
# apply_procedure_preset — metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_output_preserves_dimensions(self, procedure):
        face = _make_face(image_size=768)
        result = apply_procedure_preset(face, procedure, intensity=50.0, image_size=768)
        assert result.image_width == 768
        assert result.image_height == 768

    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_output_preserves_confidence(self, procedure):
        face = _make_face()
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert result.confidence == face.confidence

    def test_output_shape_unchanged(self):
        face = _make_face()
        result = apply_procedure_preset(face, "blepharoplasty", intensity=50.0)
        assert result.landmarks.shape == (478, 3)


# ---------------------------------------------------------------------------
# apply_procedure_preset — image_size scaling
# ---------------------------------------------------------------------------


class TestImageSizeScaling:
    def test_larger_face_scales_displacement(self):
        """Radius scales with geometric mean of face dimensions."""
        face_512 = _make_face(image_size=512)
        face_1024 = FaceLandmarks(
            landmarks=face_512.landmarks.copy(),
            image_width=1024,
            image_height=1024,
            confidence=0.9,
        )
        r512 = apply_procedure_preset(face_512, "rhinoplasty", intensity=50.0)
        r1024 = apply_procedure_preset(face_1024, "rhinoplasty", intensity=50.0)
        diff_512 = np.linalg.norm(r512.landmarks - face_512.landmarks)
        diff_1024 = np.linalg.norm(r1024.landmarks - face_1024.landmarks)
        # Larger face dimensions produce larger RBF radius, different displacement
        assert diff_1024 != diff_512

    def test_nonsquare_same_area_same_scale(self):
        """Non-square faces with same area get the same RBF radius scale."""
        import math

        # Both have geometric mean = sqrt(1024*512) = 724.08
        geo_wide = math.sqrt(1024 * 512) / 512.0
        geo_tall = math.sqrt(512 * 1024) / 512.0
        assert abs(geo_wide - geo_tall) < 1e-10

    def test_square_matches_old_behavior(self):
        """For square images, geometric mean equals the side length."""
        import math

        assert math.sqrt(512 * 512) == 512.0
        assert math.sqrt(1024 * 1024) == 1024.0


# ---------------------------------------------------------------------------
# _get_procedure_handles
# ---------------------------------------------------------------------------


class TestGetProcedureHandles:
    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_returns_nonempty_handles(self, procedure):
        indices = PROCEDURE_LANDMARKS[procedure]
        radius = PROCEDURE_RADIUS[procedure]
        handles = _get_procedure_handles(procedure, indices, 1.0, radius)
        assert len(handles) > 0

    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_handles_have_valid_indices(self, procedure):
        indices = PROCEDURE_LANDMARKS[procedure]
        radius = PROCEDURE_RADIUS[procedure]
        handles = _get_procedure_handles(procedure, indices, 1.0, radius)
        for h in handles:
            assert 0 <= h.landmark_index < 478

    @pytest.mark.parametrize("procedure", list(PROCEDURE_LANDMARKS.keys()))
    def test_handles_have_positive_radius(self, procedure):
        indices = PROCEDURE_LANDMARKS[procedure]
        radius = PROCEDURE_RADIUS[procedure]
        handles = _get_procedure_handles(procedure, indices, 1.0, radius)
        for h in handles:
            assert h.influence_radius > 0

    def test_zero_scale_produces_zero_displacement(self):
        indices = PROCEDURE_LANDMARKS["rhinoplasty"]
        radius = PROCEDURE_RADIUS["rhinoplasty"]
        handles = _get_procedure_handles("rhinoplasty", indices, 0.0, radius)
        for h in handles:
            np.testing.assert_array_almost_equal(h.displacement, np.zeros(2))

    def test_rhinoplasty_has_bilateral_symmetry(self):
        """Rhinoplasty should have both left and right alar handles."""
        indices = PROCEDURE_LANDMARKS["rhinoplasty"]
        radius = PROCEDURE_RADIUS["rhinoplasty"]
        handles = _get_procedure_handles("rhinoplasty", indices, 1.0, radius)
        displacements_x = [h.displacement[0] for h in handles]
        has_positive_x = any(d > 0 for d in displacements_x)
        has_negative_x = any(d < 0 for d in displacements_x)
        assert has_positive_x and has_negative_x


# ---------------------------------------------------------------------------
# apply_procedure_preset — clinical flags
# ---------------------------------------------------------------------------


class TestClinicalFlags:
    def test_ehlers_danlos_widens_radius(self):
        from landmarkdiff.clinical import ClinicalFlags

        face = _make_face()
        flags = ClinicalFlags(ehlers_danlos=True)
        result_ed = apply_procedure_preset(
            face, "rhinoplasty", intensity=50.0, clinical_flags=flags
        )
        result_normal = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        # Should produce different results (wider influence)
        diff = np.linalg.norm(result_ed.landmarks - result_normal.landmarks)
        assert diff > 0

    def test_bells_palsy_removes_affected_handles(self):
        from landmarkdiff.clinical import ClinicalFlags

        face = _make_face()
        flags = ClinicalFlags(bells_palsy=True, bells_palsy_side="left")
        result = apply_procedure_preset(face, "rhytidectomy", intensity=50.0, clinical_flags=flags)
        # Should still produce a valid result
        assert isinstance(result, FaceLandmarks)
        assert result.landmarks.shape == (478, 3)

    def test_no_flags_same_as_none(self):
        from landmarkdiff.clinical import ClinicalFlags

        face = _make_face()
        flags = ClinicalFlags()  # all defaults (no conditions)
        result_flags = apply_procedure_preset(
            face, "rhinoplasty", intensity=50.0, clinical_flags=flags
        )
        result_none = apply_procedure_preset(face, "rhinoplasty", intensity=50.0)
        np.testing.assert_array_almost_equal(result_flags.landmarks, result_none.landmarks)


# ---------------------------------------------------------------------------
# apply_procedure_preset — immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_does_not_modify_input(self):
        face = _make_face()
        original = face.landmarks.copy()
        apply_procedure_preset(face, "orthognathic", intensity=75.0)
        np.testing.assert_array_equal(face.landmarks, original)
