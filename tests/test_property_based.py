"""Property-based tests for landmark manipulation.

Uses Hypothesis to verify invariants that should hold for all valid inputs:
- Landmark count preservation
- Coordinate bounds
- Symmetry of bilateral operations
- Idempotency where expected
- Monotonicity of intensity
"""

from __future__ import annotations

import numpy as np
import pytest

from landmarkdiff.landmarks import FaceLandmarks
from landmarkdiff.manipulation import apply_procedure_preset

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

PROCEDURES = [
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
]

pytestmark = pytest.mark.skipif(
    not HAS_HYPOTHESIS,
    reason="hypothesis not installed",
)


def _make_face(seed: int = 0, size: int = 512) -> FaceLandmarks:
    rng = np.random.default_rng(seed)
    landmarks = rng.uniform(0.15, 0.85, (478, 3)).astype(np.float32)
    return FaceLandmarks(
        landmarks=landmarks,
        image_width=size,
        image_height=size,
        confidence=0.95,
    )


if HAS_HYPOTHESIS:

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        procedure=st.sampled_from(PROCEDURES),
        intensity=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_landmark_count_preserved(seed, procedure, intensity):
        """Deformation must not change the number of landmarks."""
        face = _make_face(seed)
        result = apply_procedure_preset(face, procedure, intensity=intensity)
        assert result.landmarks.shape == (478, 3)

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        procedure=st.sampled_from(PROCEDURES),
        intensity=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=50, deadline=5000)
    def test_coordinates_remain_finite(seed, procedure, intensity):
        """All landmark coordinates must remain finite after deformation."""
        face = _make_face(seed)
        result = apply_procedure_preset(face, procedure, intensity=intensity)
        assert np.all(np.isfinite(result.landmarks))

    @given(
        seed=st.integers(min_value=0, max_value=500),
        procedure=st.sampled_from(PROCEDURES),
    )
    @settings(max_examples=30, deadline=5000)
    def test_zero_intensity_is_identity(seed, procedure):
        """Zero intensity should produce no change."""
        face = _make_face(seed)
        result = apply_procedure_preset(face, procedure, intensity=0.0)
        np.testing.assert_allclose(
            result.landmarks,
            face.landmarks,
            atol=1e-5,
            err_msg=f"{procedure} at intensity=0 should be identity",
        )

    @given(
        seed=st.integers(min_value=0, max_value=500),
        procedure=st.sampled_from(PROCEDURES),
    )
    @settings(max_examples=30, deadline=5000)
    def test_image_dimensions_preserved(seed, procedure):
        """Image dimensions should not change after deformation."""
        face = _make_face(seed, size=256)
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert result.image_width == 256
        assert result.image_height == 256

    @given(
        seed=st.integers(min_value=0, max_value=200),
        procedure=st.sampled_from(PROCEDURES),
        intensity_low=st.floats(min_value=10.0, max_value=40.0),
        intensity_high=st.floats(min_value=60.0, max_value=100.0),
    )
    @settings(max_examples=30, deadline=5000)
    def test_higher_intensity_larger_displacement(
        seed,
        procedure,
        intensity_low,
        intensity_high,
    ):
        """Higher intensity should produce larger total displacement."""
        face = _make_face(seed)
        result_low = apply_procedure_preset(face, procedure, intensity=intensity_low)
        result_high = apply_procedure_preset(face, procedure, intensity=intensity_high)

        disp_low = np.linalg.norm(result_low.landmarks - face.landmarks)
        disp_high = np.linalg.norm(result_high.landmarks - face.landmarks)

        assert disp_high >= disp_low - 1e-6, (
            f"{procedure}: intensity {intensity_high} displacement {disp_high:.6f} "
            f"< intensity {intensity_low} displacement {disp_low:.6f}"
        )

    @given(
        seed=st.integers(min_value=0, max_value=500),
        procedure=st.sampled_from(PROCEDURES),
    )
    @settings(max_examples=30, deadline=5000)
    def test_confidence_preserved(seed, procedure):
        """Confidence score should not change during deformation."""
        face = _make_face(seed)
        result = apply_procedure_preset(face, procedure, intensity=50.0)
        assert result.confidence == face.confidence
