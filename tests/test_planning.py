"""Tests for the planning module (surgical measurement annotations)."""

from __future__ import annotations

import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


def _make_face(seed=42, width=512, height=512):
    """Create a mock FaceLandmarks object."""
    rng = np.random.default_rng(seed)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    for i in range(478):
        landmarks[i, 0] = 0.3 + rng.random() * 0.4
        landmarks[i, 1] = 0.2 + rng.random() * 0.6
        landmarks[i, 2] = rng.random() * 0.1
    return FaceLandmarks(
        landmarks=landmarks,
        confidence=0.95,
        image_width=width,
        image_height=height,
    )


class TestMeasurement:
    """Tests for Measurement dataclass."""

    def test_measurement_fields(self):
        from landmarkdiff.planning import Measurement

        m = Measurement(name="nose_width", value_mm=35.0, value_px=100.0)
        assert m.name == "nose_width"
        assert m.value_mm == 35.0
        assert m.value_px == 100.0


class TestAngleMeasurement:
    """Tests for AngleMeasurement dataclass."""

    def test_angle_fields(self):
        from landmarkdiff.planning import AngleMeasurement

        a = AngleMeasurement(name="nasal_tip", degrees=90.0, vertex=(256.0, 256.0))
        assert a.name == "nasal_tip"
        assert a.degrees == 90.0
        assert a.vertex == (256.0, 256.0)


class TestPlanningResult:
    """Tests for PlanningResult dataclass."""

    def test_summary(self):
        from landmarkdiff.planning import AngleMeasurement, Measurement, PlanningResult

        result = PlanningResult(
            measurements=[
                Measurement(name="intercanthal", value_mm=33.0, value_px=50.0),
                Measurement(name="nose_width", value_mm=35.0, value_px=55.0),
            ],
            angles=[
                AngleMeasurement(name="nasal_tip", degrees=92.5, vertex=(256.0, 256.0)),
            ],
            px_per_mm=1.5,
        )
        s = result.summary()
        assert "Surgical Planning Measurements" in s
        assert "intercanthal" in s
        assert "33.0 mm" in s
        assert "nasal_tip" in s
        assert "92.5 deg" in s
        assert "1.50 px/mm" in s


class TestComputePlanningMeasurements:
    """Tests for compute_planning_measurements."""

    def test_basic_computation(self):
        from landmarkdiff.planning import compute_planning_measurements

        face = _make_face()
        result = compute_planning_measurements(face)

        assert len(result.measurements) == 10  # 10 standard measurements
        assert len(result.angles) == 3  # 3 angular measurements
        assert result.px_per_mm > 0

    def test_measurements_are_positive(self):
        from landmarkdiff.planning import compute_planning_measurements

        face = _make_face()
        result = compute_planning_measurements(face)

        for m in result.measurements:
            assert m.value_mm > 0
            assert m.value_px > 0

    def test_angles_are_valid(self):
        from landmarkdiff.planning import compute_planning_measurements

        face = _make_face()
        result = compute_planning_measurements(face)

        for a in result.angles:
            assert 0.0 <= a.degrees <= 180.0

    def test_custom_icd_reference(self):
        from landmarkdiff.planning import compute_planning_measurements

        face = _make_face()
        result_default = compute_planning_measurements(face)
        result_custom = compute_planning_measurements(face, reference_icd_mm=40.0)
        # Different reference should change px_per_mm
        assert result_default.px_per_mm != result_custom.px_per_mm

    def test_different_faces_different_measurements(self):
        from landmarkdiff.planning import compute_planning_measurements

        face1 = _make_face(seed=42)
        face2 = _make_face(seed=99)
        r1 = compute_planning_measurements(face1)
        r2 = compute_planning_measurements(face2)
        # Different random faces should have different measurements
        assert r1.measurements[0].value_px != r2.measurements[0].value_px


class TestVisualizePlanning:
    """Tests for visualize_planning."""

    def test_basic_visualization(self):
        from landmarkdiff.planning import compute_planning_measurements, visualize_planning

        face = _make_face()
        planning = compute_planning_measurements(face)

        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        result = visualize_planning(image, face, planning)

        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8
        # Should have drawn on the image (not identical to input)
        assert not np.array_equal(result, image)

    def test_small_image(self):
        from landmarkdiff.planning import compute_planning_measurements, visualize_planning

        face = _make_face(width=64, height=64)
        planning = compute_planning_measurements(face)

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = visualize_planning(image, face, planning)
        assert result.shape == (64, 64, 3)
