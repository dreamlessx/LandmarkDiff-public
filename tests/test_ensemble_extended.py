"""Extended tests for ensemble module -- aggregation strategies and init."""

from __future__ import annotations

import numpy as np


class TestEnsembleInferenceInit:
    """Tests for EnsembleInference initialization."""

    def test_default_init(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        assert ei.mode == "controlnet"
        assert ei.n_samples == 5
        assert ei.strategy == "best_of_n"
        assert ei.base_seed == 42
        assert ei.is_loaded is False

    def test_custom_init(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference(
            mode="tps",
            n_samples=3,
            strategy="pixel_average",
            base_seed=123,
        )
        assert ei.mode == "tps"
        assert ei.n_samples == 3
        assert ei.strategy == "pixel_average"
        assert ei.base_seed == 123

    def test_generate_raises_if_not_loaded(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        import pytest

        with pytest.raises(RuntimeError, match="not loaded"):
            ei.generate(img)


class TestPixelAverage:
    """Tests for _pixel_average method."""

    def test_average_of_identical(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        outputs = [img.copy() for _ in range(5)]
        result = ei._pixel_average(outputs)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img)

    def test_average_of_two(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 0, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = ei._pixel_average([a, b])
        # Mean of 0 and 200 = 100
        assert abs(result.mean() - 100.0) < 2.0


class TestPixelMedian:
    """Tests for _pixel_median method."""

    def test_median_of_identical(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        outputs = [img.copy() for _ in range(3)]
        result = ei._pixel_median(outputs)
        assert result.shape == (32, 32, 3)
        np.testing.assert_array_equal(result, img)

    def test_median_robust_to_outlier(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        normal = np.full((32, 32, 3), 100, dtype=np.uint8)
        outlier = np.full((32, 32, 3), 250, dtype=np.uint8)
        # 2 normal + 1 outlier: median should be 100
        result = ei._pixel_median([normal.copy(), normal.copy(), outlier])
        assert abs(result.mean() - 100.0) < 2.0
