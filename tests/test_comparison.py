"""Tests for the comparison module (before/after visualization utilities)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


class TestSliderComposite:
    """Tests for create_slider_composite."""

    def test_slider_at_half(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.zeros((100, 200, 3), dtype=np.uint8)
        prediction = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = create_slider_composite(original, prediction, position=0.5)
        assert result.shape == (100, 200, 3)
        # Left half should be from original (dark), right from prediction (bright)
        assert result[:, 10, 0].mean() < 50
        assert result[:, 190, 0].mean() > 200

    def test_slider_at_zero(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = create_slider_composite(original, prediction, position=0.0)
        # Position 0 = all prediction
        assert result[:, 32, 0].mean() > 150

    def test_slider_at_one(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.full((64, 64, 3), 200, dtype=np.uint8)
        prediction = np.zeros((64, 64, 3), dtype=np.uint8)
        result = create_slider_composite(original, prediction, position=1.0)
        assert result[:, 32, 0].mean() > 150

    def test_slider_with_different_size_prediction(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.zeros((100, 200, 3), dtype=np.uint8)
        prediction = np.full((50, 100, 3), 255, dtype=np.uint8)
        result = create_slider_composite(original, prediction, position=0.5)
        assert result.shape == (100, 200, 3)

    def test_slider_no_divider_line(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.zeros((64, 64, 3), dtype=np.uint8)
        result = create_slider_composite(original, prediction, line_width=0)
        assert result.shape == (64, 64, 3)

    def test_slider_custom_line_color(self):
        from landmarkdiff.comparison import create_slider_composite

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.zeros((64, 64, 3), dtype=np.uint8)
        result = create_slider_composite(
            original, prediction, line_color=(0, 0, 255), line_width=3
        )
        assert result.shape == (64, 64, 3)


class TestSideBySide:
    """Tests for create_side_by_side."""

    def test_basic_side_by_side(self):
        from landmarkdiff.comparison import create_side_by_side

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        prediction = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = create_side_by_side(original, prediction, gap=4)
        assert result.shape == (100, 204, 3)

    def test_side_by_side_no_labels(self):
        from landmarkdiff.comparison import create_side_by_side

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        prediction = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = create_side_by_side(original, prediction, add_labels=False)
        assert result.shape == (100, 204, 3)

    def test_side_by_side_custom_gap(self):
        from landmarkdiff.comparison import create_side_by_side

        original = np.zeros((50, 50, 3), dtype=np.uint8)
        prediction = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = create_side_by_side(original, prediction, gap=10)
        assert result.shape == (50, 110, 3)

    def test_side_by_side_different_sizes(self):
        from landmarkdiff.comparison import create_side_by_side

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        prediction = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = create_side_by_side(original, prediction, gap=4)
        # Prediction resized to match original
        assert result.shape[0] == 100

    def test_side_by_side_large_image_labels(self):
        from landmarkdiff.comparison import create_side_by_side

        original = np.zeros((512, 512, 3), dtype=np.uint8)
        prediction = np.full((512, 512, 3), 200, dtype=np.uint8)
        result = create_side_by_side(original, prediction, add_labels=True)
        assert result.shape == (512, 1028, 3)


class TestDifferenceHeatmap:
    """Tests for create_difference_heatmap."""

    def test_identical_images(self):
        from landmarkdiff.comparison import create_difference_heatmap

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = create_difference_heatmap(img, img.copy())
        assert result.shape == (64, 64, 3)

    def test_different_images(self):
        from landmarkdiff.comparison import create_difference_heatmap

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 100, dtype=np.uint8)
        result = create_difference_heatmap(original, prediction, amplify=3.0)
        assert result.shape == (64, 64, 3)
        # Result should not be all zero (there are differences)
        assert result.sum() > 0

    def test_heatmap_with_different_sizes(self):
        from landmarkdiff.comparison import create_difference_heatmap

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        prediction = np.full((50, 50, 3), 100, dtype=np.uint8)
        result = create_difference_heatmap(original, prediction)
        assert result.shape == (100, 100, 3)

    def test_custom_amplify(self):
        from landmarkdiff.comparison import create_difference_heatmap

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 50, dtype=np.uint8)
        result = create_difference_heatmap(original, prediction, amplify=1.0)
        assert result.shape == (64, 64, 3)


class TestCheckerboardBlend:
    """Tests for create_checkerboard_blend."""

    def test_basic_checkerboard(self):
        from landmarkdiff.comparison import create_checkerboard_blend

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = create_checkerboard_blend(original, prediction, block_size=32)
        assert result.shape == (64, 64, 3)
        # Should have both dark and bright regions
        assert result.min() < 50
        assert result.max() > 200

    def test_small_block_size(self):
        from landmarkdiff.comparison import create_checkerboard_blend

        original = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = create_checkerboard_blend(original, prediction, block_size=8)
        assert result.shape == (64, 64, 3)

    def test_checkerboard_different_sizes(self):
        from landmarkdiff.comparison import create_checkerboard_blend

        original = np.zeros((100, 100, 3), dtype=np.uint8)
        prediction = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = create_checkerboard_blend(original, prediction)
        assert result.shape == (100, 100, 3)

    def test_non_divisible_block(self):
        from landmarkdiff.comparison import create_checkerboard_blend

        original = np.zeros((70, 70, 3), dtype=np.uint8)
        prediction = np.full((70, 70, 3), 200, dtype=np.uint8)
        result = create_checkerboard_blend(original, prediction, block_size=32)
        assert result.shape == (70, 70, 3)
