"""Tests for inference pipeline (structure only — no model download)."""

import numpy as np
import pytest
import torch

from landmarkdiff.inference import LandmarkDiffPipeline, get_device, numpy_to_pil, pil_to_numpy


class TestDeviceSelection:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_mps_available_on_apple_silicon(self):
        if torch.backends.mps.is_available():
            assert get_device().type == "mps"


class TestImageConversion:
    def test_numpy_to_pil_rgb(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue channel in BGR
        pil_img = numpy_to_pil(bgr)
        arr = np.array(pil_img)
        # Should be RGB, so blue=0, red=255 after conversion
        assert arr[0, 0, 2] == 255  # blue in RGB position

    def test_roundtrip(self):
        original = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        pil_img = numpy_to_pil(original)
        recovered = pil_to_numpy(pil_img)
        np.testing.assert_array_equal(original, recovered)

    def test_grayscale(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        pil_img = numpy_to_pil(gray)
        assert pil_img.mode == "L"


class TestPipelineInit:
    def test_default_device(self):
        pipe = LandmarkDiffPipeline()
        assert pipe.device.type in ("mps", "cuda", "cpu")

    def test_not_loaded_initially(self):
        pipe = LandmarkDiffPipeline()
        assert not pipe.is_loaded

    def test_generate_raises_if_not_loaded(self):
        pipe = LandmarkDiffPipeline()
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not loaded"):
            pipe.generate(dummy)

    def test_mps_uses_float32(self):
        if torch.backends.mps.is_available():
            pipe = LandmarkDiffPipeline(device=torch.device("mps"))
            assert pipe.dtype == torch.float32
