"""Tests for mixed-precision inference utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from landmarkdiff.inference import estimate_vram_usage, get_optimal_dtype  # noqa: E402


class TestGetOptimalDtype:
    def test_cpu_returns_float32(self):
        assert get_optimal_dtype(torch.device("cpu")) == torch.float32

    def test_returns_a_torch_dtype(self):
        result = get_optimal_dtype(torch.device("cpu"))
        assert isinstance(result, torch.dtype)

    def test_none_device_does_not_raise(self):
        # Should auto-detect device without error
        result = get_optimal_dtype(None)
        assert result in (torch.float16, torch.float32)

    def test_default_returns_valid_dtype(self):
        result = get_optimal_dtype()
        assert result in (torch.float16, torch.float32)


class TestEstimateVramUsage:
    def test_returns_dict(self):
        result = estimate_vram_usage()
        assert isinstance(result, dict)
        assert "total_gb" in result
        assert "unet_gb" in result

    def test_total_positive(self):
        result = estimate_vram_usage()
        assert result["total_gb"] > 0

    def test_fp16_smaller_than_fp32(self):
        fp16 = estimate_vram_usage(dtype=torch.float16)
        fp32 = estimate_vram_usage(dtype=torch.float32)
        assert fp16["total_gb"] < fp32["total_gb"]

    def test_controlnet_larger_than_img2img(self):
        cn = estimate_vram_usage(mode="controlnet")
        img = estimate_vram_usage(mode="img2img")
        assert cn["total_gb"] > img["total_gb"]

    def test_higher_resolution_uses_more_vram(self):
        low = estimate_vram_usage(resolution=256)
        high = estimate_vram_usage(resolution=1024)
        assert high["total_gb"] > low["total_gb"]

    def test_includes_resolution(self):
        result = estimate_vram_usage(resolution=768)
        assert result["resolution"] == 768

    def test_includes_dtype_string(self):
        result = estimate_vram_usage(dtype=torch.float16)
        assert "float16" in result["dtype"]

    def test_controlnet_gb_zero_for_non_controlnet(self):
        result = estimate_vram_usage(mode="img2img")
        assert result["controlnet_gb"] == 0

    def test_vae_always_fp32(self):
        fp16 = estimate_vram_usage(dtype=torch.float16)
        fp32 = estimate_vram_usage(dtype=torch.float32)
        # VAE size should be the same regardless of dtype
        assert fp16["vae_gb"] == fp32["vae_gb"]

    def test_all_components_nonnegative(self):
        result = estimate_vram_usage()
        for key, val in result.items():
            if isinstance(val, float):
                assert val >= 0, f"{key} is negative: {val}"
