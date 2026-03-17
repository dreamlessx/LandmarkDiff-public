"""Tests for TPS ONNX export helper components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from landmarkdiff.synthetic.tps_onnx import TPSWarpONNX
from scripts.export_onnx import (
    compute_max_pixel_diff,
    compute_tps_weights,
    export_tps_onnx,
    make_dummy_inputs,
    run_onnx,
    run_pytorch,
)


class TestTPSWarpONNX:
    """Unit tests for TPSWarpONNX module."""

    def test_forward_shape(self):
        model = TPSWarpONNX(image_size=32).eval()
        image, src, dst = make_dummy_inputs(image_size=32, num_points=12, seed=1)
        weights_x, weights_y = compute_tps_weights(src, dst)

        with torch.no_grad():
            output = model(
                torch.from_numpy(image),
                torch.from_numpy(src),
                torch.from_numpy(weights_x),
                torch.from_numpy(weights_y),
            )

        assert output.shape == (1, 3, 32, 32)
        assert output.dtype == torch.float32

    def test_identity_is_close(self):
        model = TPSWarpONNX(image_size=32).eval()
        image, src, _ = make_dummy_inputs(image_size=32, num_points=10, seed=2)
        weights_x, weights_y = compute_tps_weights(src, src.copy())
        image_t = torch.from_numpy(image)

        with torch.no_grad():
            output = model(
                image_t,
                torch.from_numpy(src),
                torch.from_numpy(weights_x),
                torch.from_numpy(weights_y),
            )

        assert torch.allclose(output, image_t, atol=1e-3)


class TestExportHelpers:
    """Tests for export/validation helper functions."""

    def test_dummy_inputs_bounds(self):
        image, src, dst = make_dummy_inputs(image_size=64, num_points=20, seed=123)
        assert image.shape == (1, 3, 64, 64)
        assert src.shape == (1, 20, 2)
        assert dst.shape == (1, 20, 2)
        assert np.all(src >= 0.0) and np.all(src <= 63.0)
        assert np.all(dst >= 0.0) and np.all(dst <= 63.0)

    def test_compute_max_pixel_diff(self):
        ref = np.zeros((1, 3, 8, 8), dtype=np.float32)
        pred = ref.copy()
        assert compute_max_pixel_diff(ref, pred) == 0.0

        pred[:, :, 0, 0] = 1.0 / 255.0
        assert compute_max_pixel_diff(ref, pred) == 1.0

    def test_compute_tps_weights_shapes(self):
        _, src, dst = make_dummy_inputs(image_size=32, num_points=15, seed=8)
        weights_x, weights_y = compute_tps_weights(src, dst)
        assert weights_x.shape == (1, 18)
        assert weights_y.shape == (1, 18)
        assert np.all(np.isfinite(weights_x))
        assert np.all(np.isfinite(weights_y))

    def test_export_and_run_onnx(self, tmp_path: Path):
        ort = pytest.importorskip("onnxruntime")
        assert ort is not None

        image, src, dst = make_dummy_inputs(image_size=16, num_points=8, seed=7)
        weights_x, weights_y = compute_tps_weights(src, dst)
        onnx_path = tmp_path / "tps_warp.onnx"

        export_tps_onnx(
            output_path=onnx_path,
            image_size=16,
            opset=17,
            image_np=image,
            control_points_np=src,
            weights_x_np=weights_x,
            weights_y_np=weights_y,
        )
        assert onnx_path.exists()

        torch_out = run_pytorch(image, src, weights_x, weights_y, image_size=16)
        onnx_out = run_onnx(onnx_path, image, src, weights_x, weights_y)
        max_diff = compute_max_pixel_diff(torch_out, onnx_out)

        assert max_diff < 1.0
