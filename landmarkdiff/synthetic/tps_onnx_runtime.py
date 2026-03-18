from __future__ import annotations

from pathlib import Path

import numpy as np

from landmarkdiff.synthetic.tps_warp import _solve_tps_weights, _subsample_control_points


def _add_edge_anchors(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append identity edge anchors to keep borders stable."""
    edge_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
            [width // 2, 0],
            [width // 2, height - 1],
            [0, height // 2],
            [width - 1, height // 2],
        ],
        dtype=np.float32,
    )
    return np.vstack([src_points, edge_points]), np.vstack([dst_points, edge_points])


def _compute_tps_weights(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TPS displacement weights for x/y channels."""
    src = src_points.astype(np.float64)
    displacement = (dst_points - src_points).astype(np.float64)
    weights_x = _solve_tps_weights(src, displacement[:, 0])
    weights_y = _solve_tps_weights(src, displacement[:, 1])
    return weights_x.astype(np.float32), weights_y.astype(np.float32)


def _to_nchw_float32(image: np.ndarray) -> np.ndarray:
    """Convert uint8 HWC image [0,255] to float32 NCHW [0,1]."""
    image_f = image.astype(np.float32) / 255.0
    return np.transpose(image_f, (2, 0, 1))[None, ...].astype(np.float32)


def _to_hwc_uint8(image_nchw: np.ndarray) -> np.ndarray:
    """Convert float32 NCHW [0,1] output to uint8 HWC [0,255]."""
    image_hwc = np.transpose(image_nchw[0], (1, 2, 0))
    image_hwc = np.clip(image_hwc, 0.0, 1.0)
    return (image_hwc * 255.0).astype(np.uint8)


class TPSONNXRuntime:
    """Thin-plate spline warper backed by ONNX Runtime (CPU)."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self._session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def warp(
        self,
        image: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
    ) -> np.ndarray:
        """Warp image with ONNX TPS backend."""
        height, width = image.shape[:2]
        src_points = src_landmarks.astype(np.float32)
        dst_points = dst_landmarks.astype(np.float32)

        src_sub, dst_sub = _subsample_control_points(src_points, dst_points)
        src_ctrl, dst_ctrl = _add_edge_anchors(src_sub, dst_sub, width=width, height=height)
        weights_x, weights_y = _compute_tps_weights(src_ctrl, dst_ctrl)

        onnx_input = {
            "image": _to_nchw_float32(image),
            "control_points": src_ctrl[None, ...].astype(np.float32),
            "weights_x": weights_x[None, ...].astype(np.float32),
            "weights_y": weights_y[None, ...].astype(np.float32),
        }
        output_nchw = self._session.run(["warped"], onnx_input)[0]
        return _to_hwc_uint8(output_nchw)
