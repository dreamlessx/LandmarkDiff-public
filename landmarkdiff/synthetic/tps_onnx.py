from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tps_kernel(radius: torch.Tensor) -> torch.Tensor:
    """Thin-plate spline kernel U(r) = r^2 * log(r), with U(0) = 0."""
    safe_radius = torch.clamp(radius, min=1e-6)
    return torch.where(
        radius > 0,
        (radius * radius) * torch.log(safe_radius),
        torch.zeros_like(radius),
    )


def _pairwise_kernel(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
    """Pairwise TPS kernel between 2D point sets.

    Args:
        points_a: (B, M, 2) points.
        points_b: (B, N, 2) points.

    Returns:
        (B, M, N) TPS kernel matrix.
    """
    diffs = points_a.unsqueeze(2) - points_b.unsqueeze(1)
    radius = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-12)
    return _tps_kernel(radius)


class TPSWarpONNX(nn.Module):
    """Torch implementation of TPS image warping for ONNX export."""

    def __init__(self, image_size: int = 128):
        super().__init__()
        self.image_size = image_size

        ys, xs = torch.meshgrid(
            torch.arange(image_size, dtype=torch.float32),
            torch.arange(image_size, dtype=torch.float32),
            indexing="ij",
        )
        eval_points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
        self.register_buffer("eval_points", eval_points, persistent=False)

    def forward(
        self,
        image: torch.Tensor,
        control_points: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
    ) -> torch.Tensor:
        """Warp image from TPS weights.

        Args:
            image: (B, C, H, W) float image in [0, 1].
            control_points: (B, N, 2) control points in pixel coordinates.
            weights_x: (B, N+3) TPS weights for X displacement.
            weights_y: (B, N+3) TPS weights for Y displacement.

        Returns:
            Warped image tensor with shape (B, C, H, W).
        """
        batch, _, height, width = image.shape
        num_points = control_points.shape[1]

        eval_points = self.eval_points.to(dtype=image.dtype, device=image.device)
        eval_points = eval_points.unsqueeze(0).expand(batch, -1, -1)
        eval_kernel = _pairwise_kernel(eval_points, control_points.to(dtype=image.dtype))

        radial_x = weights_x[:, :num_points]
        affine_x = weights_x[:, num_points:]
        radial_y = weights_y[:, :num_points]
        affine_y = weights_y[:, num_points:]

        eval_x = eval_points[..., 0]
        eval_y = eval_points[..., 1]

        disp_x = affine_x[:, 0:1] + affine_x[:, 1:2] * eval_x + affine_x[:, 2:3] * eval_y
        disp_y = affine_y[:, 0:1] + affine_y[:, 1:2] * eval_x + affine_y[:, 2:3] * eval_y

        disp_x = disp_x + torch.bmm(eval_kernel, radial_x.unsqueeze(-1)).squeeze(-1)
        disp_y = disp_y + torch.bmm(eval_kernel, radial_y.unsqueeze(-1)).squeeze(-1)

        map_x = eval_x - disp_x
        map_y = eval_y - disp_y

        normalized_x = (2.0 * map_x / (width - 1)) - 1.0
        normalized_y = (2.0 * map_y / (height - 1)) - 1.0

        grid = torch.stack([normalized_x, normalized_y], dim=-1).view(batch, height, width, 2)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
