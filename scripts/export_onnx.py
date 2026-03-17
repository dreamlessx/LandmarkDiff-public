from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from landmarkdiff.synthetic.tps_onnx import TPSWarpONNX
from landmarkdiff.synthetic.tps_warp import _solve_tps_weights


def make_dummy_inputs(
    image_size: int,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic synthetic inputs for export/validation."""
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    base_image = np.stack(
        [
            0.6 * grid_x + 0.2 * grid_y,
            0.2 * grid_x + 0.6 * grid_y,
            0.5 * (grid_x + grid_y),
        ],
        axis=0,
    )
    noise = rng.normal(0.0, 0.01, size=base_image.shape).astype(np.float32)
    image = np.clip(base_image + noise, 0.0, 1.0)[None, ...].astype(np.float32)

    # Use a grid-like control-point layout to avoid numerically unstable random
    # configurations in the TPS linear system during validation.
    grid_side = int(np.ceil(np.sqrt(num_points)))
    margin = max(4.0, image_size / 16.0)
    xs = np.linspace(margin, image_size - 1 - margin, grid_side, dtype=np.float32)
    ys = np.linspace(margin, image_size - 1 - margin, grid_side, dtype=np.float32)
    mesh = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

    src_points = mesh[:num_points][None, ...].astype(np.float32)
    delta_scale = max(0.5, image_size / 128.0)
    delta = rng.normal(0.0, delta_scale, size=(1, num_points, 2)).astype(np.float32)
    dst_points = np.clip(src_points + delta, 0.0, image_size - 1).astype(np.float32)
    return image, src_points, dst_points


def compute_tps_weights(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TPS displacement weights for X and Y in NumPy.

    Args:
        src_points: (B, N, 2).
        dst_points: (B, N, 2).

    Returns:
        Tuple of (weights_x, weights_y), each with shape (B, N+3).
    """
    if src_points.shape != dst_points.shape:
        raise ValueError("src_points and dst_points must have the same shape")
    if src_points.ndim != 3 or src_points.shape[-1] != 2:
        raise ValueError("Expected src_points shape (B, N, 2)")

    displacement = dst_points - src_points
    weights_x = []
    weights_y = []

    for idx in range(src_points.shape[0]):
        src = src_points[idx].astype(np.float64)
        dx = displacement[idx, :, 0].astype(np.float64)
        dy = displacement[idx, :, 1].astype(np.float64)

        weights_x.append(_solve_tps_weights(src, dx).astype(np.float32))
        weights_y.append(_solve_tps_weights(src, dy).astype(np.float32))

    return np.stack(weights_x, axis=0), np.stack(weights_y, axis=0)


def export_tps_onnx(
    output_path: Path,
    image_size: int,
    opset: int,
    image_np: np.ndarray,
    control_points_np: np.ndarray,
    weights_x_np: np.ndarray,
    weights_y_np: np.ndarray,
) -> Path:
    """Export TPS warp module to ONNX."""
    model = TPSWarpONNX(image_size=image_size).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_t = torch.from_numpy(image_np)
    control_points_t = torch.from_numpy(control_points_np)
    weights_x_t = torch.from_numpy(weights_x_np)
    weights_y_t = torch.from_numpy(weights_y_np)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (image_t, control_points_t, weights_x_t, weights_y_t),
            str(output_path),
            dynamo=False,
            opset_version=opset,
            input_names=["image", "control_points", "weights_x", "weights_y"],
            output_names=["warped"],
            dynamic_axes={
                "image": {0: "batch"},
                "control_points": {0: "batch"},
                "weights_x": {0: "batch"},
                "weights_y": {0: "batch"},
                "warped": {0: "batch"},
            },
            do_constant_folding=True,
        )

    return output_path


def run_pytorch(
    image_np: np.ndarray,
    control_points_np: np.ndarray,
    weights_x_np: np.ndarray,
    weights_y_np: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Run TPS module in PyTorch."""
    model = TPSWarpONNX(image_size=image_size).eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(image_np),
            torch.from_numpy(control_points_np),
            torch.from_numpy(weights_x_np),
            torch.from_numpy(weights_y_np),
        )
    return output.cpu().numpy()


def run_onnx(
    onnx_path: Path,
    image_np: np.ndarray,
    control_points_np: np.ndarray,
    weights_x_np: np.ndarray,
    weights_y_np: np.ndarray,
) -> np.ndarray:
    """Run TPS module in ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {
            "image": image_np,
            "control_points": control_points_np,
            "weights_x": weights_x_np,
            "weights_y": weights_y_np,
        },
    )[0]
    return output


def compute_max_pixel_diff(torch_output: np.ndarray, onnx_output: np.ndarray) -> float:
    """Compute max per-pixel absolute diff in 0..255 float space."""
    torch_scaled = np.clip(torch_output, 0.0, 1.0) * 255.0
    onnx_scaled = np.clip(onnx_output, 0.0, 1.0) * 255.0
    return float(np.abs(torch_scaled - onnx_scaled).max())


def benchmark_cpu(
    onnx_path: Path,
    image_np: np.ndarray,
    control_points_np: np.ndarray,
    weights_x_np: np.ndarray,
    weights_y_np: np.ndarray,
    image_size: int,
    iterations: int,
) -> dict[str, float | str]:
    """Benchmark ONNX Runtime vs PyTorch latency (CPU)."""
    import onnxruntime as ort

    model = TPSWarpONNX(image_size=image_size).eval()
    image_t = torch.from_numpy(image_np)
    control_points_t = torch.from_numpy(control_points_np)
    weights_x_t = torch.from_numpy(weights_x_np)
    weights_y_t = torch.from_numpy(weights_y_np)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    with torch.no_grad():
        for _ in range(3):
            model(image_t, control_points_t, weights_x_t, weights_y_t)
    for _ in range(3):
        session.run(
            None,
            {
                "image": image_np,
                "control_points": control_points_np,
                "weights_x": weights_x_np,
                "weights_y": weights_y_np,
            },
        )

    torch_times_ms = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(image_t, control_points_t, weights_x_t, weights_y_t)
            torch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    onnx_times_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(
            None,
            {
                "image": image_np,
                "control_points": control_points_np,
                "weights_x": weights_x_np,
                "weights_y": weights_y_np,
            },
        )
        onnx_times_ms.append((time.perf_counter() - t0) * 1000.0)

    torch_mean = float(np.mean(torch_times_ms))
    onnx_mean = float(np.mean(onnx_times_ms))

    return {
        "provider": session.get_providers()[0],
        "pytorch_mean_ms": round(torch_mean, 3),
        "pytorch_std_ms": round(float(np.std(torch_times_ms)), 3),
        "onnx_mean_ms": round(onnx_mean, 3),
        "onnx_std_ms": round(float(np.std(onnx_times_ms)), 3),
        "speedup_x": round(torch_mean / onnx_mean, 3) if onnx_mean > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TPS warp operator to ONNX")
    parser.add_argument("--output", default="exports/tps_warp.onnx", help="Output ONNX file")
    parser.add_argument("--image-size", type=int, default=128, help="Square image resolution")
    parser.add_argument("--num-points", type=int, default=80, help="Control point count")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Validate ONNX vs PyTorch")
    parser.add_argument(
        "--max-pixel-diff-threshold",
        type=float,
        default=1.0,
        help="Validation threshold; expected max pixel diff < 1",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run CPU benchmark")
    parser.add_argument("--benchmark-iterations", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()

    if args.num_points < 3:
        raise ValueError("--num-points must be >= 3")

    image_np, src_points_np, dst_points_np = make_dummy_inputs(
        image_size=args.image_size,
        num_points=args.num_points,
        seed=args.seed,
    )
    weights_x_np, weights_y_np = compute_tps_weights(src_points_np, dst_points_np)

    onnx_path = export_tps_onnx(
        output_path=Path(args.output),
        image_size=args.image_size,
        opset=args.opset,
        image_np=image_np,
        control_points_np=src_points_np,
        weights_x_np=weights_x_np,
        weights_y_np=weights_y_np,
    )

    summary: dict[str, str | float] = {"onnx_path": str(onnx_path)}

    torch_output = run_pytorch(
        image_np=image_np,
        control_points_np=src_points_np,
        weights_x_np=weights_x_np,
        weights_y_np=weights_y_np,
        image_size=args.image_size,
    )
    onnx_output = run_onnx(
        onnx_path=onnx_path,
        image_np=image_np,
        control_points_np=src_points_np,
        weights_x_np=weights_x_np,
        weights_y_np=weights_y_np,
    )

    max_pixel_diff = compute_max_pixel_diff(torch_output, onnx_output)
    summary["max_pixel_diff"] = round(max_pixel_diff, 6)

    if args.validate and max_pixel_diff >= args.max_pixel_diff_threshold:
        raise RuntimeError(
            "Validation failed: "
            f"max pixel diff {max_pixel_diff:.6f} >= threshold {args.max_pixel_diff_threshold:.6f}"
        )

    if args.benchmark:
        summary.update(
            benchmark_cpu(
                onnx_path=onnx_path,
                image_np=image_np,
                control_points_np=src_points_np,
                weights_x_np=weights_x_np,
                weights_y_np=weights_y_np,
                image_size=args.image_size,
                iterations=args.benchmark_iterations,
            )
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
