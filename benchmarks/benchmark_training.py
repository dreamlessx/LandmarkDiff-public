#!/usr/bin/env python3
"""Benchmark training loop throughput.

Measures synthetic training step latency using dummy tensors that match
the shapes used during ControlNet fine-tuning. Does not require actual
model weights -- it measures tensor operation overhead.

Usage:
    python benchmarks/benchmark_training.py --device cuda --num_steps 100
    python benchmarks/benchmark_training.py --device cpu --num_steps 20 \
        --batch_size 1 --output results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for training benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark LandmarkDiff training loop throughput",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run training on",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps to simulate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per step",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print progress every N steps",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict | None:
    """Run the training benchmark and return results dict."""
    logger.info(
        "Benchmarking training (%d steps, batch %d, %s)...",
        args.num_steps,
        args.batch_size,
        args.device,
    )

    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return None

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA not available, skipping training benchmark")
        return None

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info("Running synthetic training loop...")

    # Dummy tensors matching training shapes
    latent_shape = (args.batch_size, 4, 64, 64)
    cond_shape = (args.batch_size, 3, 512, 512)

    step_times: list[float] = []
    for step in range(args.num_steps):
        start = time.perf_counter()

        # Simulate forward pass tensors
        latents = torch.randn(latent_shape, device=device, dtype=dtype)
        _cond = torch.randn(cond_shape, device=device, dtype=dtype)
        noise = torch.randn_like(latents)

        # Simulate loss computation
        loss = torch.nn.functional.mse_loss(latents + noise, latents)
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

        if (step + 1) % args.log_interval == 0:
            logger.info(
                "  Step %d/%d - %.1fms/step",
                step + 1,
                args.num_steps,
                elapsed * 1000,
            )

    results = format_results(step_times, args)
    print_results(results)

    if args.output:
        save_results(results, args.output)

    return results


def format_results(step_times: list[float], args: argparse.Namespace) -> dict:
    """Format timing data into a results dictionary."""
    mean_t = float(np.mean(step_times))
    return {
        "benchmark": "training",
        "device": args.device,
        "batch_size": args.batch_size,
        "num_steps": len(step_times),
        "mean_ms": mean_t * 1000,
        "median_ms": float(np.median(step_times)) * 1000,
        "std_ms": float(np.std(step_times)) * 1000,
        "min_ms": float(np.min(step_times)) * 1000,
        "max_ms": float(np.max(step_times)) * 1000,
        "throughput_steps_per_sec": 1.0 / mean_t if mean_t > 0 else 0.0,
        "throughput_images_per_sec": (args.batch_size / mean_t if mean_t > 0 else 0.0),
    }


def print_results(results: dict) -> None:
    """Print formatted results to stdout."""
    logger.info("")
    logger.info("Results (batch_size=%d):", results["batch_size"])
    logger.info("  Mean:       %.1f ms/step", results["mean_ms"])
    logger.info("  Median:     %.1f ms/step", results["median_ms"])
    logger.info("  Std:        %.1f ms", results["std_ms"])
    logger.info("  Min:        %.1f ms", results["min_ms"])
    logger.info("  Max:        %.1f ms", results["max_ms"])
    logger.info("  Throughput: %.1f steps/sec", results["throughput_steps_per_sec"])
    logger.info("  Throughput: %.1f images/sec", results["throughput_images_per_sec"])


def save_results(results: dict, output_dir: str) -> None:
    """Save results to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "benchmark_training.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", filepath)


def main(argv: list[str] | None = None) -> int:
    """Entry point for training benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(args)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
