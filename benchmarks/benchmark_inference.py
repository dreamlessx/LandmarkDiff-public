#!/usr/bin/env python3
"""Benchmark inference pipeline speed.

Measures per-image generation latency across different inference modes.
Results can be saved to a JSON file for tracking over time.

Usage:
    python benchmarks/benchmark_inference.py --num_images 10 --mode tps --device cpu
    python benchmarks/benchmark_inference.py --num_images 50 --mode controlnet --output results/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for inference benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark LandmarkDiff inference pipeline speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="controlnet",
        choices=["tps", "img2img", "controlnet", "controlnet_ip"],
        help="Inference mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (square)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict | None:
    """Run the inference benchmark and return results dict."""
    logger.info(
        "Benchmarking inference (%s, %d steps, %s)...",
        args.mode,
        args.steps,
        args.device,
    )

    try:
        from landmarkdiff.inference import LandmarkDiffPipeline

        pipeline = LandmarkDiffPipeline(mode=args.mode, device=args.device)
        pipeline.load()
    except Exception as e:
        logger.error("Could not load pipeline: %s", e)
        logger.error("Make sure you have the required model weights cached")
        return None

    res = args.resolution

    # Warm-up
    logger.info("Warming up (%d iterations)...", args.warmup)
    for _ in range(args.warmup):
        dummy = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)
        with contextlib.suppress(Exception):
            pipeline.generate(dummy, procedure="rhinoplasty", intensity=50)

    # Benchmark
    times: list[float] = []
    for i in range(args.num_images):
        img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)

        start = time.perf_counter()
        try:
            pipeline.generate(img, procedure="rhinoplasty", intensity=50)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            logger.info("  [%d/%d] %.2fs", i + 1, args.num_images, elapsed)
        except Exception as e:
            logger.warning("  [%d/%d] Failed: %s", i + 1, args.num_images, e)

    if not times:
        logger.warning("No successful runs")
        return None

    results = format_results(times, args)
    print_results(results)

    if args.output:
        save_results(results, args.output)

    return results


def format_results(times: list[float], args: argparse.Namespace) -> dict:
    """Format timing data into a results dictionary."""
    return {
        "benchmark": "inference",
        "mode": args.mode,
        "device": args.device,
        "steps": args.steps,
        "resolution": args.resolution,
        "num_images": len(times),
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "std_s": float(np.std(times)),
        "throughput_ips": float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
    }


def print_results(results: dict) -> None:
    """Print formatted results to stdout."""
    logger.info("")
    logger.info("Results (%s mode, %d steps):", results["mode"], results["steps"])
    logger.info("  Mean:       %.2fs", results["mean_s"])
    logger.info("  Median:     %.2fs", results["median_s"])
    logger.info("  Min:        %.2fs", results["min_s"])
    logger.info("  Max:        %.2fs", results["max_s"])
    logger.info("  Std:        %.2fs", results["std_s"])
    logger.info("  Throughput: %.2f images/sec", results["throughput_ips"])


def save_results(results: dict, output_dir: str) -> None:
    """Save results to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "benchmark_inference.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", filepath)


def main(argv: list[str] | None = None) -> int:
    """Entry point for inference benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_benchmark(args)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
