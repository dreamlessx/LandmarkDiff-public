#!/usr/bin/env python3
"""Benchmark landmark extraction speed.

Measures per-image MediaPipe landmark extraction latency.
Results can be saved to a JSON file for tracking over time.

Usage:
    python benchmarks/benchmark_landmarks.py --num_images 100
    python benchmarks/benchmark_landmarks.py --num_images 500 --resolution 1024 --output results/
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
    """Build argument parser for landmark benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark LandmarkDiff landmark extraction speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to process",
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
        "--log_interval",
        type=int,
        default=10,
        help="Print progress every N images",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict:
    """Run the landmark extraction benchmark and return results dict."""
    from landmarkdiff.landmarks import extract_landmarks

    res = args.resolution
    logger.info(
        "Benchmarking landmark extraction (%d images, %dx%d)...",
        args.num_images,
        res,
        res,
    )

    times: list[float] = []
    detections: int = 0

    for i in range(args.num_images):
        # Random noise images -- MediaPipe may not detect faces, but we measure speed
        img = np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)

        start = time.perf_counter()
        result = extract_landmarks(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if result is not None:
            detections += 1

        if (i + 1) % args.log_interval == 0:
            avg_ms = np.mean(times) * 1000
            logger.info(
                "  %d/%d - avg: %.1fms/image",
                i + 1,
                args.num_images,
                avg_ms,
            )

    results = format_results(times, detections, args)
    print_results(results)

    if args.output:
        save_results(results, args.output)

    return results


def format_results(times: list[float], detections: int, args: argparse.Namespace) -> dict:
    """Format timing data into a results dictionary."""
    mean_t = float(np.mean(times))
    return {
        "benchmark": "landmarks",
        "resolution": args.resolution,
        "num_images": len(times),
        "detections": detections,
        "detection_rate": detections / len(times) if times else 0.0,
        "mean_ms": mean_t * 1000,
        "median_ms": float(np.median(times)) * 1000,
        "std_ms": float(np.std(times)) * 1000,
        "min_ms": float(np.min(times)) * 1000,
        "max_ms": float(np.max(times)) * 1000,
        "throughput_ips": 1.0 / mean_t if mean_t > 0 else 0.0,
    }


def print_results(results: dict) -> None:
    """Print formatted results to stdout."""
    logger.info("")
    logger.info("Results (%dx%d):", results["resolution"], results["resolution"])
    logger.info("  Mean:       %.1f ms/image", results["mean_ms"])
    logger.info("  Median:     %.1f ms/image", results["median_ms"])
    logger.info("  Std:        %.1f ms", results["std_ms"])
    logger.info("  Min:        %.1f ms", results["min_ms"])
    logger.info("  Max:        %.1f ms", results["max_ms"])
    logger.info("  Throughput: %.1f images/sec", results["throughput_ips"])
    logger.info(
        "  Detections: %d/%d (%.0f%%)",
        results["detections"],
        results["num_images"],
        results["detection_rate"] * 100,
    )


def save_results(results: dict, output_dir: str) -> None:
    """Save results to a JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "benchmark_landmarks.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", filepath)


def main(argv: list[str] | None = None) -> int:
    """Entry point for landmark benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
