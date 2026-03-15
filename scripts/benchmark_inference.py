#!/usr/bin/env python3
"""Benchmark LandmarkDiff inference pipeline by stage and mode.

Profiles each pipeline stage independently (landmark extraction, deformation,
conditioning, diffusion, post-processing) and compares the four inference
modes (tps, img2img, controlnet, controlnet_ip).

Outputs results as a markdown table and JSON file.

Usage:
    # TPS-only benchmark (no GPU needed)
    python scripts/benchmark_inference.py --modes tps --repeats 20

    # Full benchmark across all modes
    python scripts/benchmark_inference.py \
        --modes tps img2img controlnet controlnet_ip \
        --repeats 10 --output results/benchmark

    # With a real face image
    python scripts/benchmark_inference.py --input data/faces_all/000001.png
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing / memory utilities
# ---------------------------------------------------------------------------


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def cpu_rss_mb() -> float:
    """Current process RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def gpu_mem_mb() -> tuple[float, float]:
    """Return (allocated_mb, peak_mb) for CUDA. (0, 0) if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            return alloc, peak
    except ImportError:
        pass
    return 0.0, 0.0


def reset_gpu_stats():
    """Reset CUDA peak memory statistics."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Stage profilers
# ---------------------------------------------------------------------------


def _warmup_and_time(fn, repeats: int) -> dict:
    """Run fn once for warmup, then time `repeats` iterations."""
    fn()  # warmup
    times = []
    for _ in range(repeats):
        with Timer() as t:
            fn()
        times.append(t.elapsed * 1000)  # ms
    arr = np.array(times)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "repeats": repeats,
    }


def profile_stages(image: np.ndarray, repeats: int = 10) -> dict:
    """Profile individual pipeline stages (no diffusion)."""
    from landmarkdiff.conditioning import generate_conditioning
    from landmarkdiff.landmarks import extract_landmarks
    from landmarkdiff.manipulation import apply_procedure_preset
    from landmarkdiff.masking import generate_surgical_mask
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    results = {}

    # 1. Landmark extraction
    logger.info("  Profiling landmark extraction...")
    results["landmark_extraction"] = _warmup_and_time(lambda: extract_landmarks(image), repeats)

    face = extract_landmarks(image)
    if face is None:
        logger.warning("  No face detected, cannot profile remaining stages")
        return results

    # 2. Manipulation (per procedure)
    logger.info("  Profiling manipulation...")
    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    manip_results = {}
    for proc in procedures:
        manip_results[proc] = _warmup_and_time(
            lambda _p=proc: apply_procedure_preset(face, _p, 50.0, image_size=512),
            repeats,
        )
    results["manipulation"] = manip_results

    # 3. Conditioning
    logger.info("  Profiling conditioning...")
    results["conditioning"] = _warmup_and_time(
        lambda: generate_conditioning(face, 512, 512), repeats
    )

    # 4. Masking
    logger.info("  Profiling masking...")
    mask_results = {}
    for proc in procedures:
        mask_results[proc] = _warmup_and_time(
            lambda _p=proc: generate_surgical_mask(face, _p, 512, 512), repeats
        )
    results["masking"] = mask_results

    # 5. TPS warp
    logger.info("  Profiling TPS warp...")
    manip = apply_procedure_preset(face, "rhinoplasty", 50.0, image_size=512)
    results["tps_warp"] = _warmup_and_time(
        lambda: warp_image_tps(image, face.pixel_coords, manip.pixel_coords),
        repeats,
    )

    # 6. Mask composite
    logger.info("  Profiling mask composite...")
    from landmarkdiff.inference import mask_composite

    mask = generate_surgical_mask(face, "rhinoplasty", 512, 512)
    warped = warp_image_tps(image, face.pixel_coords, manip.pixel_coords)
    results["mask_composite"] = _warmup_and_time(
        lambda: mask_composite(warped, image, mask), repeats
    )

    return results


def profile_mode(
    image: np.ndarray,
    mode: str,
    repeats: int = 5,
) -> dict:
    """Profile end-to-end inference for a given mode."""
    from landmarkdiff.inference import LandmarkDiffPipeline

    logger.info("  Loading pipeline (mode=%s)...", mode)
    reset_gpu_stats()

    try:
        pipe = LandmarkDiffPipeline(mode=mode)
        pipe.load()
    except Exception as e:
        return {"mode": mode, "error": str(e)}

    _, load_peak = gpu_mem_mb()

    # Warmup
    logger.info("  Warmup (%s)...", mode)
    try:
        pipe.generate(image, procedure="rhinoplasty", intensity=50.0, seed=42)
    except Exception as e:
        return {"mode": mode, "error": f"warmup failed: {e}"}

    # Timed runs
    logger.info("  Benchmarking (%s, %d runs)...", mode, repeats)
    times = []
    for i in range(repeats):
        reset_gpu_stats()
        with Timer() as t:
            pipe.generate(
                image,
                procedure="rhinoplasty",
                intensity=50.0,
                seed=42 + i,
                postprocess=(mode != "tps"),
            )
        times.append(t.elapsed * 1000)
        logger.info("    [%d/%d] %.0f ms", i + 1, repeats, times[-1])

    _, inference_peak = gpu_mem_mb()
    arr = np.array(times)

    result = {
        "mode": mode,
        "mean_ms": round(float(np.mean(arr)), 1),
        "std_ms": round(float(np.std(arr)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "throughput_fps": round(1000.0 / float(np.mean(arr)), 2),
        "gpu_load_peak_mb": round(load_peak, 1),
        "gpu_inference_peak_mb": round(inference_peak, 1),
        "cpu_rss_mb": round(cpu_rss_mb(), 1),
        "repeats": repeats,
    }

    # Clean up
    del pipe
    gc.collect()
    reset_gpu_stats()

    return result


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_stage_table(stages: dict) -> str:
    """Format per-stage results as a markdown table."""
    lines = [
        "## Per-Stage Latency",
        "",
        "| Stage | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |",
        "|-------|-----------|----------|----------|----------|",
    ]

    simple_keys = ["landmark_extraction", "conditioning", "tps_warp", "mask_composite"]
    for key in simple_keys:
        if key in stages:
            s = stages[key]
            label = key.replace("_", " ").title()
            lines.append(
                f"| {label} | {s['mean_ms']:.2f} | {s['std_ms']:.2f} | "
                f"{s['min_ms']:.2f} | {s['max_ms']:.2f} |"
            )

    # Per-procedure stages
    for stage_key in ["manipulation", "masking"]:
        if stage_key in stages and isinstance(stages[stage_key], dict):
            for proc, s in stages[stage_key].items():
                label = f"{stage_key.title()} ({proc})"
                lines.append(
                    f"| {label} | {s['mean_ms']:.2f} | {s['std_ms']:.2f} | "
                    f"{s['min_ms']:.2f} | {s['max_ms']:.2f} |"
                )

    return "\n".join(lines)


def format_mode_table(mode_results: list[dict]) -> str:
    """Format mode comparison as a markdown table."""
    lines = [
        "## End-to-End Inference by Mode",
        "",
        "| Mode | Mean (ms) | Median (ms) | FPS | GPU Peak (MB) | CPU RSS (MB) |",
        "|------|-----------|-------------|-----|---------------|--------------|",
    ]

    for r in mode_results:
        if "error" in r:
            lines.append(f"| {r['mode']} | ERROR: {r['error']} | | | | |")
        else:
            lines.append(
                f"| {r['mode']} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | "
                f"{r['throughput_fps']:.2f} | {r['gpu_inference_peak_mb']:.0f} | "
                f"{r['cpu_rss_mb']:.0f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_synthetic_face() -> np.ndarray:
    """Create a simple synthetic face image for benchmarking."""
    img = np.full((512, 512, 3), 180, dtype=np.uint8)
    cv2.ellipse(img, (256, 256), (140, 180), 0, 0, 360, (140, 160, 190), -1)
    cv2.circle(img, (200, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (312, 220), 15, (50, 50, 50), -1)
    pts = np.array([[256, 250], [240, 300], [272, 300]], np.int32)
    cv2.fillPoly(img, [pts], (130, 150, 180))
    cv2.ellipse(img, (256, 340), (40, 15), 0, 0, 360, (100, 120, 170), -1)
    noise = np.random.default_rng(42).integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Benchmark LandmarkDiff inference pipeline")
    parser.add_argument("--input", default=None, help="Input face image path")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["tps"],
        choices=["tps", "img2img", "controlnet", "controlnet_ip"],
        help="Inference modes to benchmark",
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Number of timed iterations per stage"
    )
    parser.add_argument("--output", default="results/benchmark_inference", help="Output directory")
    parser.add_argument(
        "--skip-stages",
        action="store_true",
        help="Skip per-stage profiling, only do end-to-end",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create image
    if args.input:
        image = cv2.imread(args.input)
        if image is None:
            logger.error("Cannot read %s", args.input)
            sys.exit(1)
        image = cv2.resize(image, (512, 512))
        logger.info("Using image: %s", args.input)
    else:
        image = create_synthetic_face()
        logger.info("Using synthetic test image")

    all_results: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Per-stage profiling
    if not args.skip_stages:
        logger.info("\n=== Per-Stage Profiling ===")
        stage_results = profile_stages(image, repeats=args.repeats)
        all_results["stages"] = stage_results
        stage_md = format_stage_table(stage_results)
        logger.info("")
        logger.info(stage_md)
    else:
        stage_md = ""

    # End-to-end mode comparison
    logger.info("\n=== End-to-End Mode Comparison ===")
    mode_results = []
    for mode in args.modes:
        r = profile_mode(image, mode, repeats=args.repeats)
        mode_results.append(r)

    all_results["modes"] = mode_results
    mode_md = format_mode_table(mode_results)
    logger.info("")
    logger.info(mode_md)

    # Memory summary
    mem_summary = {
        "cpu_rss_mb": round(cpu_rss_mb(), 1),
        "gpu_allocated_mb": round(gpu_mem_mb()[0], 1),
        "gpu_peak_mb": round(gpu_mem_mb()[1], 1),
    }
    all_results["memory"] = mem_summary

    # Save JSON
    json_path = output_dir / "benchmark_inference.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nJSON results: %s", json_path)

    # Save markdown
    md_path = output_dir / "benchmark_inference.md"
    md_content = "\n\n".join(
        filter(
            None,
            [
                "# LandmarkDiff Inference Benchmark",
                f"Date: {all_results['timestamp']}",
                stage_md,
                mode_md,
                f"## Memory\n\n- CPU RSS: {mem_summary['cpu_rss_mb']} MB",
                f"- GPU peak: {mem_summary['gpu_peak_mb']} MB",
            ],
        )
    )
    md_path.write_text(md_content)
    logger.info("Markdown report: %s", md_path)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
