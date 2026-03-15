"""Profile LandmarkDiff pipeline stages for latency and memory usage.

Measures inference latency per pipeline stage, peak GPU/CPU memory,
and data loading throughput. Outputs a structured performance report.

Usage:
    python scripts/profile_pipeline.py --synthetic
    python scripts/profile_pipeline.py IMAGE
    python scripts/profile_pipeline.py --data-dir data/training_combined --batch-size 4
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Timing and memory utilities
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


class MemoryTracker:
    """Track peak CPU and GPU memory usage."""

    def __init__(self):
        self.cpu_peak_mb: float = 0.0
        self.gpu_peak_mb: float = 0.0
        self.gpu_allocated_mb: float = 0.0
        self._has_torch = False
        self._has_gpu = False

        try:
            import torch

            self._has_torch = True
            self._has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

    def snapshot(self) -> dict:
        """Take a memory snapshot."""
        import resource

        cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB

        gpu_mb = 0.0
        gpu_alloc_mb = 0.0
        if self._has_gpu:
            import torch

            gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
            gpu_alloc_mb = torch.cuda.memory_allocated() / (1024**2)

        self.cpu_peak_mb = max(self.cpu_peak_mb, cpu_mb)
        self.gpu_peak_mb = max(self.gpu_peak_mb, gpu_mb)
        self.gpu_allocated_mb = gpu_alloc_mb

        return {
            "cpu_peak_mb": round(cpu_mb, 1),
            "gpu_peak_mb": round(gpu_mb, 1),
            "gpu_current_mb": round(gpu_alloc_mb, 1),
        }

    def reset_gpu(self):
        if self._has_gpu:
            import torch

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()


# ---------------------------------------------------------------------------
# Stage profilers
# ---------------------------------------------------------------------------


def profile_landmark_extraction(image: np.ndarray, repeats: int = 5) -> dict:
    """Profile MediaPipe landmark extraction."""
    from landmarkdiff.landmarks import extract_landmarks

    # Warmup
    extract_landmarks(image)

    times = []
    for _ in range(repeats):
        t = Timer()
        with t:
            face = extract_landmarks(image)
        times.append(t.elapsed)

    return {
        "name": "Landmark Extraction",
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "min_ms": round(np.min(times) * 1000, 2),
        "max_ms": round(np.max(times) * 1000, 2),
        "repeats": repeats,
        "face_detected": face is not None,
    }


def profile_manipulation(face, repeats: int = 10) -> dict:
    """Profile landmark manipulation."""
    from landmarkdiff.manipulation import apply_procedure_preset

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    results = {}

    for proc in procedures:
        # Warmup
        apply_procedure_preset(face, proc, 50.0, image_size=512)

        times = []
        for _ in range(repeats):
            t = Timer()
            with t:
                apply_procedure_preset(face, proc, 50.0, image_size=512)
            times.append(t.elapsed)

        results[proc] = {
            "mean_ms": round(np.mean(times) * 1000, 3),
            "std_ms": round(np.std(times) * 1000, 3),
        }

    return {
        "name": "Landmark Manipulation",
        "per_procedure": results,
        "repeats": repeats,
    }


def profile_conditioning(face, image: np.ndarray, repeats: int = 10) -> dict:
    """Profile conditioning image generation."""
    from landmarkdiff.conditioning import generate_conditioning

    # Warmup
    generate_conditioning(face, image.shape[1], image.shape[0])

    times = []
    for _ in range(repeats):
        t = Timer()
        with t:
            generate_conditioning(face, image.shape[1], image.shape[0])
        times.append(t.elapsed)

    return {
        "name": "Conditioning Generation",
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "repeats": repeats,
    }


def profile_masking(face, repeats: int = 10) -> dict:
    """Profile surgical mask generation."""
    from landmarkdiff.masking import generate_surgical_mask

    procedures = [
        "rhinoplasty",
        "blepharoplasty",
        "rhytidectomy",
        "orthognathic",
        "brow_lift",
        "mentoplasty",
    ]
    results = {}

    for proc in procedures:
        # Warmup
        generate_surgical_mask(face, proc)

        times = []
        for _ in range(repeats):
            t = Timer()
            with t:
                generate_surgical_mask(face, proc)
            times.append(t.elapsed)

        results[proc] = {
            "mean_ms": round(np.mean(times) * 1000, 3),
        }

    return {
        "name": "Mask Generation",
        "per_procedure": results,
        "repeats": repeats,
    }


def profile_tps_warp(image: np.ndarray, face, repeats: int = 3) -> dict:
    """Profile TPS warp."""
    from landmarkdiff.synthetic.tps_warp import warp_image_tps

    src = face.pixel_coords
    dst = src + np.random.default_rng(42).standard_normal(src.shape).astype(np.float32) * 1.5

    # Warmup
    warp_image_tps(image, src, dst)

    times = []
    for _ in range(repeats):
        t = Timer()
        with t:
            warp_image_tps(image, src, dst)
        times.append(t.elapsed)

    return {
        "name": "TPS Warp",
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "repeats": repeats,
    }


def profile_dataloader(data_dir: str, batch_size: int = 4, num_workers: int = 4) -> dict:
    """Profile data loading throughput."""
    try:
        from landmarkdiff.data import SurgicalPairDataset, create_dataloader

        ds = SurgicalPairDataset(data_dir, resolution=512)
        loader = create_dataloader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

        # Warmup: 1 batch
        first_batch = next(iter(loader))

        # Measure throughput
        n_batches = min(20, len(loader))
        t = Timer()
        with t:
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break

        images_per_sec = (n_batches * batch_size) / t.elapsed
        ms_per_batch = (t.elapsed / n_batches) * 1000

        return {
            "name": "Data Loading",
            "images_per_sec": round(images_per_sec, 1),
            "ms_per_batch": round(ms_per_batch, 2),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "total_samples": len(ds),
            "batches_measured": n_batches,
        }
    except Exception as e:
        return {
            "name": "Data Loading",
            "error": str(e),
        }


def profile_safety_validation(image: np.ndarray, repeats: int = 5) -> dict:
    """Profile safety validation."""
    from landmarkdiff.safety import SafetyValidator

    validator = SafetyValidator()
    output = (image.astype(float) * 0.95).astype(np.uint8)

    # Warmup
    validator.validate(input_image=image, output_image=output, face_confidence=0.95)

    times = []
    for _ in range(repeats):
        t = Timer()
        with t:
            validator.validate(input_image=image, output_image=output, face_confidence=0.95)
        times.append(t.elapsed)

    return {
        "name": "Safety Validation",
        "mean_ms": round(np.mean(times) * 1000, 2),
        "std_ms": round(np.std(times) * 1000, 2),
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def format_report(results: list[dict], mem: MemoryTracker) -> str:
    """Format profiling results into a readable report."""
    lines = [
        "=" * 70,
        "LandmarkDiff Pipeline Performance Profile",
        "=" * 70,
        "",
    ]

    total_ms = 0.0

    for r in results:
        name = r["name"]
        lines.append(f"--- {name} ---")

        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
            lines.append("")
            continue

        if "mean_ms" in r:
            lines.append(f"  Mean: {r['mean_ms']:.2f} ms  (std: {r.get('std_ms', 0):.2f} ms)")
            total_ms += r["mean_ms"]

        if "per_procedure" in r:
            for proc, data in r["per_procedure"].items():
                lines.append(f"  {proc:<20s}: {data['mean_ms']:.3f} ms")
                total_ms += data["mean_ms"]

        if "images_per_sec" in r:
            lines.append(f"  Throughput: {r['images_per_sec']:.1f} images/sec")
            lines.append(f"  Per batch:  {r['ms_per_batch']:.2f} ms ({r['batch_size']} images)")
            lines.append(f"  Workers:    {r['num_workers']}")
            lines.append(f"  Dataset:    {r['total_samples']} samples")

        for key in ["face_detected", "repeats"]:
            if key in r:
                lines.append(f"  {key}: {r[key]}")

        lines.append("")

    # Memory summary
    mem_snap = mem.snapshot()
    lines.append("--- Memory Usage ---")
    lines.append(f"  CPU peak:         {mem_snap['cpu_peak_mb']:.1f} MB")
    if mem_snap["gpu_peak_mb"] > 0:
        lines.append(f"  GPU peak:         {mem_snap['gpu_peak_mb']:.1f} MB")
        lines.append(f"  GPU current:      {mem_snap['gpu_current_mb']:.1f} MB")
    lines.append("")

    # Total
    lines.append("--- Summary ---")
    lines.append(f"  Total pipeline latency (single image): ~{total_ms:.1f} ms")
    lines.append(f"  Estimated throughput: ~{1000 / max(total_ms, 1):.1f} images/sec")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_synthetic_image() -> np.ndarray:
    """Create a synthetic face-like test image."""
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
    parser = argparse.ArgumentParser(description="Profile LandmarkDiff pipeline")
    parser.add_argument("image", nargs="?", help="Input image path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test image")
    parser.add_argument("--data-dir", default=None, help="Data directory for loader profiling")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=None, help="Save report to file")
    args = parser.parse_args()

    # Load image
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Cannot read {args.image}")
            sys.exit(1)
        image = cv2.resize(image, (512, 512))
    else:
        image = create_synthetic_image()
        print("Using synthetic test image")

    mem = MemoryTracker()
    mem.reset_gpu()
    results = []

    # Profile each stage
    print("Profiling pipeline stages...")

    try:
        r = profile_landmark_extraction(image, repeats=args.repeats)
        results.append(r)
        print(f"  Landmarks: {r['mean_ms']:.2f} ms")
    except Exception as e:
        results.append({"name": "Landmark Extraction", "error": str(e)})

    # Get face for subsequent stages
    from landmarkdiff.landmarks import extract_landmarks

    face = extract_landmarks(image)

    if face is not None:
        try:
            r = profile_manipulation(face, repeats=args.repeats * 2)
            results.append(r)
            print("  Manipulation: done")
        except Exception as e:
            results.append({"name": "Landmark Manipulation", "error": str(e)})

        try:
            r = profile_conditioning(face, image, repeats=args.repeats * 2)
            results.append(r)
            print(f"  Conditioning: {r['mean_ms']:.2f} ms")
        except Exception as e:
            results.append({"name": "Conditioning Generation", "error": str(e)})

        try:
            r = profile_masking(face, repeats=args.repeats * 2)
            results.append(r)
            print("  Masking: done")
        except Exception as e:
            results.append({"name": "Mask Generation", "error": str(e)})

        try:
            r = profile_tps_warp(image, face, repeats=args.repeats)
            results.append(r)
            print(f"  TPS Warp: {r['mean_ms']:.2f} ms")
        except Exception as e:
            results.append({"name": "TPS Warp", "error": str(e)})

    try:
        r = profile_safety_validation(image, repeats=args.repeats)
        results.append(r)
        print(f"  Safety: {r['mean_ms']:.2f} ms")
    except Exception as e:
        results.append({"name": "Safety Validation", "error": str(e)})

    # Data loading (if dir specified)
    if args.data_dir:
        print(f"  Profiling data loading from {args.data_dir}...")
        r = profile_dataloader(args.data_dir, args.batch_size, args.num_workers)
        results.append(r)
        if "images_per_sec" in r:
            print(f"  DataLoader: {r['images_per_sec']:.1f} img/sec")

    # Generate report
    report = format_report(results, mem)
    print()
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
