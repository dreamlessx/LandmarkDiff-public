"""Benchmark inference pipeline speed."""

import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="controlnet")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    print(f"Benchmarking inference ({args.mode}, {args.steps} steps, {args.device})...")

    try:
        from landmarkdiff.inference import LandmarkDiffPipeline
        pipeline = LandmarkDiffPipeline.from_pretrained("checkpoints/latest", device=args.device)
    except Exception as e:
        print(f"Could not load pipeline: {e}")
        print("Make sure you have a checkpoint at checkpoints/latest")
        return

    # Warm-up
    print("Warming up...")
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        pipeline.generate(dummy, procedure="rhinoplasty", intensity=0.5, mode=args.mode)
    except Exception:
        pass

    # Benchmark
    times = []
    for i in range(args.num_images):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        start = time.perf_counter()
        try:
            pipeline.generate(img, procedure="rhinoplasty", intensity=0.5, mode=args.mode)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  [{i+1}/{args.num_images}] {elapsed:.2f}s")
        except Exception as e:
            print(f"  [{i+1}/{args.num_images}] Failed: {e}")

    if times:
        print(f"\nResults ({args.mode} mode, {args.steps} steps):")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")
        print(f"  Min: {np.min(times):.2f}s")
        print(f"  Max: {np.max(times):.2f}s")


if __name__ == "__main__":
    main()
