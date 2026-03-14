"""Benchmark landmark extraction speed."""

import argparse
import time

import numpy as np

from landmarkdiff.landmarks import extract_landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    # Create synthetic test images
    res = args.resolution
    print(f"Benchmarking landmark extraction ({args.num_images} images, {res}x{res})...")

    times = []
    for i in range(args.num_images):
        # Use random noise images (MediaPipe may not detect faces, but we're measuring speed)
        img = np.random.randint(0, 255, (args.resolution, args.resolution, 3), dtype=np.uint8)

        start = time.perf_counter()
        _ = extract_landmarks(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.num_images} - avg: {np.mean(times) * 1000:.1f}ms/image")

    print("\nResults:")
    print(f"  Mean: {np.mean(times) * 1000:.1f} ms/image")
    print(f"  Median: {np.median(times) * 1000:.1f} ms/image")
    print(f"  Std: {np.std(times) * 1000:.1f} ms")
    print(f"  Throughput: {1 / np.mean(times):.1f} images/sec")


if __name__ == "__main__":
    main()
