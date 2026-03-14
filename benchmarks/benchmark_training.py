"""Benchmark training loop throughput."""

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(
        f"Benchmarking training ({args.num_steps} steps, batch {args.batch_size}, {args.device})..."
    )

    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install training deps: pip install -e '.[train]'")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # simulate training loop timing without actual model loading
    # measures the overhead of data loading, tensor ops, and gradient steps
    print("Running synthetic training loop...")

    # dummy tensors matching training shapes
    latent_shape = (args.batch_size, 4, 64, 64)
    cond_shape = (args.batch_size, 3, 512, 512)

    step_times = []
    for step in range(args.num_steps):
        start = time.perf_counter()

        # simulate forward pass tensors
        latents = torch.randn(latent_shape, device=device, dtype=dtype)
        _cond = torch.randn(cond_shape, device=device, dtype=dtype)
        noise = torch.randn_like(latents)

        # simulate loss computation
        loss = torch.nn.functional.mse_loss(latents + noise, latents)
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{args.num_steps} - {elapsed * 1000:.1f}ms/step")

    print(f"\nResults (batch_size={args.batch_size}):")
    print(f"  Mean: {np.mean(step_times) * 1000:.1f} ms/step")
    print(f"  Median: {np.median(step_times) * 1000:.1f} ms/step")
    print(f"  Throughput: {1 / np.mean(step_times):.1f} steps/sec")
    print(f"  Throughput: {args.batch_size / np.mean(step_times):.1f} images/sec")


if __name__ == "__main__":
    main()
