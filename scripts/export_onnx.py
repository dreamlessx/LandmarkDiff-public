"""Export LandmarkDiff ControlNet to ONNX for optimized inference.

Exports the fine-tuned ControlNet model to ONNX format with optional
INT8/FP16 quantization. Includes benchmark comparison between PyTorch
and ONNX inference latency.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/phaseA/latest
    python scripts/export_onnx.py --checkpoint path/to/model --quantize fp16
    python scripts/export_onnx.py --checkpoint path/to/model --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_controlnet_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    quantize: str | None = None,
) -> Path:
    """Export ControlNet to ONNX format.

    Args:
        checkpoint_path: Path to the ControlNet checkpoint directory or file.
        output_path: Output ONNX file path.
        opset_version: ONNX opset version.
        quantize: Quantization mode ("fp16", "int8", or None).

    Returns:
        Path to the exported ONNX file.
    """
    import torch

    print(f"Loading ControlNet from {checkpoint_path}...")

    try:
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"Failed to load ControlNet: {e}")
        print("Creating a minimal ControlNet for export demonstration...")
        from diffusers import ControlNetModel
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float32,
        )

    controlnet.eval()
    device = torch.device("cpu")
    controlnet = controlnet.to(device)

    # Prepare dummy inputs matching ControlNet's forward signature
    batch_size = 1
    height = 64  # Latent space height (512/8)
    width = 64   # Latent space width (512/8)
    channels = 4  # Latent channels

    dummy_sample = torch.randn(batch_size, channels, height, width, device=device)
    dummy_timestep = torch.tensor([999], device=device, dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)
    dummy_controlnet_cond = torch.randn(batch_size, 3, 512, 512, device=device)

    # Trace the model
    print("Tracing model for ONNX export...")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            controlnet,
            (dummy_sample, dummy_timestep, dummy_encoder_hidden_states, dummy_controlnet_cond),
            str(output_file),
            opset_version=opset_version,
            input_names=["sample", "timestep", "encoder_hidden_states", "controlnet_cond"],
            output_names=[f"down_block_{i}" for i in range(13)] + ["mid_block"],
            dynamic_axes={
                "sample": {0: "batch_size"},
                "encoder_hidden_states": {0: "batch_size"},
                "controlnet_cond": {0: "batch_size"},
            },
        )
        print(f"ONNX model exported to: {output_file}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nNote: ControlNet ONNX export requires careful handling of")
        print("attention layers. Consider using optimum-diffusers instead.")
        print("\nAlternative: pip install optimum[onnxruntime]")
        print("  from optimum.onnxruntime import ORTStableDiffusionPipeline")
        return output_file

    # Apply quantization if requested
    if quantize and output_file.exists():
        print(f"\nApplying {quantize} quantization...")
        output_file = _apply_quantization(output_file, quantize)

    # Report file size
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")

    return output_file


def _apply_quantization(onnx_path: Path, mode: str) -> Path:
    """Apply quantization to an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        mode: "fp16" or "int8".

    Returns:
        Path to the quantized model.
    """
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType

        if mode == "fp16":
            from onnxruntime.transformers import float16
            model = onnx.load(str(onnx_path))
            model_fp16 = float16.convert_float_to_float16(model)
            output_path = onnx_path.with_suffix(".fp16.onnx")
            onnx.save(model_fp16, str(output_path))
            print(f"FP16 model saved to: {output_path}")
            return output_path

        elif mode == "int8":
            output_path = onnx_path.with_suffix(".int8.onnx")
            quantize_dynamic(
                str(onnx_path),
                str(output_path),
                weight_type=QuantType.QInt8,
            )
            print(f"INT8 model saved to: {output_path}")
            return output_path

        else:
            print(f"Unknown quantization mode: {mode}")
            return onnx_path

    except ImportError as e:
        print(f"Quantization requires additional packages: {e}")
        print("Install with: pip install onnxruntime onnx")
        return onnx_path


def benchmark_comparison(
    checkpoint_path: str,
    onnx_path: str | None = None,
    num_iterations: int = 10,
) -> dict:
    """Benchmark PyTorch vs ONNX inference latency.

    Args:
        checkpoint_path: Path to PyTorch ControlNet checkpoint.
        onnx_path: Path to ONNX model (optional).
        num_iterations: Number of warmup + benchmark iterations.

    Returns:
        Dictionary with benchmark results.
    """
    import torch

    results = {}

    # --- PyTorch benchmark ---
    print("\nBenchmarking PyTorch inference...")
    try:
        from diffusers import ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            checkpoint_path, torch_dtype=torch.float32,
        )
        controlnet.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controlnet = controlnet.to(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Dummy inputs
        sample = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        timestep = torch.tensor([500], device=device)
        encoder_states = torch.randn(1, 77, 768, device=device, dtype=dtype)
        cond = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                controlnet(sample, timestep, encoder_states, cond)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                controlnet(sample, timestep, encoder_states, cond)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        results["pytorch"] = {
            "mean_ms": round(np.mean(times) * 1000, 2),
            "std_ms": round(np.std(times) * 1000, 2),
            "device": str(device),
        }
        print(f"  PyTorch: {results['pytorch']['mean_ms']:.2f} ± "
              f"{results['pytorch']['std_ms']:.2f} ms ({device})")

    except Exception as e:
        print(f"  PyTorch benchmark failed: {e}")
        results["pytorch"] = {"error": str(e)}

    # --- ONNX benchmark ---
    if onnx_path and Path(onnx_path).exists():
        print("Benchmarking ONNX inference...")
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)

            sample_np = np.random.randn(1, 4, 64, 64).astype(np.float32)
            timestep_np = np.array([500], dtype=np.int64)
            encoder_np = np.random.randn(1, 77, 768).astype(np.float32)
            cond_np = np.random.randn(1, 3, 512, 512).astype(np.float32)

            inputs = {
                "sample": sample_np,
                "timestep": timestep_np,
                "encoder_hidden_states": encoder_np,
                "controlnet_cond": cond_np,
            }

            # Warmup
            for _ in range(3):
                session.run(None, inputs)

            # Benchmark
            times = []
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                session.run(None, inputs)
                times.append(time.perf_counter() - t0)

            results["onnx"] = {
                "mean_ms": round(np.mean(times) * 1000, 2),
                "std_ms": round(np.std(times) * 1000, 2),
                "provider": session.get_providers()[0],
            }
            print(f"  ONNX: {results['onnx']['mean_ms']:.2f} ± "
                  f"{results['onnx']['std_ms']:.2f} ms "
                  f"({results['onnx']['provider']})")

            if "pytorch" in results and "mean_ms" in results["pytorch"]:
                speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
                print(f"  Speedup: {speedup:.2f}x")
                results["speedup"] = round(speedup, 2)

        except ImportError:
            print("  ONNX Runtime not installed. pip install onnxruntime-gpu")
        except Exception as e:
            print(f"  ONNX benchmark failed: {e}")
            results["onnx"] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Export LandmarkDiff to ONNX")
    parser.add_argument("--checkpoint", required=True,
                        help="ControlNet checkpoint path")
    parser.add_argument("--output", default="exports/controlnet.onnx",
                        help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--quantize", choices=["fp16", "int8"],
                        help="Quantization mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparison")
    parser.add_argument("--benchmark-iterations", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("LandmarkDiff ONNX Export")
    print("=" * 60)

    output_path = export_controlnet_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        quantize=args.quantize,
    )

    if args.benchmark:
        onnx_file = str(output_path) if output_path.exists() else None
        results = benchmark_comparison(
            args.checkpoint,
            onnx_path=onnx_file,
            num_iterations=args.benchmark_iterations,
        )

        print("\n--- Benchmark Summary ---")
        import json
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
