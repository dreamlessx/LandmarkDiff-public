"""Export fine-tuned ControlNet for deployment and sharing.

Exports the EMA checkpoint to standard formats:
1. SafeTensors format (diffusers-compatible)
2. Single-file checkpoint for easy transfer
3. Inference-ready pipeline (ControlNet + SD1.5)

Usage:
    # Export from training checkpoint
    python scripts/export_model.py \\
        --checkpoint checkpoints/final/controlnet_ema \\
        --output exports/landmarkdiff_v1

    # Export with pipeline (ControlNet + SD1.5 bundled)
    python scripts/export_model.py \\
        --checkpoint checkpoints/final/controlnet_ema \\
        --output exports/landmarkdiff_v1 \\
        --include_pipeline

    # Export from training state (extracts EMA weights)
    python scripts/export_model.py \\
        --training_state checkpoints/checkpoint-50000/training_state.pt \\
        --output exports/landmarkdiff_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_from_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    include_pipeline: bool = False,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
) -> None:
    """Export a ControlNet checkpoint to deployment format."""
    from diffusers import ControlNetModel

    ckpt = Path(checkpoint_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load ControlNet
    print(f"Loading ControlNet from {ckpt}...")
    controlnet = ControlNetModel.from_pretrained(str(ckpt))

    # Save in SafeTensors format
    controlnet_out = out / "controlnet"
    controlnet.save_pretrained(controlnet_out)
    print(f"ControlNet saved to {controlnet_out}")

    # Count parameters
    total_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)

    # Create model card
    model_info = {
        "model_type": "ControlNet",
        "base_model": base_model_id,
        "conditioning": "MediaPipe FaceMesh (478 landmarks, 3-channel)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "format": "safetensors",
        "procedures": [
            "rhinoplasty",
            "blepharoplasty",
            "rhytidectomy",
            "orthognathic",
            "brow_lift",
            "mentoplasty",
        ],
        "input_resolution": 512,
    }
    with open(out / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Calculate model size
    size_mb = sum(f.stat().st_size for f in controlnet_out.rglob("*") if f.is_file()) / 1e6
    print(f"Model size: {size_mb:.1f} MB ({total_params / 1e6:.1f}M params)")

    if include_pipeline:
        _export_pipeline(controlnet, base_model_id, out)


def export_from_training_state(
    training_state_path: str,
    output_dir: str,
    controlnet_id: str = "CrucibleAI/ControlNetMediaPipeFace",
    controlnet_subfolder: str = "diffusion_sd15",
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    include_pipeline: bool = False,
) -> None:
    """Export EMA weights from a training_state.pt file."""
    from diffusers import ControlNetModel

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load training state
    print(f"Loading training state from {training_state_path}...")
    state = torch.load(training_state_path, map_location="cpu", weights_only=True)
    global_step = state.get("global_step", "unknown")
    print(f"  Training step: {global_step}")

    # Create ControlNet architecture from pretrained
    print(f"Loading ControlNet architecture from {controlnet_id}...")
    controlnet = ControlNetModel.from_pretrained(controlnet_id, subfolder=controlnet_subfolder)

    # Load EMA weights (preferred for inference)
    if "ema_controlnet" in state:
        print("Loading EMA weights...")
        controlnet.load_state_dict(state["ema_controlnet"])
    elif "controlnet" in state:
        print("Loading training weights (no EMA found)...")
        controlnet.load_state_dict(state["controlnet"])
    else:
        print("ERROR: No ControlNet weights found in training state")
        sys.exit(1)

    # Save
    controlnet_out = out / "controlnet"
    controlnet.save_pretrained(controlnet_out)
    print(f"ControlNet saved to {controlnet_out}")

    # Model info
    total_params = sum(p.numel() for p in controlnet.parameters())
    model_info = {
        "model_type": "ControlNet",
        "source": "training_state",
        "training_step": global_step,
        "weights": "ema" if "ema_controlnet" in state else "training",
        "total_parameters": total_params,
        "base_model": base_model_id,
    }
    with open(out / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    if include_pipeline:
        _export_pipeline(controlnet, base_model_id, out)


def _export_pipeline(controlnet, base_model_id: str, output_dir: Path) -> None:
    """Export complete pipeline (ControlNet + SD1.5)."""
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    print(f"Loading base model {base_model_id} for pipeline export...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    pipeline_dir = output_dir / "pipeline"
    pipe.save_pretrained(pipeline_dir)
    size_mb = sum(f.stat().st_size for f in pipeline_dir.rglob("*") if f.is_file()) / 1e6
    print(f"Full pipeline saved to {pipeline_dir} ({size_mb:.0f} MB)")


def create_inference_script(output_dir: str) -> None:
    """Create a minimal inference script alongside the exported model."""
    out = Path(output_dir)
    script = '''"""Minimal inference script for LandmarkDiff."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.inference import LandmarkDiffPipeline


def predict(image_path: str, procedure: str = "rhinoplasty", intensity: float = 50.0):
    """Run surgical outcome prediction."""
    checkpoint = str(Path(__file__).parent / "controlnet")
    pipe = LandmarkDiffPipeline(
        mode="controlnet",
        controlnet_checkpoint=checkpoint,
    )
    pipe.load()

    image = cv2.imread(image_path)
    result = pipe.generate(image, procedure=procedure, intensity=intensity, seed=42)

    cv2.imwrite("prediction.png", result["output"])
    print(f"Prediction saved to prediction.png")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input face image")
    parser.add_argument("--procedure", default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=50.0)
    args = parser.parse_args()
    predict(args.image, args.procedure, args.intensity)
'''
    with open(out / "predict.py", "w") as f:
        f.write(script)
    print(f"Inference script saved to {out / 'predict.py'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LandmarkDiff ControlNet")
    parser.add_argument(
        "--checkpoint", default=None, help="Path to ControlNet checkpoint directory"
    )
    parser.add_argument("--training_state", default=None, help="Path to training_state.pt file")
    parser.add_argument("--output", required=True, help="Output directory for exported model")
    parser.add_argument(
        "--include_pipeline", action="store_true", help="Also export full SD1.5 pipeline"
    )
    parser.add_argument(
        "--base_model", default="runwayml/stable-diffusion-v1-5", help="Base SD model ID"
    )
    args = parser.parse_args()

    if args.checkpoint:
        export_from_checkpoint(
            args.checkpoint,
            args.output,
            include_pipeline=args.include_pipeline,
            base_model_id=args.base_model,
        )
    elif args.training_state:
        export_from_training_state(
            args.training_state,
            args.output,
            include_pipeline=args.include_pipeline,
            base_model_id=args.base_model,
        )
    else:
        print("ERROR: Provide --checkpoint or --training_state")
        sys.exit(1)

    create_inference_script(args.output)
    print("\nExport complete!")
