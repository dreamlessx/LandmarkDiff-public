"""Test pre-trained CrucibleAI ControlNet vs fine-tuned on our mesh conditioning.

Generates side-by-side comparisons to see if fine-tuning helps or hurts.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

from landmarkdiff.landmarks import extract_landmarks, render_landmark_image


def test_model(controlnet, label, test_images, output_dir):
    """Generate from a ControlNet model and save results."""
    print(f"\n=== Testing: {label} ===")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = (
        "high quality professional photo of a person's face,"
        " natural skin texture, photorealistic, 8k"
    )
    neg = "blurry, distorted, deformed, bad anatomy, low quality, watermark, text"

    for i, img_path in enumerate(test_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))

        face = extract_landmarks(img)
        if face is None:
            continue

        mesh = render_landmark_image(face, 512, 512)
        mesh_pil = Image.fromarray(mesh[:, :, ::-1])

        gen = torch.Generator(device="cpu").manual_seed(42)
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            image=mesh_pil,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=gen,
        ).images[0]

        gen_np = np.array(result)[:, :, ::-1]
        comparison = np.hstack([mesh, gen_np, img])
        cv2.imwrite(str(output_dir / f"{label}_sample_{i}.png"), comparison)
        print(f"  Saved {label}_sample_{i}.png")

        if i >= 5:
            break

    del pipe
    torch.cuda.empty_cache()


def main(
    checkpoint_dir: str = "checkpoints_v2",
    test_image_dir: str = "data/ffhq",
    output_dir: str = "checkpoints_v2/comparison",
    device: str = "cuda",
    num_test_images: int = 8,
):
    """Test pre-trained vs fine-tuned ControlNet models.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        test_image_dir: Directory with test images
        output_dir: Directory to save comparison results
        device: Device to use (cuda or cpu)
        num_test_images: Number of test images to use
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Test images
    test_images = sorted(Path(test_image_dir).glob("*.png"))[:num_test_images]
    if not test_images:
        test_images = sorted(Path(test_image_dir).glob("*.jpg"))[:num_test_images]

    if not test_images:
        print(f"No test images found in {test_image_dir}")
        return

    # 1. Pre-trained CrucibleAI (no fine-tuning)
    print("Loading pre-trained CrucibleAI...")
    pretrained = ControlNetModel.from_pretrained(
        "CrucibleAI/ControlNetMediaPipeFace",
        subfolder="diffusion_sd15",
        torch_dtype=torch.float16,
    )
    test_model(pretrained, "pretrained", test_images, out)
    del pretrained
    torch.cuda.empty_cache()

    # 2. Fine-tuned (our 50K step model)
    finetuned_path = Path(checkpoint_dir) / "final" / "controlnet_ema"
    if finetuned_path.exists():
        print("Loading fine-tuned model...")
        finetuned = ControlNetModel.from_pretrained(
            str(finetuned_path),
            torch_dtype=torch.float16,
        )
        test_model(finetuned, "finetuned_50k", test_images, out)
        del finetuned
        torch.cuda.empty_cache()

    # 3. Early checkpoint (10K steps - less fine-tuning)
    early_path = Path(checkpoint_dir) / "checkpoint-10000" / "controlnet_ema"
    if early_path.exists():
        print("Loading early checkpoint (10K)...")
        early = ControlNetModel.from_pretrained(
            str(early_path),
            torch_dtype=torch.float16,
        )
        test_model(early, "finetuned_10k", test_images, out)

    print(f"\nDone! Results in {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test pre-trained CrucibleAI ControlNet vs fine-tuned on our mesh conditioning."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v2",
        help="Checkpoint directory (default: checkpoints_v2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints_v2/comparison",
        help="Output directory for comparison results (default: checkpoints_v2/comparison)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of test images to use (default: 8)",
    )

    args = parser.parse_args()
    main(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        device=args.device,
        num_test_images=args.num_images,
    )
