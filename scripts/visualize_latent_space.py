"""Latent space visualization via t-SNE/UMAP.

Generates a 2D embedding of the ControlNet latent space colored by
procedure type. This reveals whether the model has learned to separate
different surgical procedures in its internal representation.

Usage:
    python scripts/visualize_latent_space.py \
        --checkpoint checkpoints/phaseB/best \
        --test-dir data/hda_splits/test \
        --output paper/latent_tsne.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCEDURE_COLORS = {
    "rhinoplasty": "#1f77b4",  # blue
    "blepharoplasty": "#ff7f0e",  # orange
    "rhytidectomy": "#2ca02c",  # green
    "orthognathic": "#d62728",  # red
    "unknown": "#7f7f7f",  # gray
}


def extract_latent(pipe, conditioning_pil, seed=42, num_steps=20, return_step="final"):
    """Extract the latent representation at a given diffusion step.

    Returns the latent tensor (1, 4, 64, 64) flattened to a feature vector.
    """
    from diffusers import UniPCMultistepScheduler

    device = pipe.device
    dtype = pipe.unet.dtype

    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    prompt = "photo of a face after cosmetic surgery"
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)

    cond_tensor = pipe.prepare_image(
        conditioning_pil,
        width=512,
        height=512,
        batch_size=1,
        num_images_per_prompt=1,
        device=device,
        dtype=pipe.controlnet.dtype,
    )

    # Run a few steps (not full inference — we want intermediate latent)
    target_step = min(5, num_steps)  # Early step captures more structural info
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            scaled = scheduler.scale_model_input(latents, t)
            down_samples, mid_sample = pipe.controlnet(
                scaled,
                t,
                encoder_hidden_states=text_emb,
                controlnet_cond=cond_tensor,
                return_dict=False,
            )
            noise_pred = pipe.unet(
                scaled,
                t,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if i + 1 >= target_step:
                break

    return latents.flatten().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Latent space t-SNE visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default="data/hda_splits/test")
    parser.add_argument("--output", type=str, default="paper/latent_tsne.png")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap", "pca"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("Loading pipeline...")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    ckpt = Path(args.checkpoint)
    if (ckpt / "controlnet_ema").exists():
        ckpt = ckpt / "controlnet_ema"

    controlnet = ControlNetModel.from_pretrained(str(ckpt))
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Load test pairs
    test_dir = Path(args.test_dir)
    input_files = sorted(test_dir.glob("*_input.png"))
    if args.max_pairs:
        input_files = input_files[: args.max_pairs]

    latents = []
    procedures = []
    labels = []

    for i, inp_file in enumerate(input_files):
        prefix = inp_file.stem.replace("_input", "")
        cond_file = test_dir / f"{prefix}_conditioning.png"

        if not cond_file.exists():
            continue

        # Determine procedure
        proc = "unknown"
        for p in PROCEDURE_COLORS:
            if p in prefix:
                proc = p
                break

        conditioning = cv2.resize(cv2.imread(str(cond_file)), (512, 512))
        cond_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        print(f"  [{i + 1}/{len(input_files)}] {prefix} ({proc})")

        latent_vec = extract_latent(pipe, cond_pil, seed=args.seed)
        latents.append(latent_vec)
        procedures.append(proc)
        labels.append(prefix)

    if len(latents) < 5:
        print("Not enough samples for visualization")
        return

    X = np.array(latents)
    print(f"\nFeature matrix: {X.shape}")

    # Dimensionality reduction
    if args.method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(X) - 1))
        embedding = reducer.fit_transform(X)
    elif args.method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        embedding = reducer.fit_transform(X)
    else:  # PCA
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=args.seed)
        embedding = reducer.fit_transform(X)

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for proc in sorted(set(procedures)):
        mask = [p == proc for p in procedures]
        pts = embedding[mask]
        color = PROCEDURE_COLORS.get(proc, "#7f7f7f")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            label=proc,
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
        )

    method_name = {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}[args.method]
    ax.set_title(f"ControlNet Latent Space ({method_name})", fontsize=14)
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.legend(title="Procedure", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # Also save PCA variance explained if using PCA
    if args.method == "pca":
        print(f"Variance explained: {reducer.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
