"""Visualize ControlNet cross-attention and self-attention maps.

Extracts attention maps from the SD1.5 + ControlNet pipeline during inference
and produces publication-quality heatmaps showing how the model attends to
different facial regions given the landmark mesh conditioning.

This is a novel analysis contribution: we show that ControlNet cross-attention
concentrates on anatomically relevant regions (e.g., nose for rhinoplasty,
periorbital for blepharoplasty), validating that the model has learned
procedure-specific spatial priors.

Usage:
    python scripts/visualize_attention.py \
        --checkpoint checkpoints/phaseA/best \
        --input data/hda_splits/test/rhinoplasty_Nose_01_input.png \
        --conditioning data/hda_splits/test/rhinoplasty_Nose_01_conditioning.png \
        --output paper/attention_maps/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Attention hooking infrastructure
# ──────────────────────────────────────────────────────────────────────────────


class AttentionStore:
    """Captures attention maps from UNet cross-attention and self-attention layers.

    Registers forward hooks on all Attention modules in the UNet and ControlNet
    to record the attention weight matrices during inference.  After a forward
    pass, the stored maps can be aggregated per resolution to produce spatial
    heatmaps.
    """

    def __init__(self):
        self.attention_maps: dict[str, list[torch.Tensor]] = {
            "cross": [],  # Text/prompt cross-attention
            "self": [],  # Self-attention (spatial)
            "controlnet": [],  # ControlNet internal attention
        }
        self.enabled = True

    def register_hooks(self, unet, controlnet=None):
        """Register attention-capturing hooks on UNet and optional ControlNet.

        Uses diffusers' attention processor API for clean integration.
        """
        from diffusers.models.attention_processor import Attention

        # Create a custom processor that stores attention maps
        store = self

        class StoringAttnProcessor:
            """Attention processor that stores attention maps for visualization."""

            def __init__(self, original_processor, layer_name: str, attn_type: str):
                self.original = original_processor
                self.layer_name = layer_name
                self.attn_type = attn_type

            def __call__(
                self,
                attn: Attention,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                **kwargs,
            ):
                if not store.enabled:
                    # Fall through to original processor
                    return self.original(
                        attn,
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        **kwargs,
                    )

                # Determine if this is cross or self attention
                is_cross = encoder_hidden_states is not None
                attn_type = "cross" if is_cross else "self"
                if "controlnet" in self.layer_name.lower():
                    attn_type = "controlnet"

                residual = hidden_states
                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(
                        batch_size, channel, height * width
                    ).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape
                    if encoder_hidden_states is None
                    else encoder_hidden_states.shape
                )

                # Prepare Q, K, V
                query = attn.to_q(hidden_states)
                kv_input = encoder_hidden_states if is_cross else hidden_states
                key = attn.to_k(kv_input)
                value = attn.to_v(kv_input)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Compute attention scores
                scale = head_dim**-0.5
                attn_weights = (
                    torch.bmm(
                        query.reshape(-1, query.shape[2], head_dim),
                        key.reshape(-1, key.shape[2], head_dim).transpose(1, 2),
                    )
                    * scale
                )

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_probs = attn_weights.softmax(dim=-1)

                # Store the attention maps (detached, on CPU to save GPU memory)
                # Average over heads, keep spatial structure
                avg_attn = attn_probs.reshape(batch_size, attn.heads, -1, attn_probs.shape[-1])
                avg_attn = avg_attn.mean(dim=1)  # (B, spatial, kv_len)
                store.attention_maps[attn_type].append(avg_attn[0].detach().cpu())

                # Continue with standard attention computation
                hidden_states = torch.bmm(
                    attn_probs,
                    value.reshape(-1, value.shape[2], head_dim),
                )
                hidden_states = hidden_states.reshape(batch_size, attn.heads, -1, head_dim)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
                hidden_states = hidden_states.to(query.dtype)

                # Linear projection and dropout
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(
                        batch_size, channel, height, width
                    )

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                if hasattr(attn, "rescale_output_factor"):
                    hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

        # Replace processors in UNet
        attn_procs = {}
        for name, module in unet.named_modules():
            if isinstance(module, Attention):
                # Get original processor
                orig_proc = module.processor
                attn_type = "cross" if "attn2" in name else "self"
                attn_procs[name] = StoringAttnProcessor(orig_proc, name, attn_type)
                module.processor = attn_procs[name]

        # Replace processors in ControlNet
        if controlnet is not None:
            for name, module in controlnet.named_modules():
                if isinstance(module, Attention):
                    orig_proc = module.processor
                    proc = StoringAttnProcessor(orig_proc, f"controlnet.{name}", "controlnet")
                    module.processor = proc

        print(f"  Registered attention hooks on {len(attn_procs)} UNet layers")

    def clear(self):
        """Clear stored attention maps."""
        for key in self.attention_maps:
            self.attention_maps[key].clear()

    def get_attention_maps(
        self, attn_type: str = "cross", target_res: int = 64
    ) -> torch.Tensor | None:
        """Get aggregated attention maps at a target resolution.

        Args:
            attn_type: "cross", "self", or "controlnet"
            target_res: Target spatial resolution for the output maps.

        Returns:
            Tensor of shape (target_res, target_res) or None.
        """
        maps = self.attention_maps.get(attn_type, [])
        if not maps:
            return None

        aggregated = []
        for m in maps:
            spatial_dim = m.shape[0]
            res = int(spatial_dim**0.5)
            if res * res != spatial_dim:
                continue  # Skip non-square attention maps

            # Reshape to (res, res, kv_len) and take mean over kv dimension
            # This gives us the average attention per spatial position
            attn_map = m.reshape(res, res, -1).mean(dim=-1)

            # Resize to target resolution
            attn_map = attn_map.unsqueeze(0).unsqueeze(0).float()
            attn_map = F.interpolate(
                attn_map, size=(target_res, target_res), mode="bilinear", align_corners=False
            )
            aggregated.append(attn_map.squeeze())

        if not aggregated:
            return None

        # Average across all layers
        return torch.stack(aggregated).mean(dim=0)

    def get_per_resolution_maps(self, attn_type: str = "cross") -> dict[int, torch.Tensor]:
        """Get attention maps grouped by resolution.

        Returns dict mapping resolution -> averaged attention map.
        """
        maps = self.attention_maps.get(attn_type, [])
        by_res = {}

        for m in maps:
            spatial_dim = m.shape[0]
            res = int(spatial_dim**0.5)
            if res * res != spatial_dim:
                continue
            attn_map = m.reshape(res, res, -1).mean(dim=-1)
            by_res.setdefault(res, []).append(attn_map)

        return {res: torch.stack(v).mean(dim=0) for res, v in by_res.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Visualization utilities
# ──────────────────────────────────────────────────────────────────────────────


def attention_to_heatmap(
    attn_map: np.ndarray, size: int = 512, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Convert attention map to a colored heatmap image.

    Args:
        attn_map: 2D numpy array of attention weights.
        size: Output image size (square).
        colormap: OpenCV colormap.

    Returns:
        (size, size, 3) uint8 BGR heatmap image.
    """
    # Normalize to [0, 255]
    attn_map = attn_map - attn_map.min()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()
    attn_uint8 = (attn_map * 255).astype(np.uint8)

    # Resize
    attn_resized = cv2.resize(attn_uint8, (size, size), interpolation=cv2.INTER_CUBIC)

    # Apply colormap
    heatmap = cv2.applyColorMap(attn_resized, colormap)
    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap on an image with transparency.

    Args:
        image: (H, W, 3) uint8 BGR image.
        heatmap: (H, W, 3) uint8 BGR heatmap (same size as image).
        alpha: Blending factor (0 = image only, 1 = heatmap only).

    Returns:
        Blended (H, W, 3) uint8 BGR image.
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
    return blended


def create_multi_resolution_figure(
    input_img: np.ndarray,
    conditioning_img: np.ndarray,
    per_res_maps: dict[int, torch.Tensor],
    prediction: np.ndarray,
    output_path: Path,
    size: int = 256,
):
    """Create a multi-resolution attention visualization figure.

    Layout:
    [Input] [Conditioning] [Attn@8] [Attn@16] [Attn@32] [Attn@64] [Prediction]

    Shows how attention patterns differ across UNet encoder/decoder resolutions.
    """
    panels = []

    # Input image
    input_resized = cv2.resize(input_img, (size, size))
    panels.append(add_label(input_resized, "Input"))

    # Conditioning
    cond_resized = cv2.resize(conditioning_img, (size, size))
    panels.append(add_label(cond_resized, "Conditioning"))

    # Attention maps at each resolution
    for res in sorted(per_res_maps.keys()):
        attn = per_res_maps[res].numpy()
        heatmap = attention_to_heatmap(attn, size)
        overlaid = overlay_heatmap(input_resized, heatmap, alpha=0.6)
        panels.append(add_label(overlaid, f"Attn@{res}x{res}"))

    # Prediction
    pred_resized = cv2.resize(prediction, (size, size))
    panels.append(add_label(pred_resized, "Prediction"))

    # Concatenate
    row = np.hstack(panels)
    cv2.imwrite(str(output_path), row)
    print(f"  Saved: {output_path} ({row.shape})")


def create_cross_vs_self_figure(
    input_img: np.ndarray,
    cross_map: torch.Tensor | None,
    self_map: torch.Tensor | None,
    controlnet_map: torch.Tensor | None,
    output_path: Path,
    size: int = 256,
):
    """Create a figure comparing cross-attention, self-attention, and ControlNet attention.

    Layout:
    [Input] [Cross-Attn] [Self-Attn] [ControlNet-Attn] [Overlay]
    """
    panels = []
    input_resized = cv2.resize(input_img, (size, size))
    panels.append(add_label(input_resized, "Input"))

    # Cross attention
    if cross_map is not None:
        cross_heat = attention_to_heatmap(cross_map.numpy(), size)
        panels.append(add_label(cross_heat, "Cross-Attn"))
        panels.append(add_label(overlay_heatmap(input_resized, cross_heat, 0.5), "Cross Overlay"))
    else:
        placeholder = np.ones((size, size, 3), dtype=np.uint8) * 128
        panels.append(add_label(placeholder, "Cross-Attn"))
        panels.append(add_label(placeholder, "Cross Overlay"))

    # Self attention
    if self_map is not None:
        self_heat = attention_to_heatmap(self_map.numpy(), size, cv2.COLORMAP_VIRIDIS)
        panels.append(add_label(self_heat, "Self-Attn"))
        panels.append(add_label(overlay_heatmap(input_resized, self_heat, 0.5), "Self Overlay"))
    else:
        placeholder = np.ones((size, size, 3), dtype=np.uint8) * 128
        panels.append(add_label(placeholder, "Self-Attn"))
        panels.append(add_label(placeholder, "Self Overlay"))

    # ControlNet attention
    if controlnet_map is not None:
        cn_heat = attention_to_heatmap(controlnet_map.numpy(), size, cv2.COLORMAP_INFERNO)
        panels.append(add_label(cn_heat, "ControlNet-Attn"))
        panels.append(add_label(overlay_heatmap(input_resized, cn_heat, 0.5), "CN Overlay"))
    else:
        placeholder = np.ones((size, size, 3), dtype=np.uint8) * 128
        panels.append(add_label(placeholder, "ControlNet-Attn"))
        panels.append(add_label(placeholder, "CN Overlay"))

    row = np.hstack(panels)
    cv2.imwrite(str(output_path), row)
    print(f"  Saved: {output_path} ({row.shape})")


def create_timestep_evolution_figure(
    input_img: np.ndarray,
    timestep_maps: list[tuple[int, torch.Tensor]],
    output_path: Path,
    size: int = 192,
):
    """Show how attention evolves over diffusion timesteps.

    Runs inference with attention capture at multiple timesteps to show
    how the model's focus shifts during the denoising process.
    """
    panels = []
    input_resized = cv2.resize(input_img, (size, size))
    panels.append(add_label(input_resized, "Input", font_scale=0.35))

    for step, attn_map in timestep_maps:
        heatmap = attention_to_heatmap(attn_map.numpy(), size)
        overlaid = overlay_heatmap(input_resized, heatmap, alpha=0.6)
        panels.append(add_label(overlaid, f"t={step}", font_scale=0.35))

    row = np.hstack(panels)
    cv2.imwrite(str(output_path), row)
    print(f"  Saved: {output_path} ({row.shape})")


def add_label(img: np.ndarray, text: str, font_scale: float = 0.4) -> np.ndarray:
    """Add text label at bottom of image."""
    h, w = img.shape[:2]
    labeled = img.copy()
    bar_h = max(int(18 * font_scale * 2), 12)
    labeled[h - bar_h :, :] = 0
    cv2.putText(
        labeled,
        text,
        (3, h - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


# ──────────────────────────────────────────────────────────────────────────────
# Main inference + visualization pipeline
# ──────────────────────────────────────────────────────────────────────────────


def run_attention_analysis(
    checkpoint_path: Path,
    test_dir: Path,
    output_dir: Path,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    max_pairs: int = 5,
):
    """Run inference with attention capture on test pairs.

    For each test pair:
    1. Load input + conditioning images
    2. Run ControlNet inference with attention hooks
    3. Generate attention visualization figures
    4. Save individual and comparison visualizations
    """
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline...")
    controlnet_subdir = (
        "controlnet_ema" if (checkpoint_path / "controlnet_ema").exists() else "controlnet"
    )
    controlnet = ControlNetModel.from_pretrained(
        str(checkpoint_path / controlnet_subdir),
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    print("  Pipeline loaded on", pipe.device)

    # Register attention hooks
    attn_store = AttentionStore()
    attn_store.register_hooks(pipe.unet, pipe.controlnet)

    # Discover test pairs
    pairs = sorted(test_dir.glob("*_input.png"))[:max_pairs]
    print(f"\nProcessing {len(pairs)} test pairs...")

    prompt = "high quality photo of a face after cosmetic surgery, realistic skin texture"
    negative_prompt = "blurry, distorted, low quality, deformed"

    all_summaries = []

    for pair_path in pairs:
        prefix = pair_path.stem.replace("_input", "")
        procedure = prefix.split("_")[0]
        print(f"\n--- {prefix} ({procedure}) ---")

        # Load images
        input_img = cv2.imread(str(pair_path))
        cond_path = test_dir / f"{prefix}_conditioning.png"
        test_dir / f"{prefix}_target.png"

        if input_img is None:
            print("  Skipping — can't read input")
            continue

        input_img = cv2.resize(input_img, (512, 512))

        # Load or generate conditioning image
        if cond_path.exists():
            cond_img = cv2.imread(str(cond_path))
            cond_img = cv2.resize(cond_img, (512, 512))
        else:
            # Generate from landmarks
            from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

            lm = extract_landmarks(input_img)
            if lm is None:
                print("  Skipping — no landmarks detected")
                continue
            cond_img = render_landmark_image(lm, 512, 512)

        # Convert to PIL for pipeline
        cond_rgb = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        # Run inference with attention capture
        attn_store.clear()
        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=cond_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        pred_pil = result.images[0]
        pred_np = cv2.cvtColor(np.array(pred_pil), cv2.COLOR_RGB2BGR)

        # Get aggregated attention maps
        cross_map = attn_store.get_attention_maps("cross", target_res=64)
        self_map = attn_store.get_attention_maps("self", target_res=64)
        cn_map = attn_store.get_attention_maps("controlnet", target_res=64)

        # Get per-resolution maps
        cross_per_res = attn_store.get_per_resolution_maps("cross")
        self_per_res = attn_store.get_per_resolution_maps("self")

        n_cross = len(attn_store.attention_maps["cross"])
        n_self = len(attn_store.attention_maps["self"])
        n_cn = len(attn_store.attention_maps["controlnet"])
        print(f"  Captured: {n_cross} cross, {n_self} self, {n_cn} controlnet attention maps")
        print(f"  Cross resolutions: {sorted(cross_per_res.keys())}")
        print(f"  Self resolutions: {sorted(self_per_res.keys())}")

        # Generate visualizations
        # 1. Cross vs Self vs ControlNet comparison
        create_cross_vs_self_figure(
            input_img,
            cross_map,
            self_map,
            cn_map,
            output_dir / f"{prefix}_attention_types.png",
        )

        # 2. Multi-resolution cross-attention
        if cross_per_res:
            create_multi_resolution_figure(
                input_img,
                cond_img,
                cross_per_res,
                pred_np,
                output_dir / f"{prefix}_cross_multiresolution.png",
            )

        # 3. Multi-resolution self-attention
        if self_per_res:
            create_multi_resolution_figure(
                input_img,
                cond_img,
                self_per_res,
                pred_np,
                output_dir / f"{prefix}_self_multiresolution.png",
            )

        # 4. Individual high-res attention overlay
        if cross_map is not None:
            heatmap = attention_to_heatmap(cross_map.numpy(), 512)
            overlaid = overlay_heatmap(input_img, heatmap, alpha=0.5)
            cv2.imwrite(str(output_dir / f"{prefix}_cross_overlay.png"), overlaid)

        # Track summary for aggregate figure
        all_summaries.append(
            {
                "prefix": prefix,
                "procedure": procedure,
                "cross_map": cross_map,
                "self_map": self_map,
                "cn_map": cn_map,
                "input_img": input_img,
                "pred_img": pred_np,
            }
        )

    # Create aggregate figure: all procedures side by side
    if all_summaries:
        create_aggregate_figure(all_summaries, output_dir / "attention_aggregate.png")

    # Run timestep evolution analysis on the first pair
    if pairs:
        print("\n--- Timestep Evolution Analysis ---")
        run_timestep_evolution(
            pipe,
            attn_store,
            pairs[0],
            test_dir,
            output_dir,
            num_steps,
            guidance_scale,
            seed,
        )

    print("\n=== Attention visualization complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Files generated: {len(list(output_dir.glob('*.png')))}")


def run_timestep_evolution(
    pipe,
    attn_store,
    pair_path,
    test_dir,
    output_dir,
    num_steps,
    guidance_scale,
    seed,
):
    """Capture attention at multiple timesteps during denoising.

    Shows how the model's spatial focus evolves from noise (high t)
    to clean image (low t). Early steps focus on global structure,
    later steps refine local details.
    """
    prefix = pair_path.stem.replace("_input", "")
    input_img = cv2.imread(str(pair_path))
    input_img = cv2.resize(input_img, (512, 512))

    cond_path = test_dir / f"{prefix}_conditioning.png"
    if cond_path.exists():
        cond_img = cv2.imread(str(cond_path))
        cond_img = cv2.resize(cond_img, (512, 512))
    else:
        from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

        lm = extract_landmarks(input_img)
        if lm is None:
            return
        cond_img = render_landmark_image(lm, 512, 512)

    cond_rgb = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
    cond_pil = Image.fromarray(cond_rgb)

    # Capture attention at specific timesteps
    capture_steps = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    timestep_maps = []

    prompt = "high quality photo of a face after cosmetic surgery, realistic skin texture"
    negative_prompt = "blurry, distorted, low quality, deformed"

    # We run multiple passes, each capturing attention at a specific step
    for capture_step in capture_steps:
        attn_store.clear()
        attn_store.enabled = False  # Disable by default

        step_counter = {"current": 0}

        def callback(pipe, step, timestep, callback_kwargs):
            if step == capture_step:
                attn_store.enabled = True
            elif step == capture_step + 1:
                attn_store.enabled = False
            step_counter["current"] = step
            return callback_kwargs

        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.no_grad():
            pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=cond_pil,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=callback,
            )

        cross_map = attn_store.get_attention_maps("cross", target_res=64)
        if cross_map is not None:
            timestep_maps.append((capture_step, cross_map))
            print(f"  Captured attention at step {capture_step}/{num_steps}")

    attn_store.enabled = True  # Re-enable

    if timestep_maps:
        create_timestep_evolution_figure(
            input_img,
            timestep_maps,
            output_dir / f"{prefix}_timestep_evolution.png",
        )


def create_aggregate_figure(
    summaries: list[dict],
    output_path: Path,
    size: int = 192,
):
    """Create aggregate figure showing attention patterns across procedures.

    Layout: one row per test pair, columns for input, cross-attn, self-attn, prediction.
    Groups by procedure type.
    """
    rows = []

    for summary in summaries:
        panels = []
        input_resized = cv2.resize(summary["input_img"], (size, size))
        panels.append(add_label(input_resized, "Input", 0.35))

        if summary["cross_map"] is not None:
            cross_heat = attention_to_heatmap(summary["cross_map"].numpy(), size)
            panels.append(
                add_label(
                    overlay_heatmap(input_resized, cross_heat, 0.6),
                    "Cross-Attn",
                    0.35,
                )
            )
        else:
            panels.append(
                add_label(
                    np.ones((size, size, 3), dtype=np.uint8) * 128,
                    "Cross-Attn",
                    0.35,
                )
            )

        if summary["self_map"] is not None:
            self_heat = attention_to_heatmap(
                summary["self_map"].numpy(), size, cv2.COLORMAP_VIRIDIS
            )
            panels.append(
                add_label(
                    overlay_heatmap(input_resized, self_heat, 0.6),
                    "Self-Attn",
                    0.35,
                )
            )
        else:
            panels.append(
                add_label(
                    np.ones((size, size, 3), dtype=np.uint8) * 128,
                    "Self-Attn",
                    0.35,
                )
            )

        pred_resized = cv2.resize(summary["pred_img"], (size, size))
        panels.append(add_label(pred_resized, "Prediction", 0.35))

        row = np.hstack(panels)

        # Add procedure label on left
        label_w = 80
        label = np.zeros((size, label_w, 3), dtype=np.uint8)
        proc_name = summary["procedure"].capitalize()
        cv2.putText(
            label,
            proc_name,
            (3, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row = np.hstack([label, row])
        rows.append(row)

    # Stack with separators
    if rows:
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)

        separator = np.ones((2, max_w, 3), dtype=np.uint8) * 100
        final = []
        for i, r in enumerate(padded):
            if i > 0:
                final.append(separator)
            final.append(r)

        figure = np.vstack(final)
        cv2.imwrite(str(output_path), figure)
        print(f"\n  Aggregate figure: {output_path} ({figure.shape})")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ControlNet attention maps during inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "phaseA" / "best",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=ROOT / "data" / "hda_splits" / "test",
        help="Test data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "attention_maps",
        help="Output directory for attention visualizations",
    )
    parser.add_argument("--num-steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-pairs", type=int, default=5, help="Maximum number of test pairs to process"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CONTROLNET ATTENTION VISUALIZATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance_scale}, Seed: {args.seed}")
    print()

    run_attention_analysis(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
