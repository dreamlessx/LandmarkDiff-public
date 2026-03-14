"""Inference pipeline for surgical outcome prediction.

Four modes:
1. ControlNet: CrucibleAI/ControlNetMediaPipeFace + SD1.5 (requires HF auth + GPU)
2. ControlNet + IP-Adapter: ControlNet with identity preservation via face embeddings
3. Img2Img: SD1.5 img2img with mask compositing (runs on MPS, no auth needed)
4. TPS-only: Pure geometric warp — no diffusion model, instant results

Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks, render_landmark_image
from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask, mask_to_3channel
from landmarkdiff.synthetic.tps_warp import warp_image_tps


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr[:, :, ::-1].copy())


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1].copy()
    return arr


PROCEDURE_PROMPTS: dict[str, str] = {
    "rhinoplasty": (
        "clinical photograph, patient face, natural refined nose, smooth nasal bridge, "
        "realistic skin pores and texture, sharp focus, studio lighting, "
        "DSLR quality, natural skin color"
    ),
    "blepharoplasty": (
        "clinical photograph, patient face, natural eyelids, smooth periorbital area, "
        "realistic skin pores and texture, sharp focus, studio lighting, "
        "DSLR quality, natural skin color"
    ),
    "rhytidectomy": (
        "clinical photograph, patient face, defined jawline, smooth facial contour, "
        "realistic skin pores and texture, sharp focus, studio lighting, "
        "DSLR quality, natural skin color"
    ),
    "orthognathic": (
        "clinical photograph, patient face, balanced jaw and chin proportions, "
        "realistic skin pores and texture, sharp focus, studio lighting, "
        "DSLR quality, natural skin color"
    ),
}

NEGATIVE_PROMPT = (
    "painting, drawing, illustration, cartoon, anime, render, 3d, cgi, "
    "blurry, distorted, deformed, disfigured, bad anatomy, bad proportions, "
    "extra limbs, mutated, poorly drawn face, ugly, low quality, low resolution, "
    "watermark, text, signature, duplicate, artifact, noise, overexposed, "
    "plastic skin, waxy, smooth skin, airbrushed, oversaturated"
)


def mask_composite(
    warped: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    use_laplacian: bool = True,
) -> np.ndarray:
    """Composite warped image into original using ONLY the mask region.

    Uses Laplacian pyramid blending by default for seamless transitions.
    Falls back to simple alpha blend if Laplacian unavailable.
    Matches skin tone in LAB space to prevent any color shift.
    """
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0

    # Match color of warped region to original skin tone in LAB space
    corrected = _match_skin_tone(warped, original, mask_f)

    if use_laplacian:
        try:
            from landmarkdiff.postprocess import laplacian_pyramid_blend
            return laplacian_pyramid_blend(corrected, original, mask_f)
        except Exception:
            pass

    # Fallback: simple alpha blend
    mask_3ch = mask_to_3channel(mask_f)
    result = (
        corrected.astype(np.float32) * mask_3ch
        + original.astype(np.float32) * (1.0 - mask_3ch)
    ).astype(np.uint8)

    return result


def _match_skin_tone(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match source skin tone to target within mask, preserving structure.

    Works in LAB space: transfers L (luminance) and AB (color) statistics
    from the original to the warped image so skin tone is preserved exactly.
    """
    mask_bool = mask > 0.3
    if not np.any(mask_bool):
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Match each LAB channel's statistics in the mask region
    for ch in range(3):
        src_vals = src_lab[:, :, ch][mask_bool]
        tgt_vals = tgt_lab[:, :, ch][mask_bool]

        src_mean, src_std = np.mean(src_vals), np.std(src_vals) + 1e-6
        tgt_mean, tgt_std = np.mean(tgt_vals), np.std(tgt_vals) + 1e-6

        # Normalize source to match target's distribution
        src_lab[:, :, ch] = np.where(
            mask_bool,
            (src_lab[:, :, ch] - src_mean) * (tgt_std / src_std) + tgt_mean,
            src_lab[:, :, ch],
        )

    src_lab = np.clip(src_lab, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


class LandmarkDiffPipeline:
    """End-to-end pipeline: image -> landmarks -> manipulate -> generate.

    Modes:
    - 'controlnet': CrucibleAI/ControlNetMediaPipeFace + SD1.5
    - 'controlnet_ip': ControlNet + IP-Adapter for identity preservation
    - 'img2img': SD1.5 img2img with mask compositing
    - 'tps': Pure geometric TPS warp (no diffusion, instant)
    """

    # Default IP-Adapter model for SD1.5 face identity
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus-face_sd15.bin"
    IP_ADAPTER_SCALE_DEFAULT = 0.6

    def __init__(
        self,
        mode: str = "img2img",
        controlnet_id: str = "CrucibleAI/ControlNetMediaPipeFace",
        controlnet_checkpoint: str | None = None,
        base_model_id: str | None = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ip_adapter_scale: float = 0.6,
        clinical_flags: Optional["ClinicalFlags"] = None,
        displacement_model_path: str | None = None,
    ):
        self.mode = mode
        self.device = device or get_device()
        self.ip_adapter_scale = ip_adapter_scale
        self.clinical_flags = clinical_flags
        self.controlnet_checkpoint = controlnet_checkpoint

        # Load displacement model for data-driven manipulation
        self._displacement_model = None
        if displacement_model_path:
            try:
                from landmarkdiff.displacement_model import DisplacementModel
                self._displacement_model = DisplacementModel.load(displacement_model_path)
                print(f"Displacement model loaded: {self._displacement_model.procedures}")
            except Exception as e:
                print(f"WARNING: Failed to load displacement model: {e}")

        if self.device.type == "mps":
            self.dtype = torch.float32
        elif dtype:
            self.dtype = dtype
        else:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if base_model_id:
            self.base_model_id = base_model_id
        elif mode in ("controlnet", "controlnet_ip"):
            self.base_model_id = "runwayml/stable-diffusion-v1-5"
        else:
            self.base_model_id = "runwayml/stable-diffusion-v1-5"

        self.controlnet_id = controlnet_id
        self._pipe = None
        self._ip_adapter_loaded = False

    def load(self) -> None:
        if self.mode == "tps":
            print("TPS mode — no model to load")
            return
        if self.mode in ("controlnet", "controlnet_ip"):
            self._load_controlnet()
            if self.mode == "controlnet_ip":
                self._load_ip_adapter()
        else:
            self._load_img2img()

    def _load_controlnet(self) -> None:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            DPMSolverMultistepScheduler,
        )

        if self.controlnet_checkpoint:
            # Load fine-tuned ControlNet from local checkpoint
            ckpt_path = Path(self.controlnet_checkpoint)
            # Support both direct path and training checkpoint structure
            if (ckpt_path / "controlnet_ema").exists():
                ckpt_path = ckpt_path / "controlnet_ema"
            print(f"Loading fine-tuned ControlNet from {ckpt_path}...")
            controlnet = ControlNetModel.from_pretrained(
                str(ckpt_path), torch_dtype=self.dtype,
            )
        else:
            print(f"Loading ControlNet from {self.controlnet_id}...")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, subfolder="diffusion_sd15", torch_dtype=self.dtype,
            )
        print(f"Loading base model from {self.base_model_id}...")
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # DPM++ 2M Karras — produces more photorealistic output than UniPC
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        # FP32 VAE decode — prevents color banding artifacts on skin tones
        if hasattr(self._pipe, "vae") and self._pipe.vae is not None:
            self._pipe.vae.config.force_upcast = True
        self._apply_device_optimizations()

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter for identity-preserving generation.

        Uses h94/IP-Adapter-FaceID with CLIP image encoder to condition
        generation on the input face identity.
        """
        if self._pipe is None:
            raise RuntimeError("Base pipeline must be loaded before IP-Adapter")
        try:
            print(f"Loading IP-Adapter ({self.IP_ADAPTER_WEIGHT_NAME})...")
            self._pipe.load_ip_adapter(
                self.IP_ADAPTER_REPO,
                subfolder=self.IP_ADAPTER_SUBFOLDER,
                weight_name=self.IP_ADAPTER_WEIGHT_NAME,
            )
            self._pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            self._ip_adapter_loaded = True
            print(f"IP-Adapter loaded (scale={self.ip_adapter_scale})")
        except Exception as e:
            print(f"WARNING: IP-Adapter load failed: {e}")
            print("Falling back to ControlNet-only mode")
            self._ip_adapter_loaded = False

    def _load_img2img(self) -> None:
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            DPMSolverMultistepScheduler,
        )

        print(f"Loading SD1.5 img2img from {self.base_model_id}...")
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._apply_device_optimizations()

    def _apply_device_optimizations(self) -> None:
        if self.device.type == "mps":
            self._pipe = self._pipe.to(self.device)
            self._pipe.enable_attention_slicing()
        elif self.device.type == "cuda":
            try:
                self._pipe.enable_model_cpu_offload()
            except Exception:
                self._pipe = self._pipe.to(self.device)
        else:
            self._pipe.enable_sequential_cpu_offload()
        print(f"Pipeline loaded on {self.device} ({self.dtype})")

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None or self.mode == "tps"

    def generate(
        self,
        image: np.ndarray,
        procedure: str = "rhinoplasty",
        intensity: float = 50.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        controlnet_conditioning_scale: float = 0.9,
        strength: float = 0.5,
        seed: Optional[int] = None,
        clinical_flags: Optional["ClinicalFlags"] = None,
        postprocess: bool = True,
        use_gfpgan: bool = False,
    ) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        flags = clinical_flags or self.clinical_flags
        image_512 = cv2.resize(image, (512, 512))

        face = extract_landmarks(image_512)
        if face is None:
            raise ValueError("No face detected in image.")

        # Estimate face view angle for multi-view awareness
        view_info = estimate_face_view(face)

        # Use displacement model for data-driven manipulation if available
        manipulation_mode = "preset"
        if self._displacement_model and procedure in self._displacement_model.procedures:
            try:
                from landmarkdiff.displacement_model import DisplacementModel
                rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
                # Map UI intensity (0-100) to displacement model intensity (0-2)
                dm_intensity = intensity / 50.0  # 50 -> 1.0x mean displacement
                displacement = self._displacement_model.get_displacement_field(
                    procedure, intensity=dm_intensity, noise_scale=0.3, rng=rng,
                )
                # Apply displacement to landmarks
                new_lm = face.landmarks.copy()
                n = min(len(new_lm), len(displacement))
                new_lm[:n, 0] += displacement[:n, 0]
                new_lm[:n, 1] += displacement[:n, 1]
                new_lm[:, 0] = np.clip(new_lm[:, 0], 0.01, 0.99)
                new_lm[:, 1] = np.clip(new_lm[:, 1], 0.01, 0.99)
                manipulated = FaceLandmarks(
                    landmarks=new_lm,
                    image_width=512, image_height=512,
                    confidence=face.confidence,
                )
                manipulation_mode = "displacement_model"
            except Exception:
                manipulated = apply_procedure_preset(
                    face, procedure, intensity, image_size=512, clinical_flags=flags,
                )
        else:
            manipulated = apply_procedure_preset(
                face, procedure, intensity, image_size=512, clinical_flags=flags,
            )
        landmark_img = render_landmark_image(manipulated, 512, 512)
        mask = generate_surgical_mask(
            face, procedure, 512, 512, clinical_flags=flags,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt = PROCEDURE_PROMPTS.get(procedure, "a photo of a person's face")

        # Step 1: TPS geometric warp (always computed — the geometric baseline)
        tps_warped = warp_image_tps(image_512, face.pixel_coords, manipulated.pixel_coords)

        if self.mode == "tps":
            raw_output = tps_warped
        elif self.mode in ("controlnet", "controlnet_ip"):
            ip_image = numpy_to_pil(image_512) if self._ip_adapter_loaded else None
            raw_output = self._generate_controlnet(
                image_512, landmark_img, prompt, num_inference_steps,
                guidance_scale, controlnet_conditioning_scale, generator,
                ip_adapter_image=ip_image,
            )
        else:
            raw_output = self._generate_img2img(
                tps_warped, mask, prompt, num_inference_steps,
                guidance_scale, strength, generator,
            )

        # Step 2: Post-processing for photorealism (neural + classical pipeline)
        identity_check = None
        restore_used = "none"
        if postprocess and self.mode != "tps":
            from landmarkdiff.postprocess import full_postprocess
            pp_result = full_postprocess(
                generated=raw_output,
                original=image_512,
                mask=mask,
                restore_mode="codeformer" if use_gfpgan else "none",
                use_realesrgan=use_gfpgan,
                use_laplacian_blend=True,
                sharpen_strength=0.25,
                verify_identity=True,
            )
            composited = pp_result["image"]
            identity_check = pp_result["identity_check"]
            restore_used = pp_result["restore_used"]
        else:
            composited = mask_composite(raw_output, image_512, mask)

        return {
            "output": composited,
            "output_raw": raw_output,
            "output_tps": tps_warped,
            "input": image_512,
            "landmarks_original": face,
            "landmarks_manipulated": manipulated,
            "conditioning": landmark_img,
            "mask": mask,
            "procedure": procedure,
            "intensity": intensity,
            "device": str(self.device),
            "mode": self.mode,
            "view_info": view_info,
            "ip_adapter_active": self._ip_adapter_loaded,
            "identity_check": identity_check,
            "restore_used": restore_used,
            "manipulation_mode": manipulation_mode,
        }

    def _generate_controlnet(
        self, image: np.ndarray, conditioning: np.ndarray,
        prompt: str, steps: int, cfg: float, cn_scale: float,
        generator: torch.Generator | None,
        ip_adapter_image: Image.Image | None = None,
    ) -> np.ndarray:
        kwargs = dict(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=numpy_to_pil(conditioning),  # control conditioning only
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=cn_scale,
            generator=generator,
        )
        if ip_adapter_image is not None and self._ip_adapter_loaded:
            kwargs["ip_adapter_image"] = ip_adapter_image
        result = self._pipe(**kwargs)
        return pil_to_numpy(result.images[0])

    def _generate_img2img(
        self, image: np.ndarray, mask: np.ndarray,
        prompt: str, steps: int, cfg: float, strength: float,
        generator: torch.Generator | None,
    ) -> np.ndarray:
        result = self._pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=numpy_to_pil(image),
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=strength,
            generator=generator,
        )
        return pil_to_numpy(result.images[0])


def estimate_face_view(face: FaceLandmarks) -> dict:
    """Estimate face orientation from landmarks for multi-view awareness.

    Uses the nose tip (idx 1), left ear (idx 234), and right ear (idx 454) to
    estimate yaw angle. Pitch from forehead (idx 10) and chin (idx 152).

    Returns dict with yaw, pitch (degrees), and view classification.
    """
    coords = face.pixel_coords
    nose_tip = coords[1]
    left_ear = coords[234]
    right_ear = coords[454]
    forehead = coords[10]
    chin = coords[152]

    # Yaw: ratio of nose-to-ear distances (symmetric = 0 degrees)
    left_dist = np.linalg.norm(nose_tip - left_ear)
    right_dist = np.linalg.norm(nose_tip - right_ear)
    total = left_dist + right_dist
    if total < 1.0:
        yaw = 0.0
    else:
        ratio = (right_dist - left_dist) / total
        yaw = float(np.arcsin(np.clip(ratio, -1, 1)) * 180 / np.pi)

    # Pitch: nose-to-chin vs forehead-to-nose vertical ratio
    upper = np.linalg.norm(forehead - nose_tip)
    lower = np.linalg.norm(nose_tip - chin)
    if upper + lower < 1.0:
        pitch = 0.0
    else:
        pitch_ratio = (lower - upper) / (upper + lower)
        pitch = float(pitch_ratio * 45)

    # Classify view
    abs_yaw = abs(yaw)
    if abs_yaw < 15:
        view = "frontal"
    elif abs_yaw < 45:
        view = "three_quarter"
    else:
        view = "profile"

    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "view": view,
        "is_frontal": abs_yaw < 15,
        "warning": "Side-view detected: results may be less accurate" if abs_yaw > 30 else None,
    }


def run_inference(
    image_path: str,
    procedure: str = "rhinoplasty",
    intensity: float = 50.0,
    output_dir: str = "scripts/inference_output",
    seed: int = 42,
    mode: str = "img2img",
    ip_adapter_scale: float = 0.6,
    controlnet_checkpoint: str | None = None,
    displacement_model_path: str | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        sys.exit(1)

    pipe = LandmarkDiffPipeline(
        mode=mode, ip_adapter_scale=ip_adapter_scale,
        controlnet_checkpoint=controlnet_checkpoint,
        displacement_model_path=displacement_model_path,
    )
    pipe.load()

    print(f"\nGenerating {procedure} prediction (intensity={intensity}, mode={mode})...")
    result = pipe.generate(image, procedure=procedure, intensity=intensity, seed=seed)

    cv2.imwrite(str(out / "input.png"), result["input"])
    cv2.imwrite(str(out / "output.png"), result["output"])
    cv2.imwrite(str(out / "output_raw.png"), result["output_raw"])
    cv2.imwrite(str(out / "output_tps.png"), result["output_tps"])
    cv2.imwrite(str(out / "conditioning.png"), result["conditioning"])
    cv2.imwrite(str(out / "mask.png"), (result["mask"] * 255).astype(np.uint8))

    comparison = np.hstack([result["input"], result["output_tps"], result["output"]])
    cv2.imwrite(str(out / "comparison.png"), comparison)

    view = result.get("view_info", {})
    if view.get("warning"):
        print(f"WARNING: {view['warning']}")
    print(f"Face view: {view.get('view', 'unknown')} (yaw={view.get('yaw', 0)})")
    print(f"Results saved to {out}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LandmarkDiff inference")
    parser.add_argument("image", help="Path to face image")
    parser.add_argument("--procedure", default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=50.0)
    parser.add_argument("--output", default="scripts/inference_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", default="img2img",
        choices=["img2img", "controlnet", "controlnet_ip", "tps"],
    )
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned ControlNet checkpoint")
    parser.add_argument("--displacement-model", default=None,
                        help="Path to displacement_model.npz for data-driven manipulation")
    args = parser.parse_args()

    run_inference(
        args.image, args.procedure, args.intensity, args.output,
        args.seed, args.mode, args.ip_adapter_scale, args.checkpoint,
        args.displacement_model,
    )
