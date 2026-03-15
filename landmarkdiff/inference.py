"""Inference pipeline for surgical outcome prediction.

Four modes:
1. ControlNet: CrucibleAI/ControlNetMediaPipeFace + SD1.5 (requires HF auth + GPU)
2. ControlNet + IP-Adapter: ControlNet with identity preservation via face embeddings
3. Img2Img: SD1.5 img2img with mask compositing (runs on MPS, no auth needed)
4. TPS-only: Pure geometric warp -- no diffusion model, instant results

Supports MPS (Apple Silicon), CUDA, and CPU backends.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image

from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask, mask_to_3channel
from landmarkdiff.synthetic.tps_warp import warp_image_tps

if TYPE_CHECKING:
    from landmarkdiff.clinical import ClinicalFlags

logger = logging.getLogger(__name__)


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


# Extensions that cv2.imread handles natively
_CV2_NATIVE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
# Extensions that need Pillow fallback (HEIC, WebP, AVIF)
_PILLOW_EXTS = {".heic", ".heif", ".webp", ".avif"}


def load_image(path: str) -> np.ndarray | None:
    """Load an image from disk, supporting HEIC, WebP, and AVIF via Pillow.

    Falls back to Pillow for formats that OpenCV cannot read natively.
    Returns BGR numpy array or None if the file cannot be loaded.
    """
    from pathlib import Path

    ext = Path(path).suffix.lower()

    if ext in _CV2_NATIVE_EXTS or ext not in _PILLOW_EXTS:
        img = cv2.imread(path)
        if img is not None:
            return img

    # Pillow fallback for HEIC/WebP/AVIF
    try:
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        logger.warning("Could not load image: %s", path)
        return None


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
    "brow_lift": (
        "clinical photograph, patient face, elevated brow position, smooth forehead, "
        "realistic skin pores and texture, sharp focus, studio lighting, "
        "DSLR quality, natural skin color"
    ),
    "mentoplasty": (
        "clinical photograph, patient face, refined chin contour, balanced lower face, "
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

# Skin tone matching: minimum mask alpha to include in LAB stats transfer
_SKIN_TONE_MASK_THRESHOLD = 0.3
# Epsilon to avoid division by zero in std normalization
_STD_EPSILON = 1e-6
# Default SD1.5 resolution (all pipelines resize to this)
_SD15_RESOLUTION = 512
# Intensity mapping: UI scale (0-100) to displacement model scale (0-2)
_INTENSITY_UI_TO_MODEL = 50.0
# Face view classification thresholds (degrees)
_YAW_FRONTAL_MAX = 15
_YAW_THREE_QUARTER_MAX = 45
_YAW_WARNING_THRESHOLD = 30
# Max pitch scale factor (maps pitch ratio to degrees)
_PITCH_SCALE = 45
# Input quality thresholds
_MIN_RESOLUTION = 128  # minimum dimension in pixels
_BLUR_LAPLACIAN_THRESHOLD = 50.0  # below this variance, image is likely blurry
_DARK_MEAN_THRESHOLD = 40  # mean brightness below this is poorly lit


def check_image_quality(image: np.ndarray) -> list[str]:
    """Check input image quality and return a list of warning strings.

    Detects low resolution, blur (via Laplacian variance), and poor
    lighting. Returns an empty list if quality is acceptable.
    """
    warnings = []
    h, w = image.shape[:2]

    if min(h, w) < _MIN_RESOLUTION:
        warnings.append(
            f"Low resolution ({w}x{h}): minimum {_MIN_RESOLUTION}px "
            "recommended for reliable landmark detection"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < _BLUR_LAPLACIAN_THRESHOLD:
        warnings.append(
            f"Blurry input (Laplacian variance={lap_var:.1f}): sharp images produce better results"
        )

    mean_brightness = float(gray.mean())
    if mean_brightness < _DARK_MEAN_THRESHOLD:
        warnings.append(
            f"Dark input (mean brightness={mean_brightness:.0f}/255): well-lit images recommended"
        )

    return warnings


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
            logger.debug("Laplacian blend failed, using alpha blend", exc_info=True)

    # Fallback: simple alpha blend
    mask_3ch = mask_to_3channel(mask_f)
    result = (
        corrected.astype(np.float32) * mask_3ch + original.astype(np.float32) * (1.0 - mask_3ch)
    ).astype(np.uint8)

    return result


def _match_skin_tone(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match source skin tone to target within mask, preserving structure.

    Works in LAB space: transfers L (luminance) and AB (color) statistics
    from the original to the warped image so skin tone is preserved exactly.
    """
    mask_bool = mask > _SKIN_TONE_MASK_THRESHOLD
    if not np.any(mask_bool):
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Match each LAB channel's statistics in the mask region
    for ch in range(3):
        src_vals = src_lab[:, :, ch][mask_bool]
        tgt_vals = tgt_lab[:, :, ch][mask_bool]

        src_mean, src_std = np.mean(src_vals), np.std(src_vals) + _STD_EPSILON
        tgt_mean, tgt_std = np.mean(tgt_vals), np.std(tgt_vals) + _STD_EPSILON

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
    - 'controlnet': CrucibleAI/ControlNetMediaPipeFace + SD1.5 (30 steps)
    - 'controlnet_fast': ControlNet + LCM-LoRA (4 steps, CPU-viable)
    - 'controlnet_ip': ControlNet + IP-Adapter for identity preservation
    - 'img2img': SD1.5 img2img with mask compositing
    - 'tps': Pure geometric TPS warp (no diffusion, instant)
    """

    # Default IP-Adapter model for SD1.5 face identity
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_SUBFOLDER = "models"
    IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus-face_sd15.bin"
    IP_ADAPTER_SCALE_DEFAULT = 0.6

    # LCM-LoRA for fast inference (2-4 steps instead of 30)
    LCM_LORA_REPO = "latent-consistency/lcm-lora-sdv1-5"

    def __init__(
        self,
        mode: str = "img2img",
        controlnet_id: str = "CrucibleAI/ControlNetMediaPipeFace",
        controlnet_checkpoint: str | None = None,
        base_model_id: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ip_adapter_scale: float = 0.6,
        clinical_flags: ClinicalFlags | None = None,
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
                logger.info("Displacement model loaded: %s", self._displacement_model.procedures)
            except Exception as e:
                logger.warning("Failed to load displacement model: %s", e)

        if self.device.type == "mps":
            self.dtype = torch.float32
        elif dtype:
            self.dtype = dtype
        else:
            self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if base_model_id:
            self.base_model_id = base_model_id
        else:
            self.base_model_id = "runwayml/stable-diffusion-v1-5"

        self.controlnet_id = controlnet_id
        self._pipe = None
        self._ip_adapter_loaded = False
        self._lcm_loaded = False

    def load(self) -> None:
        if self.mode == "tps":
            logger.info("TPS mode -- no model to load")
            return
        if self.mode in ("controlnet", "controlnet_ip", "controlnet_fast"):
            self._load_controlnet()
            if self.mode == "controlnet_fast":
                self._load_lcm_lora()
            elif self.mode == "controlnet_ip":
                self._load_ip_adapter()
        else:
            self._load_img2img()

    def _load_controlnet(self) -> None:
        from diffusers import (
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionControlNetPipeline,
        )

        _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        _kw: dict = {"local_files_only": True} if _local_only else {}

        if self.controlnet_checkpoint:
            # Load fine-tuned ControlNet from local checkpoint
            ckpt_path = Path(self.controlnet_checkpoint)
            # Support both direct path and training checkpoint structure
            if (ckpt_path / "controlnet_ema").exists():
                ckpt_path = ckpt_path / "controlnet_ema"
            logger.info("Loading fine-tuned ControlNet from %s", ckpt_path)
            controlnet = ControlNetModel.from_pretrained(
                str(ckpt_path),
                torch_dtype=self.dtype,
            )
        else:
            logger.info("Loading ControlNet from %s", self.controlnet_id)
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id,
                subfolder="diffusion_sd15",
                torch_dtype=self.dtype,
                **_kw,
            )
        logger.info("Loading base model from %s", self.base_model_id)
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.base_model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            **_kw,
        )
        # DPM++ 2M Karras -- produces more photorealistic output than UniPC
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        # FP32 VAE decode -- prevents color banding artifacts on skin tones
        if hasattr(self._pipe, "vae") and self._pipe.vae is not None:
            self._pipe.vae.config.force_upcast = True
        self._apply_device_optimizations()

    def _load_lcm_lora(self) -> None:
        """Load LCM-LoRA for fast 4-step inference.

        LCM-LoRA (Latent Consistency Model) distills the denoising process
        into 2-4 steps, making CPU inference viable (~3-8s vs ~60s+).
        Replaces the scheduler with LCMScheduler for consistency sampling.
        """
        if self._pipe is None:
            raise RuntimeError("Base pipeline must be loaded before LCM-LoRA")
        try:
            from diffusers import LCMScheduler

            logger.info("Loading LCM-LoRA from %s", self.LCM_LORA_REPO)
            _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            _kw: dict = {"local_files_only": True} if _local_only else {}
            self._pipe.load_lora_weights(self.LCM_LORA_REPO, **_kw)
            self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)
            self._lcm_loaded = True
            logger.info("LCM-LoRA loaded -- 4-step inference enabled")
        except Exception as e:
            logger.warning("LCM-LoRA load failed: %s", e)
            logger.warning("Falling back to standard scheduler (30 steps)")
            self._lcm_loaded = False

    def _load_ip_adapter(self) -> None:
        """Load IP-Adapter for identity-preserving generation.

        Uses h94/IP-Adapter-FaceID with CLIP image encoder to condition
        generation on the input face identity.
        """
        if self._pipe is None:
            raise RuntimeError("Base pipeline must be loaded before IP-Adapter")
        try:
            logger.info("Loading IP-Adapter (%s)", self.IP_ADAPTER_WEIGHT_NAME)
            self._pipe.load_ip_adapter(
                self.IP_ADAPTER_REPO,
                subfolder=self.IP_ADAPTER_SUBFOLDER,
                weight_name=self.IP_ADAPTER_WEIGHT_NAME,
            )
            self._pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded (scale=%s)", self.ip_adapter_scale)
        except Exception as e:
            logger.warning("IP-Adapter load failed: %s", e)
            logger.warning("Falling back to ControlNet-only mode")
            self._ip_adapter_loaded = False

    def _load_img2img(self) -> None:
        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionImg2ImgPipeline,
        )

        _local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        _kw: dict = {"local_files_only": True} if _local_only else {}

        logger.info("Loading SD1.5 img2img from %s", self.base_model_id)
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            **_kw,
        )
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
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
            # Attention slicing trades compute for VRAM on low-memory GPUs
            self._pipe.enable_attention_slicing()
        else:
            self._pipe.enable_sequential_cpu_offload()
            self._pipe.enable_attention_slicing()
        logger.info("Pipeline loaded on %s (%s)", self.device, self.dtype)

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None or self.mode == "tps"

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        procedure: str = "rhinoplasty",
        intensity: float = 50.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        controlnet_conditioning_scale: float = 0.9,
        strength: float = 0.5,
        seed: int | None = None,
        clinical_flags: ClinicalFlags | None = None,
        postprocess: bool = True,
        use_gfpgan: bool = False,
        deterministic: bool = False,
    ) -> dict:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Check input quality and warn (non-blocking)
        quality_warnings = check_image_quality(image)
        for warning in quality_warnings:
            logger.warning("Input quality: %s", warning)

        flags = clinical_flags or self.clinical_flags
        res = _SD15_RESOLUTION
        image_512 = cv2.resize(image, (res, res))

        face = extract_landmarks(image_512)
        if face is None:
            raise ValueError("No face detected in image.")

        # Estimate face view angle for multi-view awareness
        view_info = estimate_face_view(face)

        # Use displacement model for data-driven manipulation if available
        manipulation_mode = "preset"
        if self._displacement_model and procedure in self._displacement_model.procedures:
            try:
                rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
                # Map UI intensity (0-100) to displacement model intensity (0-2)
                dm_intensity = intensity / _INTENSITY_UI_TO_MODEL  # 50 -> 1.0x mean displacement
                displacement = self._displacement_model.get_displacement_field(
                    procedure,
                    intensity=dm_intensity,
                    noise_scale=0.3,
                    rng=rng,
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
                    image_width=res,
                    image_height=res,
                    confidence=face.confidence,
                )
                manipulation_mode = "displacement_model"
            except Exception as exc:
                logger.warning("Displacement model failed, falling back to preset: %s", exc)
                manipulated = apply_procedure_preset(
                    face,
                    procedure,
                    intensity,
                    image_size=res,
                    clinical_flags=flags,
                )
        else:
            manipulated = apply_procedure_preset(
                face,
                procedure,
                intensity,
                image_size=res,
                clinical_flags=flags,
            )
        landmark_img = render_landmark_image(manipulated, res, res)
        mask = generate_surgical_mask(
            face,
            procedure,
            res,
            res,
            clinical_flags=flags,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        prompt = PROCEDURE_PROMPTS.get(procedure, "a photo of a person's face")

        # Step 1: TPS geometric warp (always computed -- the geometric baseline)
        tps_warped = warp_image_tps(image_512, face.pixel_coords, manipulated.pixel_coords)

        if self.mode == "tps":
            raw_output = tps_warped
        elif self.mode in ("controlnet", "controlnet_ip", "controlnet_fast"):
            # LCM mode: override to 4 steps, low guidance (LCM works best with cfg=1-2)
            if self._lcm_loaded:
                num_inference_steps = min(num_inference_steps, 4)
                guidance_scale = min(guidance_scale, 1.5)
            ip_image = numpy_to_pil(image_512) if self._ip_adapter_loaded else None
            try:
                raw_output = self._generate_controlnet(
                    image_512,
                    landmark_img,
                    prompt,
                    num_inference_steps,
                    guidance_scale,
                    controlnet_conditioning_scale,
                    generator,
                    ip_adapter_image=ip_image,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory during inference. Try reducing "
                    "num_inference_steps or switching to mode='tps' for CPU-only."
                ) from exc
        else:
            if controlnet_conditioning_scale != 0.9:
                logger.warning(
                    "controlnet_conditioning_scale has no effect in img2img mode "
                    "(no ControlNet is used). Use mode='controlnet' for "
                    "conditioning scale control."
                )
            try:
                raw_output = self._generate_img2img(
                    tps_warped,
                    mask,
                    prompt,
                    num_inference_steps,
                    guidance_scale,
                    strength,
                    generator,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU out of memory during inference. Try reducing "
                    "num_inference_steps or switching to mode='tps' for CPU-only."
                ) from exc

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
            "lcm_active": self._lcm_loaded,
            "identity_check": identity_check,
            "restore_used": restore_used,
            "manipulation_mode": manipulation_mode,
            "quality_warnings": quality_warnings,
        }

    def _generate_controlnet(
        self,
        image: np.ndarray,
        conditioning: np.ndarray,
        prompt: str,
        steps: int,
        cfg: float,
        cn_scale: float,
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
        output = pil_to_numpy(result.images[0])
        del result
        return output

    def _generate_img2img(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        steps: int,
        cfg: float,
        strength: float,
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
        output = pil_to_numpy(result.images[0])
        del result
        return output


def estimate_face_view(face: FaceLandmarks) -> dict:
    """Estimate face orientation from landmarks for multi-view awareness.

    Uses the nose tip (idx 1), left ear (idx 234), and right ear (idx 454) to
    estimate yaw angle. Pitch from forehead (idx 10) and chin (idx 152).

    Returns dict with yaw, pitch (degrees), and view classification.
    """
    coords = face.pixel_coords
    # MediaPipe landmark indices for key anatomical points
    nose_tip = coords[1]  # nose tip
    left_ear = coords[234]  # left tragion (ear)
    right_ear = coords[454]  # right tragion (ear)
    forehead = coords[10]  # forehead center
    chin = coords[152]  # chin center

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
        pitch = float(pitch_ratio * _PITCH_SCALE)

    # Classify view
    abs_yaw = abs(yaw)
    if abs_yaw < _YAW_FRONTAL_MAX:
        view = "frontal"
    elif abs_yaw < _YAW_THREE_QUARTER_MAX:
        view = "three_quarter"
    else:
        view = "profile"

    return {
        "yaw": round(yaw, 1),
        "pitch": round(pitch, 1),
        "view": view,
        "is_frontal": abs_yaw < _YAW_FRONTAL_MAX,
        "warning": "Side-view detected: results may be less accurate"
        if abs_yaw > _YAW_WARNING_THRESHOLD
        else None,
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
    deterministic: bool = False,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    if image is None:
        logger.error("Could not load %s", image_path)
        sys.exit(1)

    pipe = LandmarkDiffPipeline(
        mode=mode,
        ip_adapter_scale=ip_adapter_scale,
        controlnet_checkpoint=controlnet_checkpoint,
        displacement_model_path=displacement_model_path,
    )
    pipe.load()

    logger.info("Generating %s prediction (intensity=%s, mode=%s)", procedure, intensity, mode)
    result = pipe.generate(
        image, procedure=procedure, intensity=intensity, seed=seed, deterministic=deterministic
    )

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
        logger.warning("%s", view["warning"])
    logger.info("Face view: %s (yaw=%s)", view.get("view", "unknown"), view.get("yaw", 0))
    logger.info("Results saved to %s/", out)


# ---------------------------------------------------------------------------
# Mixed-precision inference utilities
# ---------------------------------------------------------------------------


def get_optimal_dtype(device: torch.device | None = None) -> torch.dtype:
    """Select the optimal dtype for a device.

    - CUDA with Tensor Cores (compute >= 7.0): float16
    - CUDA without Tensor Cores: float32
    - MPS (Apple Silicon): float32 (float16 is unreliable on MPS)
    - CPU: float32

    Returns:
        torch.dtype suitable for the target device.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        if cap[0] >= 7:
            return torch.float16
        return torch.float32
    return torch.float32


def estimate_vram_usage(
    resolution: int = 512,
    dtype: torch.dtype = torch.float16,
    num_inference_steps: int = 20,
    mode: str = "controlnet",
) -> dict[str, float]:
    """Estimate VRAM usage for a given inference configuration.

    Returns a dict with estimated VRAM in GB for each component.
    These are rough estimates based on SD1.5 + ControlNet architecture.
    """
    # Base model sizes in GB (SD1.5)
    unet_params = 860e6  # ~860M parameters
    controlnet_params = 361e6  # ~361M parameters
    vae_params = 83e6  # ~83M parameters
    text_encoder_params = 123e6  # ~123M parameters

    bytes_per_param = 2 if dtype == torch.float16 else 4

    unet_gb = unet_params * bytes_per_param / 1e9
    cn_gb = controlnet_params * bytes_per_param / 1e9 if "controlnet" in mode else 0
    vae_gb = vae_params * 4 / 1e9  # VAE always FP32
    text_gb = text_encoder_params * bytes_per_param / 1e9

    # Latent space activations (resolution/8 due to VAE downscale)
    latent_size = resolution // 8
    latent_channels = 4
    latent_bytes = latent_size * latent_size * latent_channels * bytes_per_param
    activations_gb = latent_bytes * num_inference_steps * 2 / 1e9  # rough

    total = unet_gb + cn_gb + vae_gb + text_gb + activations_gb

    return {
        "unet_gb": round(unet_gb, 2),
        "controlnet_gb": round(cn_gb, 2),
        "vae_gb": round(vae_gb, 2),
        "text_encoder_gb": round(text_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb": round(total, 2),
        "dtype": str(dtype),
        "resolution": resolution,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LandmarkDiff inference")
    parser.add_argument("image", help="Path to face image")
    parser.add_argument("--procedure", default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=50.0)
    parser.add_argument("--output", default="scripts/inference_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        default="img2img",
        choices=["img2img", "controlnet", "controlnet_ip", "controlnet_fast", "tps"],
    )
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument(
        "--checkpoint", default=None, help="Path to fine-tuned ControlNet checkpoint"
    )
    parser.add_argument(
        "--displacement-model",
        default=None,
        help="Path to displacement_model.npz for data-driven manipulation",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable fully deterministic mode (slower, but reproducible across GPUs)",
    )
    args = parser.parse_args()

    run_inference(
        args.image,
        args.procedure,
        args.intensity,
        args.output,
        args.seed,
        args.mode,
        args.ip_adapter_scale,
        args.checkpoint,
        args.displacement_model,
        args.deterministic,
    )
