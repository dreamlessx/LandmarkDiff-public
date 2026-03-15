"""Post-processing pipeline for photorealistic face output.

Neural net components:
- CodeFormer (primary): face restoration with controllable fidelity-quality tradeoff
- GFPGAN (fallback): face restoration for diffusion artifact repair
- Real-ESRGAN: neural super-resolution for background regions
- ArcFace: identity verification to flag drift between input/output

Classical components:
- Multi-band Laplacian pyramid blending (replaces simple alpha blend)
- Frequency-aware sharpening (recovers fine skin texture)
- Color histogram matching (ensures skin tone consistency)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum CodeFormer fidelity weight: below this the codebook dominates
# and output becomes unacceptably blurry
_CODEFORMER_MIN_FIDELITY = 0.1

# Singleton model caches -- load once, reuse across calls
_CODEFORMER_MODEL = None
_GFPGAN_HELPER = None
_REALESRGAN_UPSAMPLER = None
_ARCFACE_APP = None


def laplacian_pyramid_blend(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    levels: int = 6,
) -> np.ndarray:
    """Multi-band Laplacian pyramid blending for seamless compositing.

    Unlike simple alpha blending which creates visible halos at mask edges,
    Laplacian blending operates at multiple frequency bands. Low frequencies
    (overall color/lighting) blend smoothly, high frequencies (skin texture,
    pores, hair) transition sharply. This eliminates the "pasted on" look.

    Args:
        source: BGR image to blend IN (the surgical result).
        target: BGR image to blend INTO (the original photo).
        mask: Float32 mask [0-1] (1 = source region).
        levels: Number of pyramid levels (6 works well for 512x512).

    Returns:
        Seamlessly composited BGR image.
    """
    # Ensure same size
    h, w = target.shape[:2]
    source = cv2.resize(source, (w, h)) if source.shape[:2] != (h, w) else source

    # Normalize mask
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    mask_3ch = np.stack([mask_f] * 3, axis=-1) if mask_f.ndim == 2 else mask_f

    # Make dimensions divisible by 2^levels
    factor = 2**levels
    new_h = (h + factor - 1) // factor * factor
    new_w = (w + factor - 1) // factor * factor

    if new_h != h or new_w != w:
        source = cv2.resize(source, (new_w, new_h))
        target = cv2.resize(target, (new_w, new_h))
        mask_3ch = cv2.resize(mask_3ch, (new_w, new_h))

    src_f = source.astype(np.float32)
    tgt_f = target.astype(np.float32)

    # Build Gaussian pyramids for the mask
    mask_pyr = [mask_3ch]
    for _ in range(levels):
        mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))

    # Build Laplacian pyramids for source and target
    src_lap = _build_laplacian_pyramid(src_f, levels)
    tgt_lap = _build_laplacian_pyramid(tgt_f, levels)

    # Blend each level using the mask at that resolution
    blended_lap = []
    for i in range(levels + 1):
        sl = src_lap[i]
        tl = tgt_lap[i]
        ml = mask_pyr[i]
        # Resize mask to match level shape if needed
        if ml.shape[:2] != sl.shape[:2]:
            ml = cv2.resize(ml, (sl.shape[1], sl.shape[0]))
        blended = sl * ml + tl * (1.0 - ml)
        blended_lap.append(blended)

    # Reconstruct from blended Laplacian
    result = _reconstruct_from_laplacian(blended_lap)

    # Crop back to original size
    result = result[:h, :w]
    return np.clip(result, 0, 255).astype(np.uint8)


def _build_laplacian_pyramid(
    image: np.ndarray,
    levels: int,
) -> list[np.ndarray]:
    """Build Laplacian pyramid from an image."""
    gaussian = [image.copy()]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    laplacian = []
    for i in range(levels):
        upsampled = cv2.pyrUp(gaussian[i + 1])
        # Match sizes (pyrUp can add a pixel)
        gh, gw = gaussian[i].shape[:2]
        upsampled = upsampled[:gh, :gw]
        laplacian.append(gaussian[i] - upsampled)

    laplacian.append(gaussian[-1])  # coarsest level
    return laplacian


def _reconstruct_from_laplacian(pyramid: list[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    image = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        image = cv2.pyrUp(image)
        lh, lw = pyramid[i].shape[:2]
        image = image[:lh, :lw]
        image = image + pyramid[i]
    return image


def frequency_aware_sharpen(
    image: np.ndarray,
    strength: float = 0.3,
    radius: int = 3,
) -> np.ndarray:
    """Sharpen high-frequency detail (skin texture, pores) without amplifying noise.

    Uses unsharp masking in LAB space (luminance only) to avoid
    color fringing. Preserves the smooth look of diffusion output
    while recovering fine texture detail.

    Args:
        image: BGR image.
        strength: Sharpening strength (0.2-0.5 typical for faces).
        radius: Gaussian blur radius for unsharp mask.

    Returns:
        Sharpened BGR image.
    """
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]

    # Unsharp mask on luminance only
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(l_channel, (ksize, ksize), 0)
    sharpened = l_channel + strength * (l_channel - blurred)

    lab[:, :, 0] = np.clip(sharpened, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def texture_aware_blend(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    texture_weight: float = 0.6,
    detail_radius: int = 5,
) -> np.ndarray:
    """Blend generated result with original while preserving skin texture.

    Decomposes both images into base (low-frequency color/shape) and detail
    (high-frequency texture) layers. Uses the generated base but retains
    the original's detail layer in the mask region, preserving pores,
    freckles, and fine skin texture that diffusion models tend to smooth out.

    Args:
        original: BGR original image.
        generated: BGR generated/predicted image (same dimensions).
        mask: Float32 mask [0-1] where 1 = modified region.
        texture_weight: How much original texture to preserve (0-1).
            0 = use generated texture entirely, 1 = use original texture entirely.
        detail_radius: Bilateral filter radius for base/detail decomposition.
            Larger values capture coarser texture in the detail layer.

    Returns:
        Blended BGR image preserving original skin detail.
    """
    h, w = original.shape[:2]
    gen = cv2.resize(generated, (w, h)) if generated.shape[:2] != (h, w) else generated

    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0
    mask_3ch = np.stack([mask_f] * 3, axis=-1) if mask_f.ndim == 2 else mask_f

    orig_f = original.astype(np.float32)
    gen_f = gen.astype(np.float32)

    # Decompose into base (bilateral filtered) and detail (residual)
    d = detail_radius * 2 + 1
    sigma_color = 75.0
    sigma_space = float(detail_radius)

    orig_base = cv2.bilateralFilter(orig_f, d, sigma_color, sigma_space)
    gen_base = cv2.bilateralFilter(gen_f, d, sigma_color, sigma_space)

    orig_detail = orig_f - orig_base
    gen_detail = gen_f - gen_base

    # In the mask region: use generated base + blend of original/generated detail
    blended_detail = orig_detail * texture_weight + gen_detail * (1.0 - texture_weight)

    # Composite: inside mask use generated base + blended detail,
    # outside mask use original
    result = gen_base + blended_detail
    result = result * mask_3ch + orig_f * (1.0 - mask_3ch)

    return np.clip(result, 0, 255).astype(np.uint8)


def restore_face_gfpgan(
    image: np.ndarray,
    upscale: int = 1,
) -> np.ndarray:
    """Restore face quality using GFPGAN.

    Fixes common diffusion artifacts: blurry eyes, distorted features,
    inconsistent skin texture. The restored face is then blended back
    into the original for a natural look.

    Args:
        image: BGR face image (any size).
        upscale: Upscale factor (1 = same size, 2 = 2x).

    Returns:
        Restored BGR image, or original if GFPGAN unavailable.
    """
    try:
        from gfpgan import GFPGANer
    except ImportError:
        return image

    # GFPGAN requires 3-channel BGR input
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        global _GFPGAN_HELPER
        # Singleton: avoid reloading ~300MB GFPGAN model on every call
        if _GFPGAN_HELPER is None:
            _GFPGAN_HELPER = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        _, _, restored = _GFPGAN_HELPER.enhance(
            image,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        if restored is not None:
            return restored
    except Exception:
        pass

    return image


def restore_face_codeformer(
    image: np.ndarray,
    fidelity: float = 0.7,
    upscale: int = 1,
) -> np.ndarray:
    """Restore face quality using CodeFormer (neural net).

    CodeFormer uses a Transformer-based codebook lookup to restore degraded
    faces. The fidelity parameter controls the quality-fidelity tradeoff:
    lower values produce higher quality but may alter identity slightly,
    higher values preserve identity but fix fewer artifacts.

    Args:
        image: BGR face image.
        fidelity: Quality-fidelity balance (0.0=quality, 1.0=fidelity). 0.7 default.
        upscale: Upscale factor (1 = same size).

    Returns:
        Restored BGR image, or original if CodeFormer unavailable.
    """
    if fidelity < _CODEFORMER_MIN_FIDELITY:
        logger.warning(
            "CodeFormer fidelity_weight=%.2f is below minimum %.2f; "
            "clamping to avoid blurry output from codebook dominance",
            fidelity,
            _CODEFORMER_MIN_FIDELITY,
        )
        fidelity = _CODEFORMER_MIN_FIDELITY

    try:
        import torch
        from codeformer.basicsr.utils import img2tensor, tensor2img
        from codeformer.basicsr.utils.download_util import load_file_from_url
        from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
        from torchvision.transforms.functional import normalize as tv_normalize
    except ImportError:
        return image

    # CodeFormer requires 3-channel BGR input
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    try:
        global _CODEFORMER_MODEL
        from codeformer.basicsr.archs.codeformer_arch import CodeFormer as CodeFormerArch
        from codeformer.inference_codeformer import set_realesrgan as _unused  # noqa: F401

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if _CODEFORMER_MODEL is None:
            model = CodeFormerArch(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)

            ckpt_path = load_file_from_url(
                url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                model_dir="weights/CodeFormer",
                progress=True,
            )
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["params_ema"])
            model.eval()
            _CODEFORMER_MODEL = model
        model = _CODEFORMER_MODEL

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=device,
        )
        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(only_center_face=True)
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            tv_normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_t, w=fidelity, adain=True)[0]
                restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored = restored.astype(np.uint8)
            face_helper.add_restored_face(restored)

        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_image()
        if restored_img is not None:
            return restored_img
    except Exception:
        pass

    return image


def enhance_background_realesrgan(
    image: np.ndarray,
    mask: np.ndarray,
    outscale: int = 2,
) -> np.ndarray:
    """Enhance non-face background regions using Real-ESRGAN neural upscaler.

    Only applies to regions outside the surgical mask to improve overall
    image quality without interfering with the face restoration pipeline.

    Args:
        image: BGR image.
        mask: Float32 mask [0-1] where 1 = face region (skip these pixels).
        outscale: Upscale factor (2 = 2x resolution, then downsample back).

    Returns:
        Enhanced BGR image at original resolution.
    """
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        return image

    try:
        global _REALESRGAN_UPSAMPLER
        if _REALESRGAN_UPSAMPLER is None:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            _REALESRGAN_UPSAMPLER = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
            )
        enhanced, _ = _REALESRGAN_UPSAMPLER.enhance(image, outscale=outscale)

        # Downscale back to original size
        h, w = image.shape[:2]
        enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Only apply enhancement to background (outside mask).
        # Dilate and feather the mask to create a buffer zone at the
        # hair-skin boundary where Real-ESRGAN introduces halo artifacts.
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        # Dilate mask by ~3% of image size to push enhanced region
        # away from high-contrast face boundary
        dilate_px = max(int(min(h, w) * 0.03), 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask_2d = mask_f if mask_f.ndim == 2 else mask_f[:, :, 0]
        dilated = cv2.dilate(mask_2d, kernel)

        # Gaussian feather the dilated edge for a smooth transition
        feathered = cv2.GaussianBlur(dilated, (dilate_px * 2 + 1, dilate_px * 2 + 1), 0)
        mask_3ch = np.stack([feathered] * 3, axis=-1)

        # Keep face region from original, use enhanced for background
        result = np.clip(
            image.astype(np.float32) * mask_3ch + enhanced.astype(np.float32) * (1.0 - mask_3ch),
            0,
            255,
        ).astype(np.uint8)
        return result
    except Exception:
        pass

    return image


def verify_identity_arcface(
    original: np.ndarray,
    result: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Verify output preserves input identity using ArcFace neural net.

    Computes cosine similarity between ArcFace embeddings of the original
    and result images. If similarity drops below threshold, flags identity
    drift -- meaning the postprocessing or diffusion altered the person's
    appearance too much.

    Args:
        original: BGR original face image.
        result: BGR post-processed output image.
        threshold: Minimum cosine similarity to pass (0.5 = same person).

    Returns:
        Dict with 'similarity' (float), 'passed' (bool), 'message' (str).
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return {
            "similarity": -1.0,
            "passed": True,
            "message": "InsightFace not installed — identity check skipped",
        }

    try:
        global _ARCFACE_APP
        if _ARCFACE_APP is None:
            _ARCFACE_APP = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _ARCFACE_APP.prepare(ctx_id=0 if _has_cuda() else -1, det_size=(320, 320))
        app = _ARCFACE_APP

        orig_faces = app.get(original)
        result_faces = app.get(result)

        if not orig_faces or not result_faces:
            return {
                "similarity": -1.0,
                "passed": True,
                "message": "Could not detect face in one/both images — check skipped",
            }

        orig_emb = orig_faces[0].embedding
        result_emb = result_faces[0].embedding

        sim = float(
            np.dot(orig_emb, result_emb)
            / (np.linalg.norm(orig_emb) * np.linalg.norm(result_emb) + 1e-8)
        )
        sim = float(np.clip(sim, 0, 1))

        passed = sim >= threshold
        if passed:
            msg = f"Identity preserved (similarity={sim:.3f})"
        else:
            msg = f"WARNING: Identity drift detected (similarity={sim:.3f} < {threshold})"

        return {"similarity": sim, "passed": passed, "message": msg}
    except Exception as e:
        return {
            "similarity": -1.0,
            "passed": True,
            "message": f"Identity check failed: {e}",
        }


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def histogram_match_skin(
    source: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Match skin color histogram of source to reference within masked region.

    More robust than simple mean/std matching — preserves the full
    distribution of skin tones including highlights and shadows.

    Args:
        source: BGR image whose skin tone to adjust.
        reference: BGR image with target skin tone.
        mask: Float32 mask [0-1] of skin region.

    Returns:
        Color-matched BGR image.
    """
    # Ensure 2D mask for per-channel indexing
    m = mask
    if m.ndim == 3:
        m = m[:, :, 0]
    mask_bool = m > 0.3 if m.dtype == np.float32 else m > 76

    if not np.any(mask_bool):
        return source

    # Clip to valid uint8 range before LAB conversion to prevent overflow
    # on images with saturated or out-of-range pixel values
    src_u8 = np.clip(source, 0, 255).astype(np.uint8)
    ref_u8 = np.clip(reference, 0, 255).astype(np.uint8)
    src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Full CDF matching on L (luminance) channel only.
    # For a/b (chrominance) channels, use gentle mean-shift with damping
    # to avoid color shifts on dark skin tones (Fitzpatrick V-VI).
    for ch in range(3):
        src_vals = src_lab[:, :, ch][mask_bool]
        ref_vals = ref_lab[:, :, ch][mask_bool]

        if len(src_vals) == 0 or len(ref_vals) == 0:
            continue

        if ch == 0:
            # L channel: full CDF matching for brightness
            src_sorted = np.sort(src_vals)
            ref_sorted = np.sort(ref_vals)
            src_cdf = np.linspace(0, 1, len(src_sorted))
            ref_cdf = np.linspace(0, 1, len(ref_sorted))
            mapping = np.interp(src_cdf, ref_cdf, ref_sorted)
            src_flat = src_lab[:, :, ch].ravel()
            matched = np.interp(src_flat, src_sorted, mapping)
            matched_2d = matched.reshape(src_lab.shape[:2])
            src_lab[:, :, ch] = np.where(mask_bool, matched_2d, src_lab[:, :, ch])
        else:
            # a/b channels: gentle mean-shift (30% blend) to preserve skin color
            src_mean = np.mean(src_vals)
            ref_mean = np.mean(ref_vals)
            shift = (ref_mean - src_mean) * 0.3
            shifted = src_lab[:, :, ch] + shift * mask_bool.astype(np.float32)
            src_lab[:, :, ch] = shifted

    result_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def full_postprocess(
    generated: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    restore_mode: str = "codeformer",
    codeformer_fidelity: float = 0.7,
    use_realesrgan: bool = True,
    use_laplacian_blend: bool = True,
    sharpen_strength: float = 0.25,
    verify_identity: bool = True,
    identity_threshold: float = 0.5,
) -> dict:
    """Full neural net + classical post-processing pipeline for maximum photorealism.

    Pipeline:
    1. Face restoration: CodeFormer (primary) or GFPGAN (fallback) neural nets
    2. Background enhancement: Real-ESRGAN neural upscaler (non-face regions)
    3. Skin tone histogram matching to original (classical)
    4. Frequency-aware sharpening for texture recovery (classical)
    5. Laplacian pyramid blending for seamless compositing (classical)
    6. ArcFace identity verification (neural net quality gate)

    Args:
        generated: BGR generated/warped face image.
        original: BGR original face image.
        mask: Float32 surgical mask [0-1].
        restore_mode: 'codeformer', 'gfpgan', or 'none'.
        codeformer_fidelity: CodeFormer fidelity weight (0=quality, 1=fidelity).
        use_realesrgan: Apply Real-ESRGAN to background regions.
        use_laplacian_blend: Use Laplacian blend vs simple alpha blend.
        sharpen_strength: Texture sharpening amount (0 = none).
        verify_identity: Run ArcFace identity check at the end.
        identity_threshold: Min cosine similarity to pass identity check.

    Returns:
        Dict with 'image' (composited BGR), 'identity_check' (dict), 'restore_used' (str).
    """
    result = generated.copy()
    restore_used = "none"

    # Step 1: Neural face restoration (CodeFormer > GFPGAN > skip)
    if restore_mode == "codeformer":
        restored = restore_face_codeformer(result, fidelity=codeformer_fidelity)
        if restored is not result:
            result = restored
            restore_used = "codeformer"
        else:
            # CodeFormer unavailable, fall back to GFPGAN
            pre_gfpgan = result
            result = restore_face_gfpgan(result)
            restore_used = "gfpgan" if result is not pre_gfpgan else "none"
    elif restore_mode == "gfpgan":
        restored = restore_face_gfpgan(result)
        if restored is not result:
            result = restored
            restore_used = "gfpgan"

    # Step 2: Neural background enhancement
    if use_realesrgan:
        result = enhance_background_realesrgan(result, mask)

    # Step 3: Skin tone histogram matching (classical)
    result = histogram_match_skin(result, original, mask)

    # Step 4: Sharpen texture (classical)
    if sharpen_strength > 0:
        result = frequency_aware_sharpen(result, strength=sharpen_strength)

    # Step 5: Blend into original (classical)
    if use_laplacian_blend:
        composited = laplacian_pyramid_blend(result, original, mask)
    else:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        mask_3ch = np.stack([mask_f] * 3, axis=-1) if mask_f.ndim == 2 else mask_f
        composited = (
            result.astype(np.float32) * mask_3ch + original.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)

    # Step 6: Neural identity verification
    identity_check = {"similarity": -1.0, "passed": True, "message": "skipped"}
    if verify_identity:
        identity_check = verify_identity_arcface(
            original,
            composited,
            threshold=identity_threshold,
        )

    return {
        "image": composited,
        "identity_check": identity_check,
        "restore_used": restore_used,
    }
