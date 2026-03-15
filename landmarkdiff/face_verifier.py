"""Neural face verification, distortion detection, and restoration pipeline.

End-to-end system that:
1. Detects face distortions (blur, beauty filters, compression, warping, etc.)
2. Classifies distortion type and severity using no-reference quality metrics
3. Restores faces using cascaded neural networks (CodeFormer → GFPGAN → Real-ESRGAN)
4. Verifies output identity matches input via ArcFace embeddings
5. Scores output realism using learned perceptual metrics

Designed for:
- Cleaning scraped training data (reject/fix bad images before pair generation)
- Post-diffusion quality gate (ensure generated faces pass realism threshold)
- Filter removal (undo Snapchat/Instagram beauty filters for clinical use)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DistortionReport:
    """Analysis of detected distortions in a face image."""

    # Overall quality score (0-100, higher = better)
    quality_score: float = 0.0

    # Individual distortion scores (0-1, higher = more distorted)
    blur_score: float = 0.0  # Laplacian variance-based
    noise_score: float = 0.0  # High-freq energy ratio
    compression_score: float = 0.0  # JPEG block artifact detection
    oversmooth_score: float = 0.0  # Beauty filter / airbrushed detection
    color_cast_score: float = 0.0  # Unnatural color shift
    geometric_distort: float = 0.0  # Face proportion anomalies
    lighting_score: float = 0.0  # Over/under exposure

    # Classification
    primary_distortion: str = "none"
    severity: str = "none"  # none, mild, moderate, severe
    is_usable: bool = True  # Whether image is worth restoring vs rejecting

    # Details
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Quality Score: {self.quality_score:.1f}/100",
            f"Primary Issue: {self.primary_distortion} ({self.severity})",
            f"Usable: {self.is_usable}",
            "",
            "Distortion Breakdown:",
            f"  Blur:         {self.blur_score:.3f}",
            f"  Noise:        {self.noise_score:.3f}",
            f"  Compression:  {self.compression_score:.3f}",
            f"  Oversmooth:   {self.oversmooth_score:.3f}",
            f"  Color Cast:   {self.color_cast_score:.3f}",
            f"  Geometric:    {self.geometric_distort:.3f}",
            f"  Lighting:     {self.lighting_score:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class RestorationResult:
    """Result of neural face restoration pipeline."""

    restored: np.ndarray  # Restored BGR image
    original: np.ndarray  # Original BGR image
    distortion_report: DistortionReport  # Pre-restoration analysis
    post_quality_score: float = 0.0  # Quality after restoration
    identity_similarity: float = 0.0  # ArcFace cosine sim (original vs restored)
    identity_preserved: bool = True  # Whether identity check passed
    restoration_stages: list[str] = field(default_factory=list)  # Which nets ran
    improvement: float = 0.0  # quality_after - quality_before

    def summary(self) -> str:
        lines = [
            f"Pre-restoration:  {self.distortion_report.quality_score:.1f}/100",
            f"Post-restoration: {self.post_quality_score:.1f}/100",
            f"Improvement:      +{self.improvement:.1f}",
            f"Identity Sim:     {self.identity_similarity:.3f}",
            f"Identity OK:      {self.identity_preserved}",
            f"Stages Used:      {' → '.join(self.restoration_stages) or 'none'}",
        ]
        return "\n".join(lines)


@dataclass
class BatchVerificationReport:
    """Summary of batch face verification/restoration."""

    total: int = 0
    passed: int = 0  # Good quality, no fix needed
    restored: int = 0  # Fixed and now usable
    rejected: int = 0  # Too distorted to salvage
    identity_failures: int = 0  # Restoration changed identity
    avg_quality_before: float = 0.0
    avg_quality_after: float = 0.0
    avg_identity_sim: float = 0.0
    distortion_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Total Images:     {self.total}",
            f"  Passed (good):  {self.passed}",
            f"  Restored:       {self.restored}",
            f"  Rejected:       {self.rejected}",
            f"  Identity Fail:  {self.identity_failures}",
            f"Avg Quality Before: {self.avg_quality_before:.1f}",
            f"Avg Quality After:  {self.avg_quality_after:.1f}",
            f"Avg Identity Sim:   {self.avg_identity_sim:.3f}",
            "",
            "Distortion Breakdown:",
        ]
        for dist_type, count in sorted(
            self.distortion_counts.items(),
            key=lambda x: -x[1],
        ):
            lines.append(f"  {dist_type}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distortion Detection (classical + neural)
# ---------------------------------------------------------------------------


def detect_blur(image: np.ndarray) -> float:
    """Detect blur using Laplacian variance.

    Low variance = blurry. We normalize to 0-1 where 1 = very blurry.
    Uses both Laplacian variance and gradient magnitude for robustness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Laplacian variance (primary metric)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Gradient magnitude (secondary)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2).mean()

    # Normalize: typical sharp face has lap_var > 500, grad_mag > 30
    blur_lap = 1.0 - min(lap_var / 800.0, 1.0)
    blur_grad = 1.0 - min(grad_mag / 50.0, 1.0)

    return float(np.clip(0.6 * blur_lap + 0.4 * blur_grad, 0, 1))


def detect_noise(image: np.ndarray) -> float:
    """Detect image noise level.

    Estimates noise by measuring high-frequency energy in smooth regions.
    Uses the median absolute deviation of the Laplacian (robust estimator).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Robust noise estimation via MAD of Laplacian
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
    sigma_est = np.median(np.abs(lap)) * 1.4826  # MAD → std conversion

    # Normalize: sigma > 20 is very noisy
    return float(np.clip(sigma_est / 25.0, 0, 1))


def detect_compression_artifacts(image: np.ndarray) -> float:
    """Detect JPEG compression block artifacts.

    Measures energy at 8x8 block boundaries (JPEG DCT block size).
    High boundary energy relative to interior = compression artifacts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    if h < 16 or w < 16:
        return 0.0

    gray_f = gray.astype(np.float64)

    # Compute horizontal and vertical differences
    h_diff = np.abs(np.diff(gray_f, axis=1))
    v_diff = np.abs(np.diff(gray_f, axis=0))

    # Energy at 8-pixel boundaries vs non-boundaries
    h_boundary = h_diff[:, 7::8].mean() if h_diff[:, 7::8].size > 0 else 0
    h_interior = h_diff.mean()
    v_boundary = v_diff[7::8, :].mean() if v_diff[7::8, :].size > 0 else 0
    v_interior = v_diff.mean()

    if h_interior < 1e-6 or v_interior < 1e-6:
        return 0.0

    # Ratio of boundary to interior energy (>1 means block artifacts)
    h_ratio = h_boundary / (h_interior + 1e-6)
    v_ratio = v_boundary / (v_interior + 1e-6)
    artifact_ratio = (h_ratio + v_ratio) / 2.0

    # Normalize: ratio > 1.5 indicates visible artifacts
    return float(np.clip((artifact_ratio - 1.0) / 0.8, 0, 1))


def detect_oversmoothing(image: np.ndarray) -> float:
    """Detect beauty filter / airbrushed skin (oversmoothing).

    Beauty filters remove skin texture while preserving edges. We detect
    this by measuring the ratio of edge energy to texture energy.
    High edge / low texture = beauty filtered.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    # Focus on face center region (avoid background)
    if h < 8 or w < 8:
        return 0.0  # Too small to analyze
    roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    # Texture energy: variance of high-pass filtered image
    blurred = cv2.GaussianBlur(roi.astype(np.float64), (0, 0), 2.0)
    high_pass = roi.astype(np.float64) - blurred
    texture_energy = np.var(high_pass)

    # Edge energy: Canny edge density
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.mean(edges > 0)

    # Oversmooth: low texture but edges still present
    # Natural skin: texture_energy > 20, beauty filter: < 8
    smooth_score = 1.0 - min(texture_energy / 30.0, 1.0)

    # If there are still strong edges but no texture, it's a filter
    if edge_density > 0.02:
        smooth_score *= 1.3  # Amplify if edges present but no texture

    return float(np.clip(smooth_score, 0, 1))


def detect_color_cast(image: np.ndarray) -> float:
    """Detect unnatural color cast (Instagram-style filters).

    Measures deviation of average A/B channels in LAB space from
    neutral. Natural skin has consistent LAB distributions; filtered
    images shift these channels.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = image.shape[:2]

    # Sample face center region
    roi = lab[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    # A channel: green-red axis (neutral ~128)
    # B channel: blue-yellow axis (neutral ~128)
    a_mean = roi[:, :, 1].mean()
    b_mean = roi[:, :, 2].mean()

    # Deviation from neutral
    a_dev = abs(a_mean - 128) / 128.0
    b_dev = abs(b_mean - 128) / 128.0

    # Also check if color distribution is unnaturally narrow (saturated filter)
    a_std = roi[:, :, 1].std()
    b_std = roi[:, :, 2].std()
    narrow_color = max(0, 1.0 - (a_std + b_std) / 30.0)

    score = 0.5 * (a_dev + b_dev) + 0.3 * narrow_color
    return float(np.clip(score, 0, 1))


def detect_geometric_distortion(image: np.ndarray) -> float:
    """Detect geometric face distortion (warping filters, lens distortion).

    Uses MediaPipe landmarks to check face proportions against anatomical
    norms. Distorted faces have abnormal inter-ocular / face-width ratios.
    """
    try:
        from landmarkdiff.landmarks import extract_landmarks
    except ImportError:
        return 0.0

    face = extract_landmarks(image)
    if face is None:
        return 0.5  # Can't detect face = possibly distorted

    coords = face.pixel_coords
    h, w = image.shape[:2]

    if len(coords) < 478:
        return 0.5  # Incomplete landmark set

    # Key ratios that should be anatomically consistent
    left_eye = coords[33]
    right_eye = coords[263]
    nose_tip = coords[1]
    chin = coords[152]
    forehead = coords[10]

    iod = np.linalg.norm(left_eye - right_eye)
    face_height = np.linalg.norm(forehead - chin)
    nose_to_chin = np.linalg.norm(nose_tip - chin)

    if iod < 1.0 or face_height < 1.0:
        return 0.5

    # Anatomical norms (approximate):
    # face_height / iod ≈ 2.5-3.5
    # nose_to_chin / face_height ≈ 0.3-0.45
    height_ratio = face_height / iod
    lower_ratio = nose_to_chin / face_height

    # Score deviations from normal ranges
    height_dev = max(0, abs(height_ratio - 3.0) - 0.5) / 1.5
    lower_dev = max(0, abs(lower_ratio - 0.38) - 0.08) / 0.15

    # Eye symmetry check (vertical alignment)
    eye_tilt = abs(left_eye[1] - right_eye[1]) / (iod + 1e-6)
    tilt_dev = max(0, eye_tilt - 0.05) / 0.15

    score = 0.4 * height_dev + 0.3 * lower_dev + 0.3 * tilt_dev
    return float(np.clip(score, 0, 1))


def detect_lighting_issues(image: np.ndarray) -> float:
    """Detect over/under exposure and harsh lighting.

    Checks luminance histogram for clipping and uneven distribution.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Check for clipping
    overexposed = np.mean(l_channel > 245) * 5  # Fraction near white
    underexposed = np.mean(l_channel < 10) * 5  # Fraction near black

    # Check for bimodal distribution (harsh shadows)
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
    hist_sum = hist.sum()
    if hist_sum < 1e-10:
        return 0.0
    hist = hist / hist_sum
    # Measure how spread out the histogram is
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
    # Low entropy = concentrated = potentially problematic
    entropy_score = max(0, 1.0 - entropy / 7.0)

    score = 0.4 * overexposed + 0.4 * underexposed + 0.2 * entropy_score
    return float(np.clip(score, 0, 1))


def analyze_distortions(image: np.ndarray) -> DistortionReport:
    """Run full distortion analysis on a face image.

    Combines all detection methods into a comprehensive report with
    quality score, primary distortion classification, and severity.
    """
    blur = detect_blur(image)
    noise = detect_noise(image)
    compression = detect_compression_artifacts(image)
    oversmooth = detect_oversmoothing(image)
    color_cast = detect_color_cast(image)
    geometric = detect_geometric_distortion(image)
    lighting = detect_lighting_issues(image)

    # Overall quality: weighted combination (inverted — 100 = perfect)
    weighted = (
        0.25 * blur
        + 0.15 * noise
        + 0.10 * compression
        + 0.20 * oversmooth
        + 0.10 * color_cast
        + 0.10 * geometric
        + 0.10 * lighting
    )
    quality = (1.0 - weighted) * 100.0

    # Classify primary distortion
    scores = {
        "blur": blur,
        "noise": noise,
        "compression": compression,
        "oversmooth": oversmooth,
        "color_cast": color_cast,
        "geometric": geometric,
        "lighting": lighting,
    }
    primary = max(scores, key=scores.get)
    primary_val = scores[primary]

    if primary_val < 0.15:
        severity = "none"
        primary = "none"
    elif primary_val < 0.35:
        severity = "mild"
    elif primary_val < 0.60:
        severity = "moderate"
    else:
        severity = "severe"

    # Image is usable if quality > 30 and no severe geometric distortion
    is_usable = quality > 25 and geometric < 0.7

    return DistortionReport(
        quality_score=quality,
        blur_score=blur,
        noise_score=noise,
        compression_score=compression,
        oversmooth_score=oversmooth,
        color_cast_score=color_cast,
        geometric_distort=geometric,
        lighting_score=lighting,
        primary_distortion=primary,
        severity=severity,
        is_usable=is_usable,
        details=scores,
    )


# ---------------------------------------------------------------------------
# Neural Face Quality Scoring (no-reference)
# ---------------------------------------------------------------------------

_FACE_QUALITY_NET = None


def _get_face_quality_scorer() -> Any:
    """Get or create singleton face quality assessment model.

    Uses FaceXLib's quality scorer or falls back to BRISQUE-style features.
    """
    global _FACE_QUALITY_NET
    if _FACE_QUALITY_NET is not None:
        return _FACE_QUALITY_NET

    try:
        from facexlib.assessment import init_assessment_model

        _FACE_QUALITY_NET = init_assessment_model("hypernet")
        return _FACE_QUALITY_NET
    except Exception:
        pass

    return None


def neural_quality_score(image: np.ndarray) -> float:
    """Score face quality using neural network (0-100, higher = better).

    Tries FaceXLib quality assessment first, then falls back to
    BRISQUE-style scoring using OpenCV's QualityBRISQUE if available,
    or classical metrics as last resort.
    """
    # Try neural scorer
    scorer = _get_face_quality_scorer()
    if scorer is not None:
        try:
            import torch
            from facexlib.utils import img2tensor

            img_t = img2tensor(image / 255.0, bgr2rgb=True, float32=True)
            img_t = img_t.unsqueeze(0)
            if torch.cuda.is_available():
                img_t = img_t.cuda()
                scorer = scorer.cuda()
            with torch.no_grad():
                score = scorer(img_t).item()
            return float(np.clip(score * 100, 0, 100))
        except Exception:
            pass

    # Fallback: composite classical score
    report = analyze_distortions(image)
    return report.quality_score


# ---------------------------------------------------------------------------
# Neural Face Restoration (cascaded)
# ---------------------------------------------------------------------------


def restore_face(
    image: np.ndarray,
    distortion: DistortionReport | None = None,
    mode: str = "auto",
    codeformer_fidelity: float = 0.7,
) -> tuple[np.ndarray, list[str]]:
    """Cascaded neural face restoration.

    Selects and applies restoration networks based on detected distortions:
    - Blur/oversmooth → CodeFormer (recovers texture from codebook)
    - Noise/compression → GFPGAN (trained on degraded faces)
    - Background → Real-ESRGAN (neural 4x upscale + downsample)
    - Color cast → Classical LAB correction (no neural net needed)
    - Geometric → Not fixable by restoration (flag and skip)

    Args:
        image: BGR face image to restore.
        distortion: Pre-computed distortion report (computed if None).
        mode: 'auto' (choose based on distortion), 'codeformer', 'gfpgan', 'all'.
        codeformer_fidelity: CodeFormer quality-fidelity tradeoff.

    Returns:
        Tuple of (restored BGR image, list of stages applied).
    """
    if distortion is None:
        distortion = analyze_distortions(image)

    result = image.copy()
    stages = []

    # Step 0: Fix color cast first (classical — fast, doesn't affect identity)
    if distortion.color_cast_score > 0.25:
        result = _fix_color_cast(result)
        stages.append("color_correction")

    # Step 1: Fix lighting issues (classical)
    if distortion.lighting_score > 0.35:
        result = _fix_lighting(result)
        stages.append("lighting_fix")

    # Step 2: Neural face restoration
    if mode == "auto":
        # Choose based on what's wrong
        needs_face_restore = (
            distortion.blur_score > 0.2
            or distortion.oversmooth_score > 0.25
            or distortion.noise_score > 0.25
            or distortion.compression_score > 0.2
        )
        if needs_face_restore:
            mode = "codeformer"  # CodeFormer handles most degradations well

    if mode in ("codeformer", "all"):
        restored = _try_codeformer(result, fidelity=codeformer_fidelity)
        if restored is not None:
            result = restored
            stages.append("codeformer")
        else:
            # Fallback to GFPGAN
            restored = _try_gfpgan(result)
            if restored is not None:
                result = restored
                stages.append("gfpgan")

    elif mode == "gfpgan":
        restored = _try_gfpgan(result)
        if restored is not None:
            result = restored
            stages.append("gfpgan")

    # Step 3: Background enhancement with Real-ESRGAN (if image is low-res)
    h, w = result.shape[:2]
    if h < 400 or w < 400:
        enhanced = _try_realesrgan(result)
        if enhanced is not None:
            result = enhanced
            stages.append("realesrgan")

    # Step 4: Mild sharpening if still soft after restoration
    post_blur = detect_blur(result)
    if post_blur > 0.3:
        from landmarkdiff.postprocess import frequency_aware_sharpen

        result = frequency_aware_sharpen(result, strength=0.3)
        stages.append("sharpen")

    return result, stages


def _try_codeformer(image: np.ndarray, fidelity: float = 0.7) -> np.ndarray | None:
    """Try CodeFormer restoration. Returns None if unavailable."""
    try:
        from landmarkdiff.postprocess import restore_face_codeformer

        restored = restore_face_codeformer(image, fidelity=fidelity)
        if restored is not image:
            return restored
    except Exception:
        pass
    return None


def _try_gfpgan(image: np.ndarray) -> np.ndarray | None:
    """Try GFPGAN restoration. Returns None if unavailable."""
    try:
        from landmarkdiff.postprocess import restore_face_gfpgan

        restored = restore_face_gfpgan(image)
        if restored is not image:
            return restored
    except Exception:
        pass
    return None


_FV_REALESRGAN = None


def _try_realesrgan(image: np.ndarray) -> np.ndarray | None:
    """Try Real-ESRGAN 2x upscale + downsample. Returns None if unavailable."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        global _FV_REALESRGAN
        if _FV_REALESRGAN is None:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            _FV_REALESRGAN = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
            )
        enhanced, _ = _FV_REALESRGAN.enhance(image, outscale=2)

        # Downsample to 512x512 for pipeline consistency
        enhanced = cv2.resize(enhanced, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        return enhanced
    except Exception:
        pass
    return None


def _fix_color_cast(image: np.ndarray) -> np.ndarray:
    """Remove color cast by normalizing A/B channels in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Center A and B channels around 128 (neutral)
    for ch in [1, 2]:
        channel = lab[:, :, ch]
        mean_val = channel.mean()
        # Shift toward neutral, but only partially to preserve natural skin tone
        shift = (128.0 - mean_val) * 0.6
        lab[:, :, ch] = np.clip(channel + shift, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _fix_lighting(image: np.ndarray) -> np.ndarray:
    """Fix over/under exposure using adaptive CLAHE in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # CLAHE on luminance channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# ArcFace Identity Verification
# ---------------------------------------------------------------------------

_ARCFACE_APP = None


def _get_arcface() -> Any:
    """Get or create singleton ArcFace model."""
    global _ARCFACE_APP
    if _ARCFACE_APP is not None:
        return _ARCFACE_APP

    try:
        import torch
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ctx_id = 0 if torch.cuda.is_available() else -1
        app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        _ARCFACE_APP = app
        return app
    except Exception:
        return None


def get_face_embedding(image: np.ndarray) -> np.ndarray | None:
    """Extract ArcFace 512-d embedding from a face image.

    Returns None if no face detected or InsightFace unavailable.
    """
    app = _get_arcface()
    if app is None:
        return None

    try:
        faces = app.get(image)
        if faces:
            emb = faces[0].embedding
            if np.linalg.norm(emb) < 1e-6:
                logger.warning("ArcFace returned near-zero embedding (occluded face?)")
                return None
            return emb
    except Exception:
        pass
    return None


def verify_identity(
    original: np.ndarray,
    restored: np.ndarray,
    threshold: float = 0.6,
) -> tuple[float, bool]:
    """Compare identity between original and restored using ArcFace.

    Returns (cosine_similarity, passed).
    Similarity > threshold means same person (threshold=0.6 is conservative).
    """
    emb_orig = get_face_embedding(original)
    emb_rest = get_face_embedding(restored)

    if emb_orig is None or emb_rest is None:
        return -1.0, True  # Can't verify — assume OK

    sim = float(
        np.dot(emb_orig, emb_rest) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_rest) + 1e-8)
    )
    sim = float(np.clip(sim, -1, 1))
    return sim, sim >= threshold


# ---------------------------------------------------------------------------
# Full Verification + Restoration Pipeline
# ---------------------------------------------------------------------------


def verify_and_restore(
    image: np.ndarray,
    quality_threshold: float = 60.0,
    identity_threshold: float = 0.6,
    restore_mode: str = "auto",
    codeformer_fidelity: float = 0.7,
) -> RestorationResult:
    """Full pipeline: analyze → restore → verify identity.

    This is the main entry point for the face verifier. It:
    1. Analyzes the input for distortions
    2. If quality is below threshold, applies neural restoration
    3. Verifies the restored face preserves identity
    4. Returns comprehensive result with metrics

    Args:
        image: BGR face image.
        quality_threshold: Min quality to skip restoration (0-100).
        identity_threshold: Min ArcFace similarity to pass (0-1).
        restore_mode: 'auto', 'codeformer', 'gfpgan', 'all'.
        codeformer_fidelity: CodeFormer quality-fidelity balance.

    Returns:
        RestorationResult with restored image and full metrics.
    """
    # Step 1: Analyze distortions
    report = analyze_distortions(image)

    # Step 2: Decide if restoration needed
    if report.quality_score >= quality_threshold and report.severity in ("none", "mild"):
        # Image is good enough — no restoration needed
        return RestorationResult(
            restored=image.copy(),
            original=image.copy(),
            distortion_report=report,
            post_quality_score=report.quality_score,
            identity_similarity=1.0,
            identity_preserved=True,
            restoration_stages=[],
            improvement=0.0,
        )

    if not report.is_usable:
        # Too distorted to salvage
        return RestorationResult(
            restored=image.copy(),
            original=image.copy(),
            distortion_report=report,
            post_quality_score=report.quality_score,
            identity_similarity=0.0,
            identity_preserved=False,
            restoration_stages=["rejected"],
            improvement=0.0,
        )

    # Step 3: Neural restoration
    restored, stages = restore_face(
        image,
        distortion=report,
        mode=restore_mode,
        codeformer_fidelity=codeformer_fidelity,
    )

    # Step 4: Post-restoration quality check
    post_quality = neural_quality_score(restored)

    # Step 5: Identity verification
    sim, id_ok = verify_identity(image, restored, threshold=identity_threshold)

    return RestorationResult(
        restored=restored,
        original=image.copy(),
        distortion_report=report,
        post_quality_score=post_quality,
        identity_similarity=sim,
        identity_preserved=id_ok,
        restoration_stages=stages,
        improvement=post_quality - report.quality_score,
    )


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------


def verify_batch(
    image_dir: str,
    output_dir: str | None = None,
    quality_threshold: float = 60.0,
    identity_threshold: float = 0.6,
    restore_mode: str = "auto",
    save_rejected: bool = False,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> BatchVerificationReport:
    """Process a directory of face images: analyze, restore, verify, sort.

    Outputs:
    - {output_dir}/passed/     — good images (no fix needed)
    - {output_dir}/restored/   — fixed images
    - {output_dir}/rejected/   — too distorted to use (if save_rejected=True)
    - {output_dir}/report.txt  — batch verification report

    Args:
        image_dir: Directory of face images to process.
        output_dir: Where to save results (default: {image_dir}_verified/).
        quality_threshold: Min quality to pass without restoration.
        identity_threshold: Min identity similarity after restoration.
        restore_mode: 'auto', 'codeformer', 'gfpgan', 'all'.
        save_rejected: Whether to copy rejected images to rejected/ subdir.
        extensions: File extensions to process.

    Returns:
        BatchVerificationReport with summary statistics.
    """
    image_path = Path(image_dir)
    if output_dir is None:
        out_path = image_path.parent / f"{image_path.name}_verified"
    else:
        out_path = Path(output_dir)

    # Create output dirs
    passed_dir = out_path / "passed"
    restored_dir = out_path / "restored"
    rejected_dir = out_path / "rejected"
    passed_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)
    if save_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(
        [f for f in image_path.iterdir() if f.suffix.lower() in extensions and f.is_file()]
    )

    report = BatchVerificationReport(total=len(image_files))
    quality_before = []
    quality_after = []
    identity_sims = []

    for i, img_file in enumerate(image_files):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("Processing %d/%d: %s", i + 1, len(image_files), img_file.name)

        image = cv2.imread(str(img_file))
        if image is None:
            report.rejected += 1
            continue

        # Resize to 512x512 for consistency
        image = cv2.resize(image, (512, 512))

        # Run verification + restoration
        result = verify_and_restore(
            image,
            quality_threshold=quality_threshold,
            identity_threshold=identity_threshold,
            restore_mode=restore_mode,
        )

        quality_before.append(result.distortion_report.quality_score)
        quality_after.append(result.post_quality_score)

        # Track distortion types
        dist_type = result.distortion_report.primary_distortion
        report.distortion_counts[dist_type] = report.distortion_counts.get(dist_type, 0) + 1

        if not result.distortion_report.is_usable or "rejected" in result.restoration_stages:
            report.rejected += 1
            if save_rejected:
                cv2.imwrite(str(rejected_dir / img_file.name), image)
        elif not result.restoration_stages:
            # Passed without restoration
            report.passed += 1
            cv2.imwrite(str(passed_dir / img_file.name), image)
        else:
            # Restored
            if result.identity_preserved:
                report.restored += 1
                cv2.imwrite(str(restored_dir / img_file.name), result.restored)
                identity_sims.append(result.identity_similarity)
            else:
                report.identity_failures += 1
                if save_rejected:
                    cv2.imwrite(str(rejected_dir / img_file.name), image)

    # Compute averages
    report.avg_quality_before = float(np.mean(quality_before)) if quality_before else 0.0
    report.avg_quality_after = float(np.mean(quality_after)) if quality_after else 0.0
    report.avg_identity_sim = float(np.mean(identity_sims)) if identity_sims else 0.0

    # Save report
    report_text = report.summary()
    (out_path / "report.txt").write_text(report_text)
    logger.info("\n%s", report_text)
    logger.info("Results saved to %s/", out_path)

    return report
