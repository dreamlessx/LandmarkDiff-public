"""Training data augmentation pipeline for LandmarkDiff.

Provides domain-specific augmentations that maintain landmark consistency:
- Geometric: flip, rotation, affine (landmarks co-transformed)
- Photometric: color jitter, brightness, contrast (applied to images only)
- Skin-tone augmentation: ITA-space perturbation for Fitzpatrick balance
- Conditioning augmentation: noise injection, dropout for robustness

All augmentations preserve the correspondence between:
  input_image ↔ conditioning_image ↔ target_image ↔ mask
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentationConfig:
    """Augmentation parameters."""

    # Geometric
    random_flip: bool = True
    random_rotation_deg: float = 5.0
    random_scale: tuple[float, float] = (0.95, 1.05)
    random_translate: float = 0.02  # fraction of image size

    # Photometric (images only, not conditioning)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    saturation_range: tuple[float, float] = (0.9, 1.1)
    hue_shift_range: float = 5.0  # degrees

    # Conditioning augmentation
    conditioning_dropout_prob: float = 0.1
    conditioning_noise_std: float = 0.02

    # Skin-tone augmentation
    ita_perturbation_std: float = 3.0  # ITA angle noise

    seed: int | None = None


def augment_training_sample(
    input_image: np.ndarray,
    target_image: np.ndarray,
    conditioning: np.ndarray,
    mask: np.ndarray,
    landmarks_src: np.ndarray | None = None,
    landmarks_dst: np.ndarray | None = None,
    config: AugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Apply consistent augmentations to a training sample.

    All spatial transforms are applied to images AND landmarks together
    so correspondence is preserved.

    Args:
        input_image: (H, W, 3) original face image (uint8 BGR).
        target_image: (H, W, 3) target face image (uint8 BGR).
        conditioning: (H, W, 3) conditioning image (uint8).
        mask: (H, W) or (H, W, 1) float32 mask.
        landmarks_src: (N, 2) normalized [0,1] source landmark coords.
        landmarks_dst: (N, 2) normalized [0,1] target landmark coords.
        config: Augmentation parameters.
        rng: Random generator for reproducibility.

    Returns:
        Dict with augmented versions of all inputs.
    """
    if config is None:
        config = AugmentationConfig()
    if rng is None:
        rng = np.random.default_rng(config.seed)

    h, w = input_image.shape[:2]
    out_input = input_image.copy()
    out_target = target_image.copy()
    out_cond = conditioning.copy()
    out_mask = mask.copy()
    out_lm_src = landmarks_src.copy() if landmarks_src is not None else None
    out_lm_dst = landmarks_dst.copy() if landmarks_dst is not None else None

    # --- Geometric augmentations (applied to all) ---

    # Random horizontal flip
    if config.random_flip and rng.random() < 0.5:
        out_input = np.ascontiguousarray(out_input[:, ::-1])
        out_target = np.ascontiguousarray(out_target[:, ::-1])
        out_cond = np.ascontiguousarray(out_cond[:, ::-1])
        out_mask = np.ascontiguousarray(
            out_mask[:, ::-1] if out_mask.ndim == 2 else out_mask[:, ::-1, :]
        )
        if out_lm_src is not None:
            out_lm_src[:, 0] = 1.0 - out_lm_src[:, 0]
        if out_lm_dst is not None:
            out_lm_dst[:, 0] = 1.0 - out_lm_dst[:, 0]

    # Random rotation + scale + translate
    if config.random_rotation_deg > 0 or config.random_scale != (1.0, 1.0):
        angle = rng.uniform(-config.random_rotation_deg, config.random_rotation_deg)
        scale = rng.uniform(config.random_scale[0], config.random_scale[1])
        tx = rng.uniform(-config.random_translate, config.random_translate) * w
        ty = rng.uniform(-config.random_translate, config.random_translate) * h

        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        out_input = cv2.warpAffine(out_input, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        out_target = cv2.warpAffine(out_target, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        out_cond = cv2.warpAffine(
            out_cond, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mask_2d = out_mask if out_mask.ndim == 2 else out_mask[:, :, 0]
        mask_2d = cv2.warpAffine(mask_2d, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out_mask = mask_2d if out_mask.ndim == 2 else mask_2d[:, :, np.newaxis]

        # Transform landmarks
        if out_lm_src is not None:
            out_lm_src = _transform_landmarks(out_lm_src, M, w, h)
        if out_lm_dst is not None:
            out_lm_dst = _transform_landmarks(out_lm_dst, M, w, h)

    # --- Photometric augmentations (images only, not conditioning/mask) ---

    # Brightness
    b_factor = rng.uniform(config.brightness_range[0], config.brightness_range[1])
    out_input = np.clip(out_input.astype(np.float32) * b_factor, 0, 255).astype(np.uint8)
    out_target = np.clip(out_target.astype(np.float32) * b_factor, 0, 255).astype(np.uint8)

    # Contrast
    c_factor = rng.uniform(config.contrast_range[0], config.contrast_range[1])
    mean_in = out_input.mean()
    mean_tgt = out_target.mean()
    out_input = np.clip(
        (out_input.astype(np.float32) - mean_in) * c_factor + mean_in, 0, 255
    ).astype(np.uint8)
    out_target = np.clip(
        (out_target.astype(np.float32) - mean_tgt) * c_factor + mean_tgt, 0, 255
    ).astype(np.uint8)

    # Saturation (in HSV space)
    s_factor = rng.uniform(config.saturation_range[0], config.saturation_range[1])
    if abs(s_factor - 1.0) > 1e-4:
        out_input = _adjust_saturation(out_input, s_factor)
        out_target = _adjust_saturation(out_target, s_factor)

    # Hue shift
    if config.hue_shift_range > 0:
        hue_delta = rng.uniform(-config.hue_shift_range, config.hue_shift_range)
        if abs(hue_delta) > 0.1:
            out_input = _shift_hue(out_input, hue_delta)
            out_target = _shift_hue(out_target, hue_delta)

    # --- Conditioning augmentation ---

    # Conditioning dropout (replace with zeros to learn unconditional)
    if config.conditioning_dropout_prob > 0 and rng.random() < config.conditioning_dropout_prob:
        out_cond = np.zeros_like(out_cond)

    # Conditioning noise
    if config.conditioning_noise_std > 0:
        noise = rng.normal(0, config.conditioning_noise_std * 255, out_cond.shape)
        out_cond = np.clip(out_cond.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    result = {
        "input_image": out_input,
        "target_image": out_target,
        "conditioning": out_cond,
        "mask": out_mask,
    }
    if out_lm_src is not None:
        result["landmarks_src"] = out_lm_src
    if out_lm_dst is not None:
        result["landmarks_dst"] = out_lm_dst

    return result


def _transform_landmarks(landmarks: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    """Transform normalized landmarks with an affine matrix."""
    # Convert to pixel coords
    px = landmarks.copy()
    px[:, 0] *= w
    px[:, 1] *= h

    # Apply affine transform
    ones = np.ones((px.shape[0], 1))
    px_h = np.hstack([px, ones])  # (N, 3)
    transformed = (M @ px_h.T).T  # (N, 2)

    # Back to normalized
    transformed[:, 0] /= w
    transformed[:, 1] /= h
    return np.clip(transformed, 0.0, 1.0)


def _adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust saturation of a BGR image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _shift_hue(img: np.ndarray, delta_deg: float) -> np.ndarray:
    """Shift hue of a BGR image by delta degrees."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV hue range is [0, 180]
    hsv[:, :, 0] = (hsv[:, :, 0] + delta_deg / 2) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_skin_tone(
    image: np.ndarray,
    ita_delta: float = 0.0,
) -> np.ndarray:
    """Augment skin tone by shifting in L*a*b* space.

    This helps balance Fitzpatrick representation in training by
    simulating different skin tones from existing samples.

    Args:
        image: (H, W, 3) BGR uint8 image.
        ita_delta: ITA angle shift (positive = lighter, negative = darker).

    Returns:
        Augmented image with shifted skin tone.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Shift L channel (lightness) based on ITA delta
    # ITA = arctan((L-50)/b), so shifting ITA shifts L
    l_shift = ita_delta * 0.5  # approximate mapping
    lab[:, :, 0] = np.clip(lab[:, :, 0] + l_shift, 0, 255)

    # Slightly shift b channel too for more natural tone changes
    b_shift = -ita_delta * 0.15
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


class FitzpatrickBalancer:
    """Oversample underrepresented Fitzpatrick types during training.

    Maintains per-type counts and generates sampling weights to ensure
    equitable training across all skin types.
    """

    def __init__(self, target_distribution: dict[str, float] | None = None):
        """Initialize balancer.

        Args:
            target_distribution: Target fraction per type. Defaults to uniform.
        """
        self.target = target_distribution or {
            "I": 1 / 6,
            "II": 1 / 6,
            "III": 1 / 6,
            "IV": 1 / 6,
            "V": 1 / 6,
            "VI": 1 / 6,
        }
        self._counts: dict[str, int] = {}

    def register_sample(self, fitz_type: str) -> None:
        """Register a sample's Fitzpatrick type."""
        self._counts[fitz_type] = self._counts.get(fitz_type, 0) + 1

    def get_sampling_weights(self, fitz_types: list[str]) -> np.ndarray:
        """Compute sampling weights for a list of samples.

        Returns weights inversely proportional to type frequency,
        so underrepresented types get upsampled.
        """
        total = sum(self._counts.values()) or 1
        weights = []
        for ft in fitz_types:
            count = self._counts.get(ft, 1)
            freq = count / total
            target_freq = self.target.get(ft, 1 / 6)
            # Weight = target / actual (capped for stability)
            w = min(target_freq / max(freq, 1e-6), 5.0)
            weights.append(w)

        w = np.array(weights, dtype=np.float64)
        return w / w.sum()  # normalize to probability distribution
