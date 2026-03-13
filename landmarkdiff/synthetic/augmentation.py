"""Clinical degradation augmentation pipeline.

Degrades clean FFHQ/CelebA-HQ images to match real clinical photo distribution.
Applied from day 1 of training — domain gap prevention, not afterthought.

Each sample gets 3-5 random augmentations from the pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for a single augmentation."""

    name: str
    fn: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    probability: float


def point_source_lighting(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate point-source clinical lighting from a random direction."""
    h, w = image.shape[:2]

    # Random light source position
    lx = rng.uniform(0, w)
    ly = rng.uniform(0, h)
    intensity = rng.uniform(0.3, 0.7)

    # Distance-based falloff
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((x_grid - lx) ** 2 + (y_grid - ly) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2)
    light_map = 1.0 - (dist / max_dist) * intensity

    light_map = np.clip(light_map, 0.3, 1.0)
    light_3ch = np.stack([light_map] * 3, axis=-1)

    return np.clip(image.astype(np.float32) * light_3ch, 0, 255).astype(np.uint8)


def color_temperature_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jitter color temperature +/- 2000K equivalent."""
    shift = rng.uniform(-0.15, 0.15)

    result = image.astype(np.float32)
    if shift > 0:
        # Warmer: boost red, reduce blue
        result[:, :, 2] *= 1 + shift  # red (BGR)
        result[:, :, 0] *= 1 - shift * 0.5  # blue
    else:
        # Cooler: boost blue, reduce red
        result[:, :, 0] *= 1 + abs(shift)
        result[:, :, 2] *= 1 - abs(shift) * 0.5

    return np.clip(result, 0, 255).astype(np.uint8)


def green_fluorescent_cast(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add green fluorescent lighting cast (common in clinical settings)."""
    intensity = rng.uniform(0.05, 0.15)
    result = image.astype(np.float32)
    result[:, :, 1] *= 1 + intensity  # green channel boost
    result[:, :, 0] *= 1 - intensity * 0.3  # slight blue reduction
    result[:, :, 2] *= 1 - intensity * 0.3  # slight red reduction
    return np.clip(result, 0, 255).astype(np.uint8)


def jpeg_compression(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate JPEG compression artifacts (quality 40-85)."""
    quality = int(rng.uniform(40, 85))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def gaussian_sensor_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian sensor noise (sigma 5-25)."""
    sigma = rng.uniform(5, 25)
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def barrel_distortion(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply barrel/pincushion distortion simulating phone camera lens."""
    h, w = image.shape[:2]
    k1 = rng.uniform(-0.2, 0.2)

    fx = fy = max(w, h)
    cx, cy = w / 2, h / 2

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def motion_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slight motion blur (common in handheld clinical photos)."""
    size = int(rng.uniform(3, 7))
    angle = rng.uniform(0, 180)

    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size

    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def vignette(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add lens vignetting (darkened corners)."""
    h, w = image.shape[:2]
    strength = rng.uniform(0.3, 0.7)

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)

    mask = 1 - strength * (dist / max_dist) ** 2
    mask = np.clip(mask, 0.3, 1.0)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    return np.clip(image.astype(np.float32) * mask_3ch, 0, 255).astype(np.uint8)


# Augmentation pool with probabilities from the spec
AUGMENTATION_POOL: list[AugmentationConfig] = [
    AugmentationConfig("point_source_lighting", point_source_lighting, 0.40),
    AugmentationConfig("color_temperature", color_temperature_jitter, 0.60),
    AugmentationConfig("green_fluorescent", green_fluorescent_cast, 0.25),
    AugmentationConfig("jpeg_compression", jpeg_compression, 0.30),
    AugmentationConfig("sensor_noise", gaussian_sensor_noise, 0.40),
    AugmentationConfig("barrel_distortion", barrel_distortion, 0.30),
    AugmentationConfig("motion_blur", motion_blur, 0.20),
    AugmentationConfig("vignette", vignette, 0.25),
]


def apply_clinical_augmentation(
    image: np.ndarray,
    min_augmentations: int = 3,
    max_augmentations: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random clinical degradation augmentations to an image.

    Each sample gets min_augmentations to max_augmentations from the pool,
    selected by their individual probabilities.

    Args:
        image: BGR input image (clean FFHQ/CelebA-HQ).
        min_augmentations: Minimum number of augmentations to apply.
        max_augmentations: Maximum number of augmentations to apply.
        rng: Random number generator.

    Returns:
        Degraded image matching clinical photo distribution.
    """
    rng = rng or np.random.default_rng()

    # Select augmentations by probability
    selected = []
    for aug in AUGMENTATION_POOL:
        if rng.random() < aug.probability:
            selected.append(aug)

    # Ensure min/max bounds
    if len(selected) < min_augmentations:
        remaining = [a for a in AUGMENTATION_POOL if a not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: min_augmentations - len(selected)])

    if len(selected) > max_augmentations:
        rng.shuffle(selected)
        selected = selected[:max_augmentations]

    # Apply in random order
    rng.shuffle(selected)
    result = image.copy()
    for aug in selected:
        result = aug.fn(result, rng)

    return result
