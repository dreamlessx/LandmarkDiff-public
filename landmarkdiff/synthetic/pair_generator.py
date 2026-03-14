"""Synthetic pair generator for ControlNet fine-tuning.

FFHQ -> landmarks -> random FFD -> conditioning + mask -> augment input.
Augmentations on INPUT only, never target.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image
from landmarkdiff.manipulation import (
    apply_procedure_preset,
)
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.augmentation import apply_clinical_augmentation
from landmarkdiff.synthetic.tps_warp import warp_image_tps


@dataclass(frozen=True)
class TrainingPair:
    """A single training sample for ControlNet fine-tuning."""

    input_image: np.ndarray  # augmented input (512x512 BGR)
    target_image: np.ndarray  # clean target (512x512 BGR) - TPS-warped original
    conditioning: np.ndarray  # landmark rendering (512x512 BGR)
    canny: np.ndarray  # canny edge map (512x512 grayscale)
    mask: np.ndarray  # feathered surgical mask (512x512 float32)
    procedure: str
    intensity: float


PROCEDURES = ["rhinoplasty", "blepharoplasty", "rhytidectomy", "orthognathic"]


def generate_pair(
    image: np.ndarray,
    procedure: str | None = None,
    intensity: float | None = None,
    target_size: int = 512,
    rng: np.random.Generator | None = None,
) -> TrainingPair | None:
    """Generate a single training pair from a face image."""
    rng = rng or np.random.default_rng()

    # Resize to target
    resized = cv2.resize(image, (target_size, target_size))

    # Extract landmarks
    face = extract_landmarks(resized)
    if face is None:
        return None

    # Random procedure and intensity if not specified
    if procedure is None:
        procedure = rng.choice(PROCEDURES)
    if intensity is None:
        intensity = float(rng.uniform(30, 90))

    # Manipulate landmarks
    manipulated = apply_procedure_preset(face, procedure, intensity, target_size)

    # Generate conditioning from manipulated landmarks
    landmark_img = render_landmark_image(manipulated, target_size, target_size)
    _, canny, _ = generate_conditioning(manipulated, target_size, target_size)

    # Generate mask
    mask = generate_surgical_mask(face, procedure, target_size, target_size)

    # Generate target: TPS warp the original image to match manipulated landmarks
    src_px = face.pixel_coords
    dst_px = manipulated.pixel_coords
    target = warp_image_tps(resized, src_px, dst_px)

    # Apply clinical augmentation to INPUT only (never target)
    augmented_input = apply_clinical_augmentation(resized, rng=rng)

    return TrainingPair(
        input_image=augmented_input,
        target_image=target,
        conditioning=landmark_img,
        canny=canny,
        mask=mask,
        procedure=procedure,
        intensity=intensity,
    )


def generate_pairs_from_directory(
    image_dir: str | Path,
    num_pairs: int = 1000,
    target_size: int = 512,
    seed: int = 42,
) -> Iterator[TrainingPair]:
    """Generate training pairs from a directory of face images."""
    rng = np.random.default_rng(seed)
    image_dir = Path(image_dir)

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in extensions)

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    generated = 0
    consecutive_failures = 0
    idx = 0
    while generated < num_pairs:
        # Cycle through images
        img_path = image_files[idx % len(image_files)]
        idx += 1
        image = cv2.imread(str(img_path))
        if image is None:
            consecutive_failures += 1
            if consecutive_failures > len(image_files):
                print(f"Warning: {consecutive_failures} consecutive failures, stopping early")
                break
            continue

        pair = generate_pair(image, target_size=target_size, rng=rng)
        if pair is not None:
            yield pair
            generated += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures > len(image_files):
                print(f"Warning: {consecutive_failures} consecutive failures, stopping early")
                break


def save_pair(pair: TrainingPair, output_dir: Path, index: int) -> None:
    """Save a training pair to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{index:06d}"

    cv2.imwrite(str(output_dir / f"{prefix}_input.png"), pair.input_image)
    cv2.imwrite(str(output_dir / f"{prefix}_target.png"), pair.target_image)
    cv2.imwrite(str(output_dir / f"{prefix}_conditioning.png"), pair.conditioning)
    cv2.imwrite(str(output_dir / f"{prefix}_canny.png"), pair.canny)
    cv2.imwrite(str(output_dir / f"{prefix}_mask.png"), (pair.mask * 255).astype(np.uint8))
