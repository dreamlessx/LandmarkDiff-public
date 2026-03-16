"""Generate training pairs using real surgical displacement statistics.

Replaces hand-tuned RBF manipulations with data-driven displacements learned
from 3,544 real before/after surgery image pairs. Each generated pair applies
a displacement field sampled from the fitted per-procedure distribution.

The displacement model captures:
- Per-landmark mean and variance for each procedure
- Anatomically correct deformation patterns (e.g., rhinoplasty moves nose,
  not jaw; orthognathic moves chin, not eyes)
- Realistic magnitude ranges (observed ~1-2% of face width, vs. our synthetic
  presets which were ~3-5%)

Usage:
    # Generate 10K pairs per procedure from CelebA
    python scripts/generate_realistic_pairs.py \
        --source_dir data/celeba_hq_extracted \
        --displacement_model data/displacement_model.npz \
        --output_dir data/synthetic_surgery_pairs_v3 \
        --pairs_per_procedure 10000

    # Use custom intensity range
    python scripts/generate_realistic_pairs.py \
        --source_dir data/celeba_hq_extracted \
        --displacement_model data/displacement_model.npz \
        --output_dir data/synthetic_surgery_pairs_v3 \
        --intensity_min 0.5 --intensity_max 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.displacement_model import DisplacementModel
from landmarkdiff.landmarks import FaceLandmarks, extract_landmarks, render_landmark_image
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps


def apply_displacement_field(
    face: FaceLandmarks,
    displacement: np.ndarray,
    image_size: int = 512,
) -> FaceLandmarks:
    """Apply a displacement field to face landmarks.

    Args:
        face: Original face landmarks.
        displacement: (478, 2) displacement in normalized [0, 1] space.
        image_size: Image dimension for pixel coordinate update.

    Returns:
        New FaceLandmarks with displaced positions.
    """
    new_landmarks = face.landmarks.copy()
    n = min(len(new_landmarks), len(displacement))
    new_landmarks[:n, 0] += displacement[:n, 0]
    new_landmarks[:n, 1] += displacement[:n, 1]

    # Clamp to valid range
    new_landmarks[:, 0] = np.clip(new_landmarks[:, 0], 0.01, 0.99)
    new_landmarks[:, 1] = np.clip(new_landmarks[:, 1], 0.01, 0.99)

    return FaceLandmarks(
        landmarks=new_landmarks,
        image_width=image_size,
        image_height=image_size,
        confidence=face.confidence,
    )


def generate_pair(
    image: np.ndarray,
    face: FaceLandmarks,
    model: DisplacementModel,
    procedure: str,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.5, 1.5),
    noise_scale: float = 0.5,
    size: int = 512,
) -> dict | None:
    """Generate a single training pair using the displacement model.

    Applies a sampled displacement field to landmarks, generates conditioning,
    and creates the TPS-warped target.

    Returns:
        Dict with input, target, conditioning, mask arrays, or None on failure.
    """
    # Sample intensity uniformly from range
    intensity = rng.uniform(*intensity_range)

    # Generate displacement field
    displacement = model.get_displacement_field(
        procedure,
        intensity=intensity,
        noise_scale=noise_scale,
        rng=rng,
    )

    # Apply displacement
    manipulated = apply_displacement_field(face, displacement, size)

    # Generate conditioning image (3-channel landmark mesh)
    try:
        cond_result = generate_conditioning(manipulated, size, size)
        # generate_conditioning returns a tuple of (landmarks, edges, wireframe)
        if isinstance(cond_result, tuple):
            conditioning = cond_result[0]  # use landmark rendering
        else:
            conditioning = cond_result
    except Exception:
        conditioning = render_landmark_image(manipulated, size, size)

    # Generate surgical mask
    try:
        mask = generate_surgical_mask(face, procedure, size, size)
    except Exception:
        mask = np.ones((size, size), dtype=np.float32)

    # TPS warp to create target
    try:
        target = warp_image_tps(
            image,
            face.pixel_coords,
            manipulated.pixel_coords,
        )
    except Exception:
        return None

    return {
        "input": image,
        "target": target,
        "conditioning": conditioning,
        "mask": mask,
        "procedure": procedure,
        "intensity": intensity,
    }


def generate_pairs(
    source_dir: str,
    displacement_model_path: str,
    output_dir: str,
    pairs_per_procedure: int = 5000,
    intensity_min: float = 0.5,
    intensity_max: float = 1.5,
    noise_scale: float = 0.5,
    seed: int = 42,
    max_source_images: int = 0,
) -> None:
    """Generate training pairs using real displacement model."""

    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load displacement model
    print(f"Loading displacement model from {displacement_model_path}...")
    model = DisplacementModel.load(displacement_model_path)
    print(f"  Procedures: {model.procedures}")
    print(f"  Samples: {model.n_samples}")

    procedures = model.procedures
    intensity_range = (intensity_min, intensity_max)

    # Find source images
    extensions = {".png", ".jpg", ".jpeg"}
    source_images = sorted(
        [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    )
    if max_source_images > 0:
        source_images = source_images[:max_source_images]

    if not source_images:
        print(f"No source images found in {src}")
        return

    print(f"Source images: {len(source_images)}")
    print(f"Procedures: {procedures}")
    print(f"Pairs per procedure: {pairs_per_procedure}")
    print(f"Intensity range: [{intensity_min}, {intensity_max}]")
    print(f"Total pairs expected: {len(procedures) * pairs_per_procedure}")

    rng = np.random.default_rng(seed)

    # Pre-extract landmarks only (NOT images) to save memory.
    # Images are reloaded on-demand during generation.
    print("\nExtracting landmarks from source images...")
    face_index: list[tuple[Path, FaceLandmarks]] = []
    for i, img_path in enumerate(source_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))
        face = extract_landmarks(img)
        if face is not None:
            face_index.append((img_path, face))
        # Free image memory immediately
        del img
        if (i + 1) % 500 == 0:
            print(f"  Extracted {len(face_index)}/{i + 1} faces...")

    print(f"  Indexed {len(face_index)} faces from {len(source_images)} images")

    if not face_index:
        print("ERROR: No faces detected in source images")
        return

    # Generate pairs
    t0 = time.time()
    total_generated = 0
    metadata = {}
    stats = {proc: 0 for proc in procedures}

    for proc in procedures:
        proc_dir = out / proc
        proc_dir.mkdir(exist_ok=True)
        print(f"\nGenerating {proc} pairs...")

        generated = 0
        attempts = 0
        max_attempts = pairs_per_procedure * 3  # Allow 3x retry budget

        while generated < pairs_per_procedure and attempts < max_attempts:
            # Pick random source image -- reload from disk (memory-efficient)
            idx = rng.integers(len(face_index))
            img_path, face = face_index[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                attempts += 1
                continue
            img = cv2.resize(img, (512, 512))
            attempts += 1

            result = generate_pair(
                img,
                face,
                model,
                proc,
                rng,
                intensity_range=intensity_range,
                noise_scale=noise_scale,
            )

            if result is None:
                continue

            # Save
            prefix = f"{proc}_{generated:06d}"
            cv2.imwrite(str(out / f"{prefix}_input.png"), result["input"])
            cv2.imwrite(str(out / f"{prefix}_target.png"), result["target"])
            cv2.imwrite(str(out / f"{prefix}_conditioning.png"), result["conditioning"])

            mask_uint8 = (result["mask"] * 255).astype(np.uint8)
            cv2.imwrite(str(out / f"{prefix}_mask.png"), mask_uint8)

            metadata[prefix] = {
                "procedure": proc,
                "source": "synthetic_v3",
                "wave": 3,
                "intensity": float(result["intensity"]),
            }

            generated += 1
            stats[proc] = generated

            if generated % 500 == 0:
                elapsed = time.time() - t0
                total = sum(stats.values())
                rate = total / max(elapsed, 1)
                print(
                    f"  {proc}: {generated}/{pairs_per_procedure} "
                    f"({rate:.1f} pairs/s, {elapsed:.0f}s elapsed)"
                )

        total_generated += generated
        print(f"  {proc}: {generated} pairs generated ({attempts} attempts)")

    # Save metadata
    meta_path = out / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "stats": stats,
                "total": total_generated,
                "model_path": str(displacement_model_path),
                "intensity_range": list(intensity_range),
                "noise_scale": noise_scale,
                "seed": seed,
                "pairs": metadata,
            },
            f,
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total pairs: {total_generated}")
    print(f"  Time: {elapsed:.0f}s ({total_generated / max(elapsed, 1):.1f} pairs/s)")
    print(f"  Per procedure: {stats}")
    print(f"  Output: {out}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training pairs with real displacement model"
    )
    parser.add_argument("--source_dir", required=True, help="Directory of source face images")
    parser.add_argument(
        "--displacement_model", required=True, help="Path to displacement_model.npz"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for generated pairs")
    parser.add_argument(
        "--pairs_per_procedure", type=int, default=5000, help="Number of pairs per procedure"
    )
    parser.add_argument(
        "--intensity_min", type=float, default=0.5, help="Min intensity scale factor"
    )
    parser.add_argument(
        "--intensity_max", type=float, default=1.5, help="Max intensity scale factor"
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.5, help="Noise scale for displacement sampling"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_source_images", type=int, default=0, help="Max source images to use (0 = all)"
    )
    args = parser.parse_args()

    generate_pairs(
        args.source_dir,
        args.displacement_model,
        args.output_dir,
        args.pairs_per_procedure,
        args.intensity_min,
        args.intensity_max,
        args.noise_scale,
        args.seed,
        args.max_source_images,
    )
